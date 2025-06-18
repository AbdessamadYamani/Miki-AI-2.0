from typing import List, Dict, Optional, Tuple, Union, Any, Generator
import os
import logging,re
import google.generativeai as genai
from PIL import  Image
from tools.token_usage_tool import _get_token_usage # type: ignore
from vision.vis import _hash_pil_image,capture_full_screen # Keep these from vision.vis
from utils.image_utils import image_to_base64 # Import this from the new utility file
import json
import demjson3
from agents.ai_agent import UIAgent
from tools.shortcuts_tool import load_shortcuts_cache
from tools.web_search_tool import search_web_for_info
from datetime import datetime 
from task_exec.tasks_management import save_task_execution_to_db, find_similar_user_task_structure
from task_exec.tasks_management import _analyze_and_categorize_task
from utils.app_name import get_base_app_name
from chromaDB_management.cache import sanitize_filename,get_active_window_name
from tools.shortcuts_tool import load_shortcuts_cache, get_application_shortcuts # Import shortcuts tool functions
from task_exec.tasks_management import retrieve_similar_task_executions_from_db # Added for plan adaptation
from task_exec.task_planner import critique_action # Assuming process_next_step is also in task_planner or imported elsewhere
from tools.actions import execute_action
import time # Removed datetime from here
from chromaDB_management.credential import save_credential
from utils.reinforcement_util import retrieve_relevant_reinforcements_from_db
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ( # type: ignore
    get_model, get_critic_model, DEBUG_DIR,BLUE, CONFIRM_RISKY_COMMANDS
)
from tools.files_upload import process_files_from_urls

# Add this near the top of the file or where VALID_ACTIONS or similar is defined
VALID_ACTIONS = {
    "focus_window", "click", "type", "press_keys", "move_mouse", "run_shell_command", "run_python_script", "write_file", "navigate_web", "search_web", "search_youtube", "wait", "ask_user", "describe_screen", "capture_screenshot", "click_and_type", "multi_action", "task_complete", "read_file", "INFORM_USER", "process_local_files", "process_files_from_urls", "start_visual_listener", "edit_image_with_file", "refresh_application_shortcuts"
}

def _normalize_instruction(instruction: str) -> str:
    """Converts instruction to lowercase and removes leading/trailing whitespace."""
    if not isinstance(instruction, str): return ""
    return instruction.lower().strip()



def assess_action_outcome(
    original_instruction: str,
    action: dict,
    exec_success: bool,
    exec_message: str,
    screenshot_after: Optional[Image.Image],
    llm_model: genai.GenerativeModel,
    screenshot_before_hash: Optional[str] = None
) -> Tuple[str, str, Dict[str, int]]:
    """Assess the outcome of an action."""
    action_type = action.get("action_type", "unknown")
    logging.info(f"Assessing outcome for action: {action_type}")
    logging.debug(f"Execution Result: Success={exec_success}, Message='{exec_message}'")

    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    # Handle file processing actions first
    if action_type in ["process_local_files", "process_files_from_urls"]:
        if not exec_success:
            return "FAILURE", f"File processing failed: {exec_message}", token_usage
            
        # Handle both string and dictionary results
        if isinstance(exec_message, str):
            return "SUCCESS", exec_message, token_usage
        elif isinstance(exec_message, dict):
            # Extract the result from the dictionary
            result = exec_message.get('result', '')
            source = exec_message.get('source', '')
            url = exec_message.get('url', '')
            files = exec_message.get('files', [])
            
            # Format the message in a structured way
            message = "📄 File Processing Results\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            if source == "local":
                message += f"📂 Source: Local Files\n"
                if files:
                    message += f"📋 Files: {', '.join(files)}\n"
            elif source == "url":
                message += f"🌐 Source: URL\n"
                if url:
                    message += f"🔗 URL: {url}\n"
                    
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📝 Analysis:\n{result}\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            return "SUCCESS", message, token_usage
        else:
            logging.error(f"Internal inconsistency for {action_type}: exec_success is True, but exec_message is neither string nor dict. Message: {exec_message}")
            return "FAILURE", f"Internal error: Successful execution reported but result format is unexpected for {action_type}.", token_usage

    if not exec_success:
        logging.warning(f"Execution reported failure. Assessing as FAILURE. Reason: {exec_message}")
        return "FAILURE", f"Action execution failed: {exec_message}", token_usage

    # Add file processing actions to non-visual actions
    non_visual_actions_or_explicit_success = {
        "wait", "get_clipboard", "set_clipboard", "save_credential", "read_file", "write_file",
        "navigate_web", "search_web", "capture_screenshot", "INFORM_USER", "search_youtube",
        "generate_large_content_with_gemini", "process_local_files", "process_files_from_urls"  # Added file processing actions
    }
    
    if action_type in non_visual_actions_or_explicit_success or \
    action_type in {"run_shell_command", "run_python_script"}:
        logging.info(f"Action '{action_type}' reported success and is non-visual or has explicit success criteria. Assessing as SUCCESS.")
        return "SUCCESS", f"Action executed successfully: {exec_message}", token_usage

    if screenshot_after is None:
        logging.warning("Cannot perform visual assessment: Screenshot after action is missing. Assuming success based on execution report.")
        return "SUCCESS", f"Action executed successfully (non-visual assessment as screenshot_after is missing): {exec_message}", token_usage

    current_screenshot_hash = _hash_pil_image(screenshot_after)
    screen_changed = False
    if screenshot_before_hash and current_screenshot_hash:
        screen_changed = screenshot_before_hash != current_screenshot_hash

    # For visual actions, we need to verify the screen changed
    visual_change_expected = action_type in {
        "click", "type", "press_keys", "move_mouse", "focus_window",
        "click_and_type", "multi_action"
    }

    if visual_change_expected:
        if not screen_changed:
            logging.warning(f"Screen content hash did NOT change after action '{action_type}' which usually causes visual changes. Assessing as FAILURE.")
            return "FAILURE", f"Action '{action_type}' reported success, but screen content did not change visually, indicating it likely failed.", token_usage
        else:
            logging.info("Screen content hash changed as expected for visual action.")
            return "SUCCESS", f"Action '{action_type}' executed successfully and screen content changed as expected.", token_usage
    else:
        logging.info("Screen content hash did not change (as expected for this action type or no change expected).")
        return "SUCCESS", f"Action '{action_type}' executed successfully.", token_usage



def request_replan_from_failure(
    original_instruction: str,
    plan_history: List[Tuple[dict, bool, str, Optional[Dict]]],
    failed_action: dict,
    failure_reasoning: str,
    screenshot_after_failure: Optional[Image.Image],
    llm_model: genai.GenerativeModel
) -> Tuple[Optional[dict], Dict[str, int]]:
    logging.warning("Requesting replan from LLM due to step failure.")

    history_summary = "\nExecution History Leading to Failure:\n"
    if not plan_history: history_summary += "No steps executed before failure.\n"
    else:
        max_history = 5
        start_index = max(0, len(plan_history) - max_history)
        for i, (action, success, message, _) in enumerate(plan_history[start_index:], start=start_index): # type: ignore
            status = "[OK]" if success else "[ERROR]"
            action_type = action.get('action_type', 'Unknown') # type: ignore
            msg_parts = message.split("| Assessment:") # type: ignore
            exec_msg = msg_parts[0].strip()
            assess_msg = msg_parts[1].strip() if len(msg_parts) > 1 else "(No assessment)"
            history_summary += f"Step {i+1} ({action_type}): {status}\n  Exec: {exec_msg}\n  Assess: {assess_msg}\n"

    failed_action_type = failed_action.get('action_type')
    failed_action_desc = f"Failed Action: {failed_action_type} with params {failed_action.get('parameters')}"
    failure_context = f"Failure Assessment Reasoning: {failure_reasoning}"

    screenshot_base64 = image_to_base64(screenshot_after_failure) if screenshot_after_failure else None

    replan_guidance_list = [
        "1. Analyze the failure: Why did the previous step likely fail based on the assessment reasoning and visual context (if available)?",
        "2. Create a NEW, COMPLETE plan: Starting from the current state shown in the screenshot, devise a sequence of *available* actions to achieve the original user instruction. **The output MUST be a full plan, not just a single step.**",
        "3. Avoid the mistake: Your new plan should try a different approach or add steps to overcome the previous failure.",
    ]

    if failed_action_type == "focus_window":
        actual_focused_window_title = None
        match = re.search(r"(?:instead,\s*a|found)\s*['\"]?(.*?)['\"]?\s*(?:window)?\s*(?:is\s*in\s*the\s*foreground|active)", failure_reasoning, re.IGNORECASE)
        if not match and "Current: '" in failure_reasoning: match = re.search(r"Current:\s*'(.*?)'", failure_reasoning)
        if match:
            extracted_title = match.group(1).strip()
            sanitized_title = re.sub(r"^[a-zA-Z0-9_]+\.(?:exe|com|bat)\s*-\s*", "", extracted_title)
            if sanitized_title and len(sanitized_title) > 3 and sanitized_title.lower() != "unknown":
                actual_focused_window_title = sanitized_title[:60]
                logging.info(f"Extracted potential actual focused window title for replan: '{actual_focused_window_title}'")
            else: logging.warning(f"Extracted title '{extracted_title}' seems too generic or short after sanitization, skipping.")

        replan_guidance_list.append("   * Focus Failure Specifics: The assessment reasoning indicates the wrong window might be focused or the target wasn't found.")
        if actual_focused_window_title:
            replan_guidance_list.append(f"   * The assessment suggests the window actually focused might contain '{actual_focused_window_title}' in its title.")
            replan_guidance_list.append(f"   * **CRITICAL RECOVERY SEQUENCE:** Your NEW plan MUST start with these exact steps:")
            replan_guidance_list.append(f"     1. `focus_window` with `title_substring`: '{actual_focused_window_title}' (to target the wrong window).")
            replan_guidance_list.append(f"     2. `press_keys` with `keys`: ['alt', 'f4'] (to close it).")
            replan_guidance_list.append(f"     3. `wait` with `duration_seconds`: 1.5")
            replan_guidance_list.append("   * **AFTER THE RECOVERY SEQUENCE:** Proceed with steps to achieve the original goal (e.g., opening the correct application/file if needed, then focusing it).")
        else:
            replan_guidance_list.append("   * If the target window wasn't found or the wrong window couldn't be identified: Your NEW plan must include steps to *open* the correct application/file first, then wait, then focus it.")
        replan_guidance_list.append("   * Ensure any `focus_window` step for the *target* window later in the plan has the correct `title_substring` parameter.")
        replan_guidance_list.append("   * **CRITICAL:** Ensure the generated plan is a complete sequence of steps, following the critical recovery sequence above if applicable.")

    elif failed_action_type == "run_shell_command" or failed_action_type == "run_python_script":
        replan_guidance_list.extend([
            f"   * {failed_action_type} Failure Specifics: The assessment reasoning contains the error output.",
            "   * **If 'ENOENT', 'cannot find path/file', 'not recognized as an internal or external command', 'Code 2' error:** Command name might be misspelled, the program might not be in the system PATH, or it needs a full path. NEW plan MUST verify command spelling, use full path if needed, OR ensure the correct `working_directory` is used if the command is relative to it. **DO NOT add `cd` to `command` string.**",
            "   * **If a chained command (using `&&` or `&`) failed:** Analyze which part of the chain failed from the error message. The NEW plan should address that specific failing part, possibly by breaking the chain into smaller, verifiable steps or using `search_web` on the specific error.",
            "   * **If 'file used by another process' error:** Previous background command (`npx create-react-app`, `npm install`, complex script) likely still running. NEW plan MUST include a longer `wait` (180-300s) *before* the failed step.",
            "   * **If a creation command (e.g., `mkdir`) failed because the item 'already exists':** This is often not a true failure if the goal was to *ensure* the item exists. The NEW plan should acknowledge this and proceed with the next logical step, assuming the item is now present. DO NOT try to create it again unless the goal is to handle a name conflict (in which case, `ask_user` might be needed).",
            "   * **If a deletion command (e.g., `del`, `rm`) failed because the item was 'not found':** This might be acceptable if the goal was to *ensure* the item is gone. The NEW plan should consider if this 'failure' actually meets the objective and proceed accordingly.",
            "   * **If error creating/editing file content via shell (`echo`, `Set-Content`):** Unreliable. NEW plan MUST use `write_file` action instead.",
            "   * If command/script not found: Check typos or try full path.",
            "   * If syntax error: Check quoting/escaping or try alternative command/script logic.",
            "   * **CRITICAL:** Ensure the generated plan is a complete sequence of steps with correct structure.",
        ])
    elif failed_action_type == "click":
        replan_guidance_list.extend([
            "   * Click Failure Specifics: The element described might not be visible, the description might be ambiguous, or the wrong window is focused.",
            "   * **Check Screenshot:** Does the element seem visible in the current screenshot?",
            "   * **Refine Description:** Try a more specific `element_description` (e.g., add position 'top left', type 'button').",
            "   * **Focus Window:** Add a `focus_window` step *before* the `click` if the correct window might not be active.",
            "   * **Wait:** Add a `wait` step before the `click` if the UI might still be loading.",
            "   * **Alternative:** Consider if `press_keys` (shortcut) or `run_shell_command` could achieve the goal instead.",
            "   * **CRITICAL:** Ensure the generated plan is a complete sequence.",
        ])
    elif failed_action_type == "type":
        replan_guidance_list.extend([
            "   * Type Failure Specifics: The text might be typed into the wrong place or the input field wasn't ready.",
            "   * **Focus Window/Element:** Ensure the correct window is focused using `focus_window`. If possible, `click` the specific input field first.",
            "   * **Wait:** Add a `wait` step before `type` to ensure the field is ready.",
            "   * **CRITICAL:** Ensure the generated plan is a complete sequence.",
        ])
    elif failed_action_type == "click_and_type":
        replan_guidance_list.extend([
            "   * Click_and_Type Failure Specifics: Failed during click or type.",
            "   * Analyze reasoning: Did click fail (element not found?) or type fail?",
            "   * **If click failed:** Try more specific `element_description`, or use `focus_window` first.",
            "   * **If type failed:** Ensure correct element was clicked/focused. Consider adding a `wait` between click and type.",
            "   * **Alternative:** Break into separate `click`, `wait`, `type` steps in the new plan.",
            "   * **CRITICAL:** Ensure the generated plan is a complete sequence.",
        ])
    replan_guidance_list.extend([
        "4. Choose the most suitable method (Shell Command > Python Script > Shortcut > Visual > Web Search). **Strongly prefer `write_file` over shell commands for creating/editing file content.**",
        "   * **CRITICAL JSON Structure:** Every action step MUST have `{\"action_type\": \"...\", \"parameters\": {...}}`. `parameters` key is required even if empty or for simple actions like `wait`.",
        "5. Output Format: Provide the *entire new plan* and reasoning in JSON: {\"plan\": [...], \"reasoning\": \"...\"}. **The 'plan' value MUST be a list of action objects.**",
        "Do not invent actions. Ensure parameters are correct. Do not include text outside JSON."
    ])
    replan_guidance = "\n".join(replan_guidance_list)

    prompt_parts = [
        "Respond only with a single valid JSON object. Do not include any markdown fences, explanatory text, leading/trailing whitespace, or other characters.",
        "You are an advanced PC Automation Assistant acting as a Re-Planner.",
        "A previous plan failed. Analyze the failure and generate a *new, complete plan* to achieve the user's original goal from the *current state*.",
        "\n--- USER GOAL ---",
        f"**Original User Instruction:** \"{original_instruction}\"",
        "\n--- HISTORY & CONTEXT ---",
        "**Failure Details:**",
        history_summary,
        f"  {failed_action_desc}",
        f"  {failure_context}",
        "\n**Current Screenshot (After Failed Step):** [See attached image, if available]",
        "\n--- AVAILABLE TOOLS (Use ONLY these - Ensure correct 'parameters' structure!) ---",
        "*   `{{ \"action_type\": \"focus_window\", \"parameters\": {{ \"title_substring\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"click\", \"parameters\": {{ \"element_description\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"type\", \"parameters\": {{ \"text_to_type\": \"...\", \"interval_seconds\": 0.05 }} }}`",
        "*   `{{ \"action_type\": \"press_keys\", \"parameters\": {{ \"keys\": [...] }} }}`",
        "*   `{{ \"action_type\": \"move_mouse\", \"parameters\": {{ \"x\": ..., \"y\": ... }} }}`",
        "*   `{{ \"action_type\": \"run_shell_command\", \"parameters\": {{ \"command\": \"...\", \"working_directory\": \"...\"? }} }}`",
        "*   `{{ \"action_type\": \"run_python_script\", \"parameters\": {{ \"script_path\": \"...\", \"working_directory\": \"...\"? }} }}`",
        "*   `{{ \"action_type\": \"write_file\", \"parameters\": {{ \"file_path\": \"...\", \"content\": \"...\", \"append\": false? }} }}`",
        "*   `{{ \"action_type\": \"navigate_web\", \"parameters\": {{ \"url\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"search_web\", \"parameters\": {{ \"query\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"generate_large_content_with_gemini\", \"parameters\": {{ \"context_summary\": \"...\", \"detailed_prompt_for_gemini\": \"...\", \"target_file_path\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"wait\", \"parameters\": {{ \"duration_seconds\": 1.0 }} }}`",
        "*   `{{ \"action_type\": \"ask_user\", \"parameters\": {{ \"question\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"describe_screen\", \"parameters\": {{}} }}`",
        "*   `{{ \"action_type\": \"capture_screenshot\", \"parameters\": {{ \"file_path\": \"...\"? }} }}`",
        "*   `{{ \"action_type\": \"click_and_type\", \"parameters\": {{ \"element_description\": \"...\", \"text_to_type\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"multi_action\", \"parameters\": {{ \"sequence\": [...] }} }}`",
        "*   `{{ \"action_type\": \"read_file\", \"parameters\": {{ \"file_path\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"INFORM_USER\", \"parameters\": {{ \"message\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"search_youtube\", \"parameters\": {{ \"query\": \"...\" }} }}`",
        "*   `{{ \"action_type\": \"edit_image_with_file\", \"parameters\": {{ \"image_path\": \"OPTIONAL_PATH_TO_IMAGE_OR_EMPTY_STRING_FOR_NEW\", \"prompt\": \"Description of edits OR image to generate\" }} }}`",
        "*   `{{ \"action_type\": \"process_local_files\", \"parameters\": {{ \"file_path\": \"path/to/your/file.ext\", \"prompt\": \"Your analysis prompt\" }} }}`",
        "*   `{{ \"action_type\": \"process_files_from_urls\", \"parameters\": {{ \"url\": \"http://example.com/file.ext\", \"prompt\": \"Your analysis prompt\" }} }}`",
        "    - **To generate a NEW image:** Set `image_path` to an empty string (`\"\"`) or `null`. The `prompt` should describe the image to create (e.g., \"A photo of a red apple on a wooden table\").",
        "    - **To edit an EXISTING image:** Provide the `image_path` to the image file. The `prompt` should describe the desired edits (e.g., \"Make the background blurry\", \"Convert to grayscale\").",
        "    - Use this action for any requests involving image creation, generation, editing, or modification based on a text prompt.",
        "    Example (Generating a new image):",
        "    `{{ \"action_type\": \"edit_image_with_file\", \"parameters\": {{ \"image_path\": \"\", \"prompt\": \"A cute cat wearing a tiny hat\" }} }}`",
        "    Example (Editing an existing image):",
        "    `{{ \"action_type\": \"edit_image_with_file\", \"parameters\": {{ \"image_path\": \"uploads/user_photo.jpg\", \"prompt\": \"Increase brightness and add a vintage filter\" }} }}`",
        "\n--- INSTRUCTIONS (Replanning) ---",
        replan_guidance,
        "\n--- EXAMPLES ---",
        "**Example 1: Recovering from Focus Failure**",
        "*   **Failure:** `focus_window` for 'TargetApp' failed, 'WrongApp' was focused instead.",
        "*   **Expected Output JSON:**",
        "    `{{\"plan\": [ {{ \"action_type\": \"focus_window\", \"parameters\": {{ \"title_substring\": \"WrongApp\" }} }}, {{ \"action_type\": \"press_keys\", \"parameters\": {{ \"keys\": [\"alt\", \"f4\"] }} }}, {{ \"action_type\": \"wait\", \"parameters\": {{ \"duration_seconds\": 1.5 }} }}, {{ \"action_type\": \"run_shell_command\", \"parameters\": {{ \"command\": \"TargetApp.exe\" }} }}, {{ \"action_type\": \"wait\", \"parameters\": {{ \"duration_seconds\": 2.0 }} }}, {{ \"action_type\": \"focus_window\", \"parameters\": {{ \"title_substring\": \"TargetApp\" }} }} ], \"reasoning\": \"Focus the wrong app, close it with Alt+F4, wait, launch the correct TargetApp, wait for it to load, then focus it to recover from the previous focus failure.\"}}`",
        "**Example 2: Recovering from Click Failure**",
        "*   **Failure:** `click` on 'Submit button' failed (element not found).",
        "*   **Expected Output JSON:**",
        "    `{{\"plan\": [ {{ \"action_type\": \"focus_window\", \"parameters\": {{ \"title_substring\": \"Application Window Title\" }} }}, {{ \"action_type\": \"wait\", \"parameters\": {{ \"duration_seconds\": 1.0 }} }}, {{ \"action_type\": \"click\", \"parameters\": {{ \"element_description\": \"Green Submit button near bottom right\" }} }} ], \"reasoning\": \"The previous click failed. Ensure the correct window is focused, wait briefly for UI stability, then try clicking again using a more specific description ('Green Submit button near bottom right').\"}}`",
        "--- END EXAMPLES ---",
        "\n--- USER GOAL (Repeat) ---",
        f"**Original User Instruction:** \"{original_instruction}\"",
        "\n--- HISTORY & CONTEXT (Repeat) ---",
        "**Failure Details:**",
        history_summary,
        f"  {failed_action_desc}",
        f"  {failure_context}",
        "\n--- FINAL INSTRUCTIONS: CRITICAL OUTPUT REQUIREMENTS ---",
        "1.  **RAW JSON ONLY:** Your ENTIRE response MUST be a single, valid JSON object. Start with `{{` and end with `}}`. NO other text, no markdown (like \`\`\`json), no explanations outside the JSON.",
        "2.  **STRUCTURE:** The JSON object MUST have exactly two top-level keys:",
        "    *   `\"plan\"`: A JSON array `[...]` of action objects. This array can be empty `[]` if no actions are appropriate.",
        "    *   `\"reasoning\"`: A string explaining your plan or why no actions are planned.",
        "3.  **ACTION STRUCTURE (for each object in the \"plan\" array):**",
        "    *   Every action object MUST have:",
        "        *   `\"action_type\"`: A string (e.g., `\"run_shell_command\"`).",
        "        *   `\"parameters\"`: An object (e.g., `{{\"command\": \"echo hello\"}}`). This object can be empty `{}` if no parameters are needed.",
        "4.  **`multi_action` STRUCTURE (if used within the \"plan\"):**",
        "    *   If an `action_type` is `\"multi_action\"`, its `\"parameters\"` object MUST contain a key `\"sequence\"`.",
        "    *   The value of `\"sequence\"` MUST be a JSON array `[...]` of further action objects (which themselves follow the action structure).",
        "5.  **JSON SYNTAX (VERY IMPORTANT):**",
        "    *   **Commas:** Ensure commas `,` correctly separate key-value pairs within objects AND elements within arrays (e.g., between action objects in the \"plan\" list, or in a \"sequence\" list). A missing or misplaced comma is a common error.",
        "    *   **Quotes:** All keys and string values MUST be enclosed in double quotes (`\"`).",
        "    *   **Escaping:** Properly escape special characters within strings:",
        "        *   Double quotes: `\\\"`",
        "        *   Newlines: `\\\\n`",
        "        *   Backslashes: `\\\\\\\\`",
        "        *   Other characters as needed (e.g., `\\t` for tab). This is CRITICAL for `write_file` content and `run_shell_command` commands.",
        "6.  **VALIDATION:** Before outputting, mentally validate your JSON. Ensure all braces `{{}}` and brackets `[]` are balanced and all syntax rules are followed.",
        "",
        "Example of the required top-level structure:",
        "`{{ \"plan\": [ {{ \"action_type\": \"...\", \"parameters\": {{...}} }}, {{ \"action_type\": \"...\", \"parameters\": {{...}} }} ], \"reasoning\": \"...\" }}`",
        "Or for an empty plan:",
        "`{{ \"plan\": [], \"reasoning\": \"No actions needed because...\" }}`"
    ]

    content = [{"text": "\n".join(prompt_parts)}]
    if screenshot_base64:
        content.append({"inline_data": {"mime_type": "image/png", "data": screenshot_base64}})
    else:
        content[0]['text'] += "\n(Note: Screenshot not available for replanning)." # type: ignore

    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    try:
        safety_settings = {}
        response = llm_model.generate_content(content, safety_settings=safety_settings)
        token_usage = _get_token_usage(response)
        raw_response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            raw_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not raw_response_text:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                logging.error(f"Replanning LLM blocked: {response.prompt_feedback.block_reason}")
            else:
                logging.error("Replanning LLM empty response.")
            return None, token_usage

        logging.debug(f"LLM Raw Replan Response:\n{raw_response_text}")

        replan_dict = None
        try:
            cleaned_text = re.sub(r'^[^{\[]*', '', raw_response_text, flags=re.DOTALL).strip()
            cleaned_text = re.sub(r'[^}\]]*$', '', cleaned_text, flags=re.DOTALL).strip()
            if not cleaned_text:
                logging.error("Response was empty after aggressive cleaning for replan.")
            else:
                replan_dict = json.loads(cleaned_text)
                logging.info("JSON parsing successful for replan after aggressive cleaning.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode replan JSON after aggressive cleaning: {e}. Cleaned text: '{cleaned_text if 'cleaned_text' in locals() else 'N/A'}'") # type: ignore
            return None, token_usage

        if isinstance(replan_dict, dict) and "plan" in replan_dict and "reasoning" in replan_dict and isinstance(replan_dict["plan"], list):
            valid_plan = True
            available_actions = {"focus_window", "click", "type", "press_keys",
                                "move_mouse", "run_shell_command", "navigate_web",
                                "wait", "ask_user", "describe_screen", "click_and_type",
                                "write_file", "run_python_script", "multi_action",
                                "generate_large_content_with_gemini", "INFORM_USER", # process_local_files was already singular here
                                "capture_screenshot", "read_file", "search_web", "search_youtube", "edit_image_with_file",
                                "process_local_files", "process_files_from_urls"} # Changed process_local_files to singular
            for i, step in enumerate(replan_dict["plan"]):
                action_type_val = step.get("action_type")
                params_val = step.get("parameters")
                if (not isinstance(step, dict) or
                    not isinstance(action_type_val, str) or
                    action_type_val not in available_actions or
                    not isinstance(params_val, dict)):
                    logging.error(f"Replanning invalid step structure/type step {i+1}: {step}")
                    valid_plan = False; break
                if action_type_val == "focus_window" and "title_substring" not in params_val: # type: ignore
                    logging.error(f"Replanning invalid 'focus_window' step {i+1}: missing 'title_substring'. Step: {step}"); valid_plan = False; break
                if action_type_val == "click_and_type" and ("element_description" not in params_val or "text_to_type" not in params_val): # type: ignore
                    logging.error(f"Replanning invalid 'click_and_type' step {i+1}: missing parameters. Step: {step}"); valid_plan = False; break
                if action_type_val == "write_file" and ("file_path" not in params_val or "content" not in params_val): # type: ignore
                    logging.error(f"Replanning invalid 'write_file' step {i+1}: missing parameters. Step: {step}"); valid_plan = False; break
                if action_type_val == "run_shell_command" and "command" not in params_val: # type: ignore
                    logging.error(f"Replanning invalid 'run_shell_command' step {i+1}: missing 'command'. Step: {step}"); valid_plan = False; break
                if action_type_val == "run_python_script" and "script_path" not in params_val: # type: ignore
                    logging.error(f"Replanning invalid 'run_python_script' step {i+1}: missing 'script_path'. Step: {step}"); valid_plan = False; break
                if action_type_val == "generate_large_content_with_gemini" and not all(k in params_val for k in ["context_summary", "detailed_prompt_for_gemini", "target_file_path"]): # type: ignore
                    logging.error(f"Replanning invalid 'generate_large_content_with_gemini' step {i+1}: missing parameters. Step: {step}"); valid_plan = False; break
                if action_type_val == "edit_image_with_file" and "prompt" not in params_val: # Added check for edit_image_with_file
                    logging.error(f"Replanning invalid 'edit_image_with_file' step {i+1}: missing 'prompt'. Step: {step}"); valid_plan = False; break
                if action_type_val == "process_local_files" and not all(k in params_val for k in ["file_path", "prompt"]): # type: ignore # Ensured singular form
                    logging.error(f"Replanning invalid 'process_local_files' step {i+1}: missing parameters. Step: {step}"); valid_plan = False; break
                if action_type_val == "process_files_from_urls" and not all(k in params_val for k in ["url", "prompt"]): # type: ignore
                    logging.error(f"Replanning invalid 'process_files_from_urls' step {i+1}: missing parameters. Step: {step}"); valid_plan = False; break


            if not valid_plan:
                logging.error("Replanning failed: Generated plan contains invalid steps.")
                return None, token_usage

            logging.info("Replanning successful.")
            logging.info(f"Replan Reasoning: {replan_dict.get('reasoning')}")
            return replan_dict, token_usage
        else:
            logging.error(f"Replanning response invalid structure: {replan_dict}")
            return None, token_usage

    except ValueError as ve:
        if "response.prompt_feedback" in str(ve) and hasattr(response, 'prompt_feedback'): # type: ignore
            try:
                block_reason = response.prompt_feedback.block_reason # type: ignore
                logging.error(f"Replanning LLM blocked: {block_reason}")
            except Exception: pass
        logging.error(f"ValueError accessing replanning LLM: {ve}")
        return None, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    except Exception as e:
        logging.error(f"Error during replanning LLM call: {e}", exc_info=True)
        return None, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

def _extract_search_query_from_instruction(instruction: str) -> Optional[str]:
    """
    Attempts to extract a search query from a "play X on Y" type instruction.
    Example: "play genius by sia on youtube" -> "genius by sia"
    """
    # Regex to capture the content between "play" and "on youtube" (or similar)
    # It's a simple heuristic and might need refinement for more complex cases.
    match = re.search(r"play\s+(.+?)\s+(?:on|in)\s+(youtube|spotify|soundcloud)", instruction, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback for simpler "search for X" or "find X"
    match_search = re.search(r"(?:search for|find|look up)\s+(.+)", instruction, re.IGNORECASE)
    if match_search:
        return match_search.group(1).strip()
    return None

def _adapt_plan_from_past_task(
    past_actions_json: str,
    current_instruction: str,
    past_instruction: str,
    similarity_distance: float # Added similarity_distance
) -> Optional[List[Dict[str, Any]]]:
    """
    Adapts a plan (list of actions) from a similar past task for the current instruction.
    """
    logging.info(f"Attempting to adapt plan (Similarity: {similarity_distance:.4f}). Current: '{current_instruction}', Past: '{past_instruction}'")
    try:
        past_actions = json.loads(past_actions_json)
        # The past_actions are stored as [action_dict, success_bool, message_str, special_directive_dict_or_None]
        # We only need the action_dict part
        executable_past_actions = [item[0] for item in past_actions if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict)]
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding past_actions_json: {e}")
        return None

    current_query = _extract_search_query_from_instruction(current_instruction)
    past_query = _extract_search_query_from_instruction(past_instruction)

    if not current_query or not past_query or current_query == past_query:
        logging.info("Could not extract distinct queries or queries are identical. No adaptation performed.")
        # If queries are the same, we might still reuse but for now, let's focus on substitution.
        # Or, if no queries, maybe it's not a search-like task suitable for this simple adaptation.
        return None
    VERY_HIGH_SIMILARITY_THRESHOLD_FOR_COORDS = 0.01 # e.g., for >0.99 similarity
    SIMILARITY_THRESHOLD_FOR_COORDINATE_CLICK = 0.1 # Stricter threshold for using coordinates

    logging.info(f"Adapting based on query change: '{past_query}' -> '{current_query}'")
    adapted_plan = []
    stop_adaptation_and_replan = False

    for action_dict in executable_past_actions:
        if stop_adaptation_and_replan:
            break

        # Find the full past_action_tuple to get success status and message for coordinate extraction
        original_past_action_tuple = next((item for item in past_actions if isinstance(item, list) and len(item) > 0 and item[0] == action_dict), None)

        current_action_to_add = action_dict.copy() # Work with a copy
        action_type = current_action_to_add.get("action_type")
        params = current_action_to_add.get("parameters", {})
        action_adapted_by_coordinate = False

        # --- Adaptation logic for 'type', 'click_and_type', 'multi_action' (entity substitution) ---
        if action_type == "type" and isinstance(params.get("text_to_type"), str) and past_query and past_query.lower() in params["text_to_type"].lower():
            params["text_to_type"] = params["text_to_type"].lower().replace(past_query.lower(), current_query) # type: ignore
            logging.info(f"Adapted 'type' action. New text: {params['text_to_type']}")
        elif action_type == "click_and_type" and isinstance(params.get("text_to_type"), str) and past_query and past_query.lower() in params["text_to_type"].lower():
            params["text_to_type"] = params["text_to_type"].lower().replace(past_query.lower(), current_query) # type: ignore
            logging.info(f"Adapted 'click_and_type' action. New text: {params['text_to_type']}")
        elif action_type == "multi_action" and "sequence" in params and isinstance(params["sequence"], list):
            new_sequence = []
            for sub_action_dict_orig in params["sequence"]:
                sub_action_copy = sub_action_dict_orig.copy()
                sub_params = sub_action_copy.get("parameters", {})
                if sub_action_copy.get("action_type") == "type" and \
                   isinstance(sub_params.get("text_to_type"), str) and \
                   past_query and past_query.lower() in sub_params["text_to_type"].lower():
                    sub_params["text_to_type"] = sub_params["text_to_type"].lower().replace(past_query.lower(), current_query) # type: ignore
                    logging.info(f"Adapted 'type' within multi_action. New text: {sub_params['text_to_type']}")
                new_sequence.append(sub_action_copy)
            params["sequence"] = new_sequence
        # --- End of entity substitution ---

        # Heuristic: If we encounter a 'click' action that likely clicked a specific search result,
        # or a 'describe_screen', stop adapting and let the agent plan the rest.
        if action_type == "click":
            element_desc_from_past = params.get("element_description", "")
            is_content_specific_click = past_query and past_query.lower() in element_desc_from_past.lower()

            can_try_coords_due_to_similarity = False
            if similarity_distance < SIMILARITY_THRESHOLD_FOR_COORDINATE_CLICK:
                if not is_content_specific_click:
                    can_try_coords_due_to_similarity = True
                elif similarity_distance < VERY_HIGH_SIMILARITY_THRESHOLD_FOR_COORDS:
                    # For extremely similar tasks, risk using coordinates even if it seemed content-specific before,
                    # assuming the UI layout for that specific content item is identical.
                    logging.info(f"Very high similarity (Dist: {similarity_distance:.4f}), considering coordinate click even for potentially content-specific item: '{element_desc_from_past}'")
                    can_try_coords_due_to_similarity = True
            
            if can_try_coords_due_to_similarity and \
               original_past_action_tuple and original_past_action_tuple[1] is True and \
               isinstance(original_past_action_tuple[2], str): # past_success and past_message is string
                
                past_message_for_coords = original_past_action_tuple[2]
                coord_match = re.search(r"at \((\d+),\s*(\d+)\)", past_message_for_coords)
                if coord_match:
                    ex_x, ex_y = int(coord_match.group(1)), int(coord_match.group(2))
                    # Preserve original parameters if any, then override/add x, y
                    new_params_for_coord_click = params.copy() # Start with existing params
                    new_params_for_coord_click["x"] = ex_x
                    new_params_for_coord_click["y"] = ex_y
                    if "element_description" in new_params_for_coord_click: # Remove element_description if using coords
                        del new_params_for_coord_click["element_description"]
                    current_action_to_add = {"action_type": "click", "parameters": new_params_for_coord_click}
                    action_adapted_by_coordinate = True
                    logging.info(f"Adapted 'click' to use coordinates ({ex_x},{ex_y}) from past task due to high similarity (Dist: {similarity_distance:.4f}) and non-content-specific target.")
                else:
                    logging.info(f"High similarity for click (Dist: {similarity_distance:.4f}), but couldn't parse coordinates from past message: '{past_message_for_coords}'")
            
            if not action_adapted_by_coordinate and is_content_specific_click:
                # If we didn't adapt by coordinate (either because similarity wasn't high enough for this content-specific click,
                # or coords couldn't be parsed) AND it's still considered a content-specific click, then stop adaptation.
                logging.info(f"Stopping adaptation at content-specific 'click' action: {element_desc_from_past}")
                stop_adaptation_and_replan = True
                continue # Don't add this specific click to the adapted plan

        if action_type == "describe_screen":
            logging.info("Skipping 'describe_screen' action from past plan.")
            continue

        adapted_plan.append(current_action_to_add)

    return adapted_plan if adapted_plan else None



def iterative_task_executor(
    original_instruction: str,
    agent_state: 'AgentState',
    agent: UIAgent,
    llm_model: genai.GenerativeModel,
    initial_history: Optional[List[str]] = None
) -> Generator[Dict[str, Any], Optional[str], List[Tuple[dict, bool, str, Optional[Dict]]]]:
    # Check if llm_model is None and try to get a new instance
    if llm_model is None:
        llm_model = get_model()
        if llm_model is None:
            logging.error("LLM model is None and could not be initialized")
            # Yield or raise an error to indicate failure
            yield {"type": "error", "message": "LLM model not initialized"}
            return [] # Return empty list or handle error appropriately

    # Get critic model
    critic_model = get_critic_model()
    if critic_model is None:
        logging.error("Critic model is None and could not be initialized")
        yield {"type": "error", "message": "Critic model not initialized"}
        return []


    results = []
    max_iterations = 25
    current_iteration = agent_state.current_task.iteration_count if agent_state.current_task else 0
    task_completed = False
    current_app_base_name = "unknown"
    current_shortcuts: Union[str, List[Dict[str, str]]] = [] # Can be string or list of dicts
    last_assessment_screenshot_hash: Optional[str] = None
    action_failure_counts: Dict[str, int] = agent_state.current_task.action_failure_counts if agent_state.current_task else {}

    MAX_RETRIES_PER_STEP = 1
    MAX_CONSECUTIVE_FAILURES_BEFORE_ASK = 2
    MAX_REPLAN_ATTEMPTS = 1

    consecutive_failures_on_current_step = 0
    current_task_instruction = original_instruction
    MAX_YOUTUBE_SEARCH_ATTEMPTS = 2
    total_consecutive_failures = 0
    replan_attempts_current_cycle = 0

    pending_credential_request: Optional[Dict[str, str]] = None
    credential_consent_choice: Optional[str] = None
    plan_to_inject: Optional[List[Dict[str, Any]]] = None
    plan_injection_reason: Optional[str] = None
    plan_source = "Standard Planning"
    user_plan_used = False
    chosen_high_level_plan_description: Optional[str] = None
    current_sub_tasks: List[str] = []
    current_sub_task_index: int = 0
    instruction_for_current_planning_cycle = original_instruction
    pending_risky_action: Optional[Dict[str, Any]] = None
    user_response_to_ask: Optional[str] = None # Initialize user_response_to_ask

    logging.info(f"Starting iterative execution for: {original_instruction}")
    load_shortcuts_cache()

    # --- Attempt to use similar past task execution ---
    if not plan_to_inject: # Only if no other plan is already injected
        logging.info(f"Checking for similar past successful tasks for: {original_instruction}")
        # Ensure n_results is at least 1 if collection is not empty
        similar_past_tasks = retrieve_similar_task_executions_from_db(original_instruction, n_results=1) # type: ignore
        if similar_past_tasks:
            for past_task_data in similar_past_tasks:
                past_metadata = past_task_data.get("metadata", {})
                past_status = past_metadata.get("status")
                past_instruction_text = past_metadata.get("original_instruction")
                similarity_distance = past_task_data.get("distance", 1.0)
                SIMILARITY_THRESHOLD_FOR_ADAPTATION = 0.30

                if past_status == "success" and past_instruction_text and similarity_distance < SIMILARITY_THRESHOLD_FOR_ADAPTATION:
                    logging.info(f"Found similar past successful task (Dist: {similarity_distance:.4f}): '{past_instruction_text}'. Attempting to adapt its plan.")
                    past_actions_json_str = past_metadata.get("full_results_json")
                    if past_actions_json_str:
                        adapted_plan = _adapt_plan_from_past_task(past_actions_json_str, original_instruction, past_instruction_text, similarity_distance) # type: ignore
                        if adapted_plan:
                            plan_to_inject = adapted_plan
                            plan_injection_reason = f"Adapted from similar past task (Dist: {similarity_distance:.4f}): '{past_instruction_text}'"
                            print(f"\n{BLUE}=== Using Adapted Plan from Past Task ===")
                            print(f"Current Task: '{original_instruction}'")
                            print(f"Adapted from: '{past_instruction_text}' (Similarity Distance: {similarity_distance:.4f})")
                            print(f"Adapted Actions ({len(adapted_plan)} steps):{RESET}")
                            for i, action in enumerate(adapted_plan):
                                print(f"  {i+1}. {action.get('action_type')}: {action.get('parameters')}")
                            print(f"{BLUE}======================================{RESET}\n")
                            logging.info(f"Injecting adapted plan with {len(plan_to_inject)} steps.")
                            break
    if agent_state.current_task and current_iteration == 0 and not agent_state.current_task.initial_planning_done:
        if hasattr(agent_state.current_task, 'sub_tasks') and agent_state.current_task.sub_tasks:
            current_sub_tasks = agent_state.current_task.sub_tasks
            current_sub_task_index = agent_state.current_task.current_sub_task_index
            logging.info(f"Resuming with existing sub-tasks. Current: {current_sub_task_index + 1}/{len(current_sub_tasks)}")
        else:
            category, sub_tasks_list, analysis_reasoning, analysis_tokens = _analyze_and_categorize_task(original_instruction, llm_model) # type: ignore
            if agent_state.current_task: agent_state.current_task._accumulate_tokens(analysis_tokens)
            agent_state.add_thought(f"Task Analysis: Category='{category}', Sub-tasks={len(sub_tasks_list)}. Reasoning: {analysis_reasoning}", type="task_analysis")
            if category and sub_tasks_list:
                current_sub_tasks = sub_tasks_list
                current_sub_task_index = 0
                if hasattr(agent_state.current_task, 'sub_tasks'): agent_state.current_task.sub_tasks = current_sub_tasks
                if hasattr(agent_state.current_task, 'current_sub_task_index'): agent_state.current_task.current_sub_task_index = current_sub_task_index
                if hasattr(agent_state.current_task, 'task_category'): agent_state.current_task.task_category = category
                logging.info(f"Task broken down into {len(current_sub_tasks)} sub-tasks.")
            else:
                current_sub_tasks = [original_instruction]
        if agent_state.current_task: agent_state.current_task.initial_planning_done = True

    while current_iteration < max_iterations and not task_completed:
        action_to_execute = None # Initialize action_to_execute at the start of each iteration

        if current_sub_tasks and 0 <= current_sub_task_index < len(current_sub_tasks):
            instruction_for_current_planning_cycle = current_sub_tasks[current_sub_task_index]
            if not agent_state.current_task or not any(
                thought['type'] == 'sub_task_info' and
                thought['content'].startswith(f"Working on Sub-task {current_sub_task_index + 1}")
                for thought in agent_state.current_task.agent_thoughts[-3:] # type: ignore
            ):
                agent_state.add_thought(f"Working on Sub-task {current_sub_task_index + 1}/{len(current_sub_tasks)}: {instruction_for_current_planning_cycle}", type="sub_task_info")
        else:
            instruction_for_current_planning_cycle = current_task_instruction

        if current_sub_tasks and 0 <= current_sub_task_index < len(current_sub_tasks) and not plan_to_inject :
            sub_task_instruction_lower = instruction_for_current_planning_cycle.lower()
            required_context_marker = None
            action_to_re_establish_if_lost = None

            if "youtube" in sub_task_instruction_lower or "video" in sub_task_instruction_lower:
                required_context_marker = "youtube"
                action_to_re_establish_if_lost = {"action_type": "navigate_web", "parameters": {"url": "https://www.youtube.com"}}
            elif "notepad" in sub_task_instruction_lower:
                required_context_marker = "notepad.exe"
                action_to_re_establish_if_lost = {"action_type": "run_shell_command", "parameters": {"command": "notepad.exe"}}

            if required_context_marker:
                active_window_title_for_check = get_active_window_name().lower()
                active_app_base_name_for_check = get_base_app_name(active_window_title_for_check)
                context_match = False
                known_browsers_bases = ["chrome", "firefox", "msedge", "brave"]

                if ".exe" in required_context_marker:
                    if required_context_marker == active_app_base_name_for_check:
                        context_match = True
                elif any(browser_base in active_app_base_name_for_check for browser_base in known_browsers_bases) and \
                     required_context_marker in active_window_title_for_check:
                    context_match = True
                
                if not context_match:
                    log_message = f"Context mismatch for sub-task '{instruction_for_current_planning_cycle}'. Expected context related to '{required_context_marker}', but active window is '{active_window_title_for_check}' (Base app: '{active_app_base_name_for_check}')."
                    was_recently_in_context = False
                    if agent_state.current_task:
                        original_full_context_for_history = "youtube.com" if required_context_marker == "youtube" else required_context_marker
                        for msg in reversed(agent_state.current_task.conversation_history[-3:]):
                            if msg["role"] == "system" and "content" in msg:
                                if original_full_context_for_history in msg["content"].lower():
                                    was_recently_in_context = True
                                    break
                    if was_recently_in_context:
                        log_message += " However, required context was recently observed in history. Planner should attempt to focus correct window."
                        logging.warning(log_message)
                    elif action_to_re_establish_if_lost:
                        log_message += f" Attempting to re-establish context by injecting plan: {action_to_re_establish_if_lost}."
                        logging.warning(log_message)
                        plan_to_inject = [action_to_re_establish_if_lost]
                        plan_injection_reason = f"Re-establishing context '{required_context_marker}' (was lost) for sub-task {current_sub_task_index + 1}"
                        agent_state.add_thought(f"Injecting plan to re-establish context: {plan_injection_reason}", type="context_fix")
                    else:
                        log_message += " No specific action defined to re-establish this context if lost and not recently in history."
                        logging.error(log_message)

        while agent_state.task_is_paused:
            if pending_risky_action and user_response_to_ask is not None:
                confirmed_command_str = pending_risky_action.get("parameters", {}).get("command", "Unknown command")
                if "yes" in user_response_to_ask.lower():
                    logging.info(f"User confirmed risky command: {confirmed_command_str}")
                    action_to_execute = pending_risky_action.copy()
                    action_to_execute["_bypass_confirmation_"] = True
                    if agent_state.current_task:
                        agent_state.current_task.conversation_history.append({"role": "system", "content": f"System: User confirmed execution of risky command '{confirmed_command_str}'."})
                    pending_risky_action = None
                    user_response_to_ask = None
                else:
                    logging.warning(f"User denied risky command: {confirmed_command_str}")
                    results.append((pending_risky_action, False, f"User denied risky command execution: {confirmed_command_str}", None))
                    if agent_state.current_task:
                        agent_state.current_task.conversation_history.append({"role": "system", "content": f"System: User denied execution of risky command '{confirmed_command_str}'."})
                    pending_risky_action = None
                    user_response_to_ask = None
                    total_consecutive_failures = 0
                    consecutive_failures_on_current_step = 0
                    replan_attempts_current_cycle = 0
                    current_iteration +=1
                    if agent_state.current_task: agent_state.current_task.iteration_count = current_iteration
                    logging.info(f"--- Iteration {current_iteration}/{max_iterations} (User Denied Risky Command) ---")
                    continue
            elif pending_risky_action and user_response_to_ask is None:
                logging.warning("Resumed from pause but pending_risky_action exists and user_response_to_ask is None. This might be an issue.")
            logging.debug("Generator: Task is paused. Yielding 'paused' state.")
            yield {"type": "paused"}
            logging.debug("Generator: Resuming from pause check.")
        
        current_iteration += 1
        if agent_state.current_task: agent_state.current_task.iteration_count = current_iteration
        logging.info(f"--- Iteration {current_iteration}/{max_iterations} (Plan Source: {plan_source}, Consecutive Failures: {total_consecutive_failures}, Replans: {replan_attempts_current_cycle}) ---")
        logging.info(f"Current planning cycle instruction: '{instruction_for_current_planning_cycle}'")

        if not action_to_execute:
            if plan_to_inject is not None:
                logging.info(f"Injecting new plan (reason: {plan_injection_reason}, {len(plan_to_inject)} steps).")
                if not plan_to_inject:
                    logging.warning("Injected plan was empty or exhausted. Continuing normal planning.")
                    plan_to_inject = None
                    plan_injection_reason = None
        
        try:
            full_app_name = get_active_window_name()
            app_base_name = get_base_app_name(full_app_name)
            name_for_shortcuts_lookup = app_base_name
            is_website_shortcut_lookup = False
            known_browsers_bases_exec = ["chrome.exe", "firefox.exe", "msedge.exe", "brave.exe"]

            if app_base_name.lower() in known_browsers_bases_exec:
                if agent_state.current_task and agent_state.current_task.conversation_history:
                    for msg in reversed(agent_state.current_task.conversation_history[-5:]):
                        if msg["role"] == "system" and "content" in msg and "System Observation: Opened web URL:" in msg["content"]:
                            url_match = re.search(r"Opened web URL:\s*(https?://[^/\s]+)", msg["content"])
                            if url_match:
                                full_url = url_match.group(1)
                                domain_match = re.search(r"https?://(?:www\.)?([^/]+)", full_url)
                                if domain_match:
                                    domain = domain_match.group(1)
                                    name_for_shortcuts_lookup = domain
                                    logging.info(f"Browser '{app_base_name}' is active. Identified active website '{domain}' from history. Will fetch shortcuts for '{domain}'.")
                                    is_website_shortcut_lookup = True
                                    break
                    else:
                        logging.info(f"Browser '{app_base_name}' is active, but no recent 'Opened web URL' found in history. Using browser name for shortcuts.")
                else:
                    logging.info(f"Browser '{app_base_name}' is active, but no conversation history to check for URLs. Using browser name for shortcuts.")

            if app_base_name != current_app_base_name:
                logging.info(f"Application context changed. Old: '{current_app_base_name}', New: '{app_base_name}', Full title: '{full_app_name}'. Shortcut lookup term: '{name_for_shortcuts_lookup}'")
                current_app_base_name = app_base_name
                current_shortcuts_str, shortcut_tokens = get_application_shortcuts(name_for_shortcuts_lookup) # type: ignore
                current_shortcuts = current_shortcuts_str # type: ignore
                if current_shortcuts_str and current_shortcuts_str.strip(): # type: ignore
                    from config import GREEN, RESET
                    logging.info(f"{GREEN}Using shortcuts for '{name_for_shortcuts_lookup}':\n{current_shortcuts_str}{RESET}") # type: ignore
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(shortcut_tokens)
                if is_website_shortcut_lookup:
                    from config import YELLOW, RESET
                    logging.info(f"{YELLOW}Fetching shortcuts for WEBSITE: {name_for_shortcuts_lookup}{RESET}")
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System Observation: Active application changed to '{app_base_name}' (Window: '{full_app_name}')."})
        except Exception as app_detect_err:
            logging.error(f"Error during application detection/shortcut handling: {app_detect_err}", exc_info=True)
            current_app_base_name = "unknown"; current_shortcuts = ""

        planning_screenshot = capture_full_screen()
        screenshot_before_action_hash = _hash_pil_image(planning_screenshot)

        if last_assessment_screenshot_hash and screenshot_before_action_hash and screenshot_before_action_hash != last_assessment_screenshot_hash:
            logging.info("Detected screen change since last assessment.")
            if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": "System Observation: Screen content changed since last action."})

        # action_to_execute = None # This was moved to the top of the loop
        planning_reasoning = "Planning skipped due to pending credential request or injected plan."
        next_step_data = None

        if not action_to_execute:
            if chosen_high_level_plan_description:
                logging.info(f"Using chosen high-level plan description for detailed planning: {chosen_high_level_plan_description[:200]}...")
                instruction_for_current_planning_cycle = chosen_high_level_plan_description
                plan_source = "Detailed planning of chosen high-level path"
                chosen_high_level_plan_description = None

            if plan_to_inject:
                action_to_execute = plan_to_inject.pop(0)
                planning_reasoning = f"Executing step from injected plan (original reason: {plan_injection_reason})"
                if not plan_to_inject:
                    plan_to_inject = None
                    plan_injection_reason = None
                logging.info(f"Using action from injected plan: {action_to_execute.get('action_type') if isinstance(action_to_execute, dict) else 'Invalid injected action'}")
            else:
                string_history_for_planning = [f"{msg['role']}: {msg['content']}" for msg in (agent_state.current_task.conversation_history if agent_state.current_task else [])]
                executed_actions_summary_parts = []
                if results:
                    for i, (action_dict_res, success_res, msg_res, _) in reversed(list(enumerate(results[-5:]))):
                        action_type_res = action_dict_res.get('action_type', 'Unknown')
                        outcome_res = "OK" if success_res else "FAIL"
                        exec_msg_only_res = msg_res.split("| Assessment:")[0].strip()
                        summary_line = f"  Prev. Action {len(results) - i}: {action_type_res}, Outcome: {outcome_res} - Msg: {exec_msg_only_res[:70]}{'...' if len(exec_msg_only_res)>70 else ''}"
                        executed_actions_summary_parts.append(summary_line)
                executed_actions_summary_str_for_prompt = "\n".join(executed_actions_summary_parts) if executed_actions_summary_parts else "No actions executed yet in this task."

                next_step_data, planning_tokens = process_next_step(
                    instruction_for_current_planning_cycle,
                    string_history_for_planning,
                    llm_model,
                    current_shortcuts, # type: ignore
                    [],
                    planning_screenshot,
                    executed_actions_summary=executed_actions_summary_str_for_prompt,
                    overall_original_instruction=original_instruction,
                    all_sub_tasks_count=len(current_sub_tasks) if current_sub_tasks else 0,
                    current_sub_task_idx=current_sub_task_index if current_sub_tasks else -1
                )
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(planning_tokens)
                if next_step_data is None or "next_action" not in next_step_data or not isinstance(next_step_data["next_action"], dict):
                    reason = f"Planning failed: Invalid or no 'next_action' from process_next_step: {next_step_data}"
                    logging.error(reason)
                    results.append((({'action_type': 'STOP', 'parameters': {'reason': 'planning_failed_structure'}}, False, reason, None)))
                    break
                action_to_execute = next_step_data["next_action"]
                planning_reasoning = next_step_data.get("reasoning", "N/A")

        action_type = action_to_execute.get("action_type") # type: ignore
        params = action_to_execute.get("parameters", {}) # type: ignore
        logging.info(f"Proposed Action: {action_type}. Reasoning: {planning_reasoning}")

        if action_type == "refresh_application_shortcuts":
            logging.info(f"Executing special action: refresh_application_shortcuts for '{name_for_shortcuts_lookup}'") # type: ignore
            current_shortcuts_str, shortcut_tokens = get_application_shortcuts(name_for_shortcuts_lookup, force_refresh=True) # type: ignore
            current_shortcuts = current_shortcuts_str # type: ignore
            if agent_state.current_task: agent_state.current_task._accumulate_tokens(shortcut_tokens)
            refresh_message = f"Application shortcuts for '{name_for_shortcuts_lookup}' have been refreshed." # type: ignore
            if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System Observation: {refresh_message}"})
            results.append(((action_to_execute, True, refresh_message, {"type": "shortcuts_refreshed", "token_usage": shortcut_tokens}))) # type: ignore
            logging.info(refresh_message + " Continuing to next planning cycle.")
            total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
            action_to_execute = None
            continue
        
        if not action_to_execute.get("_bypass_confirmation_"): # type: ignore
            if next_step_data and isinstance(next_step_data, dict):
                thought_content_json_str = json.dumps(next_step_data)
                agent_state.add_thought(thought_content_json_str, type="planning_json")
        elif not next_step_data :
            agent_state.add_thought(f"Iteration {current_iteration}: Plan was injected. Action: {action_type}, Params: {params}. Original Injection Reason: {plan_injection_reason}", type="planning_info")

        critique_passed = True
        critique_feedback = "Critique skipped for credential value request or injected plan."
        if not (pending_credential_request and credential_consent_choice in ['onetime', 'remember']) and not plan_injection_reason:
            critique_screenshot = planning_screenshot
            string_history_for_critique = [f"{msg['role']}: {msg['content']}" for msg in (agent_state.current_task.conversation_history if agent_state.current_task else [])]
            if not action_to_execute.get("_bypass_confirmation_"): # type: ignore
                logging.info(f"About to call critique_action with critic_model: {critic_model}")
                critique_passed, critique_feedback, critique_tokens = critique_action(
                    instruction_for_current_planning_cycle, string_history_for_critique, action_to_execute, critic_model, critique_screenshot # type: ignore
                )
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(critique_tokens)
                if not critique_passed:
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System: Proposed action {action_type} critiqued: {critique_feedback}. Reconsidering."})
                    results.append((({'action_type': 'CRITIQUE_FAILED', 'parameters': {'failed_action': action_type}}, False, f"Critique failed: {critique_feedback}", None)))
                    total_consecutive_failures = 0
                    consecutive_failures_on_current_step = 0
                    replan_attempts_current_cycle = 0
                    if pending_credential_request and not credential_consent_choice:
                        pending_credential_request = None
                    continue

        is_legitimate_overall_completion_check_for_action = False
        if action_type == "task_complete":
            is_legitimate_overall_completion_check_for_action = not current_sub_tasks or \
                                               (current_sub_tasks and current_sub_task_index == len(current_sub_tasks) - 1)
            if is_legitimate_overall_completion_check_for_action:
                task_completed = True
                logging.info(f"LLM planned 'task_complete' for the last/only sub-task. Flagging task as complete. Reason: {planning_reasoning}")
            else:
                logging.info(f"LLM planned 'task_complete' for sub-task {current_sub_task_index + 1}/{len(current_sub_tasks)} (not the last). Treating as current sub-task success signal.")
        
        exec_result = execute_action(action_to_execute, agent) # type: ignore
        exec_success = False; exec_message = "Execution error"; special_directive = None
        if isinstance(exec_result, tuple) and len(exec_result) == 3 and exec_result[0] == -2: exec_result = (True, exec_result[1], exec_result[2]) # type: ignore
        if isinstance(exec_result, tuple) and len(exec_result) == 3: exec_success, exec_message, special_directive = exec_result # type: ignore
        elif isinstance(exec_result, tuple) and len(exec_result) == 2: 
            exec_success, exec_message = exec_result # type: ignore
            special_directive = None

        if special_directive and isinstance(special_directive, dict):
            directive_type = special_directive.get("type")
            if exec_success is True and exec_message.startswith("Confirmation required for risky command.") and directive_type == "ask_user":
                question_text = special_directive.get('question', "No question provided for risky command.")
                logging.info(f"Execution yielded for risky command confirmation: {question_text}")
                pending_risky_action = action_to_execute.copy() # type: ignore
                results.append(((action_to_execute, True, f"Confirmation required: {question_text}", special_directive))) # type: ignore
                yield_result = {"type": "ask_user", "question": question_text}
                user_response_to_ask = yield yield_result
                if user_response_to_ask is None:
                    logging.warning("User cancelled during risky command confirmation.")
                    results[-1] = (action_to_execute, False, "User cancelled risky command confirmation.", None) # type: ignore
                    pending_risky_action = None; break
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_ask})
                continue

            if isinstance(special_directive, dict) and "token_usage" in special_directive:
                action_tokens = special_directive.get("token_usage")
                if agent_state.current_task and isinstance(action_tokens, dict): agent_state.current_task._accumulate_tokens(action_tokens)

            if directive_type == "ask_user":
                question_text = special_directive.get('question', '')
                results.append(((action_to_execute, exec_success, exec_message, special_directive))) # type: ignore
                yield_result = {"type": "ask_user", "question": question_text}
                user_response_to_ask = yield yield_result
                if user_response_to_ask is None:
                    logging.warning("User cancelled during ask_user.")
                    results[-1] = (action_to_execute, False, "User cancelled during ask_user.", None) # type: ignore
                    break
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_ask})
                total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
                current_task_instruction = user_response_to_ask
                action_to_execute = None
                continue
            elif directive_type == "inform_user":
                message_to_display = special_directive.get('message', '')
                agent_state.task_is_paused = True
                if agent_state.current_task: agent_state.current_task.status = "paused"
                yield {"type": "inform_user", "message": message_to_display, "youtube_references": special_directive.get("youtube_references", []), "image_data": special_directive.get("image_data")}
                total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
                action_to_execute = None
                continue
            elif directive_type == "inject_plan":
                plan_to_inject = special_directive.get("plan", [])
                plan_injection_reason = special_directive.get("reason", "Unknown directive")
                total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0

        string_history_for_assessment = [f"{msg['role']}: {msg['content']}" for msg in (agent_state.current_task.conversation_history if agent_state.current_task else [])]
        assessment_status, assessment_reasoning, assessment_tokens = assess_action_outcome( # type: ignore
            instruction_for_current_planning_cycle, action_to_execute, exec_success, exec_message, # type: ignore
            capture_full_screen(), llm_model, screenshot_before_hash=screenshot_before_action_hash
        )
        if agent_state.current_task: agent_state.current_task._accumulate_tokens(assessment_tokens)
        final_message = f"{exec_message} | Assessment: {assessment_status} - {assessment_reasoning}"
        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System Observation: {final_message}"})
        final_success = (assessment_status == "SUCCESS")
        results.append(((action_to_execute, final_success, final_message, special_directive))) # type: ignore
        logging.info(f"Assessment Result: {assessment_status} - {assessment_reasoning}")
        if agent_state.current_task: agent_state.current_task.agent_thoughts.append({"timestamp": datetime.now().isoformat(), "content": f"Step {current_iteration} Action: {action_type} {params}\nOutcome: {assessment_status} - {assessment_reasoning}", "type": "step_outcome"}) # type: ignore

        if assessment_status == "SUCCESS":
            total_consecutive_failures = 0
            if action_type: action_failure_counts[action_type] = 0 # type: ignore
            consecutive_failures_on_current_step = 0
            replan_attempts_current_cycle = 0
            if plan_injection_reason and not plan_to_inject:
                logging.info(f"Successfully completed injected plan (Reason: {plan_injection_reason}).")
                plan_injection_reason = None
            
            if action_type == "task_complete" and not is_legitimate_overall_completion_check_for_action: # type: ignore
                logging.info(f"Intermediate sub-task {current_sub_task_index + 1} ('{current_sub_tasks[current_sub_task_index]}') successfully completed via 'task_complete' action.")
            
            if current_sub_tasks and 0 <= current_sub_task_index < len(current_sub_tasks):
                logging.info(f"Sub-task {current_sub_task_index + 1} ('{current_sub_tasks[current_sub_task_index]}') processing was successful.")
                current_sub_task_index += 1
                if agent_state.current_task and hasattr(agent_state.current_task, 'current_sub_task_index'):
                    agent_state.current_task.current_sub_task_index = current_sub_task_index

                if current_sub_task_index < len(current_sub_tasks):
                    next_sub_task_instruction = current_sub_tasks[current_sub_task_index]
                    next_sub_task_log_info = f"System: Advancing to sub-task {current_sub_task_index + 1}/{len(current_sub_tasks)}: '{next_sub_task_instruction}'."
                    logging.info(next_sub_task_log_info)
                    if agent_state.current_task:
                        agent_state.current_task.conversation_history.append({"role": "system", "content": next_sub_task_log_info})
                else:
                    logging.info("All sub-tasks completed.")
                    if not task_completed:
                        logging.info("All sub-tasks finished. Marking overall task as complete.")
                        task_completed = True
            
            if task_completed:
                if agent_state.current_task and (not results or results[-1][0].get("action_type") != "task_complete"): # type: ignore
                     agent_state.current_task.conversation_history.append({"role": "system", "content": "Overall task completed."})
                logging.info(f"Overall task '{original_instruction}' completed.")
                break
            continue

        elif assessment_status == "RETRY_POSSIBLE":
            if consecutive_failures_on_current_step < MAX_RETRIES_PER_STEP:
                consecutive_failures_on_current_step += 1
                logging.warning(f"Assessment RETRY_POSSIBLE. Retrying step (Attempt {consecutive_failures_on_current_step + 1}/{MAX_RETRIES_PER_STEP + 1}). Reason: {assessment_reasoning}")
                results[-1] = (action_to_execute, False, f"{exec_message} | Assessment: RETRY_POSSIBLE - {assessment_reasoning} (Retry {consecutive_failures_on_current_step})", special_directive) # type: ignore
                time.sleep(1.0)
                current_iteration -=1
                continue
            else:
                logging.error(f"Max retries ({MAX_RETRIES_PER_STEP}) for current step reached after RETRY_POSSIBLE. Treating as FAILURE.")
                assessment_status = "FAILURE"
                total_consecutive_failures +=1

        if assessment_status == "FAILURE":
            logging.error(f"Assessment FAILURE for step. Reason: {assessment_reasoning}")
            total_consecutive_failures += 1
            consecutive_failures_on_current_step = 0
            is_planning_failure = "Planning failed:" in assessment_reasoning
            if is_planning_failure:
                pass
            elif replan_attempts_current_cycle < MAX_REPLAN_ATTEMPTS:
                pass
            
            if total_consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_ASK:
                pass
            else:
                logging.warning("Failure encountered. Proceeding to next iteration for standard planning or further failure handling.")
                continue
        else:
            if current_sub_tasks and current_sub_task_index >= len(current_sub_tasks):
                logging.info("All defined sub-tasks processed or unexpected assessment on last one.")
                task_completed = True
                break
            logging.error(f"Assessment returned unexpected status '{assessment_status}'. Stopping. Reason: {assessment_reasoning}")
            break

    if agent_state.current_task and current_iteration >= max_iterations:
        logging.warning(f"Execution stopped: Max iterations ({max_iterations}) reached.")
        results.append((({'action_type': 'STOP', 'parameters': {'reason': 'max_iterations'}}, False, f"Max iterations reached.", None)))
        if agent_state.current_task: agent_state.current_task.agent_thoughts.append({"timestamp": datetime.now().isoformat(), "content": f"Execution stopped: Max iterations ({max_iterations}) reached.", "type": "stop"})

    final_status = "incomplete"
    execution_summary_for_db = f"Instruction: {original_instruction}\nOutcome: "
    if task_completed and not results:
        final_status = "success"
        execution_summary_for_db += f"{final_status.upper()} - Task marked complete with no actions or all sub-tasks completed."
    elif results:
        last_action_tuple = results[-1]
        last_action_dict, last_success, last_message, _ = last_action_tuple # type: ignore
        if last_action_dict.get("action_type") == "task_complete" and last_success:
            final_status = "success"
        elif not last_success :
            final_status = "failure"
        elif task_completed:
            final_status = "success"
        execution_summary_for_db += f"{final_status.upper()} - Last action: {last_action_dict.get('action_type', 'N/A')}. Message: {last_message}"
    else:
        execution_summary_for_db += "No actions attempted or task did not complete."
        final_status = "failure" if not task_completed else "incomplete"

    serializable_results = []
    for r_action, r_success, r_message, r_directive in results:
        serializable_action = r_action if isinstance(r_action, dict) else {"error": "invalid_action_format", "original": str(r_action)}
        serializable_directive = r_directive if isinstance(r_directive, dict) else None
        serializable_results.append([serializable_action, r_success, r_message, serializable_directive])
    full_results_json = json.dumps(serializable_results)

    if agent_state.current_task:
        agent_state.current_task.action_failure_counts = action_failure_counts
        save_task_execution_to_db(original_instruction, execution_summary_for_db, full_results_json, final_status, plan_source, user_feedback=None) # type: ignore
        agent_state.current_task.status = "completed" if final_status == "success" else "failed"
        agent_state.current_task.end_time = datetime.now()
        agent_state.is_task_running = False
        logging.info(f"Finished iterative execution. Total iterations: {current_iteration}")
        logging.info(f"Total LLM tokens for this task (from TaskSession): {agent_state.current_task.total_tokens}")

    return results # type: ignore



def process_next_step(
    original_instruction: str,
    history: List[str],
    llm_model: genai.GenerativeModel,
    current_shortcuts: Optional[str] = None,
    current_reinforcements: List[str] = None,
    planning_screenshot_pil: Optional[Image.Image] = None,
    benchmark_mode: bool = False,
    executed_actions_summary: Optional[str] = None,
    # New parameters for sub-task context
    overall_original_instruction: Optional[str] = None, # The user's very first instruction for the whole task
    all_sub_tasks_count: int = 0, # Total number of sub-tasks if they exist
    current_sub_task_idx: int = -1 # 0-based index of the current sub-task being planned
) -> Tuple[Optional[dict], Dict[str, int]]:
    # Check if we already have a successful image generation
    for line in reversed(history):
        if "System Observation:" in line:
            if "IMAGE_GENERATED:" in line:  # Updated check
                return {
                    "next_action": { # Changed from "plan": [{...}] to "next_action": {...}
                        "action_type": "INFORM_USER",
                        "parameters": {
                            "message": "The image has already been successfully generated. Task is complete."
                        }
                    },
                    "reasoning": "Image generation task was already completed successfully."
                }, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    logging.info("Determining next step using standard planning logic.")
    if current_shortcuts is None: current_shortcuts = "" # Should be an empty string if no shortcuts
    
    max_history_turns_for_prompt = 10
    string_history = [str(item) for item in history]
    important_indices = [i for i, item in enumerate(string_history) if "System Observation:" in item or "User responded:" in item or "ask_user" in item or "Critique failed" in item or "User Goal:" in item]
    if len(string_history) > max_history_turns_for_prompt * 2:
        cutoff = len(string_history) - (max_history_turns_for_prompt * 2)
        keep_indices = {0} if len(string_history) > 0 and "User Goal:" in string_history[0] else set()
        keep_indices.update(range(cutoff, len(string_history)))
        keep_indices.update(idx for idx in important_indices if idx >= cutoff)
        keep_indices = {idx for idx in keep_indices if 0 <= idx < len(string_history)}
        truncated_history = [string_history[i] for i in sorted(list(keep_indices))]
    else:
        truncated_history = string_history
    history_str = "\n".join(truncated_history)

    screenshot_base64 = image_to_base64(planning_screenshot_pil) if planning_screenshot_pil else None

    shortcuts_str = "No specific shortcuts known for the current app."
    if current_shortcuts and isinstance(current_shortcuts, str) and current_shortcuts.strip():
        shortcuts_str = current_shortcuts.strip()
    else:
        shortcuts_str = "No specific shortcuts known for the current app or an error occurred fetching them."

    query_for_reinforcements = original_instruction
    if history:
        for i in range(len(history) -1, -1, -1):
            hist_item_lower = history[i].lower()
            if hist_item_lower.startswith("user goal:") or hist_item_lower.startswith("user responded:"):
                query_for_reinforcements = history[i].split(":", 1)[-1].strip()
                break

    retrieved_reinforcements = retrieve_relevant_reinforcements_from_db(query_for_reinforcements, n_results=5)
    reinforcements_str = "No relevant learnings/reinforcements found in DB."
    if retrieved_reinforcements:
        max_reinforcements_to_show = 5
        relevant_reinforcements = retrieved_reinforcements[:max_reinforcements_to_show]
        reinforcements_str = "Consider These Learnings/Reinforcements from DB:\n"
        for i, learning in enumerate(relevant_reinforcements):
            reinforcements_str += f"- {learning}\n"

    reinforcements_str += """

 **Leveraging Past Successes (from Learnings/Reinforcements):**
 *   If a retrieved Learning/Reinforcement from the DB is highly relevant and describes a specific tool, method, or application that was successful or explicitly liked by the user in a *verifiably similar past task context* (e.g., "User preferred using 'VS Code' for Python editing on task 'Project X' and it was successful"),
 *   THEN, your `next_action` SHOULD BE `{{ "action_type": "ask_user", "parameters": {{ "question": "I recall that for a similar task ('Project X'), you successfully used 'VS Code' for Python editing. Would you like to use that approach again for this task?" }} }}`.
 *   Adapt the question to the specific learning and the past task context if mentioned in the learning.
 *   Only do this if the learning is specific, actionable, and refers to a clear past success. If the learning is general (e.g., "wait after opening apps"), just consider it in your plan without asking.
 *   If the user agrees to reuse the approach, incorporate that into your subsequent plan steps.
 *   If the user declines, proceed with standard planning, potentially avoiding that specific recalled approach if it was the point of contention.
 """
    
    executed_actions_summary_str = executed_actions_summary if executed_actions_summary else "No actions executed yet in this task."

    # original_instruction for this function is the current sub-task's instruction.
    # overall_original_instruction is the main goal.
    sub_task_info_for_prompt = ""
    if overall_original_instruction and all_sub_tasks_count > 0 and current_sub_task_idx != -1:
        sub_task_info_for_prompt = f"""
--- SUB-TASK CONTEXT ---
You are currently working on a specific sub-task as part of a larger user request.
Overall User Goal: "{overall_original_instruction}"
Total Sub-tasks: {all_sub_tasks_count}
Current Sub-Task Number: {current_sub_task_idx + 1}
Instruction for THIS Current Sub-Task: "{original_instruction}"
---
"""

    task_name_base = sanitize_filename(original_instruction)[:30]

    prompt = rf"""
--- CURRENT SUB-TASK OBJECTIVE / USER GOAL ---
**Instruction for this step (current sub-task):**
"{original_instruction}"

{sub_task_info_for_prompt}
--- HISTORY & CONTEXT ---
**Recent History (Actions, Outcomes, Observations, User Responses, Critiques):**
{history_str}
*   **CRITICAL: Focus on the LATEST user message/response.** This defines the current objective. If the user just agreed to create a specialized agent or selected an LLM, plan the next step for *that* process.
*   **IF A "Critique failed" message is in the LATEST history entries:**
    *   The `reasoning` for that critique (e.g., "action was unrelated to the LATEST user instruction", "action was redundant") is the MOST IMPORTANT piece of information for your next plan.
    *   Your `next_action` MUST directly address the LATEST user instruction and AVOID the specific pitfall highlighted by the critique.
    *   DO NOT repeat the same type of irrelevant or redundant action.
*   **CRITICAL: Learn from critiques.** Avoid repeating actions that failed critique due to redundancy, speculation, or irrelevance.
*   Note: "System Observation: Screen content changed..." indicates the visual state changed.
*   Note: "System Observation: Active application changed to 'app_name.exe'." or "System Observation: Opened web URL: https://example.com" indicates the current application context.
*   **VERY IMPORTANT CONTEXT CHECK:** Before planning `navigate_web` or `run_shell_command` to open an application, check the LATEST "System Observation" in history. If it indicates you are already on the target URL (e.g., "System Observation: Opened web URL: https://www.youtube.com") or the target application is already active (e.g., "System Observation: Active application changed to 'youtube.com' or 'chrome.exe' showing YouTube"), DO NOT plan to navigate or open it again if the sub-task is to interact *within* that page/app. Instead, plan actions *within* that context (e.g., `click_and_type` in a search bar, `click` on a video).
*   Note: Check history for user responses to 'ask_user', especially regarding automation, project details, LLM choices, or credentials.

**Actions Already Executed in Current Task Iteration (Most Recent First):**
{executed_actions_summary_str}
*   Note: Check history for user responses to 'ask_user', especially regarding automation, project details, LLM choices, or credentials.

**Application Shortcuts:**
{shortcuts_str}

**Learnings/Reinforcements from DB:**
{reinforcements_str}

**Current Visual Context:** [See attached screenshot if available - Use ONLY if planning a visual action (Priority 4) or assessing task completion visually.]
*   **IMPORTANT:** Do not assume you need to see the screen for every step. Only consider the visual context if the next logical step requires interacting with the GUI (e.g., `click`, `type`) or if you need to verify a visual change. Avoid unnecessary `describe_screen` or visual analysis if the task can be done via shell or script.

--- TASK PRIORITIZATION & RESPONSE STRATEGY ---

--- META-PLANNING FOR COMPLEX PROJECTS (e.g., "create a project", "build an app", "design a system") ---
IF the LATEST user instruction describes a complex, multi-stage project (especially involving creation or design of software, systems, or substantial files) AND you are at the beginning of this project (no significant project-specific actions taken yet in history for this goal):

1.  **Initial Check & User Response Handling:**
    *   **IF the PREVIOUS action was `INFORM_USER` presenting multiple paths with "--- START MERMAID CODE ---" markers:**
        *   Your `next_action` MUST be `{{ "action_type": "ask_user", "parameters": {{ "question": "Which of the above approaches (e.g., 'Path 1' or by title) would you like to proceed with? Or, would you like me to research further or suggest alternatives?" }} }}`.
        *   `reasoning`: "Asking user to select a development path after presenting options."
        *   **STOP HERE. Do not proceed to other planning steps below if this condition is met.**
    *   **IF the LATEST user message in history is a response to the path selection question above:**
        *   Analyze the user's choice.
        *   Your `next_action` should be to start implementing the CHOSEN path. This might involve a `multi_action` for the first few steps of that path, or a single action like `write_file` for the main project file.
        *   Your `reasoning` should state: "Proceeding with user's chosen path: [User's Choice]. Starting with [first action of chosen path]."
        *   **STOP HERE. Do not proceed to other planning steps below if this condition is met.**

2.  **Research Phase (If no paths presented yet, or if user asked for more research):**
    *   **Check History:** Have you already performed web searches for "how to [do the project]", "technologies for [project type]", "steps to create [X]" in the very recent history for THIS specific project goal?
    *   **IF NOT (no recent relevant research for this project):**
        *   Your `next_action` SHOULD be `search_web` or a `multi_action` of `search_web` and/or `search_youtube` actions.
        *   **Example Query:** "best ways to create a Python script for web scraping and data storage" or "modern tech stack for a simple blog website".
        *   `reasoning`: "Researching best practices, technologies, and common steps for creating the requested project/file before proposing implementation paths."
        *   **STOP HERE. Do not proceed to other planning steps below if this condition is met.**

3.  **Path Proposal Phase (After research results are in history and no paths presented yet):**
    *   **Check History:** Are there recent `search_web` or `search_youtube` results in history relevant to this project?
    *   **IF YES (and you haven't presented paths yet):**
        *   Your `next_action` MUST be `{{ "action_type": "INFORM_USER", "parameters": {{ "message": "..." }} }}`.
        *   The `message` parameter should be a string containing:
            *   An introduction, e.g., "Okay, I've researched how to approach '[original_instruction]'. Here are a few potential paths:\n\n"
            *   **Path 1:**
                *   "**Path 1: [Concise Title for Approach 1, e.g., Using Flask and SQLite]**\n"
                *   "[One-paragraph description of this approach, key steps, major technologies, and brief pros/cons. If based on web search, you can briefly mention key findings.]\n"
                *   "--- START MERMAID CODE ---\n"
                *   "sequenceDiagram\n"
                *   "    User->>Agent: Request to build X\n"
                *   "    Agent->>System: Plan initial setup (e.g., Flask app.py)\n"
                *   "    System->>Agent: Execute setup\n"
                *   "    Agent->>System: Plan database models\n"
                *   "    System->>Agent: Execute DB setup\n"
                *   "    Agent->>User: Show progress / Ask for next feature\n"
                *   "--- END MERMAID CODE ---\n\n"
            *   **Path 2 (and optionally Path 3):** Similar structure as Path 1, presenting a different approach.
                *   "**Path 2: [Concise Title for Approach 2, e.g., Static Site Generator with Netlify]**\n"
                *   "[Description...]\n"
                *   "--- START MERMAID CODE ---\n"
                *   "sequenceDiagram\n"
                *   "    ...\n"
                *   "--- END MERMAID CODE ---"
            *   (Ensure Mermaid code is valid and uses `sequenceDiagram`. Use `\n` for newlines within the message string.)
        *   `reasoning`: "Presenting researched high-level strategic paths to the user for selection, including Mermaid diagrams for clarity."
        *   **STOP HERE. Do not proceed to other planning steps below if this condition is met.**

ELSE (if not a complex project initiation, or if a path has been chosen and implementation is underway):
    Proceed with the standard planning logic below (Informational Queries, Task Completion, etc.).
--- END META-PLANNING FOR COMPLEX PROJECTS ---

--- STANDARD PLANNING LOGIC ---
1.  **Information Gathering (If Needed):**
    *   **Consider the project's nature:** Is it a common type of project with well-known architectures (e.g., a standard web app)? Or is it niche, requiring research?
    *   **Check Learnings/Reinforcements:** Do any past learnings from the DB suggest specific architectures, tools, or user preferences for similar complex tasks?
    *   **IF information is lacking OR the project is novel/niche OR you want to ensure up-to-date best practices:**
        *   Your `next_action` SHOULD be to gather information.
        *   This might involve a single `search_web` or `search_youtube` action, OR, if multiple pieces of information are needed, a `multi_action` containing a sequence of search actions.
        *   **Example of `multi_action` for diverse research:**
            `{{ "action_type": "multi_action", "parameters": {{ "sequence": [
                {{ "action_type": "search_web", "parameters": {{ "query": "best architectures for e-commerce platform with AI chatbot 2024" }} }},
                {{ "action_type": "search_youtube", "parameters": {{ "query": "tutorial integrating Rasa chatbot with Flask backend" }} }},
                {{ "action_type": "search_web", "parameters": {{ "query": "comparison of React vs Vue for e-commerce frontend" }} }},
                {{ "action_type": "read_file", "parameters": {{ "file_path": "C:\\Users\\MyUser\\Documents\\project_notes\\past_ecommerce_learnings.txt" }} }}
            ] }} }}`
        *   Your `reasoning` should explain why you are performing this research (e.g., "To comprehensively research architectures, AI tools, and frontend technologies for the e-commerce project before proposing high-level plans.").
    *   **AFTER information gathering (in a subsequent turn, once search results are in history):** The "META-PLANNING FOR COMPLEX PROJECTS" section (Path Proposal Phase) will handle proposing paths.

--- COMPLEX TASK - MULTIPLE PATH PROPOSAL (INFORM & ASK) ---
IF you are at the stage of proposing multiple paths (either directly or after information gathering),
THEN, your `next_action` MUST be `INFORM_USER`. The `message` parameter for this `INFORM_USER` action should contain 2-3 distinct high-level strategic approaches (paths) based on your knowledge and any research done.

For each proposed path within the `message`, you MUST include:
1.  **Title**: A short, descriptive title (e.g., "**Path 1: Serverless Stack**").
2.  **Description**: A paragraph explaining the key stages, technologies, and pros/cons. If based on web search, briefly mention the source or key findings.
3.  **Mermaid Sequence Diagram**: Valid Mermaid code for a `sequenceDiagram` visualizing the path. Enclose this code clearly use mermaid version 11.6.0 :
    `--- START MERMAID CODE ---`
    `sequenceDiagram`
    `    participant User`
    `    participant WebApp`
    `    User->>WebApp: Action`
    `    WebApp-->>User: Result`
    `--- END MERMAID CODE ---`
    (Ensure newlines `\n` are correctly represented if generating this as part of a JSON string, but for the `INFORM_USER` message, literal newlines are fine).
Your `reasoning` for this `INFORM_USER` action should state that you are presenting multiple high-level paths for the complex project, possibly informed by prior research.
**IMPORTANT FOLLOW-UP:** If the PREVIOUS action was `INFORM_USER` and its message contained these multi-path proposals (identifiable by "--- START MERMAID CODE ---" markers), your *current* `next_action` MUST be `{{ "action_type": "ask_user", "parameters": {{ "question": "Which of the above approaches (e.g., 'Path 1' or by title) would you like to proceed with?" }} }}`.
--- END COMPLEX TASK SECTIONS ---

1.  **Informational Queries vs. Actionable Tasks:**
    *   **IF the LATEST user message is primarily an informational question** (e.g., "What is X?", "How does Y work?", "Tell me about Z", "What are the best ways to do A?", "will X release Y?", "Konami upcoming games?"), consider the following:
        *   **CRITICAL CHECK FOR PRIOR SEARCH:**
            *   Examine the **Recent History**.
*   **Shortcut Prioritization (VERY IMPORTANT):**
    *   **IF `Application Shortcuts` are available AND contain a relevant shortcut for the current sub-task** (e.g., a shortcut for 'search', 'find', 'open search bar', 'focus search' when the sub-task is to search for something; or 'play/pause', 'next track' for media control):
        *   Your **FIRST ATTEMPT** to achieve the sub-task **MUST** be to use the relevant shortcut via the `press_keys` action.
        *   Your `reasoning` for choosing `press_keys` in this case **MUST** state which shortcut you are using and why (e.g., "Using the '/' shortcut to focus the YouTube search bar as per available shortcuts.").
        *   **Example:** If sub-task is "Search for 'genius by sia'" and shortcuts include "/: Focus search bar", your `next_action` should be `{{ "action_type": "press_keys", "parameters": {{ "keys": ["/"] }} }}`.
        *   **IF the shortcut only focuses an input field (like a search bar):** Your `next_action` should be a `multi_action` sequence:
            1.  `press_keys` (to activate/focus the field using the shortcut).
            2.  `wait` (e.g., 0.1 to 0.3 seconds, ONLY if truly necessary for the field to become active after the shortcut. If the shortcut directly allows typing, this wait can be omitted or very short).
            3.  `type` (to input the text, e.g., the search query).
            4.  (Optional) `press_keys` (e.g., ["enter"] to submit the search/form).
        *   Example (multi_action sequence to search in youtube):
         ```json
         {{
           "action_type": "multi_action",
           "parameters": {{
             "sequence": [{{ "action_type": "press_keys", "parameters": {{ "keys": ["/"] }} }}, {{ "action_type": "wait", "parameters": {{ "duration_seconds": 0.5 }} }}, {{ "action_type": "type", "parameters": {{ "text_to_type": "genius by sia" }} }},{{ "action_type": "press_keys", "parameters": {{ "keys": ["enter"] }} }}]
           }}
         }}
         ```
        *   Your `reasoning` for such a `multi_action` should also explain the shortcut's role.
        *   Only if no relevant shortcut is found, or if a shortcut attempt has clearly failed (based on history), should you then consider visual actions like `click`, `click_and_type`.
            *   **IF the IMMEDIATELY PRECEDING successful action in the history** was `search_web` or `search_youtube` AND its output (visible in history as 'System Observation: Web Search Result: ...' or 'System Observation: YouTube Search Result: ...') is relevant to the LATEST user's informational question:
                *   Your `next_action` **MUST** be `{{ "action_type": "INFORM_USER", "parameters": {{ "message": "Based on my recent search: [Concise summary of the relevant search results FROM HISTORY. If the search result indicates no definitive answer, state that clearly, e.g., 'My search did not find a specific release date, but indicated X...']" }} }}`.
                *   Your `reasoning` MUST state that you are using information from the immediately preceding search result.
                *   **DO NOT** plan another `search_web` or `search_youtube` for the same or a very similar query if a relevant search result is already in the immediate history.
            *   **ELSE (no immediately preceding relevant search result in history, OR the preceding search was clearly insufficient and the user is asking for more/different info, OR if this is the very first step for this informational query):**
                *   Plan to use `search_web` or `search_youtube`.
                *   **For "how-to", "tutorial", "best ways to", or visually demonstrable topics, STRONGLY prefer using the `search_youtube` tool.**
                *   If you use `search_youtube`, the next step will typically be `INFORM_USER` with the findings.
                *   You can use `search_youtube` in conjunction with `search_web` (e.g., in a `multi_action`) if you want to provide comprehensive information from both sources, or ask the user if they prefer web or video results before searching.
                *   After the search action(s), the next step will typically be `INFORM_USER` with the findings.
                *   Your `reasoning` should explain why you are choosing the search tool(s).
                *   **IMPORTANT SELF-CORRECTION:** If your reasoning for planning a search includes a phrase like "results are not yet available in the history" or "planning the search again ensures it is the intended action", this is likely an error. You should only plan a search if one hasn't *just been performed* or if the user is asking for *new/different* information. If a search was the last action, the next step is to process its results.
                *   **CRITICAL JSON OUTPUT FOR SEARCH:** If planning a search, your `next_action` (e.g., `search_web`, `search_youtube`, or `multi_action` containing them) and `reasoning` MUST be part of the overall JSON object as specified in the main "Output Format" section.
    *   **ELSE (if it's an actionable task for PC automation AND NOT a complex project initiation phase as described in "META-PLANNING"):** Proceed with the planning logic below.
    *   **CRITICAL PLANNING CONSTRAINT:** Review the **Recent History** carefully. If you see a "System Note: [Tool Name] tool disabled..." message, you **MUST NOT** plan to use that specific tool (`action_type`) for the remainder of this task. Choose an alternative tool or strategy, or use `ask_user` if no viable alternative exists.

2.  **Follow-up After `INFORM_USER` (Non-Project Path Proposal):**
    *   **IF the PREVIOUS action you took was `INFORM_USER` and it was successful, AND the LATEST user message is not a new actionable task or a direct follow-up question to your information, AND the INFORM_USER message did NOT contain '--- START MERMAID CODE ---' markers (meaning it wasn't a multi-path proposal):**
        *   Your `next_action` MUST be `{{ "action_type": "ask_user", "parameters": {{ "question": "Is there anything else I can help you automate today?" }} }}`.
        *   Your `reasoning` should be "Following up after providing information."
    *   **ELSE:** Proceed with other checks.

3.  **Task Completion Flow (Revised):**
    *   You are currently planning for the instruction: "{original_instruction}".
    *   This is {(f'sub-task {current_sub_task_idx + 1} of {all_sub_tasks_count} for the overall goal: "{overall_original_instruction}".') if all_sub_tasks_count > 0 and current_sub_task_idx != -1 else "the main/only task."}
    *   **A. Check for User Response to "Anything else?" question:**
        *   **IF the LATEST user message in history is a response to a question like "Is there anything else I can help you with?" or "Is there anything more?":**
            *   **IF the user's response is negative** (e.g., "no", "that's all", "nope", "all good"):
                *   Your `next_action` MUST be `{{ "action_type": "task_complete", "parameters": {{}} }}`.
                *   `reasoning`: "User confirmed no further assistance needed. Completing the task."
                *   **STOP HERE. This is the final action.**
            *   **ELSE (user's response is affirmative or a new request):**
                *   Treat the user's new response as the NEW `original_instruction` for this planning cycle.
                *   `reasoning`: "User provided a new request or follow-up. Planning for that now."
                *   Proceed to plan for this new instruction (e.g., it might be a simple action, or trigger complex project planning if it's a new project).
                *   **STOP HERE if this condition was met and a new instruction is being processed. The rest of the completion check below is skipped.**
    *   **B. Standard Completion Check (if not handling response to "Anything else?"):**
        *   **IF the current instruction ("{original_instruction}") appears fully addressed by the PREVIOUS successful action(s) in the history:**
            *   **AND IF this is the LAST sub-task** (this means {(f'current sub-task {current_sub_task_idx + 1} IS THE LAST of {all_sub_tasks_count} sub-tasks') if all_sub_tasks_count > 0 and current_sub_task_idx != -1 and (current_sub_task_idx + 1 == all_sub_tasks_count) else (f'this is NOT the last sub-task (currently {current_sub_task_idx + 1} of {all_sub_tasks_count})' if all_sub_tasks_count > 0 and current_sub_task_idx != -1 else 'this is the only task')}).
                *   Then, your `next_action` MUST be `{{ "action_type": "ask_user", "parameters": {{ "question": "I believe I've completed [provide a very brief, 2-5 word summary of what was just accomplished, e.g., 'playing the video', 'creating the Python script', 'searching for news']. Is there anything else I can help you with today?" }} }}`.
                *   `reasoning`: "The current goal appears to be met. Confirming with the user and asking if further assistance is needed before marking the overall task complete."
                *   **STOP HERE. This `ask_user` is the planned action.**
            *   **ELSE (if this is NOT the last sub-task, OR if there's no sub-tasking context and the goal isn't fully met by previous actions):**
                *   DO NOT use `task_complete` or the "anything else" `ask_user` yet.
                *   If the current sub-task is complete, and more sub-tasks remain, the executor system will handle advancing. Your focus is on the *current* instruction.
                *   Proceed to plan the next action for the current instruction ("{original_instruction}").
    *   **ELSE:** Proceed with planning the next sub-step towards the user's goal.

4.  **Handling Multiple Distinct Tasks in One Instruction:**
    *   **IF the LATEST user instruction appears to contain multiple, clearly distinct, and non-trivial automation tasks** (e.g., "Open Chrome and search for news, then open Notepad and write a summary, then save it to my_summary.txt") AND it's NOT a complex project requiring multiple path proposals:
        *   Your `next_action` MUST be `{{ "action_type": "ask_user", "parameters": {{ "question": "I see you've asked for several things (e.g., Task A, Task B). Which one should I start with, or would you like me to try them in order?" }} }}` (Adapt the question to the detected tasks).
        *   DO NOT attempt to create a single giant plan for many distinct complex tasks. Clarify first.
    *   **EXCEPTION:** If the tasks are very simple and form a natural, tight sequence (e.g., "open notepad and type hello"), you MAY use a `multi_action` for them. Use your judgment. If in doubt, clarify with `ask_user`.

--- AVAILABLE TOOLS (Action Types & Parameters) ---
*   `{{ "action_type": "focus_window", "parameters": {{ "title_substring": "Notepad" }} }}`
*   `{{ "action_type": "click", "parameters": {{ "element_description": "File menu button" }} }}`
*   `{{ "action_type": "type", "parameters": {{ "text_to_type": "Hello World!", "interval_seconds": 0.05 }} }}`
*   `{{ "action_type": "press_keys", "parameters": {{ "keys": ["ctrl", "s"] }} }}`
*   `{{ "action_type": "move_mouse", "parameters": {{ "x": 100, "y": 200, "duration_seconds": 0.25 }} }}`
*   `{{ "action_type": "run_shell_command", "parameters": {{ "command": "mkdir my_folder", "working_directory": "C:\\Users\\User\\Desktop" }} }}`
*   `{{ "action_type": "run_python_script", "parameters": {{ "script_path": "scripts/process_data.py", "working_directory": "OPTIONAL_PATH" }} }}`
*   `{{ "action_type": "write_file", "parameters": {{ "file_path": "output.txt", "content": "Data processed." }} }}`
*   `{{ "action_type": "navigate_web", "parameters": {{ "url": "https://google.com" }} }}`
*   `{{ "action_type": "search_web", "parameters": {{ "query": "latest AI news" }} }}`
*   `{{ "action_type": "search_youtube", "parameters": {{ "query": "how to bake a cake" }} }}`
*   `{{ "action_type": "wait", "parameters": {{ "duration_seconds": 2.0 }} }}`
*   `{{ "action_type": "ask_user", "parameters": {{ "question": "What filename should I use?" }} }}`
*   `{{ "action_type": "describe_screen", "parameters": {{}} }}`
*   `{{ "action_type": "capture_screenshot", "parameters": {{ "file_path": "debug_screenshot.png" }} }}`
*   `{{ "action_type": "click_and_type", "parameters": {{ "element_description": "Search input field", "text_to_type": "query text", "press_enter_after": false }} }}` 
    *   Set `press_enter_after` to `true` if you want to automatically press Enter after typing (e.g., for submitting a search query).
*   `{{ "action_type": "multi_action", "parameters": {{ "sequence": [ {{...action1...}}, {{...action2...}} ] }} }}`
*   `{{ "action_type": "task_complete", "parameters": {{}} }}`
*   `{{ "action_type": "read_file", "parameters": {{ "file_path": "path/to/file.txt" }} }}`
*   `{{ "action_type": "INFORM_USER", "parameters": {{ "message": "Information for the user." }} }}`
*   `{{ "action_type": "process_local_files", "parameters": {{ "file_path": "path/to/file.txt", "prompt": "Analyze this file" }} }}`
*   `{{ "action_type": "process_files_from_urls", "parameters": {{ "url": "https://example.com/file.txt", "prompt": "Analyze this file" }} }}`
*   `{{ "action_type": "start_visual_listener", "parameters": {{ "description_of_change": "...", "polling_interval_seconds": 10, "timeout_seconds": 300, "actions_on_detection": [{{...plan...}}], "actions_on_timeout": [{{...plan...}}]? }} }}`
*   `{{ "action_type": "refresh_application_shortcuts", "parameters": {{}} }}`
*   `{{ "action_type": "edit_image_with_file", "parameters": {{ "image_path": "OPTIONAL_PATH_TO_IMAGE_OR_EMPTY_STRING_FOR_NEW", "prompt": "Description of edits OR image to generate" }} }}`

*   `{{ "action_type": "edit_image_with_file", "parameters": {{ "image_path": "OPTIONAL_PATH_TO_IMAGE_OR_EMPTY_STRING_FOR_NEW", "prompt": "Description of edits OR image to generate" }} }}`
    - **To generate a NEW image:** Set `image_path` to an empty string (`\"\"`) or `null`. The `prompt` should describe the image to create (e.g., "A photo of a red apple on a wooden table").
    - **To edit an EXISTING image:** Provide the `image_path` to the image file. The `prompt` should describe the desired edits (e.g., "Make the background blurry", "Convert to grayscale").
    - Use this action for any requests involving image creation, generation, editing, or modification based on a text prompt.

    Example (Generating a new image):
    `{{
      \"action_type\": \"edit_image_with_file\",
      \"parameters\": {{
        \"image_path\": \"\", 
        \"prompt\": \"A cute cat wearing a tiny hat\"
      }}
    }}`

    Example (Editing an existing image):
    `{{
      \"action_type\": \"edit_image_with_file\",
      \"parameters\": {{
        \"image_path\": \"uploads/user_photo.jpg\",
        \"prompt\": \"Increase brightness and add a vintage filter\"
      }}
    }}`

**General Planning & Reasoning Strategy:**
*   **Context is Key:** Always consider the conversation history and the implied state of the system.
    *   **Check Current URL/Application:** Before planning `navigate_web` or `run_shell_command` to open an application, check the LATEST "System Observation" in history.
        *   If the history shows you are already on the target URL (e.g., "System Observation: Opened web URL: https://www.youtube.com" or the active window title in screenshot implies it) AND the current sub-task is to perform an action *on that page* (like searching, clicking a button), **DO NOT plan `navigate_web` to the same URL again.** Instead, plan the interaction (e.g., `click_and_type` for search, `click` for a button).
        *   Similarly, if the target application is already active (e.g., "System Observation: Active application changed to 'notepad.exe'"), DO NOT plan `run_shell_command` to open it again if the sub-task is to interact with the already open instance.
    *   **Critique Adherence:** If a previous action was critiqued as "redundant" (especially for navigation), explicitly avoid repeating that navigation.
    *   **Handling Lost Focus:** If the **Recent History** indicates you were recently in the correct application or on the correct URL for the current sub-task, BUT the **Current Visual Context (Screenshot)** or the LATEST "System Observation: Active application changed to..." shows a *different, incorrect* application is now active (e.g., you were on YouTube, now VS Code is active):
        *   Your **FIRST `next_action`** MUST be `{{ "action_type": "focus_window", "parameters": {{ "title_substring": "..." }} }}` to attempt to switch back to the correct application window (e.g., browser window containing "YouTube", or "Notepad").
        *   The `title_substring` should be specific enough to target the correct window (e.g., "YouTube - Brave", "Untitled - Notepad", or a more general "Brave" or "Chrome" if the specific title part is uncertain).
        *   Only if this `focus_window` action is known to have failed (e.g., from a previous attempt in history for the same situation) OR if there's no known window to focus, should you consider re-opening the application (`run_shell_command`) or re-navigating (`navigate_web`).
        *   Your `reasoning` should explicitly state that you are attempting to regain focus on the correct application due to a mismatch between recent history and current active window.

*   **Simplicity First:** Prefer simpler, more direct actions (like shell commands or keyboard shortcuts) if they can reliably achieve the sub-goal.
*   **For simple, direct tasks, you might only need a single action or a short `multi_action` sequence.**
    *   Examples of simple tasks:
        *   'open notepad and type hello' (could be a `multi_action` with `run_shell_command`, `wait`, `focus_window`, `type`)
        *   'what is the weather in London?' (could be a single `search_web` followed by `INFORM_USER` in the next turn)
        *   'save the current document' (could be a single `press_keys` like `ctrl+s` if shortcuts are known)
        *   'create a folder named temp on the desktop' (could be a single `run_shell_command` like `mkdir C:\\Users\\User\\Desktop\\temp`)
    *   Do not overcomplicate simple requests. If a task can be done in 1-3 steps, plan accordingly.

*   **Visual Actions for GUI:** Use visual interaction tools (`focus_window`, `click`, `type`) when dealing with graphical user interfaces that lack direct command-line or shortcut control.
    *   **Focus Before Interaction:** Always use `focus_window` before `click` or `type` within a specific application window to ensure the action targets the correct application.
    *   **Specificity for Clicks:** When using `click`, provide a highly specific `element_description` based on visual cues (text label, icon type, relative position) to ensure the correct element is targeted.
*   **Shell Commands for Backend/Files:** Use `run_shell_command` for file operations (creating, moving, deleting), running scripts, launching applications, or executing background tasks. Be mindful of quoting and escaping within the command string. Avoid using `echo` for creating files with complex or multi-line content unless absolutely necessary and simple.
    *   **Idempotency/Pre-checks:** When planning to create a directory (`mkdir`) or a file, consider if its pre-existence is an issue. **Prefer `write_file` for creating/editing file content over shell commands like `echo`.**
        *   If the goal is to *ensure* it exists (and pre-existence is fine), the plan can proceed after the creation attempt, even if it reports "already exists".
        *   If pre-existence is a problem (e.g., must be a fresh directory), the plan might need to include steps to delete/rename the existing one first (use `ask_user` if this is destructive and not explicitly requested).
        *   For `write_file`, if overwriting is not desired, plan to check for the file's existence first (e.g., using `run_shell_command` with `dir` or `ls`, then `ask_user` or `read_file` to decide).
*   **Keyboard Shortcuts:** Use `press_keys` for common application functions (Save, Copy, Paste, Close Tab/Window, etc.) or OS-level shortcuts.
*   **Waiting:** Use `wait` judiciously after actions that might take time to complete, such as launching an application, opening a file, or allowing a web page to load, to prevent subsequent steps from failing.
*   **Clarification:** If the user's instruction is ambiguous, or if a critical piece of information is missing (like which editor to use for a complex file), use `ask_user` to request clarification before proceeding.
*   **Screen Awareness:** Use `describe_screen` only if the user explicitly asks what is visible or if you need to understand the current visual state before planning complex interactions.

**For Complex Software Development Tasks (Act like a Senior Software Architect/Lead):**
(This section is CRITICAL for when the agent is doing detailed planning *after* a high-level path has been chosen by the user.)
IF the current goal is to implement a chosen high-level path for a software project:
1.  **Holistic Breakdown:** Decompose the project into major components and phases (e.g., Project Setup & Scaffolding, Backend API Development, Frontend UI Development, Database Design & Migrations, DevOps & CI/CD Setup, Unit & Integration Testing, Deployment Strategy).
2.  **Detailed, Professional Plan:** For each phase, plan specific, actionable steps using the available tools.
    *   **Project Setup:** Use `run_shell_command` for creating project directories, initializing version control (e.g., `git init`), setting up virtual environments (e.g., `python -m venv .venv`), installing linters/formatters. Use `write_file` for `README.md`, `.gitignore`, initial configuration files (e.g., `config.py`, `settings.json`), `requirements.txt` or `package.json`.
    *   **Backend (e.g., Python/Flask/Django, Node/Express):** Plan `write_file` actions for creating individual source files (e.g., `app.py`, `models.py`, `views.py`, `routes.js`, `controllers.js`). Generate well-structured, professional-quality code. Plan `run_shell_command` for installing framework dependencies.
    *   **Frontend (e.g., React/Vue/Angular):** Plan `run_shell_command` to initialize the frontend project (e.g., `npx create-react-app my-app`). Plan `write_file` actions for creating components, services, CSS/SCSS files with professional code.
    *   **Database:** Plan `write_file` for SQL schema definitions, ORM models, or migration scripts. Plan `run_shell_command` to apply migrations if applicable.
    *   **DevOps/Deployment:** Plan `write_file` for `Dockerfile`, `docker-compose.yml`, CI/CD pipeline configuration files (e.g., GitHub Actions `.github/workflows/main.yml`, Jenkinsfile), and deployment scripts.
    *   **Testing:** Plan `write_file` actions to create unit tests, integration tests, and end-to-end test stubs.
3.  **Code Generation Quality:** When using `write_file` for code, generate complete, functional, and professional-quality code. Adhere to best practices, include comments, and ensure proper structure for the language/framework. Leverage your large context window to generate substantial code blocks.
4.  **File Modification Strategy (Read-Modify-Write):**
    *   To modify an existing file: Use a `multi_action` sequence:
        a. `read_file` with the `file_path`.
        b. (Optional) `wait` or `start_visual_listener` if needed.
        c. `write_file` for the *same* `file_path`. The `content` for this `write_file` MUST be the complete, new version of the file, incorporating changes based on the read content and user instructions. Ensure JSON escaping for the `content`.
5.  **Iterative Approach for Very Large Tasks:** While the goal is a comprehensive upfront plan, for extremely large software, the plan might cover the initial major phases, with an understanding that further detailed planning might occur as these phases complete.

**Output Format:**
Provide ONLY the raw JSON object containing 'next_action' and 'reasoning'. Your entire response MUST be ONLY the valid JSON data, starting with `{{` and ending with `}}`.
DO NOT include markdown fences like \`\`\`json or \`\`\`, and DO NOT add any introductory text or explanations.
- The `next_action` must be a single action dictionary: `{{ "action_type": "...", "parameters": {{...}} }}`.
  - If you need to perform a sequence of tightly coupled actions (like the "Open Notepad, go to Format menu..." example), use `{{ "action_type": "multi_action", "parameters": {{ "sequence": [{{...action1...}}, {{...action2...}} ] }} }}` as the `next_action`.
- The `reasoning` should be a string explaining your thought process for choosing this next action.

Example of the exact output format required (single action):
User: open notepad
{{
  "next_action": {{ "action_type": "run_shell_command", "parameters": {{ "command": "notepad.exe" }} }},
  "reasoning": "The user wants to open Notepad. The 'run_shell_command' action is suitable for launching applications."
}}

Example of the exact output format required (multi_action for a sequence):
User: open notepad and type 'hello'
{{
  "next_action": {{
    "action_type": "multi_action",
    "parameters": {{
      "sequence": [
        {{ "action_type": "run_shell_command", "parameters": {{ "command": "notepad.exe" }} }},
        {{ "action_type": "wait", "parameters": {{ "duration_seconds": 1.5 }} }},
        {{ "action_type": "focus_window", "parameters": {{ "title_substring": "Notepad" }} }},
        {{ "action_type": "type", "parameters": {{ "text_to_type": "hello" }} }}
      ]
    }}
  }},
  "reasoning": "The user wants to open Notepad and type 'hello'. This requires a sequence: launch Notepad, wait for it to open, focus the window, then type. A 'multi_action' is appropriate."
}}

Some hints:
- If you having a conversation in an application use enter key to send messages.
- You are in windows so do not use cmds of linux or macos
- If you are not sure about the action type, use 'ask_user' to clarify.
- with the tools you have you can literally do any task, you have only to make smart plans and use them wisely
- Sometime you can use actions to do other purposes , for exemple you can use the click action to toggle the screenshot so you can see, so it s not nessesary use click only to see
- Shortcuts can be used also in websites if you want to search for shortcuts list of a website you can rely on the tool navigate_web
"""

    content = [{"text": prompt}]
    if screenshot_base64:
        content.append({"inline_data": {"mime_type": "image/png", "data": screenshot_base64}})

    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    step_dict = None # Initialize step_dict
    raw_response_text = "" # Initialize raw_response_text
    cleaned_text = "" # Initialize cleaned_text

    try:
        safety_settings = {
             'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        }
        response = llm_model.generate_content(content, safety_settings=safety_settings)
        token_usage = _get_token_usage(response)

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    raw_response_text = part.text.strip()
                    break
        elif hasattr(response, 'text') and response.text: # Fallback for older response structures
            raw_response_text = response.text.strip()

        if not raw_response_text:
             block_reason_str = "Unknown (No text part in response or empty text)"
             finish_reason_str = "Unknown"
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                 block_reason_str = str(response.prompt_feedback.block_reason)
                 if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                     finish_reason_str = f"{str(response.candidates[0].finish_reason)} (may confirm prompt block)"
                 else:
                     finish_reason_str = "(Prompt blocked)"
             if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                 finish_reason_str = str(response.candidates[0].finish_reason)
                 if finish_reason_str == "SAFETY" and "SAFETY" not in block_reason_str:
                     block_reason_str = "SAFETY (from finish_reason)"
                 elif finish_reason_str != "STOP" and block_reason_str.startswith("Unknown"):
                     block_reason_str = f"Non-STOP finish_reason: {finish_reason_str}"
             logging.error(f"LLM next step planning blocked or empty. Block Reason: {block_reason_str}, Finish Reason: {finish_reason_str}")
             return {
                 "reasoning": f"Planning failed: LLM response blocked or empty (Block: {block_reason_str}, Finish: {finish_reason_str}). Asking user for help.",
                 "next_action": {"action_type": "ask_user", "parameters": {"question": f"My planning was blocked (Reason: {block_reason_str}, Finish: {finish_reason_str}). How should I proceed?"}}
             }, token_usage

        logging.debug(f"LLM raw response for next step:\n{raw_response_text}")

        text_to_attempt_parse = raw_response_text.strip()
        parse_error_details = ""

        if not text_to_attempt_parse:
            parse_error_details = "LLM response was empty or only whitespace."
            logging.error(parse_error_details)
        else:
            if text_to_attempt_parse.startswith(("{", "[")):
                try:
                    step_dict = json.loads(text_to_attempt_parse)
                    logging.info("Direct JSON parsing successful.")
                except json.JSONDecodeError as e1:
                    logging.warning(f"Direct JSON parsing failed: {e1}. Cleaning markdown/markers...")
                    parse_error_details = f"Direct JSON parsing failed: {e1}."
                    try:
                        step_dict = demjson3.decode(text_to_attempt_parse)
                        logging.info("JSON parsing successful with demjson3 on original text.")
                    except demjson3.JSONDecodeError as e_dem:
                        parse_error_details += f" demjson3 also failed: {e_dem}."
                        logging.warning(f"demjson3 parsing failed: {e_dem}")
                        step_dict = None
            
            if step_dict is None: 
                if not text_to_attempt_parse.startswith(("{", "[")):
                     logging.info(f"Response did not start with '{{' or '['. Content: '{text_to_attempt_parse[:200]}...'. Will attempt cleaning.")
                
                cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', text_to_attempt_parse, flags=re.MULTILINE | re.IGNORECASE).strip()
                cleaned_text = re.sub(r'^----|----$', '', cleaned_text, flags=re.MULTILINE).strip()

                if not cleaned_text:
                    if not parse_error_details: parse_error_details = "Response was empty after cleaning."
                    logging.error(parse_error_details)
                elif cleaned_text == text_to_attempt_parse and not text_to_attempt_parse.startswith(("{", "[")):
                    if not parse_error_details: parse_error_details = f"Response not JSON and cleaning had no effect. Cleaned: '{cleaned_text[:200]}...'"
                    logging.error(parse_error_details)
                elif cleaned_text.startswith(("{", "[")):
                    logging.info("Trying to parse after cleaning markdown/markers.")
                    try:
                        step_dict = json.loads(cleaned_text)
                        logging.info("JSON parsing successful after cleaning.")
                    except json.JSONDecodeError as e2:
                        logging.warning(f"JSON parsing failed after cleaning: {e2}. Trying demjson3 on cleaned text...")
                        if not parse_error_details: parse_error_details = f"JSON parsing failed after cleaning: {e2}."
                        else: parse_error_details += f" JSON parsing failed after cleaning: {e2}."
                        try:
                            step_dict = demjson3.decode(cleaned_text)
                            logging.info("JSON parsing successful with demjson3 after cleaning.")
                        except demjson3.JSONDecodeError as e_dem_cleaned:
                            parse_error_details += f" demjson3 also failed on cleaned text: {e_dem_cleaned}."
                            logging.error(f"demjson3 failed on cleaned text: {e_dem_cleaned}")
                            step_dict = None
                else: 
                    if not parse_error_details: parse_error_details = f"Response after cleaning still not JSON. Cleaned: '{cleaned_text[:200]}...'"
                    logging.error(parse_error_details)

        if step_dict is None:
             logging.error(f"Planning failed: Could not parse LLM response as JSON. {parse_error_details}")
             if 'DEBUG_DIR' in globals() and DEBUG_DIR and os.path.isdir(DEBUG_DIR):
                 problematic_json_path = os.path.join(DEBUG_DIR, f"failed_json_{time.strftime('%Y%m%d-%H%M%S')}.txt")
                 try:
                     with open(problematic_json_path, 'w', encoding='utf-8') as f_err:
                         f_err.write(f"--- Original Raw Response ---\n{raw_response_text}\n")
                         f_err.write(f"--- String Attempted for Parsing (final state) ---\n{cleaned_text if cleaned_text else text_to_attempt_parse}\n")
                         f_err.write(f"--- Errors ---\n{parse_error_details}\n")
                     logging.error(f"Saved problematic JSON string to {problematic_json_path}")
                 except Exception as save_err:
                     logging.error(f"Could not save problematic JSON string: {save_err}")
             else:
                 logging.error("DEBUG_DIR not defined or accessible, cannot save problematic JSON.")

             return {
                 "reasoning": "Planning failed: Could not parse valid action from LLM response. Asking user for help.",
                 "next_action": {"action_type": "ask_user", "parameters": {"question": "My planning failed (invalid response format). How should I proceed?"}}
             }, token_usage

        if 'DEBUG_DIR' in globals() and DEBUG_DIR and os.path.isdir(DEBUG_DIR):
            results_log_path = os.path.join(DEBUG_DIR, "resultss.txt")
            try:
                with open(results_log_path, 'a', encoding='utf-8') as f_results:
                    f_results.write(f"--- Successfully Parsed JSON Plan at {time.strftime('%Y%m%d-%H%M%S')} ---\n")
                    json.dump(step_dict, f_results, indent=2)
                    f_results.write("\n\n")
                logging.info(f"Successfully parsed JSON plan logged to {results_log_path}")
            except Exception as e_log_results:
                logging.error(f"Could not log successfully parsed JSON plan to {results_log_path}: {e_log_results}")
        else:
            logging.warning("DEBUG_DIR not defined or accessible, cannot log successfully parsed JSON.")

        if (isinstance(step_dict, dict) and
            "next_action" in step_dict and
            isinstance(step_dict["next_action"], dict) and
            "action_type" in step_dict["next_action"] and
            "parameters" in step_dict["next_action"] and
            isinstance(step_dict["next_action"]["parameters"], dict) and
            "reasoning" in step_dict):

            action_to_execute = step_dict["next_action"]
            action_type = action_to_execute.get("action_type")
            params = action_to_execute.get("parameters", {})

            if action_type not in VALID_ACTIONS: # Using the VALID_ACTIONS set
                 logging.error(f"Invalid action_type '{action_type}' in next step.")
                 return {
                     "reasoning": f"Planning failed: Invalid action type '{action_type}' proposed. Asking user for help.",
                     "next_action": {"action_type": "ask_user", "parameters": {"question": f"My planning failed (invalid action type '{action_type}'). How should I proceed?"}}
                 }, token_usage

            if action_type == "multi_action":
                sequence = params.get("sequence")
                if not isinstance(sequence, list) or not sequence:
                    logging.error("Invalid 'multi_action': 'sequence' parameter missing, not a list, or empty.")
                    return {
                        "reasoning": "Planning failed: Invalid 'multi_action' structure proposed. Asking user for help.",
                        "next_action": {"action_type": "ask_user", "parameters": {"question": "My planning failed (invalid multi_action structure). How should I proceed?"}}
                    }, token_usage

                available_actions_set_for_multi = VALID_ACTIONS - {"ask_user", "multi_action", "task_complete", "start_visual_listener"}
                for i, sub_action in enumerate(sequence):
                    if not isinstance(sub_action, dict) or \
                       sub_action.get("action_type") not in available_actions_set_for_multi or \
                       not isinstance(sub_action.get("parameters"), dict):
                        logging.error(f"Invalid sub-action structure at index {i} in multi_action sequence: {sub_action}")
                        return {
                            "reasoning": f"Planning failed: Invalid sub-action structure in 'multi_action' at index {i}. Asking user for help.",
                            "next_action": {"action_type": "ask_user", "parameters": {"question": f"My planning failed (invalid sub-action in multi_action at index {i}). How should I proceed?"}}
                        }, token_usage

            if action_type == "run_python_script":
                 script_path = params.get("script_path")
                 if not script_path or not isinstance(script_path, str):
                     logging.error("Invalid 'run_python_script': 'script_path' parameter missing or not a string.")
                     return {
                         "reasoning": "Planning failed: Invalid 'run_python_script' parameters. Asking user for help.",
                         "next_action": {"action_type": "ask_user", "parameters": {"question": "My planning failed (invalid run_python_script parameters). How should I proceed?"}}
                     }, token_usage
            
            if action_type == "process_files_from_urls": 
                if not all(k in params for k in ["url", "prompt"]):
                    logging.error("Invalid 'process_files_from_urls': missing 'url' or 'prompt'.")
                    return {"reasoning": "Planning failed: Invalid 'process_files_from_urls' parameters (missing 'url' or 'prompt'). Asking user for help.",
                            "next_action": {"action_type": "ask_user", "parameters": {"question": "My planning failed (invalid process_files_from_urls parameters). How should I proceed?"}}}, token_usage

            if benchmark_mode and action_type == "ask_user":
                logging.error("Planning generated 'ask_user' in benchmark mode. Overriding with STOP.")
                return {
                    "reasoning": f"Original plan was 'ask_user' ({step_dict.get('reasoning', 'N/A')}), but overridden to STOP due to benchmark mode.",
                    "next_action": {"action_type": "STOP", "parameters": {"reason": "ask_user_in_benchmark_plan"}}
                }, token_usage

            logging.info(f"Determined next action: {action_type}")
            logging.info(f"Reasoning: {step_dict.get('reasoning', 'N/A')}")
            return step_dict, token_usage
        else:
            logging.error(f"LLM response for next step has invalid structure after successful parsing. Content: {step_dict}")
            if not isinstance(step_dict, dict):
                logging.error("Validation failed: step_dict is not a dictionary.")
            elif "next_action" not in step_dict:
                logging.error("Validation failed: 'next_action' key missing in step_dict.")
            elif not isinstance(step_dict.get("next_action"), dict):
                logging.error(f"Validation failed: 'next_action' is not a dictionary. Type: {type(step_dict.get('next_action'))}, Value: {step_dict.get('next_action')}")
            elif "action_type" not in step_dict.get("next_action", {}):
                logging.error("Validation failed: 'action_type' key missing in next_action.")
            elif "parameters" not in step_dict.get("next_action", {}):
                logging.error("Validation failed: 'parameters' key missing in next_action.")
            elif not isinstance(step_dict.get("next_action", {}).get("parameters"), dict):
                logging.error(f"Validation failed: 'parameters' in next_action is not a dictionary. Type: {type(step_dict.get('next_action', {}).get('parameters'))}, Value: {step_dict.get('next_action', {}).get('parameters')}")
            elif "reasoning" not in step_dict:
                logging.error("Validation failed: 'reasoning' key missing in step_dict.")

            return {
                "reasoning": "Planning failed: LLM response structure was invalid after parsing. Asking user for help.",
                "next_action": {"action_type": "ask_user", "parameters": {"question": "My planning failed (invalid response structure after parsing). How should I proceed?"}}
            }, token_usage

    except ValueError as ve: # Catches LLM blocking due to safety, etc.
         block_reason = "Unknown"
         finish_reason_val = "Unknown"
         try:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
             if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                 finish_reason_val = str(response.candidates[0].finish_reason)
                 if finish_reason_val == "SAFETY" and not block_reason: 
                     block_reason = "SAFETY (finish_reason)"
         except Exception: pass
         logging.error(f"ValueError during LLM call for next step (Block: {block_reason}, Finish: {finish_reason_val}): {ve}", exc_info=True)
         return {
             "reasoning": f"Planning failed due to LLM error (Block: {block_reason}, Finish: {finish_reason_val}). Asking user for help.",
             "next_action": {"action_type": "ask_user", "parameters": {"question": f"My planning failed (LLM error: Block: {block_reason}, Finish: {finish_reason_val}). How should I proceed?"}}
         }, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    except Exception as e:
        logging.error(f"Unexpected error determining next step with LLM: {e}", exc_info=True)
        return {
            "reasoning": f"Planning failed due to an unexpected error: {e}. Asking user for help.",
            "next_action": {"action_type": "ask_user", "parameters": {"question": f"An error occurred during planning ({e}). How should I proceed?"}}
        }, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    # This line should not be reached if all paths return, but as a fallback:
    return step_dict, token_usage

def clean_llm_json_response(raw_response: str) -> str:
    """Removes markdown code fences (```json, ```) and trims whitespace from LLM JSON responses."""
    import re
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
