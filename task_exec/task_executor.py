from typing import List, Dict, Optional, Tuple, Union, Any, Generator
import os
import logging,re
import google.generativeai as genai
from PIL import  Image
from tools.token_usage_tool import _get_token_usage
from vision.vis import image_to_base64,_hash_pil_image,capture_full_screen
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

from task_exec.task_planner import critique_action
from tools.actions import execute_action
import time # Removed datetime from here
from chromaDB_management.credential import save_credential
from utils.reinforcement_util import retrieve_relevant_reinforcements_from_db
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    get_model, get_critic_model, DEBUG_DIR
)
from tools.files_upload import process_files_from_urls

# Add this near the top of the file or where VALID_ACTIONS or similar is defined
VALID_ACTIONS = {
    "focus_window", "click", "type", "press_keys", "move_mouse", "run_shell_command", "run_python_script", "write_file", "navigate_web", "search_web", "search_youtube", "wait", "ask_user", "describe_screen", "capture_screenshot", "click_and_type", "multi_action", "task_complete", "read_file", "INFORM_USER", "process_local_files", "process_files_from_urls", "start_visual_listener", "edit_image_with_file"
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
            message = "File Processing Results\n"
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
            message += f"Analysis:\n{result}\n"
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
            return

    # Get critic model
    critic_model = get_critic_model()
    if critic_model is None:
        logging.error("Critic model is None and could not be initialized")
        return

    results = []
    max_iterations = 25
    current_iteration = agent_state.current_task.iteration_count if agent_state.current_task else 0
    task_completed = False
    current_app_base_name = "unknown"
    current_shortcuts = [] # This should be a string
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

    # Use agent_state.current_task._accumulate_tokens directly
    # Ensure agent_state.current_task is not None before calling _accumulate_tokens

    logging.info(f"Starting iterative execution for: {original_instruction}")
    load_shortcuts_cache()

    # --- Task Analysis and Sub-tasking ---
    if agent_state.current_task and current_iteration == 0 and not agent_state.current_task.initial_planning_done:
        # Check if sub_tasks are already populated (e.g., from a resumed task)
        if hasattr(agent_state.current_task, 'sub_tasks') and agent_state.current_task.sub_tasks: # type: ignore
            current_sub_tasks = agent_state.current_task.sub_tasks # type: ignore
            current_sub_task_index = agent_state.current_task.current_sub_task_index # type: ignore
            logging.info(f"Resuming with existing sub-tasks. Current: {current_sub_task_index + 1}/{len(current_sub_tasks)}")
        else:
            category, sub_tasks_list, analysis_reasoning, analysis_tokens = _analyze_and_categorize_task(original_instruction, llm_model)
            if agent_state.current_task: agent_state.current_task._accumulate_tokens(analysis_tokens)

            agent_state.add_thought(f"Task Analysis: Category='{category}', Sub-tasks={len(sub_tasks_list)}. Reasoning: {analysis_reasoning}", type="task_analysis")

            if category and sub_tasks_list:
                current_sub_tasks = sub_tasks_list
                current_sub_task_index = 0
                # Store in agent_state.current_task if these attributes are added to TaskSession
                if hasattr(agent_state.current_task, 'sub_tasks'): agent_state.current_task.sub_tasks = current_sub_tasks # type: ignore
                if hasattr(agent_state.current_task, 'current_sub_task_index'): agent_state.current_task.current_sub_task_index = current_sub_task_index # type: ignore
                if hasattr(agent_state.current_task, 'task_category'): agent_state.current_task.task_category = category # type: ignore
                logging.info(f"Task broken down into {len(current_sub_tasks)} sub-tasks.")
            else:
                current_sub_tasks = [original_instruction] # Treat original instruction as the only sub-task

        similar_user_task = find_similar_user_task_structure(current_task_instruction)
        if similar_user_task:
            executed_actions_summary_parts_initial = []
            if results: 
                for i, (action_dict, success, msg, _) in reversed(list(enumerate(results[-5:]))): 
                    action_type = action_dict.get('action_type', 'Unknown')
                    _ = str(action_dict.get('parameters', {})) # params_str not used
                    outcome = "OK" if success else "FAIL"
                    exec_msg_only = msg.split("| Assessment:")[0].strip()
                    summary_line = f"  Prev. Action {len(results) - i}: {action_type}, Outcome: {outcome} - Msg: {exec_msg_only[:70]}{'...' if len(exec_msg_only)>70 else ''}"
                    executed_actions_summary_parts_initial.append(summary_line)
            _ = "\n".join(executed_actions_summary_parts_initial) if executed_actions_summary_parts_initial else "No actions executed yet in this task." # executed_actions_summary_str_initial not used

            task_name = similar_user_task.get('task_name', 'Unnamed Task')
            ask_question = (
                f"I found a saved task structure named '{task_name}' that seems similar to your request. "
                "Would you like me to use your saved plan for this task instead of figuring it out? (Type 'yes' to use saved plan, 'no' to use standard planning)"
            )
            ask_action = {"action_type": "ask_user", "parameters": {"question": ask_question}}
            current_iteration += 1
            agent_state.add_thought(f"Iteration {current_iteration}/{max_iterations} (User Plan Suggestion): Asking user about saved task '{task_name}'.")

            yield_result = {"type": "ask_user", "question": ask_question} 
            user_response_to_ask = yield yield_result

            if user_response_to_ask is None:
                logging.warning(
                    "Generator received None when expecting input for user plan suggestion. "
                    "Treating as 'no' or proceeding with standard planning."
                )
                user_response_to_ask = "no"

            results.append((ask_action, True, f"Asked user about saved plan. Response: '{user_response_to_ask}'", None))
            if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_ask})

            if "yes" in user_response_to_ask.lower():
                logging.info(f"User chose to use saved plan '{task_name}'. Injecting plan.")
                try:
                    plan_text_from_user = similar_user_task.get('plan_text', '')
                    if plan_text_from_user:
                        logging.info(f"Using user-defined plan text as new instruction for '{task_name}':\n{plan_text_from_user}")
                        current_task_instruction = plan_text_from_user
                        plan_source = f"User Defined Plan: {task_name}"
                        user_plan_used = True

                        inform_message = f"Okay, I will follow your saved instructions for '{task_name}': \"{plan_text_from_user}\"."
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": inform_message})
                        yield {"type": "inform_user", "message": inform_message}
                    else:
                        logging.warning(f"User-defined plan text for '{task_name}' is empty. Falling back to standard planning.")
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"Saved plan for '{task_name}' is empty. Proceeding with standard planning."})
                        plan_source = "Standard Planning"
                        user_plan_used = False

                    if not user_plan_used:
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"Could not use saved plan '{task_name}' (format issue or empty). Proceeding with standard planning."})
                        plan_source = "Standard Planning"
                except Exception as e_load_user_plan:
                    logging.error(f"Unexpected error processing user-defined plan '{task_name}': {e_load_user_plan}", exc_info=True)
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"Error decoding saved plan '{task_name}'. Proceeding with standard planning."})
                    plan_source = "Standard Planning"
            else:
                agent_state.add_thought(f"User declined saved plan '{task_name}'. Proceeding with standard planning.")
                plan_source = "Standard Planning"

        if not user_plan_used:
            complex_keywords = ["create", "build", "set up", "install", "configure", "pipeline", "system", "server", "database", "analyze", "summarize", "generate code", "translate", "design", "write code", "visual charts"]
            is_complex_task = any(keyword in current_task_instruction.lower() for keyword in complex_keywords)

            if is_complex_task:
                logging.info("Instruction seems complex. Searching for AI services.")
                plan_source = "AI Service Search" 
                search_success, search_summary, search_tokens = search_web_for_info(f"AI service for {current_task_instruction}")
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(search_tokens)
                if search_success and search_summary and len(search_summary) > 30:
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"System: Searched web for AI services. Found: {search_summary[:100]}..."})
                    ask_question_intro = (
                        f"For your task: '{current_task_instruction}', I've looked into AI tools. "
                        "Here's a summary of what I found (this might include multiple tools, their features, and mentions of free/paid plans if available):\n\n"
                        f"\"{search_summary}\"\n\n"
                    )
                    ask_question_options = (
                        "Based on this, what would you like to do?\n"
                        "1. Try to use an AI tool mentioned above? (If so, please specify which tool if multiple are mentioned, e.g., 'use ToolName')\n"
                        "2. Search the web for detailed step-by-step instructions for this task?\n"
                        "3. Let me try to accomplish this task directly (e.g., by writing a script or using standard methods)?\n"
                        "Please type your choice (e.g., 'use ToolName', 'search web steps', 'do it myself')."
                    )
                    ask_question = ask_question_intro + ask_question_options
                    ask_action = {"action_type": "ask_user", "parameters": {"question": ask_question}}
                    agent_state.add_thought(f"Iteration {current_iteration}/{max_iterations} (AI Service Query): Asking user about AI tools.")
                    yield_result = {"type": "ask_user", "question": ask_question}
                    user_response_to_ask = yield yield_result

                    if user_response_to_ask is None:
                        logging.warning(
                            "Generator received None when expecting input for AI service query. "
                            "Defaulting to standard planning."
                        )
                        user_response_to_ask = "do it myself"

                    results.append((ask_action, True, f"Asked user about AI service. Response: '{user_response_to_ask}'", None))

                    response_lower = user_response_to_ask.lower()
                    if response_lower.startswith("use ") and len(user_response_to_ask.split()) > 1:
                        chosen_tool_name = user_response_to_ask.split(" ", 1)[1].strip()
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"Okay, I will try to use the AI tool: '{chosen_tool_name}'."})
                        plan_source = f"AI Service Plan - Tool: {chosen_tool_name}"
                    elif "search web steps" in response_lower:
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": "Okay, I will search the web for detailed steps."})
                        plan_source = "Web Search Suggestion"
                    elif "do it myself" in response_lower or "figure it out" in response_lower:
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": "Okay, I will proceed with standard planning."})
                        plan_source = "Standard Planning"
                    else:
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"Your response ('{user_response_to_ask}') was unclear. Defaulting to standard planning."})
                        plan_source = "Standard Planning"
                else:
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"System: No specific AI services found for '{current_task_instruction}'. Will try general web search or standard planning."})
                    plan_source = "Web Search Suggestion"

            if plan_source == "Web Search Suggestion" and not user_plan_used:
                logging.info("Searching web for general plan steps.")
                search_success, search_summary, search_tokens = search_web_for_info(f"steps to {current_task_instruction}")
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(search_tokens)
                if search_success and search_summary and len(search_summary) > 50:
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"System: Searched web for general steps. Found: {search_summary[:100]}..."})
                    ask_question = (
                        f"I found some general steps online for '{current_task_instruction}':\n\n"
                        f"{search_summary}\n\n"
                        "Follow these steps or figure it out? (Type 'follow steps' or 'figure it out')"
                    )
                    ask_action = {"action_type": "ask_user", "parameters": {"question": ask_question}}
                    current_iteration += 1
                    agent_state.add_thought(f"Iteration {current_iteration}/{max_iterations} (General Web Plan Query): Asking user about web steps.")
                    yield_result = {"type": "ask_user", "question": ask_question}
                    user_response_to_ask = yield yield_result

                    if user_response_to_ask is None:
                        logging.warning(
                            "Generator received None when expecting input for general web plan query. "
                            "Defaulting to 'figure it out'."
                        )
                        user_response_to_ask = "figure it out"

                    results.append((ask_action, True, f"Asked user about general web plan. Response: '{user_response_to_ask}'", None))
                    if "follow steps" in user_response_to_ask.lower():
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": "System: User chose to follow general web steps."})
                        plan_source = "Web Search Plan"
                    else:
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": "System: User chose standard planning."})
                        plan_source = "Standard Planning"
                else:
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": "System: General web search did not yield useful results. Proceeding with standard planning."})
                    plan_source = "Standard Planning"
            elif not user_plan_used and plan_source not in ["AI Service Plan", "Web Search Suggestion", "Web Search Plan"]: # type: ignore
                plan_source = "Standard Planning"
        if agent_state.current_task: agent_state.current_task.initial_planning_done = True

    # Determine the current instruction for this iteration (either a sub-task or the original)
    if current_sub_tasks and 0 <= current_sub_task_index < len(current_sub_tasks):
        instruction_for_current_planning_cycle = current_sub_tasks[current_sub_task_index]
        agent_state.add_thought(f"Working on Sub-task {current_sub_task_index + 1}/{len(current_sub_tasks)}: {instruction_for_current_planning_cycle}", type="sub_task_info")
    else:
        instruction_for_current_planning_cycle = original_instruction

    # --- Check for user-defined structure for the current sub-task (if it's a sub-task) ---
    if current_sub_tasks and 0 <= current_sub_task_index < len(current_sub_tasks) and instruction_for_current_planning_cycle != original_instruction:
        logging.info(f"Checking for user-defined structures for sub-task: {instruction_for_current_planning_cycle}")
        similar_sub_task_structure = find_similar_user_task_structure(instruction_for_current_planning_cycle)
        if similar_sub_task_structure:
            sub_task_structure_name = similar_sub_task_structure.get('task_name', 'Unnamed Sub-Task Structure')
            ask_sub_task_question = (
                f"For the current sub-task '{instruction_for_current_planning_cycle}', "
                f"I found a saved task structure named '{sub_task_structure_name}'. "
                "Would you like to use its plan text to guide this sub-task? (Type 'yes' or 'no')"
            )
            ask_sub_task_action = {"action_type": "ask_user", "parameters": {"question": ask_sub_task_question}}
            yield_sub_task_result = {"type": "ask_user", "question": ask_sub_task_question}
            user_response_to_sub_task_ask = yield yield_sub_task_result

            if user_response_to_sub_task_ask is None: user_response_to_sub_task_ask = "no"
            results.append((ask_sub_task_action, True, f"Asked user about saved structure for sub-task. Response: '{user_response_to_sub_task_ask}'", None))
            if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_sub_task_ask})

            if "yes" in user_response_to_sub_task_ask.lower():
                sub_task_plan_text = similar_sub_task_structure.get('plan_text', '')
                if sub_task_plan_text:
                    logging.info(f"User chose to use saved structure '{sub_task_structure_name}' for sub-task. New instruction for sub-task: {sub_task_plan_text}")
                    instruction_for_current_planning_cycle = sub_task_plan_text # Update the instruction for this cycle
                    agent_state.add_thought(f"Using user-defined structure '{sub_task_structure_name}' for sub-task '{current_sub_tasks[current_sub_task_index]}'.", type="sub_task_planning")
                else:
                    logging.warning(f"Saved structure '{sub_task_structure_name}' for sub-task is empty. Proceeding with original sub-task description.")
            else:
                logging.info(f"User declined saved structure for sub-task. Proceeding with original sub-task description.")
    # instruction_for_current_planning_cycle = original_instruction # This was a bug, should not reset here

    while current_iteration < max_iterations and not task_completed:

        while agent_state.task_is_paused:
            logging.debug("Generator: Task is paused. Yielding 'paused' state.")
            yield {"type": "paused"}
            logging.debug("Generator: Resuming from pause check.")
        current_iteration += 1
        if agent_state.current_task: agent_state.current_task.iteration_count = current_iteration
        logging.info(f"--- Iteration {current_iteration}/{max_iterations} (Plan Source: {plan_source}, Consecutive Failures: {total_consecutive_failures}, Replans: {replan_attempts_current_cycle}) ---")

        if plan_to_inject is not None:
            logging.info(f"Injecting new plan (reason: {plan_injection_reason}, {len(plan_to_inject)} steps).")
            if not plan_to_inject:
                logging.warning("Injected plan was empty. Continuing normal planning.")
                plan_to_inject = None; plan_injection_reason = None

        try:
            full_app_name = get_active_window_name()
            app_base_name = get_base_app_name(full_app_name)
            if app_base_name != current_app_base_name:
                logging.info(f"Application context changed to: {app_base_name}")
                current_app_base_name = app_base_name
                current_shortcuts_str, shortcut_tokens = get_application_shortcuts(current_app_base_name) # Capture tokens
                current_shortcuts = current_shortcuts_str # Assign string to current_shortcuts
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(shortcut_tokens)
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System Observation: Active application changed to '{app_base_name}'."})
        except Exception as app_detect_err:
            logging.error(f"Error during application detection/shortcut handling: {app_detect_err}", exc_info=True)
            current_app_base_name = "unknown"; current_shortcuts = "" # Ensure current_shortcuts is a string

        planning_screenshot = capture_full_screen()
        screenshot_before_action_hash = _hash_pil_image(planning_screenshot)

        if last_assessment_screenshot_hash and screenshot_before_action_hash and screenshot_before_action_hash != last_assessment_screenshot_hash:
            logging.info("Detected screen change since last assessment.")
            if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": "System Observation: Screen content changed since last action."})

        action_to_execute = None
        planning_reasoning = "Planning skipped due to pending credential request or injected plan."

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
                for i, (action_dict, success, msg, _) in reversed(list(enumerate(results[-5:]))):
                    action_type = action_dict.get('action_type', 'Unknown')
                    outcome = "OK" if success else "FAIL"
                    exec_msg_only = msg.split("| Assessment:")[0].strip() 
                    summary_line = f"  Prev. Action {len(results) - i}: {action_type}, Outcome: {outcome} - Msg: {exec_msg_only[:70]}{'...' if len(exec_msg_only)>70 else ''}"
                    executed_actions_summary_parts.append(summary_line)
            executed_actions_summary_str_for_prompt = "\n".join(executed_actions_summary_parts) if executed_actions_summary_parts else "No actions executed yet in this task."

            next_step_data, planning_tokens = process_next_step(
                instruction_for_current_planning_cycle,
                string_history_for_planning,
                llm_model,
                current_shortcuts, # Pass the string here
                [], # current_reinforcements - assuming this is handled by the prompt
                planning_screenshot,
                executed_actions_summary=executed_actions_summary_str_for_prompt
            )
            if agent_state.current_task: agent_state.current_task._accumulate_tokens(planning_tokens)
            if next_step_data is None or "next_action" not in next_step_data or not isinstance(next_step_data["next_action"], dict):
                reason = f"Planning failed: Invalid or no 'next_action' from process_next_step: {next_step_data}"
                logging.error(reason)
                results.append((({'action_type': 'STOP', 'parameters': {'reason': 'planning_failed_structure'}}, False, reason, None)))
                break
            action_to_execute = next_step_data["next_action"]
            planning_reasoning = next_step_data.get("reasoning", "N/A")

        action_type = action_to_execute.get("action_type")
        params = action_to_execute.get("parameters", {})
        logging.info(f"Proposed Action: {action_type}. Reasoning: {planning_reasoning}")
        # agent_state.add_thought(f"Iteration {current_iteration}: Proposed Action: {action_type}, Params: {params}, Reasoning: {planning_reasoning}", type="planning")
        
        # Store the raw plan JSON as the thought content for planning
        if next_step_data and isinstance(next_step_data, dict):
            thought_content_json_str = json.dumps(next_step_data)
            agent_state.add_thought(thought_content_json_str, type="planning_json") # Use a distinct type
        else:
            agent_state.add_thought(f"Iteration {current_iteration}: Planning data not in expected format. Reasoning: {planning_reasoning}", type="planning_error")

        critique_passed = True
        critique_feedback = "Critique skipped for credential value request or injected plan."
        if not (pending_credential_request and credential_consent_choice in ['onetime', 'remember']) and not plan_injection_reason:
            critique_screenshot = planning_screenshot
            string_history_for_critique = [f"{msg['role']}: {msg['content']}" for msg in (agent_state.current_task.conversation_history if agent_state.current_task else [])]
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

        if action_type == "task_complete":
            task_completed = True
            results.append(((action_to_execute, True, f"Task completed. Reason: {planning_reasoning}", None)))
            break

        exec_result = execute_action(action_to_execute, agent)

        exec_success = False
        exec_message = "Execution result processing error."
        special_directive = None

        if isinstance(exec_result, tuple) and len(exec_result) == 3 and exec_result[0] == -2: # type: ignore
            _, exec_message, ask_info = exec_result # type: ignore
            exec_result = (True, exec_message, ask_info) # type: ignore

        if isinstance(exec_result, tuple) and len(exec_result) == 3:
            exec_success, exec_message, special_directive = exec_result # type: ignore
            directive_type = special_directive.get("type") if isinstance(special_directive, dict) else None

            if isinstance(special_directive, dict) and "token_usage" in special_directive:
                action_tokens = special_directive.get("token_usage")
                if agent_state.current_task and isinstance(action_tokens, dict) and all(k in action_tokens for k in ["prompt_tokens", "candidates_tokens", "total_tokens"]): # type: ignore
                    agent_state.current_task._accumulate_tokens(action_tokens) # type: ignore

            if directive_type == "ask_user":
                question_text = special_directive.get('question', '') # type: ignore
                logging.info(f"Execution yielded for user input: {exec_message}")
                results.append(((action_to_execute, exec_success, exec_message, special_directive)))

                is_credential_consent_question = "'manual', 'onetime', or 'remember'" in question_text.lower()
                is_plan_choice_question = "Which of the above approaches" in question_text and "would you like to proceed with?" in question_text

                if is_credential_consent_question:
                    match_service = re.search(r"for '([^']+)'", question_text)
                    match_user = re.search(r"\(user: '([^']+)'\)", question_text)
                    match_type = re.search(r"need the \[(password|api key|email)\]", question_text, re.IGNORECASE)
                    if match_service and match_user and match_type:
                        pending_credential_request = {
                            'service': match_service.group(1),
                            'username': match_user.group(1),
                            'type': match_type.group(1).lower()
                        }
                        credential_consent_choice = None
                        logging.info(f"Identified credential consent request: {pending_credential_request}")
                    else:
                        logging.warning("Could not parse service/user/type from credential consent question.")
                        pending_credential_request = None
                elif is_plan_choice_question:
                    logging.info("Identified 'ask_user' as a request for choosing a high-level plan.")


                yield_result = {"type": "ask_user", "question": question_text}
                user_response_to_ask = yield yield_result

                if user_response_to_ask is None:
                    logging.warning("User cancelled during ask_user.")
                    results[-1] = (action_to_execute, False, "User cancelled during ask_user.", None) # type: ignore
                    pending_credential_request = None; credential_consent_choice = None
                    break

                logging.info(f"Received user response: '{user_response_to_ask}'.")
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_ask})
                is_risky_command_response = "[RISKY COMMAND]" in question_text

                if pending_credential_request and not credential_consent_choice:
                    response_lower = user_response_to_ask.lower().strip()
                    if response_lower in ['manual', 'onetime', 'remember']:
                        credential_consent_choice = response_lower
                        logging.info(f"User chose '{response_lower}' for credential handling.")
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": f"{user_response_to_ask} (Chose {response_lower})"})
                        if response_lower == 'manual': pending_credential_request = None
                    else:
                        logging.warning(f"Invalid response ('{user_response_to_ask}') to credential consent.")
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": f"{user_response_to_ask} (Invalid choice for consent)"})
                        pending_credential_request = None
                    continue
                elif pending_credential_request and credential_consent_choice in ['onetime', 'remember']:
                    credential_value = user_response_to_ask
                    service = pending_credential_request.get('service', 'unknown_service')
                    username = pending_credential_request.get('username', 'unknown_user')
                    logging.info(f"Received credential value for {service} (user: {username}).")
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": f"[CREDENTIAL PROVIDED FOR {service}]"})
                    if credential_consent_choice == 'remember':
                        save_credential(service, username, credential_value)
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System Internal Note: Credential value provided='{credential_value}'"})
                    pending_credential_request = None; credential_consent_choice = None
                    continue
                elif is_risky_command_response:
                    response_lower = user_response_to_ask.lower().strip()
                    if response_lower == 'yes':
                        logging.info("User confirmed risky command execution.")
                    else:
                        logging.warning(f"User denied risky command execution (Response: '{user_response_to_ask}').")
                    total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
                    continue
                elif is_plan_choice_question:
                    logging.info(f"User chose plan option: '{user_response_to_ask}'")
                    if "stop" not in user_response_to_ask.lower():
                        if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": f"Okay, proceeding with the approach: {user_response_to_ask}"})
                        yield {"type": "inform_user", "message": f"Okay, I will now work on the approach: '{user_response_to_ask[:100]}...'"}
                        current_task_instruction = f"User chose to proceed with the approach: {user_response_to_ask}. Now, create a detailed plan for this approach."
                        plan_source = f"Chosen High-Level Plan: {user_response_to_ask}"
                    else:
                        logging.info("User chose to stop after being presented with plan options.")
                        task_completed = True
                        break
                    total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
                    continue
                else:
                    logging.info("Received user response to a general 'ask_user'. Continuing loop for replanning.")
                    total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
                    continue

            elif directive_type == "inform_user":
                message_to_display = special_directive.get('message', '') # type: ignore
                logging.info(f"Execution yielded for informing user: {message_to_display}")

                if "--- START MERMAID CODE ---" in message_to_display and "--- END MERMAID CODE ---" in message_to_display:
                    logging.info("INFORM_USER message appears to contain high-level plan options.")

                youtube_refs = special_directive.get("youtube_references", []) # type: ignore
                if agent_state.current_task:
                    if youtube_refs:
                        agent_state.current_task.youtube_references.extend(youtube_refs) # type: ignore
                        message_with_refs = {
                            "role": "assistant",
                            "content": f"ℹ️ {message_to_display}",
                            "references": youtube_refs
                        }
                        agent_state.current_task.conversation_history.append(message_with_refs)
                    else:
                        agent_state.current_task.conversation_history.append({
                            "role": "assistant",
                            "content": f"ℹ️ {message_to_display}"
                        })

                agent_state.task_is_paused = True
                agent_state.is_task_running = True # This seems contradictory, should be False if paused
                if agent_state.current_task: agent_state.current_task.status = "paused"
                _ = {"type": "inform_user", "message": message_to_display, "youtube_references": youtube_refs} # yield_result not used
                total_consecutive_failures = 0; consecutive_failures_on_current_step = 0; replan_attempts_current_cycle = 0
                continue

            elif directive_type == "inject_plan":
                logging.info(f"Action '{action_type}' directed to inject a new plan. Reason: {special_directive.get('reason')}") # type: ignore
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System: {exec_message} - New plan will be injected."})
                plan_to_inject = special_directive.get("plan", []) # type: ignore
                plan_injection_reason = special_directive.get("reason", "Unknown directive") # type: ignore
                total_consecutive_failures = 0
                consecutive_failures_on_current_step = 0
                replan_attempts_current_cycle = 0
            elif directive_type in ["search_result", "youtube_search_result", "screen_description", "generation_complete", "generation_failed", "generation_error", "listener_timeout_no_actions", "image_generation_result", "file_processing_result"]: # Added file_processing_result
                logging.info(f"Action '{action_type}' completed with data directive: {directive_type}. Original success status and message preserved for assessment.")
                if directive_type == "image_generation_result":
                    details = special_directive.get("details", {})
                    if details.get("success", False):
                        # If image generation was successful, add the result to conversation history
                        if agent_state.current_task:
                            image_data = details.get("image_data")  # Get image_data from details
                            message = details.get("message", "Image generated successfully")
                            # Format the message to include the image in a chat-friendly way
                            chat_message = {
                                "role": "assistant",
                                "content": f"✅ {message}",
                                "image_data": image_data  # Use image_data instead of image
                            }
                            agent_state.current_task.conversation_history.append(chat_message)
                        return "TASK_COMPLETE", exec_message, token_usage
                    else:
                        # If image generation failed, add error to conversation history
                        if agent_state.current_task:
                            error_message = details.get("message", "Image generation failed")
                            agent_state.current_task.conversation_history.append({
                                "role": "assistant",
                                "content": f"❌ {error_message}"
                            })
                        exec_success = False
                        exec_message = error_message
            elif special_directive is None:
                pass
            else:
                logging.error(f"Unexpected and unhandled special directive type: {directive_type} from action {action_type}. Treating as execution failure.")
                exec_success = False
                exec_message = f"Execution failed due to unexpected and unhandled special directive: {directive_type}"
        elif isinstance(exec_result, tuple) and len(exec_result) == 2:
            exec_success, exec_message = exec_result # type: ignore
            special_directive = None
        else:
            exec_success = False
            exec_message = f"Unexpected return type from execute_action: {type(exec_result)}"
            logging.error(exec_message)
            results.append(((action_to_execute, exec_success, exec_message, None))) # type: ignore

        logging.info(f"Execution Result: {'OK' if exec_success else 'FAIL'} - {exec_message}")

        screenshot_after_action = None
        if action_type in {"click", "type", "focus_window", "click_and_type", "move_mouse"} or not exec_success:
            time.sleep(0.5)
            screenshot_after_action = capture_full_screen()
            if screenshot_after_action is None: logging.warning("Failed to capture screenshot for assessment.")

        string_history_for_assessment = [f"{msg['role']}: {msg['content']}" for msg in (agent_state.current_task.conversation_history if agent_state.current_task else [])]
        assessment_status, assessment_reasoning, assessment_tokens = assess_action_outcome(
            instruction_for_current_planning_cycle, action_to_execute, exec_success, exec_message,
            screenshot_after_action, llm_model, screenshot_before_hash=screenshot_before_action_hash
        )
        if agent_state.current_task: agent_state.current_task._accumulate_tokens(assessment_tokens)
        final_message = f"{exec_message} | Assessment: {assessment_status} - {assessment_reasoning}"
        
        # After action execution and assessment:
        system_observation = f"System Observation: {final_message}"
        if agent_state.current_task: 
            agent_state.current_task.conversation_history.append({
                "role": "system", 
                "content": system_observation
            })
        
        final_success = (assessment_status == "SUCCESS")
        results.append(((action_to_execute, final_success, final_message, special_directive)))
        logging.info(f"Assessment Result: {assessment_status} - {assessment_reasoning}")
        last_assessment_screenshot_hash = _hash_pil_image(screenshot_after_action)

        if agent_state.current_task and (assessment_status == "FAILURE" or (assessment_status == "RETRY_POSSIBLE" and consecutive_failures_on_current_step >= MAX_RETRIES_PER_STEP)):
            action_type_failed = action_to_execute.get('action_type')
            action_failure_counts[action_type_failed] = action_failure_counts.get(action_type_failed, 0) + 1 # type: ignore
            logging.warning(f"Incremented failure count for '{action_type_failed}': {action_failure_counts.get(action_type_failed)}") # type: ignore

            if action_type_failed == "search_youtube" and action_failure_counts.get(action_type_failed, 0) > MAX_YOUTUBE_SEARCH_ATTEMPTS: # type: ignore
                agent_state.current_task.conversation_history.append({"role": "system", "content": "System Note: YouTube search tool disabled for this task due to repeated failures."})
                logging.warning(f"YouTube search tool disabled for this task after {MAX_YOUTUBE_SEARCH_ATTEMPTS} failures.")

        if agent_state.current_task: agent_state.current_task.agent_thoughts.append({"timestamp": datetime.now().isoformat(), "content": f"Step {current_iteration} Action: {action_type} {params}\nOutcome: {assessment_status} - {assessment_reasoning}", "type": "step_outcome"})

        if agent_state.current_task and agent_state.current_task.conversation_history and \
        agent_state.current_task.conversation_history[-1].get("role") == "system" and \
        "System Internal Note: Credential value provided=" in agent_state.current_task.conversation_history[-1].get("content", ""):
            agent_state.current_task.conversation_history.pop()

        if assessment_status == "SUCCESS":
            total_consecutive_failures = 0
            if action_type: action_failure_counts[action_type] = 0 # type: ignore
            consecutive_failures_on_current_step = 0
            replan_attempts_current_cycle = 0
            if plan_injection_reason and not plan_to_inject:
                logging.info(f"Successfully completed injected plan (Reason: {plan_injection_reason}).")
                plan_injection_reason = None
            if current_sub_tasks and 0 <= current_sub_task_index < len(current_sub_tasks):
                logging.info(f"Sub-task {current_sub_task_index + 1} ('{current_sub_tasks[current_sub_task_index]}') appears complete.")
                current_sub_task_index += 1
                if agent_state.current_task and hasattr(agent_state.current_task, 'current_sub_task_index'): agent_state.current_task.current_sub_task_index = current_sub_task_index # type: ignore

                if current_sub_task_index < len(current_sub_tasks):
                    next_sub_task_info = f"Completed sub-task. Moving to sub-task {current_sub_task_index + 1}/{len(current_sub_tasks)}: '{current_sub_tasks[current_sub_task_index]}'."
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "assistant", "content": next_sub_task_info})
                    yield {"type": "inform_user", "message": next_sub_task_info} 
                else:
                    logging.info("All sub-tasks completed.") 
            continue
        elif assessment_status == "RETRY_POSSIBLE":
            if consecutive_failures_on_current_step < MAX_RETRIES_PER_STEP:
                consecutive_failures_on_current_step += 1
                logging.warning(f"Assessment RETRY_POSSIBLE. Retrying step (Attempt {consecutive_failures_on_current_step + 1}/{MAX_RETRIES_PER_STEP + 1}). Reason: {assessment_reasoning}")
                results[-1] = (action_to_execute, False, f"{exec_message} | Assessment: RETRY_POSSIBLE - {assessment_reasoning} (Retry {consecutive_failures_on_current_step})", special_directive) # type: ignore
                time.sleep(1.0)
                current_iteration -= 1
                continue
            else:
                logging.error(f"Max retries ({MAX_RETRIES_PER_STEP}) for current step reached after RETRY_POSSIBLE. Treating as FAILURE.")
                assessment_status = "FAILURE"
                total_consecutive_failures +=1

        if assessment_status == "FAILURE":
            logging.error(f"Assessment FAILURE for step. Reason: {assessment_reasoning}")
            total_consecutive_failures += 1
            consecutive_failures_on_current_step = 0

            is_planning_failure = "Planning failed: Could not parse valid action from LLM response" in assessment_reasoning or \
                                "Planning failed: LLM response structure was invalid" in assessment_reasoning or \
                                "Planning failed: Invalid 'multi_action' structure" in assessment_reasoning or \
                                "Planning failed: Invalid action type" in assessment_reasoning or \
                                "Planning failed: Invalid 'run_python_script' parameters" in assessment_reasoning

            if is_planning_failure:
                logging.warning("Planning failed due to invalid LLM response format/structure. Asking user for guidance directly.")
                ask_question = f"My planning failed. The error was: {assessment_reasoning}. How should I proceed? (e.g., 'stop', 'retry planning', or provide specific instructions)"
                ask_action = {"action_type": "ask_user", "parameters": {"question": ask_question}}
                yield_result = {"type": "ask_user", "question": ask_question}
                user_response_to_ask = yield yield_result

                if user_response_to_ask is None:
                    logging.warning("Generator received None when expecting input for planning failure guidance. Defaulting to 'stop'.")
                    user_response_to_ask = "stop"

                results.append(((ask_action, True, f"Asked user for guidance on planning failure. Response: '{user_response_to_ask}'", None)))
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_ask})

                response_lower = user_response_to_ask.lower()
                if "stop" in response_lower:
                    logging.info("User requested to stop after planning failure.")
                    break
                else:
                    logging.info("User provided guidance after planning failure. Will attempt to re-plan in the next iteration using this new input.")
                    total_consecutive_failures = 0
                    replan_attempts_current_cycle = 0
                    current_task_instruction = user_response_to_ask
                    continue
            elif replan_attempts_current_cycle < MAX_REPLAN_ATTEMPTS:
                replan_attempts_current_cycle += 1
                logging.warning(f"Attempting replan ({replan_attempts_current_cycle}/{MAX_REPLAN_ATTEMPTS}) due to failure.")

                string_history_for_replan = [f"{msg['role']}: {msg['content']}" for msg in (agent_state.current_task.conversation_history if agent_state.current_task else [])]
                new_plan_dict, replan_tokens = request_replan_from_failure(
                    instruction_for_current_planning_cycle, results, action_to_execute, assessment_reasoning,
                    screenshot_after_action, llm_model
                )
                if agent_state.current_task: agent_state.current_task._accumulate_tokens(replan_tokens)
                if new_plan_dict and new_plan_dict.get("plan"):
                    logging.info(f"Replanning successful. New plan has {len(new_plan_dict['plan'])} steps. Restarting execution cycle with new plan.")
                    plan_to_inject = new_plan_dict.get("plan")
                    plan_injection_reason = f"Replanned after failure (Attempt {replan_attempts_current_cycle})"
                    plan_source = "Replanned"
                    if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "system", "content": f"System: Replanning attempt {replan_attempts_current_cycle} successful. New plan generated."})
                    results.append((({'action_type': 'REPLAN', 'parameters': {'attempt': replan_attempts_current_cycle, 'reason': 'failure'}}, True, f"Replanned. New plan: {len(plan_to_inject) if plan_to_inject else 0} steps.", None))) # type: ignore
                    total_consecutive_failures = 0
                    continue
                else:
                    logging.error(f"Replanning attempt {replan_attempts_current_cycle} failed to generate a valid new plan.")
                    results.append((({'action_type': 'REPLAN', 'parameters': {'attempt': replan_attempts_current_cycle}}, False, "Replanning failed to produce plan.", None)))

            if total_consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_ASK:
                logging.warning(f"Reached {total_consecutive_failures} total consecutive failures (including failed replans). Asking user for guidance.")
                ask_question = f"I've encountered repeated failures. The last error was: {assessment_reasoning}. How should I proceed? (e.g., 'stop', 'retry differently', 'skip step', or provide specific instructions)"
                ask_action = {"action_type": "ask_user", "parameters": {"question": ask_question}}
                yield_result = {"type": "ask_user", "question": ask_question}
                user_response_to_ask = yield yield_result

                if user_response_to_ask is None:
                    logging.warning("Generator received None when expecting input for failure guidance. Defaulting to 'stop'.")
                    user_response_to_ask = "stop"

                results.append(((ask_action, True, f"Asked user for guidance on failure. Response: '{user_response_to_ask}'", None)))
                if agent_state.current_task: agent_state.current_task.conversation_history.append({"role": "user", "content": user_response_to_ask})

                response_lower = user_response_to_ask.lower()
                if "stop" in response_lower:
                    logging.info("User requested to stop after failure.")
                    break
                else:
                    logging.info("User provided guidance. Will attempt to re-plan in the next iteration using this new input.")
                    total_consecutive_failures = 0
                    replan_attempts_current_cycle = 0
                    current_task_instruction = user_response_to_ask
                    continue
            else:
                logging.warning("Failure encountered, but not a planning failure, replan attempts available, and total failures below threshold. Proceeding to next iteration for standard planning.")
                continue
        else:
            if current_sub_tasks and current_sub_task_index >= len(current_sub_tasks):
                logging.info("All defined sub-tasks have been processed.")
                task_completed = True 
                break
            logging.error(f"Assessment returned unexpected status '{assessment_status}'. Stopping. Reason: {assessment_reasoning}")
            break

    if agent_state.current_task and current_iteration >= max_iterations:
        logging.warning(f"Execution stopped: Max iterations ({max_iterations}) reached.")
        results.append((({'action_type': 'STOP', 'parameters': {'reason': 'max_iterations'}}, False, f"Max iterations reached.", None)))
        agent_state.current_task.agent_thoughts.append({"timestamp": datetime.now().isoformat(), "content": f"Execution stopped: Max iterations ({max_iterations}) reached.", "type": "stop"})

    final_status = "incomplete"
    execution_summary_for_db = f"Instruction: {original_instruction}\nOutcome: "

    if task_completed and not results: 
        final_status = "success"
        execution_summary_for_db += f"{final_status.upper()} - All sub-tasks completed."
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
        execution_summary_for_db += "No actions attempted or recorded."
        final_status = "failure"

    serializable_results = []
    for r_action, r_success, r_message, r_directive in results:
        serializable_action = r_action if isinstance(r_action, dict) else {"error": "invalid_action_format", "original": str(r_action)}
        serializable_directive = r_directive if isinstance(r_directive, dict) else None
        serializable_results.append([serializable_action, r_success, r_message, serializable_directive])
    full_results_json = json.dumps(serializable_results)

    if agent_state.current_task:
        agent_state.current_task.action_failure_counts = action_failure_counts
        save_task_execution_to_db(original_instruction, execution_summary_for_db, full_results_json, final_status, plan_source, user_feedback=None)

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
    executed_actions_summary: Optional[str] = None
) -> Tuple[Optional[dict], Dict[str, int]]:
    # Check if we already have a successful image generation
    for line in reversed(history):
        if "System Observation:" in line:
            if "IMAGE_GENERATED:" in line:  # Updated check
                return {
                    "plan": [{
                        "action_type": "INFORM_USER",
                        "parameters": {
                            "message": "The image has already been successfully generated. Task is complete."
                        }
                    }],
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


    task_name_base = sanitize_filename(original_instruction)[:30]

    prompt = rf"""
--- USER GOAL ---
**Initial User Goal:**
"{original_instruction}"

--- HISTORY & CONTEXT ---
**Recent History (Actions, Outcomes, Observations, User Responses, Critiques):**
{history_str}
*   **CRITICAL: Focus on the LATEST user message/response.** This defines the current objective. If the user just agreed to create a specialized agent or selected an LLM, plan the next step for *that* process.
*   **IF A "Critique failed" message is in the LATEST history entries:**
    *   The `reasoning` for that critique (e.g., "action was unrelated to the LATEST user instruction") is the MOST IMPORTANT piece of information for your next plan.
    *   Your `next_action` MUST directly address the LATEST user instruction and AVOID the specific pitfall highlighted by the critique.
    *   DO NOT repeat the same type of irrelevant action or an action that led to the critique.
*   **CRITICAL: Learn from critiques.** Avoid repeating actions that failed critique due to redundancy, speculation, or irrelevance.
*   Note: "System Observation: Screen content changed..." indicates the visual state changed.
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

--- META-PLANNING FOR COMPLEX PROJECTS ---
IF the user's LATEST instruction describes a very complex, multi-stage project (e.g., "build an ecommerce website with an AI chatbot", "create a data analysis pipeline for X", "develop a mobile app for Y", "design and implement a customer support system with AI features"),
THEN, your FIRST step is to consider if you have enough information to propose high-level strategic paths OR if you need to gather more information first.

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
    *   **AFTER information gathering (in a subsequent turn, once search results are in history):** You will then proceed to the "COMPLEX TASK - MULTIPLE PATH PROPOSAL" stage below, using the gathered information to inform your proposed paths.

2.  **Proceed to Path Proposal (If sufficient info or after gathering info):**
    *   If you have sufficient information (either from prior knowledge, learnings, or recent search results in history), then proceed directly to the "COMPLEX TASK - MULTIPLE PATH PROPOSAL (INFORM & ASK)" section below.

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
    *   **ELSE (if it's an actionable task for PC automation AND NOT a complex project requiring the above meta-planning/path proposal flow):** Proceed with the planning logic below.
    *   **CRITICAL PLANNING CONSTRAINT:** Review the **Recent History** carefully. If you see a "System Note: [Tool Name] tool disabled..." message, you **MUST NOT** plan to use that specific tool (`action_type`) for the remainder of this task. Choose an alternative tool or strategy, or use `ask_user` if no viable alternative exists.

 2.  **Follow-up After `INFORM_USER`:**
    *   **IF the PREVIOUS action you took was `INFORM_USER` and it was successful, AND the LATEST user message is not a new actionable task or a direct follow-up question to your information, AND the INFORM_USER message did NOT contain '--- START MERMAID CODE ---' markers (meaning it wasn't a multi-path proposal):**
        *   Your `next_action` MUST be `{{ "action_type": "ask_user", "parameters": {{ "question": "Is there anything else I can help you automate today?" }} }}`.
        *   Your `reasoning` should be "Following up after providing information."
    *   **ELSE:** Proceed with other checks.

3.  **Task Completion Check:**
    *   **IF the LATEST user instruction seems fully addressed by the PREVIOUS successful action(s) in the history (and it wasn't an informational query that was just answered by `INFORM_USER`):**
        *   Your `next_action` MUST be `{{ "action_type": "task_complete", "parameters": {{}} }}` (unless the previous action was `INFORM_USER`, in which case see point 2).
        *   Your `reasoning` should state that the user's goal appears to be met.
        *   Example: If user said "open YouTube" and the history shows YouTube was just successfully opened, the next action is `task_complete`.
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
*   `{{ "action_type": "click_and_type", "parameters": {{ "element_description": "Search input field", "text_to_type": "query text" }} }}`
*   `{{ "action_type": "multi_action", "parameters": {{ "sequence": [ {{...action1...}}, {{...action2...}} ] }} }}`
*   `{{ "action_type": "task_complete", "parameters": {{}} }}`
*   `{{ "action_type": "read_file", "parameters": {{ "file_path": "path/to/file.txt" }} }}`
*   `{{ "action_type": "INFORM_USER", "parameters": {{ "message": "Information for the user." }} }}`
*   `{{ "action_type": "process_local_files", "parameters": {{ "file_path": "path/to/file.txt", "prompt": "Analyze this file" }} }}`
*   `{{ "action_type": "process_files_from_urls", "parameters": {{ "url": "https://example.com/file.txt", "prompt": "Analyze this file" }} }}`
*   `{{ "action_type": "start_visual_listener", "parameters": {{ "description_of_change": "...", "polling_interval_seconds": 10, "timeout_seconds": 300, "actions_on_detection": [{{...plan...}}], "actions_on_timeout": [{{...plan...}}]? }} }}`
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
    *   **Idempotency/Pre-checks:** When planning to create a directory (`mkdir`) or a file, consider if its pre-existence is an issue.
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

    try:
        safety_settings = {
             'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        }
        response = llm_model.generate_content(content, safety_settings=safety_settings)
        token_usage = _get_token_usage(response)

        raw_response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    raw_response_text = part.text.strip()
                    break
        elif hasattr(response, 'text') and response.text:
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

        step_dict = None
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
            
            if step_dict is None: # If direct parsing failed or it didn't look like JSON
                # This log will appear if it didn't start with { or [ initially,
                # OR if it did start with {/[ but parsing failed and we are now trying to clean.
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
             if 'DEBUG_DIR' in globals() and os.path.isdir(DEBUG_DIR):
                 problematic_json_path = os.path.join(DEBUG_DIR, f"failed_json_{time.strftime('%Y%m%d-%H%M%S')}.txt")
                 try:
                     with open(problematic_json_path, 'w', encoding='utf-8') as f_err:
                         f_err.write(f"--- Original Raw Response ---\n{raw_response_text}\n")
                         f_err.write(f"--- String Attempted for Parsing (final state) ---\n{cleaned_text if 'cleaned_text' in locals() and cleaned_text else text_to_attempt_parse}\n")
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

        if 'DEBUG_DIR' in globals() and os.path.isdir(DEBUG_DIR):
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

            available_actions = {
                "run_shell_command", "write_file", "ask_user", "run_python_script",
                "multi_action", "press_keys", "focus_window", "click", "type",
                "click_and_type", "move_mouse", "search_web", "navigate_web", "INFORM_USER",
                "wait", "describe_screen", "task_complete", # process_local_files was already singular here
                "capture_screenshot", "read_file", "start_visual_listener", "search_youtube",
                "edit_image_with_file", "process_local_files", "process_files_from_urls"
            }
            if action_type not in available_actions:
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

                available_actions_set = available_actions - {"ask_user", "multi_action", "task_complete", "start_visual_listener"}
                for i, sub_action in enumerate(sequence):
                    if not isinstance(sub_action, dict) or \
                       sub_action.get("action_type") not in available_actions_set or \
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
            
            if action_type == "process_files_from_urls": # Added validation for process_files_from_urls
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
            # Detailed validation logging
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

    except ValueError as ve:
         block_reason = "Unknown"
         finish_reason_val = "Unknown"
         try:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
             if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                 finish_reason_val = str(response.candidates[0].finish_reason)
                 if finish_reason_val == "SAFETY" and not block_reason: # Check if block_reason is already set to avoid redundancy
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

    return step_dict, token_usage # This line was missing in the original diff, added for completeness

def clean_llm_json_response(raw_response: str) -> str:
    """Removes markdown code fences (```json, ```) and trims whitespace from LLM JSON responses."""
    import re
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
