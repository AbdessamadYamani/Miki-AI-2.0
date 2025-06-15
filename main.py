import os
import re
import sys
import time
import cv2
import pyautogui
from PIL import Image
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple
import json
from tools.token_usage_tool import _get_token_usage
import logging 
import os, sys
import logging 
from vision.xga import visualize_ui_elements, UIElementCollection
import sys, os,io
from utils.tesseract import ensure_tesseract_windows 
from chromaDB_management.cache import UICache,get_active_window_name
from vision.vis import capture_full_screen, pil_to_cv2, image_to_base64
from utils.file_util import save_debug_data
ensure_tesseract_windows()
from agents.ai_agent import UIAgent
from config import API_KEY,model


shortcuts_cache = {}










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
                                "generate_large_content_with_gemini", "INFORM_USER", # Added
                                "capture_screenshot", "read_file", "search_web", "search_youtube"}
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





def locate_and_click_ui_element(element_desc: str, agent: UIAgent) -> Tuple[bool, str]:
    """
    Pipeline for clicking a specific UI element. Captures screen, detects/caches elements,
    uses UIAgent to select, and performs the click.
    Returns Tuple[bool, str] indicating success and a message.
    """
    logging.info(f"Attempting to locate and click UI element: '{element_desc}'")
    error_prefix = "Click failed: "


    pil_screenshot = capture_full_screen()
    if pil_screenshot is None:
        return False, error_prefix + "Could not capture screen."
    cv2_screenshot = pil_to_cv2(pil_screenshot)
    if cv2_screenshot is None:
        return False, error_prefix + "Could not convert screenshot."


    app_name = get_active_window_name()
    logging.info(f"Active application context for UI elements: {app_name}")




    ui_elements = ui_cache.get_ui_elements(cv2_screenshot, app_name)
    if not ui_elements:
        msg = f"No UI elements detected/cached for the current screen ('{app_name}')."
        logging.warning(msg)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_debug_data(app_name, timestamp, cv2_screenshot, None, UIElementCollection())
        return False, error_prefix + msg

    logging.info(f"Working with {len(ui_elements)} UI elements for '{app_name}'")


    vis_img = visualize_ui_elements(cv2_screenshot, ui_elements)


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    debug_session_dir = save_debug_data(
        app_name, timestamp, cv2_screenshot, vis_img, ui_elements
    )


    matching_idx = agent.select_ui_element_for_click(
        ui_elements, element_desc, cv2_screenshot, vis_img
    )


    logging.info(f"UI Element Selection Reasoning:\n{agent.last_reasoning}\n")

    if matching_idx is None:
        msg = f"No matching element found by LLM for '{element_desc}'"
        logging.warning(f"[ERROR] {msg}")
        if debug_session_dir:
            failed_search_path = os.path.join(debug_session_dir, "search_failed.txt")
            try:
                with open(failed_search_path, "w", encoding="utf-8") as f:
                    f.write(f"Search term: {element_desc}\nResult: No match\nReasoning:\n{agent.last_reasoning}")
            except Exception as e:
                logging.error(f"Could not write failed search info: {e}")
        return False, error_prefix + msg


    try:
        matching_element = ui_elements[matching_idx]
        x = int(matching_element.center[0])
        y = int(matching_element.center[1])
    except IndexError:
        msg = f"Internal error: Index {matching_idx} out of bounds (size {len(ui_elements)})."
        logging.error(msg)
        return False, error_prefix + msg
    except (TypeError, ValueError) as e:
        msg = f"Error processing coords for index {matching_idx}: {e}. Center: {getattr(matching_element, 'center', 'N/A')}"
        logging.error(msg)
        return False, error_prefix + msg
    except Exception as e:
        msg = f"Error accessing element at index {matching_idx}: {e}"
        logging.error(msg)
        return False, error_prefix + msg


    element_label_short = matching_element.label[:50] + (
        "..." if len(matching_element.label) > 50 else ""
    )
    match_log_msg = f"Match found: Index={matching_idx}, Type='{matching_element.element_type}', Label='{element_label_short}', Center=({x},{y})"
    logging.info(f"[OK] {match_log_msg}")


    if debug_session_dir:
        match_info_path = os.path.join(debug_session_dir, "match_successful.txt")
        try:
            with open(match_info_path, "w", encoding="utf-8") as f:
                f.write(f"Search term: {element_desc}\n")
                f.write(f"Matched index: {matching_idx}\n")
                f.write(f"Element: {vars(matching_element)}\n")
                f.write(f"Reasoning:\n{agent.last_reasoning}\n")

            match_vis_img = cv2_screenshot.copy()
            half_width, half_height = int(matching_element.width / 2), int(
                matching_element.height / 2
            )
            x1, y1, x2, y2 = (
                x - half_width,
                y - half_height,
                x + half_width,
                y + half_height,
            )
            cv2.rectangle(match_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label_text = f"Match ({matching_idx}): {matching_element.label[:30]}"
            (tw, th), bl = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            ty = y1 - 10 if y1 - 10 > th else y2 + th + 5
            cv2.rectangle(
                match_vis_img,
                (x1, ty - th - bl),
                (x1 + tw, ty + bl),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                match_vis_img,
                label_text,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            match_vis_path = os.path.join(debug_session_dir, "matched_element_highlight.png")
            cv2.imwrite(match_vis_path, match_vis_img)
        except Exception as e:
            logging.error(f"Could not write successful match info/vis: {e}")


    try:
        logging.info(f"Performing click action at ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.25)
        pyautogui.click(x, y)
        success_msg = f"Successfully clicked '{element_label_short}' at ({x}, {y})."
        logging.info(success_msg)
        time.sleep(0.5)
        return True, success_msg
    except Exception as e:
        msg = f"Error during pyautogui click at ({x},{y}): {e}"
        logging.error(msg)
        return False, error_prefix + msg





if __name__ == "__main__":
    if not API_KEY:
         print("\nFATAL ERROR: Gemini API Key Missing. Set GEMINI_API_KEY environment variable.")
         sys.exit(1)
    try:
        if sys.platform == "win32":
            import win32gui
            import win32com.client
    except ImportError:
         if sys.platform == "win32":
             logging.warning("'pywin32' not found. Window focusing ('focus_window' action) will not work reliably.")


    ui_cache = UICache()
    ui_agent = UIAgent(model) 
