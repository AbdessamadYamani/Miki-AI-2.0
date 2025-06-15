import logging
from typing import Optional, Tuple, Dict
import google.generativeai as genai
import time
from typing import  Dict, Optional, Tuple, Any

from agents.ai_agent import UIAgent
from vis import capture_full_screen, _check_visual_condition_with_llm


def _execute_visual_listener(
    params: Dict[str, Any],
    agent_object: 'UIAgent', 
    llm_model: genai.GenerativeModel
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    description = params.get("description_of_change")
    polling_interval = params.get("polling_interval_seconds", 5.0)
    timeout_seconds = float(params.get("timeout_seconds", 300.0)) 
    actions_on_detection = params.get("actions_on_detection", []) 
    actions_on_timeout = params.get("actions_on_timeout", [])     

    if not description:
        return False, "Visual listener failed: 'description_of_change' is missing.", None
    if not isinstance(actions_on_detection, list):
        return False, "Visual listener failed: 'actions_on_detection' must be a list (plan).", None
    if not isinstance(actions_on_timeout, list):
        return False, "Visual listener failed: 'actions_on_timeout' must be a list (plan).", None

    accumulated_listener_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    start_time = time.time()
    logging.info(f"Starting visual listener: Looking for '{description}' for up to {timeout_seconds}s (polling every {polling_interval}s).")

    while time.time() - start_time < timeout_seconds:
        screenshot_pil = capture_full_screen()
        if not screenshot_pil:
            logging.warning("Listener: Failed to capture screen during poll.")
            time.sleep(polling_interval) # type: ignore
            continue

        condition_met, reasoning, check_tokens = _check_visual_condition_with_llm(
            screenshot_pil, description, llm_model
        )
        for k in accumulated_listener_tokens: accumulated_listener_tokens[k] += check_tokens[k] # type: ignore

        logging.info(f"Listener poll: Condition '{description}' met? {'YES' if condition_met else 'NO'}. Reasoning: {reasoning}")

        if condition_met:
            logging.info(f"Visual condition '{description}' MET.")
            directive = {
                "type": "inject_plan",
                "plan": actions_on_detection,
                "reason": f"visual_listener_condition_met: {description}",
                "token_usage": accumulated_listener_tokens # Include accumulated tokens
            }
            return True, f"Condition '{description}' met. Triggering detection actions.", directive

        time.sleep(polling_interval) # type: ignore

    logging.warning(f"Visual listener TIMEOUT for condition: '{description}'.")
    directive = None
    if actions_on_timeout:
        directive = {
            "type": "inject_plan",
            "plan": actions_on_timeout,
            "reason": f"visual_listener_timeout: {description}",
            "token_usage": accumulated_listener_tokens # Include accumulated tokens
        }
        return True, f"Listener for '{description}' timed out. Triggering timeout actions.", directive
    else:
        return False, f"Listener for '{description}' timed out. No timeout actions defined.", {"type": "listener_timeout_no_actions", "token_usage": accumulated_listener_tokens}

