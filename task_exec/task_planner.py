import logging
from typing import List, Optional, Tuple, Dict
from PIL import Image
import google.generativeai as genai
import json
from vision.vis import image_to_base64
from tools.token_usage_tool import _get_token_usage
import re,os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_model





def process_user_instruction(instruction: str, conversation_history: List[str]) -> Tuple[Optional[dict], Dict[str, int]]:
    """
    Process user instruction and break it down into sub-tasks if needed.
    Returns a plan with sub-tasks and their execution order.
    """
    prompt = f"""You are an AI task planner. Your job is to break down complex tasks into sub-tasks and create a detailed execution plan.

User Instruction: {instruction}

Conversation History:
{chr(10).join(conversation_history) if conversation_history else "No previous conversation."}

First, analyze if this is a complex task that needs to be broken down into sub-tasks.
If it is complex, break it down into logical sub-tasks.
For each sub-task, create a detailed plan of actions.

Example of a complex task breakdown:
Task: "Open YouTube and search for the latest song of Zara Larsson then play it"
Sub-tasks:
1. Open YouTube
   - Focus on browser window
   - Navigate to youtube.com
2. Search for Zara Larsson
   - Click search bar
   - Type search query
   - Press enter
3. Play the first video
   - Wait for search results
   - Click first video
   - Verify video starts playing

Return your plan as a JSON object with this structure:
{{
    "is_complex_task": true/false,
    "sub_tasks": [
        {{
            "description": "Sub-task description",
            "actions": [
                {{
                    "action_type": "action_name",
                    "parameters": {{
                        "param1": "value1",
                        "param2": "value2"
                    }}
                }}
            ]
        }}
    ],
    "reasoning": "Explanation of why this breakdown was chosen"
}}

If the task is simple (doesn't need sub-tasks), return:
{{
    "is_complex_task": false,
    "actions": [
        {{
            "action_type": "action_name",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ],
    "reasoning": "Explanation of why this is a simple task"
}}
"""

    # Generate the plan using the model instance
    response = llm_model.generate_content(prompt)
    token_usage = _get_token_usage(response)
    
    if not response.candidates or not response.candidates[0].content:
        return None, token_usage
        
    plan_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
    
    try:
        plan_dict = json.loads(plan_text)
        
        # Validate the plan structure
        if not isinstance(plan_dict, dict):
            raise ValueError("Plan must be a dictionary")
            
        if "is_complex_task" not in plan_dict:
            raise ValueError("Plan must specify if it's a complex task")
            
        if plan_dict["is_complex_task"]:
            if "sub_tasks" not in plan_dict:
                raise ValueError("Complex task must have sub_tasks")
            for sub_task in plan_dict["sub_tasks"]:
                if "description" not in sub_task or "actions" not in sub_task:
                    raise ValueError("Each sub-task must have description and actions")
        else:
            if "actions" not in plan_dict:
                raise ValueError("Simple task must have actions")
                
        return plan_dict, token_usage
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse plan JSON: {e}")
        return None, token_usage
    except ValueError as e:
        logging.error(f"Invalid plan structure: {e}")
        return None, token_usage
    except Exception as e:
        logging.error(f"Unexpected error processing plan: {e}")
        return None, token_usage



def critique_action(
    original_instruction: str,
    history: List[str],
    action_to_critique: dict,
    llm_model: genai.GenerativeModel,
    critique_screenshot_pil: Optional[Image.Image] = None
) -> Tuple[bool, str, Dict[str, int]]:
    action_type = action_to_critique.get('action_type', 'N/A')
    action_params = action_to_critique.get('parameters', {})
    logging.info(f"Critiquing proposed action: {action_type} with params: {action_params}")
    logging.info(f"LLM Model for critique: {llm_model}")

    if llm_model is None:
        logging.error("LLM model is None in critique_action")
        return False, "Critique failed: LLM model not initialized", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    max_history_turns_for_prompt = 6
    string_history = [str(item) for item in history]
    important_indices = [i for i, item in enumerate(string_history) if "System Observation:" in item or "User responded:" in item or "ask_user" in item or "Critique failed" in item or "User Goal:" in item]
    if len(string_history) > max_history_turns_for_prompt * 2:
        cutoff = len(string_history) - (max_history_turns_for_prompt * 2)
        keep_indices = {0} if len(string_history) > 0 and "User Goal:" in string_history[0] else set()
        keep_indices.update(range(cutoff, len(string_history)))
        keep_indices.update(idx for idx in important_indices if idx >= cutoff)
        keep_indices = {idx for idx in keep_indices if 0 <= idx < len(string_history)} # type: ignore
        truncated_history = [string_history[i] for i in sorted(list(keep_indices))]
    else:
        truncated_history = string_history
    history_str = "\n".join(truncated_history)

    screenshot_base64 = image_to_base64(critique_screenshot_pil) if critique_screenshot_pil else None

    prompt = f"""
You are an Action Critic for a PC Automation Assistant. Your task is to evaluate a single proposed action for safety, sensibility, and appropriateness given the context.

--- CONTEXT ---
**Initial User Goal (May be outdated):**
"{original_instruction}"

**Recent History (Actions, Outcomes, User Responses, Critiques):**
{history_str}
*   **CRITICAL: Identify the LATEST user instruction or clarification in the history.** This might override the initial goal. The proposed action MUST align with this LATEST request.

**Proposed Action to Critique:**
Action Type: {action_type}
Parameters: {action_params}

**Current Visual Context:** [See attached screenshot if available - CRUCIAL for visual actions like 'click', 'type']

--- YOUR TASK ---
Analyze the **Proposed Action** based on the **LATEST User Instruction** (from history), **Recent History**, and **Current Visual Context** (if provided and relevant).

1.  **Safety Check:** Is the action potentially harmful or destructive (e.g., deleting files unexpectedly, risky shell commands)?
2.  **Sensibility Check:** Does this action logically follow the history and contribute towards the **LATEST user instruction/clarification**?
    *   Is it redundant given the history?
    *   **EXCEPTION:** If the LATEST user instruction was purely conversational (e.g., "hi", "hello", "thanks", "ok") AND the Proposed Action is `ask_user` with a simple greeting/acknowledgement (e.g., "Hello!", "Okay, what next?"), this is **NOT** redundant and should PASS the sensibility check.
    *   Does it make sense given previous critiques (if any)? Avoid repeating failed patterns.
    *   Is it targeting the correct application/context based on the screenshot or history?
3.  **Parameter Check:** Are the parameters reasonable? (e.g., Is the `element_description` for `click` specific enough? Is the `command` for `run_shell_command` valid? Is the `wait` duration appropriate?)
4.  **Visual Context Check (if screenshot provided):**
    *   For `edit_image_with_file`:
        *   If the LATEST user instruction is to **create or generate** an image (e.g., "create a picture of a cat"), then `image_path` being `null` or an empty string (`""`) is **CORRECT and EXPECTED**. The `prompt` parameter should describe the image to be generated.
        *   If the LATEST user instruction is to **edit or modify** an existing image, then `image_path` **MUST** be a valid path to an image file.
        *   Critique based on this dual capability. Do not fail a generation request solely because `image_path` is empty/null.
4.  **Visual Context Check (if screenshot provided):**
    *   For `click`/`type`/`click_and_type`: Does the target element described in `element_description` seem visible and appropriate in the screenshot for the **LATEST user goal**?
    *   For `focus_window`: Does a window matching `title_substring` appear plausible based on the screenshot or history for the **LATEST user goal**?
    *   For `run_shell_command`: Does the command make sense given the visible application state and the **LATEST user goal**?

**Output Format:**
Provide ONLY these two lines:
CRITIQUE: [Your concise reasoning for PASS or FAIL, highlighting any concerns, specifically mentioning if it aligns with the LATEST user instruction]
DECISION: [PASS or FAIL]
"""

    content = [{"text": prompt}]
    if screenshot_base64:
        content.append({"inline_data": {"mime_type": "image/png", "data": screenshot_base64}})
    else:
        content[0]['text'] += "\n(Note: Screenshot not available for critique)" # type: ignore

    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    try:
        response = llm_model.generate_content(
            content,
            safety_settings={}
        )
        token_usage = _get_token_usage(response)
        txt = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            txt = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not txt:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                logging.error(f"Critique LLM response blocked: {block_reason}")
                return False, f"Critique failed: LLM response blocked ({block_reason}).", token_usage
            else:
                logging.error("Critique LLM response was empty.")
                return False, "Critique failed: LLM returned empty response.", token_usage

        logging.debug(f"[Critic Raw Response]\n{txt}")

        critique_reasoning = "No reasoning extracted."
        decision = "FAIL" 

        critique_match = re.search(r'CRITIQUE:\s*(.*?)(?=\nDECISION:|$)', txt, re.DOTALL | re.IGNORECASE)
        decision_match = re.search(r'DECISION:\s*(PASS|FAIL)', txt, re.IGNORECASE)

        if critique_match:
            critique_reasoning = critique_match.group(1).strip()
        else:
            decision_keyword_pos = txt.upper().find("DECISION:")
            critique_reasoning = txt[:decision_keyword_pos].strip() if decision_keyword_pos != -1 else txt

        if decision_match:
            decision = decision_match.group(1).strip().upper()
            if decision not in ["PASS", "FAIL"]:
                logging.warning(f"Extracted decision '{decision}' is invalid. Defaulting to FAIL.")
                decision = "FAIL"
        else:
            logging.warning(f"Could not parse DECISION from critique. Defaulting to FAIL. Response: {txt}")
            critique_reasoning += "\n(Agent Note: Could not parse DECISION, defaulted to FAIL)"

        critique_passed = (decision == "PASS")

        if not critique_passed:
            logging.warning(f"Critique failed: {critique_reasoning}")
        else:
            logging.info(f"Critique passed: {critique_reasoning}")

        return critique_passed, critique_reasoning, token_usage

    except Exception as e:
        logging.error(f"Error during critique LLM call: {e}", exc_info=True)
        return False, f"Critique failed: API error occurred. {str(e)}", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

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
    if llm_model is None:
        llm_model = get_model()
        if llm_model is None:
            logging.error("LLM model is None and could not be initialized")
            return None, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    # ... rest of the function ...
