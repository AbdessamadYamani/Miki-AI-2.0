import logging
import os
from typing import Optional
import numpy as np
import cv2
from PIL import Image, ImageGrab
from typing import Dict, Optional, Tuple, Any
import google.generativeai as genai
from tools.token_usage_tool import _get_token_usage 
import hashlib
from utils.file_util import save_debug_data
from vision.xga import  visualize_ui_elements, UIElementCollection
import pyautogui

import re,time,sys
from agents.ai_agent import UIAgent
from chromaDB_management.cache import UICache,get_active_window_name
from utils.image_utils import image_to_base64 , pil_to_cv2# type: ignore

ui_cache = UICache()






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




def capture_full_screen() -> Optional[Image.Image]:
    """
    Capture the entire virtual desktop (all monitors) as a PIL Image.
    Returns None on failure.
    """
    try:

        img = ImageGrab.grab(all_screens=True)
        logging.info("Captured full screen.")
        if img is None:
             logging.error("ImageGrab.grab returned None.")
             return None
        return img
    except ImportError:
         logging.error("Pillow library not found or ImageGrab is not available.")
         return None
    except AttributeError:

         try:
             logging.warning("Pillow version might be old or 'all_screens' not supported. Capturing primary screen only.")
             img = ImageGrab.grab()
             if img is None:
                 logging.error("ImageGrab.grab (primary screen) returned None.")
                 return None
             return img
         except Exception as e_fallback:
             logging.error(f"Error capturing primary screen: {e_fallback}")
             return None
    except Exception as e:

        logging.error(f"Error capturing screen: {e}")
        return None



def _check_visual_condition_with_llm(
    screenshot_pil: Image.Image,
    condition_description: str,
    llm_model: genai.GenerativeModel
) -> Tuple[bool, str, Dict[str, int]]: # Returns (condition_met, reasoning, token_usage)
    """
    Uses LLM to check if a visual condition is met on the screenshot.
    """
    if screenshot_pil is None:
        return False, "Screenshot missing for visual condition check.", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    screenshot_base64 = image_to_base64(screenshot_pil)
    if not screenshot_base64:
        return False, "Screenshot conversion error for visual condition check.", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    prompt = f"""
You are a precise visual analysis agent.
The user is waiting for a specific visual change to occur on the screen.
Your task is to determine if this change is visible in the CURRENT screenshot.

Condition to check for: "{condition_description}"

Current Screen: [See attached image]

Instructions:
1. Carefully analyze the current screenshot.
2. Compare what you see against the "Condition to check for".
3. Provide concise reasoning for your decision.

Output Format (MUST be these two lines ONLY):
REASONING: [Your step-by-step reasoning for why the condition is met or not met based on the image]
CONDITION_MET: [YES or NO]
"""
    content = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": screenshot_base64}}
    ]
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    try:
        safety_settings = { 
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        }
        response = llm_model.generate_content(content, safety_settings=safety_settings)
        token_usage = _get_token_usage(response)
        response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not response_text:
            block_reason = "Unknown"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
            logging.error(f"Visual condition check LLM response blocked or empty: {block_reason}")
            return False, f"LLM check failed (blocked/empty: {block_reason}).", token_usage

        logging.debug(f"Visual Condition Check LLM Raw Response:\n{response_text}")

        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\nCONDITION_MET:|$)', response_text, re.DOTALL | re.IGNORECASE)
        condition_met_match = re.search(r'CONDITION_MET:\s*(YES|NO)', response_text, re.IGNORECASE)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning extracted from LLM."

        if condition_met_match:
            condition_met_text = condition_met_match.group(1).upper()
            return condition_met_text == "YES", reasoning, token_usage
        else:
            logging.warning(f"Could not parse CONDITION_MET from LLM response for visual check. Assuming NO. Response: {response_text}")
            return False, reasoning + " (Agent Note: Could not parse CONDITION_MET, assumed NO)", token_usage

    except Exception as e:
        logging.error(f"Error during visual condition check LLM call: {e}", exc_info=True)
        return False, f"Exception during LLM check: {e}", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}


def _hash_pil_image(image: Optional[Image.Image]) -> Optional[str]:
    """Generate a hash for a PIL image."""
    if image is None:
        return None
    try:

        np_image = np.array(image.convert('RGB'))


        return hashlib.md5(np_image.tobytes()).hexdigest()
    except Exception as e:
        logging.error(f"Error hashing PIL image: {e}")
        return None



def _execute_visual_listener(
    params: Dict[str, Any],
    agent_object: 'UIAgent', 
    llm_model: genai.GenerativeModel
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    description = params.get("description_of_change")
    polling_interval = params.get("polling_interval_seconds", 5.0)
    timeout_seconds = float(params.get("timeout_seconds", 300.0)) # Ensure float
    actions_on_detection = params.get("actions_on_detection", []) 
    actions_on_timeout = params.get("actions_on_timeout", [])     
    # area_to_monitor = params.get("area_to_monitor")

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



def get_screen_description_from_gemini(screenshot_pil: Image.Image, llm_model: genai.GenerativeModel) -> Tuple[bool, str, Dict[str, int]]:
    """Sends screenshot to Gemini and asks for a description."""
    if screenshot_pil is None:
        return False, "Description failed: Missing screenshot.", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    screenshot_base64 = image_to_base64(screenshot_pil)
    if not screenshot_base64:
        return False, "Description failed: Screenshot conversion error.", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    prompt = "Please describe in detail what you see in this screenshot of a computer screen. Focus on open applications, windows, icons, and any visible text."
    content = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": screenshot_base64}}
    ]
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    try:
        safety_settings = {}
        response = llm_model.generate_content(content, safety_settings=safety_settings)
        token_usage = _get_token_usage(response)
        description = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            description = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not description:
            if response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                logging.error(f"Screen description LLM response blocked: {reason}")
                return False, f"Description failed: LLM blocked ({reason}).", token_usage
            else:
                logging.error("Screen description LLM returned empty response.")
                return False, "Description failed: LLM empty response.", token_usage

        logging.info("Successfully obtained screen description from Gemini.")
        logging.debug(f"Full Screen Description:\n{description}")
        summary = (description[:300] + '...') if len(description) > 303 else description
        return True, f"Screen description: {summary}", token_usage

    except ValueError as ve:
        logging.error(f"ValueError accessing description LLM response: {ve}")
        return False, f"Description failed: ValueError ({ve}).", token_usage 
    except Exception as e:
        logging.error(f"Error during description LLM call: {e}", exc_info=True)
        return False, f"Description failed: Exception ({e}).", token_usage




def focus_window_by_title(title_substring: str, timeout: int = 5) -> bool:
    """
    Attempts to find and focus a window whose title contains the given substring.
    Includes verification after attempting focus.
    Primarily for Windows using win32gui. Basic fallback for other platforms.
    Returns True if focus was successfully set and verified, False otherwise.
    """
    logging.info(f"Attempting to focus window containing title: '{title_substring}'")
    if sys.platform == "win32":
        try:
            import win32gui
            import win32com.client
            import time

            hwnd = None
            start_time = time.time()


            while hwnd is None and (time.time() - start_time) < timeout:
                try:

                    def callback(h, extra):
                        if win32gui.IsWindowVisible(h) and title_substring.lower() in win32gui.GetWindowText(
                            h
                        ).lower():
                            extra.append(h)
                        return True

                    hwnds = []
                    win32gui.EnumWindows(callback, hwnds)
                    if hwnds:
                        hwnd = hwnds[0]
                        logging.debug(
                            f"Found potential window handle {hwnd} with title '{win32gui.GetWindowText(hwnd)}'"
                        )
                        break
                except Exception as enum_e:
                    logging.debug(f"Error during window enumeration: {enum_e}")
                time.sleep(0.2)

            if hwnd:
                logging.info(f"Found window handle {hwnd}. Attempting to set focus.")
                try:
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.SendKeys("%")
                    time.sleep(0.1)
                    win32gui.SetForegroundWindow(hwnd)
                    time.sleep(0.5)


                    try:
                        current_hwnd = win32gui.GetForegroundWindow()
                        if current_hwnd == hwnd:
                            logging.info("Focus successfully set and verified by handle.")
                            return True
                        else:

                            current_title = win32gui.GetWindowText(current_hwnd)
                            if title_substring.lower() in current_title.lower():
                                logging.info(
                                    f"Focus likely set (current window title '{current_title}' matches substring)."
                                )
                                return True
                            else:
                                logging.warning(
                                    f"SetForegroundWindow called, but focus verification failed (Current HWND: {current_hwnd}, Title: '{current_title}')."
                                )
                                return False
                    except Exception as verify_e:
                        logging.warning(
                            f"Error during focus verification: {verify_e}. Assuming focus attempt was made but unverified."
                        )

                        return False

                except Exception as focus_e:
                    logging.error(f"Error setting focus to window {hwnd}: {focus_e}")
                    return False
            else:
                logging.warning(
                    f"Could not find visible window with title containing '{title_substring}' within {timeout} seconds."
                )
                return False

        except ImportError:
            logging.warning(
                "Cannot perform reliable window focus: 'pywin32'/'win32com' not installed."
            )
            return False
        except Exception as e:
            logging.error(f"Unexpected error during window focus attempt: {e}")
            return False
    else:
        logging.warning(
            f"Window focusing by title is not implemented for platform: {sys.platform}"
        )
        return False
