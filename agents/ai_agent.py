import logging
import re
from typing import Optional, Tuple, Dict, Union
import google.generativeai as genai
import numpy as np
from utils.image_utils import cv2_to_pil, image_to_base64 # cv2_to_pil is used
from tools.token_usage_tool import _get_token_usage # Assuming this is in tools
from vision.xga import UIElementCollection # Assuming this is in vision.xga

class UIAgent:
    def __init__(self, llm_model: genai.GenerativeModel):
        self.model = llm_model
        self.last_reasoning: str = "Selection process not started."

    def select_ui_element_for_click(
            self,
            elements: UIElementCollection,
            element_desc: str,
            cv2_screenshot: Optional[np.ndarray], # Changed type hint
            vis_img: Optional[np.ndarray] # Changed type hint
        ) -> Tuple[Optional[int], Dict[str, int]]:

            llm_call_token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
            self.last_reasoning = "Selection process not started."

            if not elements:
                logging.warning("No UI elements provided to select_ui_element_for_click.")
                self.last_reasoning = "No UI elements were detected on the screen to select from."
                return None, llm_call_token_usage

            if cv2_screenshot is None:
                logging.error("Cannot select UI element without a screenshot (cv2_screenshot is None).")
                self.last_reasoning = "Missing screenshot for visual analysis (cv2_screenshot was None)."
                return None, llm_call_token_usage

            elements_description = []
            for i, elem in enumerate(elements):
                label = elem.label.strip() if elem.label else "[No Label]"
                desc_str = f"Element {i}: Type='{elem.element_type}', Label='{label[:50]}{'...' if len(label)>50 else ''}', Center=({int(elem.center[0])},{int(elem.center[1])})"
                elements_description.append(desc_str)

            if not elements_description:
                logging.warning("UI elements list was empty after formatting descriptions.")
                self.last_reasoning = "Detected UI elements list was empty or could not be processed."
                return None, llm_call_token_usage

            elements_text = "\n".join(elements_description)            
            # Convert cv2_screenshot (NumPy array) to PIL Image
            orig_pil = None
            if cv2_screenshot is not None: # cv2_screenshot is np.ndarray
                try:
                    orig_pil = cv2_to_pil(cv2_screenshot) 
                    if orig_pil is None:
                        logging.error("cv2_to_pil returned None for original screenshot.")
                        self.last_reasoning = "Error processing original screenshot (conversion to PIL failed)."
                        return None, llm_call_token_usage
                except Exception as e:
                    logging.error(f"Failed to convert cv2_screenshot (NumPy array) to PIL: {e}")
                    self.last_reasoning = "Error processing original screenshot (conversion exception)."
                    return None, llm_call_token_usage
            else: # This case should have been caught earlier, but as a safeguard
                logging.error("cv2_screenshot became None unexpectedly before PIL conversion.")
                self.last_reasoning = "Internal error: screenshot became unavailable."
                return None, llm_call_token_usage
            
            orig_base64 = image_to_base64(orig_pil)

            # Convert vis_img (NumPy array) to PIL Image
            vis_pil = None
            if vis_img is not None: # vis_img is np.ndarray or None
                try:
                    vis_pil = cv2_to_pil(vis_img) 
                except Exception as e:
                    logging.warning(f"Failed to convert vis_img (NumPy array) to PIL: {e}")
            vis_base64 = image_to_base64(vis_pil) if vis_pil else None


            if not orig_base64:
                logging.error("Failed to convert original screenshot to base64.")
                self.last_reasoning = "Error processing original screenshot for LLM."
                return None, llm_call_token_usage

            prompt_parts = [
                "You are an advanced UI Navigation Agent. Your task is to identify the exact element a user wants to click based on a description, using visual analysis and a list of detected elements.",
                "\nI'm providing image(s) and a list of detected UI elements:",
                "\n1. FIRST IMAGE: The original screenshot.",
            ]
            content_for_llm = [{"inline_data": {"mime_type": "image/png", "data": orig_base64}}]

            if vis_base64:
                prompt_parts.append("\n2. SECOND IMAGE: Visualization with numbered boxes highlighting detected elements (numbers match indices below).")
                content_for_llm.append({"inline_data": {"mime_type": "image/png", "data": vis_base64}})
            else:
                prompt_parts.append("\n(Note: Visualization image is not available.)")

            prompt_parts.extend([
                f"\n\nUser's request: Find and click on '{element_desc}'",
                f"\n\nDetected UI elements with their indices:\n{elements_text}\n",
                "\nINSTRUCTIONS FOR ANALYSIS:",
                "- Analyze the FIRST IMAGE (original screenshot) to visually locate what the user is asking for based on the description.",
            ])
            if vis_base64:
                prompt_parts.append("- Use the SECOND IMAGE (visualization) to map your visual finding to an element index from the list.")
            else:
                prompt_parts.append("- Rely heavily on element labels, types, and positions in the list compared to the original screenshot.")

            prompt_parts.extend([
                "\n\n**Specific Guidance for Common Elements:**",
                "- **Video Thumbnails/Links:** Often appear as rectangular images with titles. The clickable area is usually the image or the title text. Look for elements with labels matching video titles or generic descriptions like 'video thumbnail'. If multiple similar items exist (e.g., search results), use relative position (e.g., 'first', 'top-most') if specified in the user's request.",
                "\nIMPORTANT: Provide step-by-step reasoning.",
                "1. Describe what you visually identify in the original screenshot matching the request.",
                "2. Examine the element list for candidates based on label, type, and location.",
                "3. If visualization is available, confirm the index using the numbered boxes.",
                "4. Explain your choice for the best match or state if no clear match exists.",
                "\nFormat your response ONLY with these two lines:",
                "REASONING: [Your detailed step-by-step reasoning here]",
                "SELECTED: [The index number of the best match, or 'NOT FOUND']"
            ])

            prompt_text = "\n".join(prompt_parts)
            content_for_llm.insert(0, {"text": prompt_text})

            try:
                safety_settings = {} # Define safety settings if needed
                response = self.model.generate_content(content_for_llm, safety_settings=safety_settings)
                llm_call_token_usage = _get_token_usage(response)
                txt = ""
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    txt = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

                if not txt:
                    block_reason = "Unknown"
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                        block_reason = str(response.prompt_feedback.block_reason)
                    logging.error(f"LLM response blocked: {block_reason}")
                    self.last_reasoning = f"LLM response blocked ({block_reason})."
                    return None, llm_call_token_usage

                logging.info(f"[UI Agent Raw Response]\n{txt}")
            except ValueError as ve:
                logging.error(f"ValueError accessing LLM response: {ve}")
                self.last_reasoning = f"LLM response likely blocked. ValueError: {ve}"
                return None, llm_call_token_usage
            except Exception as e:
                logging.error(f"Error getting response from Gemini: {e}")
                self.last_reasoning = f"Error communicating with LLM: {e}"
                return None, llm_call_token_usage

            reasoning = "No reasoning extracted."
            selection = None

            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\nSELECTED:|$)', txt, re.DOTALL | re.IGNORECASE)
            selection_match = re.search(r'SELECTED:\s*(\d+|NOT FOUND)', txt, re.IGNORECASE)

            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                logging.warning("Could not extract reasoning.")
                sel_keyword_pos = txt.upper().find("SELECTED:")
                reasoning = txt[:sel_keyword_pos].strip() if sel_keyword_pos != -1 and sel_keyword_pos > 0 else txt

            if selection_match:
                selection_text = selection_match.group(1).strip()
                if selection_text.isdigit():
                    try:
                        selected_idx = int(selection_text)
                        if 0 <= selected_idx < len(elements):
                            selection = selected_idx
                        else:
                            logging.warning(f"LLM selected invalid index: {selected_idx} (max: {len(elements)-1}).")
                            reasoning += f"\n(Agent Note: LLM selected invalid index {selected_idx}.)"
                            selection = None # Explicitly set to None
                    except ValueError:
                        logging.warning(f"Could not convert selected text '{selection_text}' to int.")
                        selection = None
                elif "NOT FOUND" in selection_text.upper():
                    logging.info("LLM indicated element not found.")
                    selection = None
                else:
                    logging.warning(f"Could not parse selection: '{selection_text}'")
                    selection = None
            else:
                logging.warning("Could not extract selection.")
                if "NOT FOUND" not in txt.upper():
                    lines = txt.strip().split('\n')
                    last_line = lines[-1] if lines else ""
                    numbers_in_last_line = re.findall(r'\b(\d+)\b$', last_line)
                    if not numbers_in_last_line:
                        numbers_in_text = re.findall(r'\b(\d+)\b', txt)
                        if numbers_in_text:
                            numbers_in_last_line = [numbers_in_text[-1]]

                    if numbers_in_last_line:
                        try:
                            potential_idx = int(numbers_in_last_line[-1])
                            if 0 <= potential_idx < len(elements):
                                logging.warning(f"Used fallback extraction, selected index: {potential_idx}")
                                selection = potential_idx
                                reasoning += f"\n(Agent Note: Used fallback extraction, selected index {potential_idx}.)"
                            else:
                                logging.warning(f"Fallback number {potential_idx} out of bounds.")
                        except ValueError:
                            pass
            
            self.last_reasoning = reasoning
            logging.info(f"LLM selected element index: {selection}" if selection is not None else "LLM did not select a valid index.")
            return selection, llm_call_token_usage
