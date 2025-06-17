import logging
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any, Generator
from tools.token_usage_tool import _get_token_usage
from vision.xga import detect_ui_elements_from_image, visualize_ui_elements, UIElementCollection, UIElement # type: ignore # type: ignore
from utils.image_utils import cv2_to_pil, image_to_base64 # type: ignore



class UIAgent:
    def __init__(self, llm_model):
        self.model = llm_model
        self.last_reasoning = ""

    def select_ui_element_for_click(
            self,
            elements: UIElementCollection,
            element_desc: str,
            cv2_screenshot: Optional[np.ndarray],
            vis_img: Optional[np.ndarray]
        ) -> Tuple[Optional[int], Dict[str, int]]:

            # Initialize token_usage for this specific LLM call
            llm_call_token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
            self.last_reasoning = "Selection process not started."

            if not elements:
                logging.warning("No UI elements provided to select_ui_element_for_click.")
                self.last_reasoning = "No UI elements were detected on the screen to select from."
                return None, llm_call_token_usage

            if cv2_screenshot is None:
                logging.error("Cannot select UI element without a screenshot.")
                self.last_reasoning = "Missing screenshot for visual analysis."
                return None, llm_call_token_usage

            # Try pattern matching first for common cases
            element_desc_lower = element_desc.lower()
            
            # Common button patterns
            button_patterns = {
                'ok': ['ok', 'okay', 'confirm', 'done', 'finish'],
                'cancel': ['cancel', 'close', 'exit', 'back'],
                'yes': ['yes', 'confirm', 'agree', 'accept'],
                'no': ['no', 'decline', 'reject', 'deny'],
                'next': ['next', 'continue', 'proceed', 'forward'],
                'previous': ['previous', 'back', 'return'],
                'save': ['save', 'store', 'keep'],
                'delete': ['delete', 'remove', 'erase', 'clear']
            }

            # Check for exact matches first
            for i, elem in enumerate(elements):
                elem_label_lower = elem.label.lower()
                
                # Exact match
                if element_desc_lower == elem_label_lower:
                    self.last_reasoning = f"Found exact match for '{element_desc}'"
                    return i, llm_call_token_usage
                
                # Check button patterns
                for pattern, keywords in button_patterns.items():
                    if any(keyword in elem_label_lower for keyword in keywords) and pattern in element_desc_lower:
                        self.last_reasoning = f"Found button pattern match for '{element_desc}'"
                        return i, llm_call_token_usage

            # If no pattern match found, proceed with AI model
            elements_description = []
            for i, elem in enumerate(elements):
                label = elem.label.strip() if elem.label else "[No Label]"
                desc = f"Element {i}: Type='{elem.element_type}', Label='{label[:50]}{'...' if len(label)>50 else ''}', Center=({int(elem.center[0])},{int(elem.center[1])})"
                elements_description.append(desc)

            if not elements_description:
                logging.warning("UI elements list was empty after formatting descriptions.")
                self.last_reasoning = "Detected UI elements list was empty or could not be processed."
                return None, llm_call_token_usage

            elements_text = "\n".join(elements_description)

            orig_pil = cv2_to_pil(cv2_screenshot)
            orig_base64 = image_to_base64(orig_pil)

            vis_base64 = None
            if vis_img is not None:
                vis_pil = cv2_to_pil(vis_img)
                vis_base64 = image_to_base64(vis_pil)

            if not orig_base64:
                logging.error("Failed to convert original screenshot to base64.")
                self.last_reasoning = "Error processing original screenshot for LLM."
                return None, llm_call_token_usage

            prompt_parts = [
                "You are an advanced UI Navigation Agent. Your task is to identify the exact element a user wants to click based on a description, using visual analysis and a list of detected elements.",
                "\nI'm providing image(s) and a list of detected UI elements:",
                "\n1. FIRST IMAGE: The original screenshot.",
            ]
            content = [{"inline_data": {"mime_type": "image/png", "data": orig_base64}}]

            if vis_base64:
                prompt_parts.append("\n2. SECOND IMAGE: Visualization with numbered boxes highlighting detected elements (numbers match indices below).")
                content.append({"inline_data": {"mime_type": "image/png", "data": vis_base64}})
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
            content.insert(0, {"text": prompt_text})

            try:
                safety_settings = {}
                response = self.model.generate_content(content, safety_settings=safety_settings)
                llm_call_token_usage = _get_token_usage(response)
                txt = ""
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    txt = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

                if not txt:
                    if response.prompt_feedback.block_reason:
                        block_reason = response.prompt_feedback.block_reason
                        logging.error(f"LLM response blocked: {block_reason}")
                        self.last_reasoning = f"LLM response blocked ({block_reason})."
                    else:
                        logging.error("LLM response was empty.")
                        self.last_reasoning = "LLM returned empty response."
                    return None, llm_call_token_usage

                logging.info(f"[UI Agent Raw Response]\n{txt}")
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
                            selection = None
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

