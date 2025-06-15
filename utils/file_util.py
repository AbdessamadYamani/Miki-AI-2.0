from typing import Optional, Tuple
import os
import logging
import numpy as np
import json
from vision.xga import UIElementCollection
from chromaDB_management.cache import sanitize_filename,UICache
import cv2
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DEBUG_DIR



ui_cache = UICache()
def _execute_read_file(file_path: str) -> Tuple[bool, str]:
    """
    Reads the content of a specified file.

    Args:
        file_path: The full path to the target file.

    Returns:
        A tuple (success: bool, content_or_error_message: str).
    """
    logging.info(f"Attempting to read file: {file_path}")
    try:
        expanded_path = os.path.expandvars(file_path)
        if not os.path.exists(expanded_path):
            logging.error(f"File not found: {expanded_path}")
            return False, f"File not found: {expanded_path}"
        if not os.path.isfile(expanded_path):
            logging.error(f"Path is not a file: {expanded_path}")
            return False, f"Path is not a file: {expanded_path}"

        # Add a size check to prevent trying to read extremely large files into memory for the prompt
        # Adjust MAX_FILE_SIZE_BYTES as needed, e.g., 1MB
        MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024
        file_size = os.path.getsize(expanded_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            logging.error(f"File is too large to read ({file_size} bytes > {MAX_FILE_SIZE_BYTES} bytes): {expanded_path}")
            return False, f"File is too large to read ({file_size/(1024*1024):.2f} MB). Maximum allowed is {MAX_FILE_SIZE_BYTES/(1024*1024):.2f} MB."

        with open(expanded_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        logging.info(f"Successfully read file: {expanded_path} ({len(content)} characters)")
        return True, content
    except FileNotFoundError:
        logging.error(f"Error reading file: Path not found: {expanded_path}")
        return False, f"Error reading file: Path not found: {expanded_path}"
    except PermissionError:
        logging.error(f"Error reading file: Permission denied for path: {expanded_path}")
        return False, f"Error reading file: Permission denied for path: {expanded_path}"
    except IsADirectoryError:
         logging.error(f"Error reading file: Specified path is a directory: {expanded_path}")
         return False, f"Error reading file: Specified path is a directory: {expanded_path}"
    except OSError as e:
        logging.error(f"Error reading file: OS error for path '{expanded_path}': {e}")
        return False, f"Error reading file: OS error for path '{expanded_path}': {e}"
    except Exception as e:
        logging.error(f"Unexpected error reading file '{expanded_path}': {e}", exc_info=True)
        return False, f"Unexpected error reading file '{expanded_path}': {e}"

def save_debug_data(app_name: str, timestamp: str, cv2_screenshot: Optional[np.ndarray],
                    vis_img: Optional[np.ndarray], ui_elements: UIElementCollection) -> Optional[str]:
    """Save debug data including screenshot, visualization, and UI elements info"""
    if cv2_screenshot is None:
        logging.warning("Cannot save debug data: screenshot is missing.")
        return None

    try:
        sanitized_app_name = sanitize_filename(app_name)
        debug_session_dir = os.path.join(DEBUG_DIR, f"{sanitized_app_name}_{timestamp}")
        os.makedirs(debug_session_dir, exist_ok=True)

        screenshot_path = os.path.join(debug_session_dir, "original_screenshot.png")
        success = cv2.imwrite(screenshot_path, cv2_screenshot)
        if not success: logging.warning(f"Failed to write screenshot to {screenshot_path}")

        if vis_img is not None:
            vis_path = os.path.join(debug_session_dir, "ui_elements_visualization.png")
            success = cv2.imwrite(vis_path, vis_img)
            if not success: logging.warning(f"Failed to write visualization to {vis_path}")
        else:
            logging.info("No visualization image provided to save_debug_data.")

        elements_path = os.path.join(debug_session_dir, "ui_elements.json")
        serialized_elements = ui_cache._serialize_ui_elements(ui_elements)
        with open(elements_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_elements, f, indent=2)

        logging.info(f"Debug data saved to: {debug_session_dir}")
        return debug_session_dir
    except OSError as e:
         logging.error(f"OS error saving debug data to {debug_session_dir}: {e}")
         return None
    except Exception as e:
        logging.error(f"Failed to save debug data: {e}")
        return None



