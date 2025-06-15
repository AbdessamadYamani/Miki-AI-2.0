import os
import json
import logging
import sys
from vision.xga import detect_ui_elements_from_image, visualize_ui_elements, UIElementCollection, UIElement # type: ignore # type: ignore
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any, Generator
import re
import sys
import psutil
import hashlib
import cv2
import time
from utils.sanitize_util import sanitize_filename
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CACHE_DIR

def get_active_window_name() -> str:
    """Get the name of the currently active application window."""
    active_app_name = "unknown_app"
    try:
        if sys.platform == "win32":
            import win32gui
            import win32process
            window_handle = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(window_handle)

            _, pid = win32process.GetWindowThreadProcessId(window_handle)
            process = psutil.Process(pid)
            app_name = process.name()

            full_name = f"{app_name} - {window_title}" if window_title else app_name
            active_app_name = sanitize_filename(full_name) or "unknown_app_sanitized"
        elif sys.platform == "darwin":
             from AppKit import NSWorkspace
             active_app_info = NSWorkspace.sharedWorkspace().activeApplication()
             app_name = active_app_info.get('NSApplicationName', 'unknown_app')

             active_app_name = sanitize_filename(app_name) or "unknown_app_sanitized"
        else:
            logging.warning("Linux active window detection is basic; using psutil fallback.")


            try:
                active_pid = os.getpid()
                process = psutil.Process(active_pid)
                active_app_name = sanitize_filename(process.name()) or "unknown_app_sanitized"
            except Exception as e:
                logging.error(f"Could not determine active window name using psutil: {e}")
                active_app_name = "unknown_app_linux"

    except ImportError as e:
        logging.warning(f"Required library for active window detection not found ({e}). Using basic fallback.")
        try:
            process = psutil.Process(os.getpid())
            active_app_name = sanitize_filename(process.name()) or "unknown_app_fallback"
        except Exception as fallback_e:
            logging.error(f"Fallback active window detection failed: {fallback_e}")
            active_app_name = "unknown_app_error"
    except Exception as e:
        logging.error(f"Error getting active window name: {e}")
        active_app_name = "unknown_app_error"


    return active_app_name



class UICache:
    def __init__(self):
        self.cache = {}
        self.last_screenshot_hash = None
        self.last_app_name = None
        self.load_cache()

    def load_cache(self):
        """Load cached UI elements from disk"""
        cache_file = os.path.join(CACHE_DIR, "ui_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    serialized_cache = json.load(f)


                for app_name, data in serialized_cache.items():

                    if isinstance(data, dict) and 'screenshot_hash' in data and 'ui_elements' in data:
                        self.cache[app_name] = {
                            'screenshot_hash': data['screenshot_hash'],
                            'ui_elements': self._deserialize_ui_elements(data['ui_elements'])
                        }
                    else:
                        logging.warning(f"Skipping invalid cache entry for app '{app_name}' during load.")

                logging.info(f"Loaded cache with {len(self.cache)} applications from {cache_file}")
            except json.JSONDecodeError as e:
                 logging.error(f"Error decoding cache file {cache_file}: {e}. Cache will be rebuilt.")
                 self.cache = {}
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
                self.cache = {}

    def save_cache(self):
        """Save cached UI elements to disk"""
        cache_file = os.path.join(CACHE_DIR, "ui_cache.json")


        serialized_cache = {}
        for app_name, data in self.cache.items():

             if isinstance(data, dict) and 'screenshot_hash' in data and 'ui_elements' in data:
                 serialized_cache[app_name] = {
                     'screenshot_hash': data['screenshot_hash'],
                     'ui_elements': self._serialize_ui_elements(data['ui_elements'])
                 }
             else:
                 logging.warning(f"Skipping invalid cache entry for app '{app_name}' during save.")


        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_cache, f, indent=2)
            logging.info(f"Saved cache to {cache_file}")
        except TypeError as e:
             logging.error(f"Error serializing cache data: {e}. Cache not saved.")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def _serialize_ui_elements(self, ui_elements):
        """Convert UIElementCollection to serializable format"""
        serialized = []
        if not isinstance(ui_elements, UIElementCollection):
             logging.warning(f"Attempted to serialize non-UIElementCollection: {type(ui_elements)}")
             return []

        for elem in ui_elements:
            try:

                elem_dict = {
                    'element_type': getattr(elem, 'element_type', 'unknown'),
                    'label': getattr(elem, 'label', ''),
                    'confidence': getattr(elem, 'confidence', 0.0),
                    'width': getattr(elem, 'width', 0),
                    'height': getattr(elem, 'height', 0),
                    'center': getattr(elem, 'center', [0, 0])
                }


                bbox_attr = getattr(elem, 'bbox', getattr(elem, 'bounding_box', None))
                if bbox_attr is not None:
                    elem_dict['bbox'] = bbox_attr
                else:
                    elem_dict['bbox'] = [0,0,0,0]


                if hasattr(elem, 'data') and isinstance(elem.data, dict):
                    serializable_data = {}
                    for k, v in elem.data.items():
                        try:
                            json.dumps({k: v})
                            serializable_data[k] = v
                        except (TypeError, OverflowError):
                            logging.debug(f"Skipping non-serializable key '{k}' in UIElement data during cache save.")
                            pass
                    if serializable_data:
                        elem_dict['data'] = serializable_data

                serialized.append(elem_dict)
            except Exception as e:
                 logging.error(f"Error serializing individual UI element: {e}. Element data: {vars(elem) if hasattr(elem, '__dict__') else elem}")

        return serialized

    def _deserialize_ui_elements(self, serialized_elements):
        """Convert serialized data back to UIElementCollection"""
        ui_elements = UIElementCollection()

        if not isinstance(serialized_elements, list):
            logging.error(f"Cannot deserialize non-list data: {type(serialized_elements)}")
            return ui_elements

        for elem_data in serialized_elements:
            if not isinstance(elem_data, dict):
                 logging.warning(f"Skipping non-dictionary item during deserialization: {elem_data}")
                 continue
            try:

                bbox = elem_data.get('bbox', [0, 0, 0, 0])
                center = elem_data.get('center', [0, 0])
                width = elem_data.get('width', 0)
                height = elem_data.get('height', 0)
                element_type = elem_data.get('element_type', 'unknown')
                label = elem_data.get('label', '')
                confidence = elem_data.get('confidence', 0.0)
                extra_data = elem_data.get('data', {})


                center = [float(c) for c in center]
                bbox = [float(b) for b in bbox]
                width = float(width)
                height = float(height)



                data_for_constructor = {
                    'type': element_type,
                    'text': label,
                    'bbox': bbox,
                    'confidence': confidence,
                    'width': width,
                    'height': height,
                    'center': center,
                    **extra_data
                }


                ui_element = UIElement(data=data_for_constructor)


                ui_elements.append(ui_element)

            except KeyError as ke:
                 logging.error(f"KeyError deserializing UI element: Missing key {ke}")
                 logging.error(f"Problematic Element data: {elem_data}")
            except (TypeError, ValueError) as ve:
                 logging.error(f"TypeError/ValueError deserializing UI element: {ve}")
                 logging.error(f"Problematic Element data: {elem_data}")
            except Exception as e:
                logging.error(f"Unexpected error deserializing UI element: {e}")
                logging.error(f"Problematic Element data: {elem_data}")

        return ui_elements


    def get_ui_elements(self, screenshot: np.ndarray, app_name: Optional[str] = None) -> UIElementCollection:
        """Get UI elements from cache if screenshot is the same, otherwise detect new ones"""

        img_hash = self._hash_image(screenshot)


        current_app_name = app_name or get_active_window_name()

        self.last_screenshot_hash = img_hash
        self.last_app_name = current_app_name


        cached_data = self.cache.get(current_app_name)
        if cached_data and cached_data.get('screenshot_hash') == img_hash:
            logging.info(f"Using cached UI elements for {current_app_name}")

            elements = cached_data.get('ui_elements')
            return elements if isinstance(elements, UIElementCollection) else UIElementCollection()



        logging.info(f"Detecting new UI elements for {current_app_name}")
        try:
            ui_elements = detect_ui_elements_from_image(screenshot)
            if not isinstance(ui_elements, UIElementCollection):
                 logging.error(f"detect_ui_elements_from_image did not return a UIElementCollection (got {type(ui_elements)}).")
                 ui_elements = UIElementCollection()
        except Exception as e:
            logging.error(f"Error detecting UI elements: {e}")
            return UIElementCollection()


        self.cache[current_app_name] = {
            'screenshot_hash': img_hash,
            'ui_elements': ui_elements
        }


        self.save_cache()

        return ui_elements

    def _hash_image(self, image: np.ndarray) -> str:
        """Generate a hash for the image to check if it has changed"""
        try:

            small_img = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
            return hashlib.md5(small_img.tobytes()).hexdigest()
        except cv2.error as e:
             logging.error(f"OpenCV error during image hashing: {e}. Returning random hash.")
             return hashlib.md5(str(time.time()).encode()).hexdigest()
        except Exception as e:
             logging.error(f"Unexpected error during image hashing: {e}. Returning random hash.")
             return hashlib.md5(str(time.time()).encode()).hexdigest()

