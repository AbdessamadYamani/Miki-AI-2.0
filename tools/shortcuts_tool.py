import os
import json
import logging
import sys
import time
import hashlib
from typing import Dict
from tools.token_usage_tool import _get_token_usage
from utils.sanitize_util import sanitize_filename
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BLUE, GREEN, RESET, YELLOW # Import colors
from config import SHORTCUT_CACHE_FILE, SHORTCUT_DEBUG_DIR, GOOGLE_SEARCH_TOOL, MODEL_NAME, get_client, GenerateContentConfig, chroma_client, app_shortcuts_collection

def load_shortcuts_cache():
    """Loads the shortcut cache from the JSON file."""
    global shortcuts_cache
    if os.path.exists(SHORTCUT_CACHE_FILE):
        try:
            with open(SHORTCUT_CACHE_FILE, 'r', encoding='utf-8') as f:
                shortcuts_cache = json.load(f)
            logging.info(f"Loaded shortcuts for {len(shortcuts_cache)} apps from cache.")
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading shortcuts cache: {e}. Starting with empty cache.")
            shortcuts_cache = {}
    else:
        logging.info("Shortcuts cache file not found. Starting with empty cache.")
        shortcuts_cache = {}

def save_shortcuts_cache():
    """Saves the current shortcut cache to the JSON file."""
    global shortcuts_cache
    try:
        with open(SHORTCUT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(shortcuts_cache, f, indent=2)
        logging.debug(f"Saved shortcuts cache to {SHORTCUT_CACHE_FILE}")
    except IOError as e:
        logging.error(f"Error saving shortcuts cache: {e}")


def get_application_shortcuts(app_name_base: str, force_refresh: bool = False) -> Tuple[str, Dict[str, int]]:
    try:
        from google.api_core import exceptions as google_exceptions 
    except ImportError:
        logging.error("Failed to import google.generativeai or google.generativeai.types.")
        logging.error("Please install the library: pip install google-generativeai")
        return "", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    global shortcuts_cache
    logging.info(f"Fetching shortcuts for application: {app_name_base}")

    os.makedirs(SHORTCUT_DEBUG_DIR, exist_ok=True)

    sanitized_app_name = sanitize_filename(app_name_base)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_llm_response_debug_path = os.path.join(SHORTCUT_DEBUG_DIR, f"{sanitized_app_name}_{timestamp}_llm_raw_response.txt")
    final_shortcuts_text_debug_path = os.path.join(SHORTCUT_DEBUG_DIR, f"{sanitized_app_name}_{timestamp}_shortcuts_text.txt")

    # 1. Try fetching from ChromaDB first if not force_refresh
    if not force_refresh and chroma_client and app_shortcuts_collection:
        try:
            shortcut_id = hashlib.md5(app_name_base.encode('utf-8')).hexdigest()
            results = app_shortcuts_collection.get(ids=[shortcut_id], include=['metadatas'])
            if results and results['ids'] and results['metadatas']:
                cached_shortcuts_text = results['metadatas'][0].get('shortcuts_text')
                if cached_shortcuts_text and isinstance(cached_shortcuts_text, str) and cached_shortcuts_text.strip():
                    logging.info(f"Using cached shortcuts for '{app_name_base}' from ChromaDB.")
                    # Print notification for ChromaDB cache hit
                    print(f"{YELLOW}[ChromaDB Cache] Using shortcuts for: {app_name_base}{RESET}")
                    # Also update the file-based cache if ChromaDB had it and file cache didn't, or if different
                    if app_name_base not in shortcuts_cache or shortcuts_cache[app_name_base] != cached_shortcuts_text:
                        shortcuts_cache[app_name_base] = cached_shortcuts_text
                        save_shortcuts_cache()
                    return cached_shortcuts_text, {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
        except Exception as e:
            logging.error(f"Error fetching shortcuts from ChromaDB for '{app_name_base}': {e}")

    # 2. Fallback to file-based cache if not found in ChromaDB or ChromaDB error, and not force_refresh
    if not force_refresh and app_name_base in shortcuts_cache and isinstance(shortcuts_cache[app_name_base], str) and shortcuts_cache[app_name_base].strip():
        logging.info(f"Using cached textual shortcuts for {app_name_base} from file-based cache.")
        return shortcuts_cache[app_name_base], {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    # 3. If not in any cache or force_refresh is True, fetch from LLM
    if GOOGLE_SEARCH_TOOL is None:
        logging.error("Google Search tool not available for fetching shortcuts.")
        return "", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    prompt = f"""
    Provide a list of the most common and useful keyboard shortcuts for the application '{app_name_base}'.
    Focus on actions like saving, opening, closing, navigating, copying, pasting, finding, switching tabs/windows, confirming dialogs (Enter), canceling (Esc).

    Format the list clearly as plain text, for example:
    - Ctrl+S: Save the current file/document
    - Ctrl+O: Open a file
    - Alt+F4: Close the current window

    Include All shortcuts for '{app_name_base}' in windows. If unsure, provide common cross-application shortcuts. Do not invent shortcuts.
    Keep the response concise and directly list the shortcuts. Do not include any other conversational text, introductions, or summaries beyond the shortcut list itself.
    """
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    if force_refresh:
        logging.info(f"Force refreshing shortcuts for {app_name_base} from LLM.")

    try:
        client= get_client()  # type: ignore
        response = client.models.generate_content( # type: ignore
            model=MODEL_NAME, # type: ignore
            contents=[{"text": prompt}], 
            config=GenerateContentConfig( # type: ignore
                tools=[GOOGLE_SEARCH_TOOL], # type: ignore
                response_modalities=["TEXT"], # type: ignore
            )
        )
        token_usage = _get_token_usage(response)

        raw_response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    raw_response_text = part.text.strip()
                    break 
        else:
            logging.error(f"Unexpected LLM response structure or empty candidates for {app_name_base}.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                logging.error(f"LLM response blocked: {block_reason}")
            return "", token_usage

        try:
            with open(raw_llm_response_debug_path, 'w', encoding='utf-8') as f_raw:
                f_raw.write(raw_response_text)
            logging.info(f"Saved raw LLM shortcut response to {raw_llm_response_debug_path}")
        except Exception as save_err:
            logging.error(f"Failed to save raw shortcut response: {save_err}")

        if not raw_response_text:
            logging.error(f"LLM returned empty text content when fetching shortcuts for {app_name_base}.")
            return "", token_usage

        logging.debug(f"LLM raw response for shortcuts:\n{raw_response_text}")

        if raw_response_text.strip(): 
            logging.info(f"Successfully fetched textual shortcuts for {app_name_base}.")
            try:
                with open(final_shortcuts_text_debug_path, 'w', encoding='utf-8') as f_text:
                    f_text.write(raw_response_text)
                logging.info(f"Saved final textual shortcuts to {final_shortcuts_text_debug_path}")
            except Exception as save_err:
                logging.error(f"Failed to save final textual shortcuts: {save_err}")

            shortcuts_cache[app_name_base] = raw_response_text 
            save_shortcuts_cache()

            # Save to ChromaDB as well
            if chroma_client and app_shortcuts_collection:
                try:
                    shortcut_id = hashlib.md5(app_name_base.encode('utf-8')).hexdigest()
                    app_shortcuts_collection.upsert(
                        ids=[shortcut_id],
                        documents=[app_name_base], # Document is the app name for searching
                        metadatas=[{
                            "app_name": app_name_base,
                            "shortcuts_text": raw_response_text,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }]
                    )
                    logging.info(f"Saved/Updated shortcuts for '{app_name_base}' to ChromaDB.")
                except Exception as e:
                    logging.error(f"Error saving shortcuts to ChromaDB for '{app_name_base}': {e}")
            return raw_response_text, token_usage
        else:
            logging.error(f"LLM returned an empty or whitespace-only response for textual shortcuts for {app_name_base}.")
            shortcuts_cache[app_name_base] = "" 
            return "", token_usage

    except google_exceptions.GoogleAPIError as api_err: # type: ignore
        logging.error(f"Google API Error fetching shortcuts: {api_err}", exc_info=True)
        return "", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0} # type: ignore
    except Exception as e:
        logging.error(f"Unexpected error fetching shortcuts with LLM Client: {e}", exc_info=True)
        return "", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0} # type: ignore
