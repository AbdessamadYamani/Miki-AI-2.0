
import logging
import time
import re,sys,os
import webbrowser
from typing import Dict, Tuple
from google.genai.types import GenerateContentConfig

from tools.token_usage_tool import _get_token_usage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import GOOGLE_SEARCH_TOOL,get_client,MODEL_NAME



def search_web_for_info(query: str) -> Tuple[bool, str, Dict[str, int]]:
    logging.info(f"Performing web search for query: '{query}'")
    try: 
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        logging.error("Failed to import google.genai components for web search.")
        return False, "Search failed: Required libraries not found.", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    if GOOGLE_SEARCH_TOOL is None:
        logging.error("Google Search tool not available for web search.")
        return False, "Search failed: Google Search tool not initialized.", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    prompt = f"""
Please use the Google Search tool to find the answer to the following question:

"{query}"

Provide a concise summary based on the search results.
"""
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    try:
        client= get_client() 
        response = client.models.generate_content( # type: ignore
            model=MODEL_NAME, # type: ignore
            contents=[{"text": prompt}], 
            config=GenerateContentConfig( # type: ignore
                tools=[GOOGLE_SEARCH_TOOL], # type: ignore
                response_modalities=["TEXT"], # type: ignore
            )
        )
        token_usage = _get_token_usage(response)
        logging.info(f"Token usage reported by API for search_web_for_info (query: '{query[:30]}...'): {token_usage}")

        raw_response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    raw_response_text = part.text.strip()
                    break

        if not raw_response_text:
            block_reason = "Unknown"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
            logging.error(f"Web search LLM returned empty/blocked response. Reason: {block_reason}")
            return False, f"Search failed: LLM response empty or blocked ({block_reason}).", token_usage

        logging.info("Web search successful.")
        logging.debug(f"Web search result summary:\n{raw_response_text}")
        logging.info(f"Returning search results with token usage: {token_usage}")
        return True, raw_response_text, token_usage

    except google_exceptions.GoogleAPIError as api_err: # type: ignore
        logging.error(f"Google API Error during web search: {api_err}", exc_info=True)
        return False, f"Search failed: API Error ({api_err}).", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    except Exception as e:
        logging.error(f"Unexpected error during web search: {e}", exc_info=True)
        return False, f"Search failed: Unexpected error ({e}).", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}




def navigate_web(url: str) -> Tuple[bool, str]:
    """Opens the URL in the default web browser."""
    logging.info(f"Opening URL: {url}")
    try:
        if not isinstance(url, str) or not url: return False, "Invalid URL."
        if not re.match(r'^[a-zA-Z]+://', url):
             logging.debug(f"Prepending 'https://' to URL: {url}")
             url = 'https://' + url
        webbrowser.open(url)
        time.sleep(2)
        return True, f"Opened web URL: {url}"
    except webbrowser.Error as e:
         logging.error(f"webbrowser error opening {url}: {e}")
         return False, f"Failed via webbrowser: {e}"
    except Exception as e:
        logging.error(f"Unexpected error opening {url}: {e}")
        return False, f"Unexpected error opening URL: {e}"
