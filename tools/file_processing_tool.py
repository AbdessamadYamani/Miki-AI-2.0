import logging
import mimetypes
import pathlib
import httpx
import tempfile
from typing import Dict, Tuple, List
from tools.token_usage_tool import _get_token_usage
from config import get_client

# Supported file types for Gemini API
SUPPORTED_MIME_TYPES = [
    'application/pdf',
    'application/x-javascript', 'text/javascript',
    'application/x-python', 'text/x-python',
    'text/plain',
    'text/html',
    'text/css',
    'text/md',
    'text/csv',
    'text/xml',
    'text/rtf'
]

def get_mime_type_from_path(filepath: str) -> str:
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type

def get_mime_type_from_url(url: str) -> str:
    mime_type, _ = mimetypes.guess_type(url)
    if not mime_type:
        try:
            response = httpx.head(url, follow_redirects=True)
            mime_type = response.headers.get('content-type', '').split(';')[0]
        except Exception:
            mime_type = None
    return mime_type

def process_local_files(filepaths: List[str], prompt: str) -> Tuple[bool, str, Dict[str, int]]:
    """Process local files and return analysis results."""
    logging.info(f"Processing local files: {filepaths}")
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    
    try:
        # Get client instance
        client = get_client()
        if client is None:
            return False, "Failed to initialize Gemini client", token_usage
            
        file_refs = []
        for filepath in filepaths:
            mime_type = get_mime_type_from_path(filepath)
            if mime_type not in SUPPORTED_MIME_TYPES:
                error_msg = f"File {filepath} type {mime_type} is not supported."
                logging.error(error_msg)
                return False, error_msg, token_usage
                
            file_ref = client.files.upload(
                file=filepath,
                config=dict(mime_type=mime_type)
            )
            file_refs.append(file_ref)
            
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[*file_refs, prompt]
        )
        
        token_usage = _get_token_usage(response)
        logging.info(f"Token usage for local file processing: {token_usage}")
        
        if not response.text:
            logging.error("Empty response from Gemini API")
            return False, "File processing failed: Empty response from API", token_usage
            
        logging.info("Local file processing successful")
        logging.debug(f"Processing result:\n{response.text}")
        
        return True, {
            "status": "success",
            "result": response.text,
            "source": "file",
            "filepath": filepaths[0] if filepaths else None
        }, token_usage
        
    except Exception as e:
        error_msg = f"File processing failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return False, error_msg, token_usage

def process_files_from_urls(urls: List[str], prompt: str) -> Tuple[bool, str, Dict[str, int]]:
    """Process files from URLs and return analysis results."""
    logging.info(f"Processing files from URLs: {urls}")
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    
    try:
        # Get client instance
        client = get_client()
        if client is None:
            return False, "Failed to initialize Gemini client", token_usage
            
        file_refs = []
        for url in urls:
            mime_type = get_mime_type_from_url(url)
            if mime_type not in SUPPORTED_MIME_TYPES:
                error_msg = f"URL {url} type {mime_type} is not supported."
                logging.error(error_msg)
                return False, error_msg, token_usage
                
            try:
                file_data = httpx.get(url).content
            except Exception as e:
                error_msg = f"Failed to download file from {url}: {str(e)}"
                logging.error(error_msg)
                return False, error_msg, token_usage
                
            # Save to a temporary file for upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(url).suffix) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
                
            try:
                file_ref = client.files.upload(
                    file=tmp_path,
                    config=dict(mime_type=mime_type)
                )
                file_refs.append(file_ref)
            finally:
                pathlib.Path(tmp_path).unlink(missing_ok=True)
                
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[*file_refs, prompt]
        )
        
        token_usage = _get_token_usage(response)
        logging.info(f"Token usage for URL file processing: {token_usage}")
        
        if not response.text:
            logging.error("Empty response from Gemini API")
            return False, "File processing failed: Empty response from API", token_usage
            
        logging.info("URL file processing successful")
        logging.debug(f"Processing result:\n{response.text}")
        
        return True, {
            "status": "success",
            "result": response.text,
            "source": "url",
            "url": urls[0] if urls else None
        }, token_usage
        
    except Exception as e:
        error_msg = f"File processing failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return False, error_msg, token_usage 