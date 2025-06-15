import mimetypes
import pathlib
import httpx
import tempfile
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

def get_mime_type_from_path(filepath):
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type

def get_mime_type_from_url(url):
    mime_type, _ = mimetypes.guess_type(url)
    if not mime_type:
        try:
            response = httpx.head(url, follow_redirects=True)
            mime_type = response.headers.get('content-type', '').split(';')[0]
        except Exception:
            mime_type = None
    return mime_type

def process_local_files(filepaths, prompt):
    file_refs = []
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    print("DONE FILES")
    try:
        for filepath in filepaths:
            mime_type = get_mime_type_from_path(filepath)
            if mime_type not in SUPPORTED_MIME_TYPES:
                error_msg = f"File {filepath} type {mime_type} is not supported."
                return False, error_msg, token_usage
                
            file_ref = get_client().files.upload(
                file=filepath,
                config=dict(mime_type=mime_type)
            )
            file_refs.append(file_ref)
            
        response = get_client().models.generate_content(
            model="gemini-2.0-flash",
            contents=[*file_refs, prompt]
        )
        
        if not response.text:
            return False, "Empty response from API", token_usage
            
        result_dict = {
            "status": "success",
            "result": response.text,
            "source": "local",
            "files": filepaths
        }
        return True, result_dict, token_usage
        
    except Exception as e:
        error_msg = f"File processing failed: {str(e)}"
        return False, error_msg, token_usage

def process_files_from_urls(urls, prompt):
    """Process files from URLs and return analysis results."""
    file_refs = []
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    
    try:
        for url in urls:
            mime_type = get_mime_type_from_url(url)
            if mime_type not in SUPPORTED_MIME_TYPES:
                error_msg = f"URL {url} type {mime_type} is not supported."
                return False, error_msg, token_usage
                
            file_data = httpx.get(url).content
            # Save to a temporary file for upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(url).suffix) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            file_ref = get_client().files.upload(
                file=tmp_path,
                config=dict(mime_type=mime_type)
            )
            file_refs.append(file_ref)
            pathlib.Path(tmp_path).unlink(missing_ok=True)
            
        response = get_client().models.generate_content(
            model="gemini-2.0-flash",
            contents=[*file_refs, prompt]
        )
        
        if not response.text:
            return False, "Empty response from API", token_usage
            
        result_dict = {
            "status": "success",
            "result": response.text,
            "source": "url",
            "url": urls[0] if urls else None
        }
        return True, result_dict, token_usage
        
    except Exception as e:
        error_msg = f"File processing failed: {str(e)}"
        return False, error_msg, token_usage

# ===========================
# Example usage
# ===========================

if __name__ == "__main__":

    urls = [
        'https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf'
    ]
    prompt = "Summarize these documents."
    # Uncomment to use
    success, result, token_usage = process_files_from_urls(urls, prompt)
    if success:
        print(f"Analysis result: {result['result']}")
        print(f"Token usage: {token_usage}")
    else:
        print(f"Error: {result}")
