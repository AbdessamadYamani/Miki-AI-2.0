import os
import re
import hashlib

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to avoid issues with invalid characters and length."""
    if not isinstance(filename, str) or not filename:
        return "invalid_or_empty_filename"


    sanitized = re.sub(r'[<>:"/\|?*\x00-\x1F]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip(' ._')

    max_len = 150
    if len(sanitized.encode('utf-8')) > max_len:
        name_part, ext_part = os.path.splitext(sanitized)
        hash_suffix = "_" + hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
        allowed_len = max_len - len(hash_suffix.encode('utf-8')) - len(ext_part.encode('utf-8'))

        if allowed_len > 0:
             truncated_name = ""
             current_len = 0
             for char in name_part:
                 char_len = len(char.encode('utf-8'))
                 if current_len + char_len <= allowed_len:
                     truncated_name += char
                     current_len += char_len
                 else:
                     break
             sanitized = truncated_name.strip(' ._') + hash_suffix + ext_part
        else:
             sanitized = hash_suffix.strip('_') + ext_part

    return sanitized if sanitized else "sanitized_empty"

