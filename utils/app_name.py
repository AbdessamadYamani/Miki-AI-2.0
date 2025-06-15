import re



def get_base_app_name(full_app_name: str) -> str:
    """Extracts a base name for an application (e.g., 'chrome' from 'chrome.exe - Google Search')."""
    if not full_app_name or full_app_name == "unknown_app":
        return "unknown"


    name = full_app_name.lower()
    name = re.sub(r'\.(exe|com|bat)$', '', name)
    name = name.split(' - ')[0]
    name = name.split(':')[0]
    name = name.strip()


    common_bases = ["chrome", "firefox", "edge", "safari", "explorer", "code", "notepad", "word", "excel", "powerpoint", "outlook", "terminal", "cmd", "powershell", "slack", "discord", "zoom"]
    for base in common_bases:
        if base in name:
            return base


    return name if name else "unknown"
