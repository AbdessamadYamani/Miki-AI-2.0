import logging
import io
import base64
from typing import Optional
import numpy as np
import cv2
from PIL import Image

def pil_to_cv2(pil_image: Image.Image) -> Optional[np.ndarray]:
    """Convert PIL Image to OpenCV format (BGR)"""
    if pil_image is None:
        return None
    try:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except cv2.error as e:
         logging.error(f"OpenCV error converting PIL to CV2: {e}")
         return None
    except Exception as e:
         logging.error(f"Unexpected error converting PIL to CV2: {e}")
         return None

def cv2_to_pil(cv2_image: np.ndarray) -> Optional[Image.Image]:
    """Convert OpenCV image (BGR) to PIL format (RGB)"""
    if cv2_image is None:
        return None
    try:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
         logging.error(f"OpenCV error converting CV2 to PIL: {e}")
         return None
    except Exception as e:
         logging.error(f"Unexpected error converting CV2 to PIL: {e}")
         return None

def image_to_base64(image: Image.Image) -> Optional[str]:
    """Convert PIL image to base64 string for Gemini API"""
    if image is None:
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error converting image to base64: {e}")
        return None