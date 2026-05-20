import cv2
import numpy as np
from typing import Any, Dict

from .detection import detect_landmarks
from .skin import get_skin_color
from .eyes import get_eye_color
from .hair import get_hair_color


def load_image_bgr_from_bytes(data: bytes) -> np.ndarray:
    """Decode an image from raw bytes into BGR (OpenCV convention)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image bytes (unsupported or corrupt format).")
    return image_bgr


def analyze_facial_colors(
    image_bgr: np.ndarray,
    *,
    allow_model_download: bool = True,
) -> Dict[str, Any]:
    """Run face landmarks + skin / iris / hair color sampling.

    ``allow_model_download``: when true and the MediaPipe ``.task`` file is missing,
    it is downloaded once (see ``app/cv/detection.py`` and ``MEDIAPIPE_*`` env vars).
    """
    landmarks = detect_landmarks(image_bgr, allow_model_download=allow_model_download)

    if landmarks is None:
        return {
            "success": False,
            "error": "No face detected",
            "skin_tone": None,
            "eye_color": None,
            "hair_color": None,
        }

    try:
        skin = get_skin_color(image_bgr, landmarks)
        eyes = get_eye_color(image_bgr, landmarks)
        hair = get_hair_color(image_bgr, landmarks)

        return {
            "success": True,
            "error": None,
            "skin_tone": skin,
            "eye_color": eyes,
            "hair_color": hair,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "skin_tone": None,
            "eye_color": None,
            "hair_color": None,
        }
