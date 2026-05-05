from pathlib import Path

import cv2
import numpy as np
from typing import Any, List, Optional

from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision.core import image as image_lib

# Repo root: StylePalette-Backend/ (parent of app/)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
FACE_LANDMARKER_MODEL_PATH = _BACKEND_ROOT / "models" / "face_landmarker.task"

_landmarker = None


def get_face_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = FaceLandmarker.create_from_model_path(
            str(FACE_LANDMARKER_MODEL_PATH)
        )
    return _landmarker


def detect_landmarks(image_bgr: np.ndarray) -> Optional[List[Any]]:
    landmarker = get_face_landmarker()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = image_lib.Image(
        image_format=image_lib.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]