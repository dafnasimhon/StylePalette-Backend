import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from typing import Any, List, Optional

from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision.core import image as image_lib

# Repo root: StylePalette-Backend/ (parent of app/)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_REL = Path("models") / "face_landmarker.task"
_DEFAULT_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_landmarker = None


def _resolve_model_path() -> Path:
    from app.config import get_settings

    cfg = get_settings()
    p = (cfg.mediapipe_face_landmarker_model or "").strip()
    if p:
        return Path(p).expanduser().resolve()
    env = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_BACKEND_ROOT / _DEFAULT_MODEL_REL).resolve()


def _ensure_model_file(path: Path, *, allow_download: bool) -> None:
    if path.is_file():
        return
    if not allow_download:
        raise FileNotFoundError(
            f"Face landmarker model missing at {path}. "
            "Place face_landmarker.task there, set MEDIAPIPE_FACE_LANDMARKER_MODEL in .env, "
            "or set MEDIAPIPE_ALLOW_MODEL_DOWNLOAD=true."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    from app.config import get_settings

    cfg = get_settings()
    url = (cfg.mediapipe_face_landmarker_url or "").strip()
    if not url:
        url = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_URL", _DEFAULT_DOWNLOAD_URL).strip()
    tmp = path.with_suffix(path.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — URL from env or Google static host
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def get_face_landmarker(allow_model_download: bool = True):
    global _landmarker
    if _landmarker is None:
        model_path = _resolve_model_path()
        _ensure_model_file(model_path, allow_download=allow_model_download)
        _landmarker = FaceLandmarker.create_from_model_path(str(model_path))
    return _landmarker


def detect_landmarks(image_bgr: np.ndarray, *, allow_model_download: bool = True) -> Optional[List[Any]]:
    landmarker = get_face_landmarker(allow_model_download)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = image_lib.Image(
        image_format=image_lib.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]