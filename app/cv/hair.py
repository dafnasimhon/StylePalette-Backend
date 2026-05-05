import cv2
import numpy as np
from typing import Any, Sequence, Tuple


# -----------------------------
# עזר: המרת נקודות לפיקסלים
# -----------------------------
def _landmark_xy(landmark: Any, w: int, h: int):
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return x, y


def _get_face_bounds(landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    return min(xs), max(xs), min(ys), max(ys)


# -----------------------------
# יצירת אזור שיער (פשוט!)
# -----------------------------
def build_hair_region(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    h, w = image_bgr.shape[:2]

    x_min, x_max, y_min, y_max = _get_face_bounds(landmarks, w, h)

    # אזור מעל הפנים (שם נמצא השיער)
    hair_top = max(0, y_min - int(0.3 * (y_max - y_min)))

    mask = np.zeros((h, w), dtype=np.uint8)

    mask[hair_top:y_min, x_min:x_max] = 255

    return mask


# -----------------------------
# חישוב צבע שיער (קצת יותר חכם)
# -----------------------------
def extract_hair_color(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = image_bgr[mask > 0]

    if len(pixels) == 0:
        raise ValueError("No hair pixels found")

    # מסנן קצת רעש (מוריד קצוות)
    pixels = pixels.astype(np.float32)

    mean = np.mean(pixels, axis=0)

    b, g, r = int(mean[0]), int(mean[1]), int(mean[2])

    return (r, g, b)


# -----------------------------
# פונקציה ראשית
# -----------------------------
def get_hair_color(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    mask = build_hair_region(image_bgr, landmarks)

    rgb = extract_hair_color(image_bgr, mask)

    return {
        "rgb": rgb,
        "hex": "#{:02x}{:02x}{:02x}".format(*rgb)
    }