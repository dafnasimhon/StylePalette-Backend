import cv2
import numpy as np
from typing import Any, Sequence, Tuple


# אותם אינדקסים מהקוד שלך
_SKIN_FOREHEAD_CHEEK_INDICES = (
    10, 151, 9, 8, 168, 6,
    195, 5, 4, 1,
    123, 50, 205, 187, 200,
    147, 213, 192, 204,
    352, 266, 425, 411, 427,
    376, 433, 416, 436,
)


# -----------------------------
# עזר: המרה נקודות לפיקסלים
# -----------------------------
def _landmark_xy(landmark: Any, w: int, h: int):
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return x, y


def _points_from_indices(landmarks, indices, w, h):
    pts = []
    for i in indices:
        pts.append(_landmark_xy(landmarks[i], w, h))
    return np.array(pts, dtype=np.int32)


# -----------------------------
# יצירת מסכה של עור
# -----------------------------
def build_skin_mask(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    h, w = image_bgr.shape[:2]

    pts = _points_from_indices(landmarks, _SKIN_FOREHEAD_CHEEK_INDICES, w, h)

    mask = np.zeros((h, w), dtype=np.uint8)

    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


# -----------------------------
# חישוב צבע ממוצע
# -----------------------------
def extract_skin_color(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = image_bgr[mask > 0]

    if len(pixels) == 0:
        raise ValueError("No skin pixels found")

    mean = np.mean(pixels, axis=0)

    b, g, r = int(mean[0]), int(mean[1]), int(mean[2])

    return (r, g, b)


# -----------------------------
# פונקציה ראשית
# -----------------------------
def get_skin_color(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    mask = build_skin_mask(image_bgr, landmarks)

    rgb = extract_skin_color(image_bgr, mask)

    return {
        "rgb": rgb,
        "hex": "#{:02x}{:02x}{:02x}".format(*rgb)
    }