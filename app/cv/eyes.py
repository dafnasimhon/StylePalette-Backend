import cv2
import numpy as np
from typing import Any, Sequence, Tuple


# אינדקסים של הקשתיות (מהקוד שלך)
_RIGHT_IRIS = (469, 470, 471, 472, 468)
_LEFT_IRIS = (474, 475, 476, 477, 473)


# -----------------------------
# עזר: המרת נקודות לפיקסלים
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
# יצירת מסכה של העיניים
# -----------------------------
def build_iris_mask(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for indices in (_RIGHT_IRIS, _LEFT_IRIS):
        pts = _points_from_indices(landmarks, indices, w, h)

        if len(pts) < 3:
            continue

        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask


# -----------------------------
# חישוב צבע ממוצע
# -----------------------------
def extract_eye_color(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = image_bgr[mask > 0]

    if len(pixels) == 0:
        raise ValueError("No eye pixels found")

    mean = np.mean(pixels, axis=0)

    b, g, r = int(mean[0]), int(mean[1]), int(mean[2])

    return (r, g, b)


# -----------------------------
# פונקציה ראשית
# -----------------------------
def get_eye_color(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    mask = build_iris_mask(image_bgr, landmarks)

    rgb = extract_eye_color(image_bgr, mask)

    return {
        "rgb": rgb,
        "hex": "#{:02x}{:02x}{:02x}".format(*rgb)
    }