import cv2
import numpy as np
from typing import Any, Sequence, Tuple


# -----------------------------
# Utilities: normalized landmark → pixel coordinates
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
# Build rectangular hair ROI above the face bbox
# -----------------------------
def build_hair_region(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    h, w = image_bgr.shape[:2]

    x_min, x_max, y_min, y_max = _get_face_bounds(landmarks, w, h)

    # Strip above face top edge (typical hair location)
    hair_top = max(0, y_min - int(0.3 * (y_max - y_min)))

    mask = np.zeros((h, w), dtype=np.uint8)

    mask[hair_top:y_min, x_min:x_max] = 255

    return mask


# -----------------------------
# Average hair color from ROI (robust to lighting)
# -----------------------------
def extract_hair_color(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = image_bgr[mask > 0]

    if len(pixels) == 0:
        raise ValueError("No hair pixels found")

    # HSV to separate strand color from outliers. Do NOT bias to the darkest
    # pixels: that grabs roots/shadows and wrongly turns blonde hair "brown".
    hsv = cv2.cvtColor(pixels.reshape((-1, 1, 3)), cv2.COLOR_BGR2HSV).reshape((-1, 3))
    s = hsv[:, 1].astype(np.float32)
    v = hsv[:, 2].astype(np.float32)

    median_v = float(np.median(v))
    if median_v >= 118:
        # Bright ROI overall → blonde / light brown / highlighted hair: use mid–bright band.
        v_lo = float(np.percentile(v, 28.0))
        v_hi = float(np.percentile(v, 96.0))
        hair_like = (v >= v_lo) & (v <= v_hi) & (s >= 8.0)
    else:
        # Darker hair: drop extreme highlights (sky/wall blowout) and deep shadow.
        v_lo = float(np.percentile(v, 12.0))
        v_hi = float(np.percentile(v, 88.0))
        hair_like = (v >= v_lo) & (v <= v_hi) & (s >= 12.0)

    if np.count_nonzero(hair_like) < 80:
        hair_like = (v >= float(np.percentile(v, 18.0))) & (v <= float(np.percentile(v, 92.0)))

    selected = pixels[hair_like] if np.count_nonzero(hair_like) > 0 else pixels
    selected = selected.astype(np.float32)
    mean = np.mean(selected, axis=0)

    b, g, r = int(mean[0]), int(mean[1]), int(mean[2])

    return (r, g, b)


# -----------------------------
# Public entry
# -----------------------------
def get_hair_color(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    mask = build_hair_region(image_bgr, landmarks)

    rgb = extract_hair_color(image_bgr, mask)

    return {
        "rgb": rgb,
        "hex": "#{:02x}{:02x}{:02x}".format(*rgb)
    }