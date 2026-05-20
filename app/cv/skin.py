import cv2
import numpy as np
from typing import Any, Sequence, Tuple


# Forehead/cheek skin convex hull landmark indices
_SKIN_FOREHEAD_CHEEK_INDICES = (
    10, 151, 9, 8, 168, 6,
    195, 5, 4, 1,
    123, 50, 205, 187, 200,
    147, 213, 192, 204,
    352, 266, 425, 411, 427,
    376, 433, 416, 436,
)


# -----------------------------
# Utilities: normalized landmark → pixel coordinates
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
# Build skin region mask from landmark hull
# -----------------------------
def build_skin_mask(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    h, w = image_bgr.shape[:2]

    pts = _points_from_indices(landmarks, _SKIN_FOREHEAD_CHEEK_INDICES, w, h)

    mask = np.zeros((h, w), dtype=np.uint8)

    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


# -----------------------------
# Average skin color (mid-luminance band; shadow/highlight resistant)
# -----------------------------
def extract_skin_color(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = image_bgr[mask > 0]

    if len(pixels) == 0:
        raise ValueError("No skin pixels found")

    # Simple mean pulls toward shadowed jaw/cheek/neck; brightest-only pushes too light.
    # Use a mid-luminance band to avoid both deep shadows and shiny highlights.
    if len(pixels) < 400:
        mean = np.mean(pixels, axis=0)
        b, g, r = int(mean[0]), int(mean[1]), int(mean[2])
        return (r, g, b)

    b_ch = pixels[:, 0].astype(np.float32)
    g_ch = pixels[:, 1].astype(np.float32)
    r_ch = pixels[:, 2].astype(np.float32)
    # Perceptual-ish luminance weights (sRGB linearization omitted; good enough for sorting)
    lum = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch
    low = float(np.percentile(lum, 35.0))
    high = float(np.percentile(lum, 75.0))
    mid = (lum >= low) & (lum <= high)
    if np.count_nonzero(mid) < 50:
        mean = np.mean(pixels, axis=0)
        b, g, r = int(mean[0]), int(mean[1]), int(mean[2])
        return (r, g, b)

    rb = float(np.mean(r_ch[mid]))
    gb = float(np.mean(g_ch[mid]))
    bb = float(np.mean(b_ch[mid]))
    return (int(rb), int(gb), int(bb))


# -----------------------------
# Public entry
# -----------------------------
def get_skin_color(image_bgr: np.ndarray, landmarks: Sequence[Any]):
    mask = build_skin_mask(image_bgr, landmarks)

    rgb = extract_skin_color(image_bgr, mask)

    return {
        "rgb": rgb,
        "hex": "#{:02x}{:02x}{:02x}".format(*rgb)
    }