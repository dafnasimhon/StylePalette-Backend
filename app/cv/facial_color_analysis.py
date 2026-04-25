#!/usr/bin/env python3
"""
StylePalette — facial color analysis from a selfie.

Uses MediaPipe Face Landmarker (Tasks API) and OpenCV to estimate skin, eye,
and hair colors as structured JSON (hex + RGB).

MediaPipe 0.10+ ships the Tasks API only; the legacy ``solutions`` stack is
not used. A ``face_landmarker.task`` bundle is resolved from an environment
variable, a local ``models/`` path, or downloaded once to a user cache.

**Hair color** uses a multi-zone (Crown / Sides / Lower) HSV pigment pipeline
with weighted median; side-near-ears zones use the highest weight
(``_HAIR_ZONE_WEIGHT_SIDE``). Set ``DEBUG_HAIR_ANALYSIS=1`` to log per-zone
sampling. Used by the FastAPI app in this package (``app.services.analysis``)
as the source of skin / eye / hair hex and RGB. Keep this file as the single CV
source for those swatches; ``SeasonAnalyzer`` maps traits to seasons separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional MediaPipe (clear error if missing)
# ---------------------------------------------------------------------------

try:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarksConnections
    from mediapipe.tasks.python.vision.core import image as image_lib
except ImportError as exc:  # pragma: no cover - environment dependent
    FaceLandmarker = None  # type: ignore[misc, assignment]
    FaceLandmarksConnections = None  # type: ignore[misc, assignment]
    image_lib = None  # type: ignore[misc, assignment]
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None

# Official MediaPipe face landmarker (float16) — includes iris landmarks
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_MODEL = "MEDIAPIPE_FACE_LANDMARKER_MODEL"
_DEFAULT_LOCAL_MODEL = _SCRIPT_DIR / "models" / "face_landmarker.task"
_HOME_CACHE_MODEL = Path.home() / ".cache" / "stylepalette" / "face_landmarker.task"
_PACKAGE_CACHE_MODEL = _SCRIPT_DIR / ".cache" / "face_landmarker.task"
DEBUG_HAIR_ANALYSIS = os.environ.get("DEBUG_HAIR_ANALYSIS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Multi-zone hair aggregation (Crown / Sides near ears / Lower). Side gets highest weight.
_HAIR_ZONE_WEIGHT_CROWN = 0.4
_HAIR_ZONE_WEIGHT_SIDE = 3.0
_HAIR_ZONE_WEIGHT_LOWER = 2.2

# ---------------------------------------------------------------------------
# Landmark index sets (MediaPipe face mesh topology, iris from landmarker)
# ---------------------------------------------------------------------------

_SKIN_FOREHEAD_CHEEK_INDICES: Tuple[int, ...] = (
    10,
    151,
    9,
    8,
    168,
    6,
    195,
    5,
    4,
    1,
    123,
    50,
    205,
    187,
    200,
    147,
    213,
    192,
    204,
    352,
    266,
    425,
    411,
    427,
    376,
    433,
    416,
    436,
)

# Forehead center + medial cheek “apples” (even lit skin; avoids nose sides / jaw shadow).
_SKIN_WELL_LIT_INDICES: Tuple[Tuple[int, ...], ...] = (
    (10, 151, 9, 8, 168, 6),  # upper face band
    (205, 50, 123, 187, 200),  # left mid-face
    (425, 411, 352, 266, 427, 416, 433),  # right mid-face
)

_RIGHT_IRIS_INDICES: Tuple[int, ...] = (469, 470, 471, 472, 468)
_LEFT_IRIS_INDICES: Tuple[int, ...] = (474, 475, 476, 477, 473)

# Upper face / hairline arc (subset of FACE_OVAL): mesh-guided crown boundary
_HAIR_UPPER_FACE_ARC: Tuple[int, ...] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
)
_EYEBROW_TOP_INDICES: Tuple[int, ...] = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)

_landmarker_singleton: Optional[Any] = None


@dataclass(frozen=True)
class ColorSample:
    hex: str
    rgb: Tuple[int, int, int]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _clip_uint8(value: float) -> int:
    return int(max(0, min(255, round(value))))


def _bgr_to_rgb(b: int, g: int, r: int) -> Tuple[int, int, int]:
    return r, g, b


def resolve_face_landmarker_model_path(*, allow_download: bool = True) -> Path:
    """
    Return path to ``face_landmarker.task``.

    Order: ``MEDIAPIPE_FACE_LANDMARKER_MODEL``, ``models/face_landmarker.task``
    next to this script, then ``~/.cache/stylepalette/face_landmarker.task``
    (downloaded if allowed and missing).
    """
    env = os.environ.get(_ENV_MODEL)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"{_ENV_MODEL} is set to {p!s} but that file does not exist."
            )
        return p

    if _DEFAULT_LOCAL_MODEL.is_file():
        return _DEFAULT_LOCAL_MODEL

    if _HOME_CACHE_MODEL.is_file():
        return _HOME_CACHE_MODEL
    if _PACKAGE_CACHE_MODEL.is_file():
        return _PACKAGE_CACHE_MODEL

    if not allow_download:
        raise FileNotFoundError(
            "Face landmarker model not found. Place face_landmarker.task in "
            f"{_DEFAULT_LOCAL_MODEL.parent}/ or set {_ENV_MODEL}."
        )

    dest_candidates = [_HOME_CACHE_MODEL, _PACKAGE_CACHE_MODEL]
    errors: List[str] = []
    with urllib.request.urlopen(_FACE_LANDMARKER_URL, timeout=120) as resp:
        data = resp.read()

    for dest in dest_candidates:
        tmp = dest.with_suffix(".tmp")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except OSError as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            errors.append(f"{dest}: {e}")

    raise RuntimeError(
        "Could not save the face landmarker model after download. "
        f"Tried:\n  " + "\n  ".join(errors) + "\n"
        f"Download manually from:\n  {_FACE_LANDMARKER_URL}\n"
        f"Save as {_DEFAULT_LOCAL_MODEL} or set {_ENV_MODEL} to the file path."
    )


def get_face_landmarker() -> Any:
    global _landmarker_singleton
    if FaceLandmarker is None:
        raise RuntimeError(
            "mediapipe is not installed or failed to import. "
            "Install with: pip install mediapipe\n"
            f"Import error: {_MEDIAPIPE_IMPORT_ERROR}"
        )
    if _landmarker_singleton is None:
        model_path = str(resolve_face_landmarker_model_path())
        _landmarker_singleton = FaceLandmarker.create_from_model_path(model_path)
    return _landmarker_singleton


def load_image_bgr(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image (unsupported or corrupt): {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")
    return image


def load_image_bgr_from_bytes(data: bytes) -> np.ndarray:
    """
    Decode a JPEG/PNG (etc.) in memory for ``analyze_facial_colors`` (e.g. FastAPI ``UploadFile``).
    """
    if not data:
        raise ValueError("Empty image bytes.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from bytes (unsupported or corrupt).")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")
    return image


def _landmark_xy(landmark: Any, width: int, height: int) -> Tuple[int, int]:
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def _points_from_indices(
    landmarks: Sequence[Any], indices: Iterable[int], w: int, h: int
) -> np.ndarray:
    pts: List[Tuple[int, int]] = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append(_landmark_xy(lm, w, h))
    return np.array(pts, dtype=np.int32)


def _unique_indices_from_face_oval() -> Tuple[int, ...]:
    if FaceLandmarksConnections is None:
        raise RuntimeError("MediaPipe FaceLandmarksConnections unavailable.")
    s: set[int] = set()
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
        s.add(conn.start)
        s.add(conn.end)
    return tuple(sorted(s))


_FACE_OVAL_INDICES: Optional[Tuple[int, ...]] = None


def _get_face_oval_indices() -> Tuple[int, ...]:
    global _FACE_OVAL_INDICES
    if _FACE_OVAL_INDICES is None:
        _FACE_OVAL_INDICES = _unique_indices_from_face_oval()
    return _FACE_OVAL_INDICES


def _convex_hull_mask(
    shape: Tuple[int, int], points: np.ndarray, pad: int = 4
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if points.size == 0:
        return mask
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def _trimmed_mean_bgr(bgr_pixels: np.ndarray, trim_ratio: float = 0.1) -> Tuple[int, int, int]:
    """
    Robust average sRGB (returned as R,G,B ints): trim outliers by grayscale rank,
    then mean in BGR. OpenCV's LAB conversion on float32 expects 0–1 data, so we
    stay in uint8 BGR for numerical stability.
    """
    if bgr_pixels.size == 0:
        raise ValueError("No pixels to sample.")
    px = bgr_pixels.reshape(-1, 3).astype(np.uint8, copy=False)
    n = px.shape[0]
    gray = cv2.cvtColor(px.reshape(n, 1, 3), cv2.COLOR_BGR2GRAY).ravel()
    order = np.argsort(gray, kind="mergesort")
    lo = int(n * trim_ratio)
    hi = max(lo + 1, n - int(n * trim_ratio))
    sel = px[order[lo:hi]]
    mean_bgr = np.mean(sel.astype(np.float64), axis=0)
    b, g, r = (
        _clip_uint8(mean_bgr[0]),
        _clip_uint8(mean_bgr[1]),
        _clip_uint8(mean_bgr[2]),
    )
    return _bgr_to_rgb(b, g, r)


def _sample_color_masked(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = image_bgr[mask > 0]
    if len(pixels) < 30:
        if len(pixels) == 0:
            raise ValueError("Empty mask for color sampling.")
        mean_bgr = np.mean(pixels.astype(np.float32), axis=0)
        b, g, r = (
            _clip_uint8(mean_bgr[0]),
            _clip_uint8(mean_bgr[1]),
            _clip_uint8(mean_bgr[2]),
        )
        return _bgr_to_rgb(b, g, r)
    return _trimmed_mean_bgr(pixels)


def _build_skin_well_lit_mask(
    image_bgr: np.ndarray, landmarks: Sequence[Any]
) -> np.ndarray:
    """Well-lit skin ROI: small hulls on forehead and cheek apples (not full face oval)."""
    h, w = image_bgr.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for group in _SKIN_WELL_LIT_INDICES:
        pts = _points_from_indices(landmarks, group, w, h)
        if pts.shape[0] < 3:
            continue
        out = cv2.bitwise_or(out, _convex_hull_mask((h, w), pts, pad=3))
    return out


def _sample_skin_color_robust(
    image_bgr: np.ndarray, skin_work_mask: np.ndarray
) -> Tuple[int, int, int]:
    """
    Re-sample from evenly lit skin: forehead / cheek region mask plus luminance
    gate so we skip lateral shadows and heavy saturation pockets.
    """
    pixels = image_bgr[skin_work_mask > 0]
    if len(pixels) == 0:
        raise ValueError("Empty skin mask for color sampling.")
    px = pixels.reshape(-1, 3).astype(np.uint8, copy=False)
    n = px.shape[0]
    if n < 20:
        return _trimmed_mean_bgr(px, trim_ratio=0.08)
    gray = cv2.cvtColor(px.reshape(n, 1, 3), cv2.COLOR_BGR2GRAY).ravel().astype(np.float32)
    order = np.argsort(gray, kind="mergesort")
    g_lo = float(np.percentile(gray, 40.0))
    g_hi = float(np.percentile(gray, 88.0))
    kept = (gray >= g_lo) & (gray <= g_hi)
    if np.sum(kept) < 15:
        lo = int(n * 0.15)
        hi = max(lo + 1, n - int(n * 0.10))
        sel = px[order[lo:hi]]
    else:
        sel = px[kept]
    return _trimmed_mean_bgr(sel, trim_ratio=0.1)


def _hair_filter_skin_like_pixels(
    sel_bgr: np.ndarray, skin_median_bgr: Optional[np.ndarray]
) -> np.ndarray:
    """Drop hair-mask pixels that match skin chroma + luminance (forehead bleed)."""
    if skin_median_bgr is None or len(sel_bgr) < 8:
        return sel_bgr
    sk = skin_median_bgr.reshape(1, 1, 3).astype(np.uint8)
    sk_lab = cv2.cvtColor(sk, cv2.COLOR_BGR2LAB).reshape(3).astype(np.int16)
    lab = (
        cv2.cvtColor(sel_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        .reshape(-1, 3)
        .astype(np.int16)
    )
    da = np.abs(lab[:, 1] - sk_lab[1])
    db = np.abs(lab[:, 2] - sk_lab[2])
    chroma_dist = np.sqrt(da.astype(np.float64) ** 2 + db.astype(np.float64) ** 2)
    lum_hair = lab[:, 0].astype(np.float64)
    lum_skin = float(sk_lab[0])
    skin_like = (chroma_dist < 22.0) & (np.abs(lum_hair - lum_skin) < 28.0)
    if np.sum(~skin_like) > 3:
        return sel_bgr[~skin_like]
    return sel_bgr


def _circular_median_hue_hsv(h: np.ndarray) -> float:
    """OpenCV H in 0..179: median direction on the hue circle (degrees 0..358)."""
    h = h.astype(np.float64).ravel()
    if h.size == 0:
        return 0.0
    t = np.radians((h * 2.0) % 360.0)
    mx = float(np.median(np.cos(t)))
    my = float(np.median(np.sin(t)))
    ang = float(np.degrees(np.arctan2(my, mx)))
    if ang < 0.0:
        ang += 360.0
    out = (ang * 0.5) % 180.0
    return max(0.0, min(179.0, out))


def _hsv_pigment_median_bgr(
    bgr_block: np.ndarray,
    skin_median_bgr: Optional[np.ndarray],
    region: str = "default",
) -> Optional[np.ndarray]:
    """
    Convert BGR pixels to HSV, remove glare (low S / extreme V) and deep shadow,
    return median H/S/V in pigment space then a single BGR uint8.
    ``region`` == ``crown`` uses stricter cuts for overhead specular (ash) hair.
    """
    bgr = _hair_filter_skin_like_pixels(bgr_block, skin_median_bgr)
    n = bgr.shape[0]
    if n < 4:
        return None
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.float64)
    s = hsv[:, 1].astype(np.float64)
    v = hsv[:, 2].astype(np.float64)
    # High-V + low-S = overexposed crown / gray reflection — not true pigment
    not_specular = ~((s < 42.0) & (v > 188.0))
    if int(np.sum(not_specular)) > max(3, n // 3):
        h, s, v = h[not_specular], s[not_specular], v[not_specular]
    if region == "crown":
        p_lo, p_hi, s_floor = 12.0, 88.0, 8.0
    else:
        p_lo, p_hi, s_floor = 6.0, 94.0, 5.0
    s_lo = max(s_floor, float(np.percentile(s, p_lo)))
    s_hi = max(s_lo + 1.0, float(np.percentile(s, p_hi)))
    v_lo = max(12.0, float(np.percentile(v, 8.0)))
    v_hi = min(250.0, float(np.percentile(v, 90.0 if region == "crown" else 92.0)))
    if region == "crown":
        v_hi = min(v_hi, 215.0)
    # Very low S = desaturated / glare; very low or high V = shadow / specular
    ok = (s >= s_lo) & (s <= s_hi) & (v >= v_lo) & (v <= v_hi) & (s >= 5.0)
    if int(np.sum(ok)) < 4:
        ok = (
            (s >= max(4.0, float(np.percentile(s, 4.0))))
            & (v >= 10.0)
            & (v <= 250.0)
            & (s + 0.2 * v > 30.0)
        )
    if int(np.sum(ok)) < 4:  # relax if crown / trim stripped too much
        ok2 = (s > 4.0) & (v > 20.0) & (v < 245.0) & (s + 0.18 * v > 32.0)
        if int(np.sum(ok2)) >= 3:
            ok = ok2
    if int(np.sum(ok)) < 3:
        return None
    h, s, v = h[ok], s[ok], v[ok]
    h_med = _circular_median_hue_hsv(h)
    s_med = float(np.median(s))
    v_med = float(np.median(v))
    bgr1 = np.array(
        [[[_clip_uint8(h_med), _clip_uint8(s_med), _clip_uint8(v_med)]]], dtype=np.uint8
    )
    return cv2.cvtColor(bgr1, cv2.COLOR_HSV2BGR).reshape(3).astype(np.float64)


def _hair_zones_to_masks(
    h_img: int,
    w: int,
    hair_mask: np.ndarray,
    landmarks: Optional[Sequence[Any]],
) -> Tuple[Tuple[Optional[np.ndarray], float], ...]:
    """
    Return ((mask, weight), ...): crown, side, lower. Masks are uint8 0/255;
    we use weights for weighted median; side = highest.
    """
    hair = hair_mask > 0
    ys, xs = np.where(hair)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    hb = max(1, y1 - y0)
    xw = max(1, x1 - x0)

    w_lat = int(max(0.10 * w, 0.24 * xw, 0.18 * w))
    if landmarks is not None and len(landmarks) > 0:
        xs_l = [landmarks[i].x * w for i in range(len(landmarks))]
        face_w = max(1.0, max(xs_l) - min(xs_l))
        w_lat = max(w_lat, int(0.32 * face_w), int(0.20 * w))

    zone_c = np.zeros((h_img, w), dtype=np.uint8)
    y_crown = y0 + int(0.36 * hb)
    inner_l = x0 + int(0.18 * xw)
    inner_r = x1 - int(0.18 * xw)
    zone_c[y0 : min(y_crown, y1 + 1), :] = 255
    if inner_l < inner_r:
        zone_c[:, : max(0, inner_l)] = 0
        zone_c[:, min(w, inner_r) :] = 0
    zone_c = cv2.bitwise_and(zone_c, hair_mask)

    zone_s = np.zeros((h_img, w), dtype=np.uint8)
    y_mid0, y_mid1 = y0 + int(0.08 * hb), y0 + int(0.90 * hb)
    y_mid0 = min(max(0, y_mid0), h_img)
    y_mid1 = min(max(0, y_mid1 + 1), h_img)
    if y_mid0 < y_mid1:
        zone_s[y_mid0:y_mid1, 0 : min(w, x0 + w_lat)] = 255
        zone_s[y_mid0:y_mid1, max(0, x1 - w_lat) : w] = 255
    zone_s = cv2.bitwise_and(zone_s, hair_mask)

    zone_l = np.zeros((h_img, w), dtype=np.uint8)
    y_lo = y0 + int(0.68 * hb)
    if y_lo <= y1:
        zone_l[y_lo : y1 + 1, :] = 255
    zone_l = cv2.bitwise_and(zone_l, hair_mask)

    if int(np.sum(zone_s > 0)) < 30:
        if y_mid0 < y_mid1:
            z2 = np.zeros((h_img, w), dtype=np.uint8)
            z2[y_mid0:y_mid1, 0 : min(w, x0 + w_lat + 8)] = 255
            z2[y_mid0:y_mid1, max(0, x1 - w_lat - 8) : w] = 255
            zone_s = cv2.bitwise_and(z2, hair_mask)

    return (
        # Side / lower = most pigment-accurate; crown downweighted (overhead glare)
        (
            zone_c if int(np.sum(zone_c > 0)) > 0 else None,
            float(_HAIR_ZONE_WEIGHT_CROWN),
        ),
        (
            zone_s if int(np.sum(zone_s > 0)) > 0 else None,
            float(_HAIR_ZONE_WEIGHT_SIDE),
        ),
        (
            zone_l if int(np.sum(zone_l > 0)) > 0 else None,
            float(_HAIR_ZONE_WEIGHT_LOWER),
        ),
    )


def _weighted_median_bgr(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """rows: (N,3) BGR; weights positive; expand rows by rounded weights, median."""
    w_rep = np.maximum(1, np.rint(weights * 5.0).astype(int))
    parts = [np.tile(rows[i : i + 1], (w_rep[i], 1)) for i in range(len(w_rep))]
    if not parts:
        return np.median(rows, axis=0)
    ex = np.vstack(parts)
    return np.median(ex, axis=0)


def _zone_bgr_reject_outliers(
    bgrs: list[np.ndarray], wts: list[float]
) -> Tuple[list[np.ndarray], list[float]]:
    """Remove zones with very low S vs peers (stray glare) in HSV."""
    if len(bgrs) < 2:
        return bgrs, wts
    svals = []
    for row in bgrs:
        u8 = np.clip(np.rint(row), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(u8.reshape(1, 1, 3), cv2.COLOR_BGR2HSV).reshape(3)
        svals.append(float(hsv[1]))
    s_med = float(np.median(svals))
    keep = [i for i, s in enumerate(svals) if s >= 0.35 * s_med or s >= 18.0]
    if not keep or len(keep) == len(bgrs):
        return bgrs, wts
    return [bgrs[i] for i in keep], [wts[i] for i in keep]


def _sample_hair_color_robust(
    image_bgr: np.ndarray,
    hair_mask: np.ndarray,
    skin_median_bgr: Optional[np.ndarray],
    landmarks: Optional[Sequence[Any]] = None,
) -> Tuple[int, int, int]:
    """
    Multi-zone (crown, side, lower) HSV-based pigment, side-weighted median,
    with glare / shadow rejection in S and V per zone.
    """
    im_h, im_w = image_bgr.shape[:2]
    if int(np.sum(hair_mask > 0)) < 20:
        raise ValueError("Empty hair mask for color sampling.")

    zones = _hair_zones_to_masks(im_h, im_w, hair_mask, landmarks)
    region_names = ("crown", "side", "lower")
    bgrs: list[np.ndarray] = []
    wts: list[float] = []
    debug_pre: list[dict[str, Any]] = []
    for zi, item in enumerate(zones):
        m, weight = item[0], item[1]
        if m is None or int(np.sum(m > 0)) < 12:
            if DEBUG_HAIR_ANALYSIS and m is not None:
                print(
                    f"[DEBUG_HAIR_ANALYSIS] hair_zone_skip {region_names[zi]}: "
                    f"mask_pixels={int(np.sum(m > 0))} (<12, skipped)"
                )
            continue
        n_m = int(np.sum(m > 0))
        block = image_bgr[m > 0].reshape(-1, 3).astype(np.uint8, copy=False)
        rname = region_names[zi] if zi < 3 else "default"
        bgr1 = _hsv_pigment_median_bgr(block, skin_median_bgr, region=rname)
        if bgr1 is not None:
            bgrs.append(bgr1)
            wts.append(float(weight))
            if DEBUG_HAIR_ANALYSIS:
                u8d = np.clip(np.rint(bgr1), 0, 255).astype(np.uint8)
                hsvd = cv2.cvtColor(u8d.reshape(1, 1, 3), cv2.COLOR_BGR2HSV).reshape(3)
                debug_pre.append(
                    {
                        "name": rname,
                        "n_mask": n_m,
                        "weight": weight,
                        "H": float(hsvd[0]),
                        "S": float(hsvd[1]),
                        "V": float(hsvd[2]),
                    }
                )
        elif DEBUG_HAIR_ANALYSIS:
            print(
                f"[DEBUG_HAIR_ANALYSIS] hair_zone_skip {rname}: "
                f"mask_pixels={n_m} (HSV pigment filter returned no sample)"
            )

    if DEBUG_HAIR_ANALYSIS and debug_pre:
        for row in debug_pre:
            print(
                "[DEBUG_HAIR_ANALYSIS] hair_zone_pretest "
                f"{row['name']}: n_mask={row['n_mask']} w={row['weight']} "
                f"H={row['H']:.1f} S={row['S']:.1f} V={row['V']:.1f}"
            )

    n_pre = len(bgrs)
    bgrs, wts = _zone_bgr_reject_outliers(bgrs, wts)
    if DEBUG_HAIR_ANALYSIS and n_pre != len(bgrs):
        print(
            f"[DEBUG_HAIR_ANALYSIS] hair_outlier_reject: zones {n_pre} -> {len(bgrs)} (low-S glare drop)"
        )
    if not bgrs:
        px = image_bgr[hair_mask > 0].reshape(-1, 3).astype(np.uint8, copy=False)
        px = _hair_filter_skin_like_pixels(px, skin_median_bgr)
        bgr1 = _hsv_pigment_median_bgr(px, skin_median_bgr, region="default")
        if bgr1 is not None:
            b = _clip_uint8(float(bgr1[0]))
            g = _clip_uint8(float(bgr1[1]))
            r = _clip_uint8(float(bgr1[2]))
            return _bgr_to_rgb(b, g, r)
        med = np.median(px.astype(np.float64), axis=0)
        b, g, r = _clip_uint8(med[0]), _clip_uint8(med[1]), _clip_uint8(med[2])
        return _bgr_to_rgb(b, g, r)

    stack = np.stack(bgrs, axis=0)
    w_arr = np.array(wts, dtype=np.float64)
    final_bgr = _weighted_median_bgr(stack, w_arr)
    b, g, r = (
        _clip_uint8(float(final_bgr[0])),
        _clip_uint8(float(final_bgr[1])),
        _clip_uint8(float(final_bgr[2])),
    )
    if DEBUG_HAIR_ANALYSIS:
        s_dbg, h_dbg = [], []
        for row in bgrs:
            u8 = np.clip(np.rint(row), 0, 255).astype(np.uint8)
            hx = cv2.cvtColor(u8.reshape(1, 1, 3), cv2.COLOR_BGR2HSV).reshape(3)
            h_dbg.append(float(hx[0]))
            s_dbg.append(float(hx[1]))
        print(
            "[DEBUG_HAIR_ANALYSIS] hair_final "
            f"n_zones={len(bgrs)} after_outlier_reject w={wts} H_medians={h_dbg} S={s_dbg} "
            f"bgr_final=({b:.0f},{g:.0f},{r:.0f})"
        )
    return _bgr_to_rgb(b, g, r)


def _sample_eye_color_robust(image_bgr: np.ndarray, iris_mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Sweet-spot iris pool, then:
    - **Brown / dark eyes** (red–brown hue or R-led BGR): robust median in BGR; no green/LAB nudge.
    - **Green / hazel** (green hue): saturation-weighted + secondary green-leaning fix.
    - Mild LAB in uint8 only; no aggressive L floor (avoids gray artifacts on brown).
    """
    pixels = image_bgr[iris_mask > 0]
    if len(pixels) == 0:
        raise ValueError("Empty iris mask for color sampling.")

    px = pixels.reshape(-1, 3).astype(np.uint8, copy=False)
    n = px.shape[0]
    gray = cv2.cvtColor(px.reshape(n, 1, 3), cv2.COLOR_BGR2GRAY).ravel().astype(np.float32)

    lo = float(np.percentile(gray, 15.0))
    hi = float(np.percentile(gray, 95.0))
    sweet = (gray >= lo) & (gray <= hi)
    if np.sum(sweet) < 8:
        sweet = gray >= float(np.percentile(gray, 25.0))
    sel = px[sweet]
    if len(sel) < 5:
        return _trimmed_mean_bgr(px, trim_ratio=0.08)

    hsv = cv2.cvtColor(sel.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mh, ms, mv = float(np.median(hue)), float(np.median(sat)), float(np.median(val))
    mb = np.median(sel.astype(np.float64), axis=0)
    b_m, g_m, r_m = float(mb[0]), float(mb[1]), float(mb[2])

    on_brown_hue = (mh <= 32.0) or (mh >= 148.0)  # OpenCV: red / orange / brown / wrap
    on_brown_bgr = (r_m + 1.0 >= b_m) and (r_m + 3.0 >= g_m) and (ms < 120.0)
    dark_eye = (ms < 88.0) and (r_m + 0.5 >= b_m)  # very dark but not blue-dominated
    if (on_brown_hue and ms < 110.0) or (on_brown_bgr) or dark_eye:
        vv = val.astype(np.float64)
        ss = sat.astype(np.float64)
        okb = (vv >= float(np.percentile(vv, 8.0))) & (vv <= float(np.percentile(vv, 91.0))) & (
            ss > float(np.percentile(ss, 3.0)) + 2.0
        )
        if int(np.sum(okb)) < 4:
            okb = (vv >= 18.0) & (vv <= 235.0)
        s2 = sel[okb] if int(np.sum(okb)) > 2 else sel
        mm = np.median(s2.astype(np.float64), axis=0)
        r_o, g_o, b_o = _clip_uint8(mm[2]), _clip_uint8(mm[1]), _clip_uint8(mm[0])
        if max(r_o, g_o, b_o) >= 6:
            if DEBUG_HAIR_ANALYSIS:
                print(
                    f"[DEBUG_HAIR_ANALYSIS] iris_path=brown h_med={mh:.1f} s_med={ms:.1f} rgb=({r_o},{g_o},{b_o})"
                )
            return (r_o, g_o, b_o)

    # Green / hazel: sat-weight; moderate green channel boost (not 3x)
    red = sel[:, 2].astype(np.float32)
    green = sel[:, 1].astype(np.float32)
    blue = sel[:, 0].astype(np.float32)
    sat_n = sat / 255.0
    green_dominant = (green > red) & (green > blue)
    weights = (0.58 * sat_n) + (0.32 * (green / 255.0)) + 1e-6
    weights = weights * np.where(green_dominant, 1.5, 1.0)
    mean_bgr = np.sum(sel.astype(np.float32) * weights[:, None], axis=0) / np.sum(weights)
    b0, g0, r0 = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])

    if float(np.mean(sat_n)) < 0.30 and 30.0 <= np.median(hue) <= 80.0:
        green_hue = (hue >= 30.0) & (hue <= 75.0) & (sat > float(np.percentile(sat, 40.0)))
        if int(np.sum(green_hue)) >= 3:
            sg = sel[green_hue].astype(np.float32)
            b0, g0, r0 = map(float, np.median(sg, axis=0))

    spread = max(r0, g0, b0) - min(r0, g0, b0)
    rh = cv2.cvtColor(
        np.array([[[int(round(b0)), int(round(g0)), int(round(r0))]]], dtype=np.uint8),
        cv2.COLOR_BGR2HSV,
    ).reshape(3)
    if spread < 26.0 and float(rh[1]) < 70.0 and 28.0 <= float(np.median(hue)) <= 78.0:
        k = (hue >= 30.0) & (hue <= 75.0) & (sat >= float(np.percentile(sat, 45.0)))
        if int(np.sum(k)) >= 4:
            c = sel[k].astype(np.float32)
            b0, g0, r0 = map(float, np.median(c, axis=0))

    final_bgr_u8 = np.array(
        [
            [
                [
                    int(np.clip(round(b0), 0, 255)),
                    int(np.clip(round(g0), 0, 255)),
                    int(np.clip(round(r0), 0, 255)),
                ]
            ]
        ],
        dtype=np.uint8,
    )
    flab = cv2.cvtColor(final_bgr_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    l_s, a_s, b_s = flab[0], flab[1], flab[2]
    mhd = float(np.median(hue) * 2.0)
    if 70.0 <= mhd <= 150.0 and b_s > 135.0:
        a_s = max(98.0, a_s - min(10.0, (b_s - 132.0) * 0.35))
    # Light lift for dark readout only (not brown-blocked range)
    if l_s < 70.0:
        l_s = min(115.0, 68.0 + 0.45 * (l_s - 40.0))
    l_s = max(38.0, min(220.0, l_s))
    lab_u8 = np.array(
        [
            [
                [
                    int(np.clip(round(l_s), 0, 255)),
                    int(np.clip(round(a_s), 0, 255)),
                    int(np.clip(round(b_s), 0, 255)),
                ]
            ]
        ],
        dtype=np.uint8,
    )
    bgr_out = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR).reshape(3)
    r_out, g_out, b_out = _bgr_to_rgb(
        _clip_uint8(float(bgr_out[0])),
        _clip_uint8(float(bgr_out[1])),
        _clip_uint8(float(bgr_out[2])),
    )
    if max(r_out, g_out, b_out) < 30:
        r_fb, g_fb, b_fb = _trimmed_mean_bgr(sel, trim_ratio=0.1)
        return (r_fb, g_fb, b_fb)
    if DEBUG_HAIR_ANALYSIS:
        print(
            f"[DEBUG_HAIR_ANALYSIS] iris_path=hazel_green h_med={mhd:.1f} rgb=({r_out},{g_out},{b_out})"
        )
    return (r_out, g_out, b_out)


def _build_skin_mask(
    image_bgr: np.ndarray, landmarks: Sequence[Any]
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    pts = _points_from_indices(landmarks, _SKIN_FOREHEAD_CHEEK_INDICES, w, h)
    return _convex_hull_mask((h, w), pts, pad=6)


def _build_iris_mask(
    image_bgr: np.ndarray, landmarks: Sequence[Any]
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for indices in (_RIGHT_IRIS_INDICES, _LEFT_IRIS_INDICES):
        pts = _points_from_indices(landmarks, indices, w, h)
        if pts.shape[0] < 3:
            continue
        hull = _convex_hull_mask((h, w), pts, pad=2)
        mask = cv2.bitwise_or(mask, hull)
    return mask


def _face_vertical_span(landmarks: Sequence[Any], w: int, h: int) -> int:
    ys = [_landmark_xy(landmarks[i], w, h)[1] for i in range(len(landmarks))]
    return max(1, int(max(ys) - min(ys)))


def _build_hair_mask(image_bgr: np.ndarray, landmarks: Sequence[Any]) -> np.ndarray:
    """
    Mesh-guided crown ROI: band from image top down to just above the brows,
    minus an **eroded** face-oval fill (keeps real hair at the hairline; drops
    forehead/cheeks inside the mesh). Horizontal crop to head bbox avoids
    distant background in wide shots.
    """
    h, w = image_bgr.shape[:2]
    fh = _face_vertical_span(landmarks, w, h)

    eyebrow_ys = [_landmark_xy(landmarks[i], w, h)[1] for i in _EYEBROW_TOP_INDICES]
    brow_y = min(eyebrow_ys)
    # Horizontal band ends **above** the brows (forehead / hair only).
    y_bottom = max(1, brow_y - max(8, int(0.05 * fh)))

    arc_pts = _points_from_indices(landmarks, _HAIR_UPPER_FACE_ARC, w, h)
    arc_y_min = int(np.min(arc_pts[:, 1]))
    arc_y_max = int(np.max(arc_pts[:, 1]))
    y_cap = max(arc_y_max, 1)
    # End band at the higher of (just above brows) vs (mesh arc bottom): smaller
    # y-extent into the image, so we rarely include brow skin or cheek shadows.
    y_bottom = min(y_bottom, y_cap)
    min_band = max(10, int(0.10 * fh))
    y_bottom = max(y_bottom, min(arc_y_min + min_band, y_cap))

    xs = [_landmark_xy(landmarks[i], w, h)[0] for i in range(len(landmarks))]
    face_w = max(1, int(max(xs) - min(xs)))
    # Wide lateral extent so ear-adjacent hair is included (not just face-width box).
    x_margin = int(max(0.20 * w, 0.48 * face_w))
    x0 = max(0, int(min(xs)) - x_margin)
    x1 = min(w, int(max(xs)) + x_margin)

    if y_bottom <= 4:
        raise ValueError("Hair region degenerate; landmarks may be unreliable.")

    band = np.zeros((h, w), dtype=np.uint8)
    band[0:y_bottom, :] = 255
    band[:, 0:x0] = 0
    band[:, x1:] = 0

    oval_xy = _points_from_indices(landmarks, _get_face_oval_indices(), w, h)
    face_fill = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_fill, cv2.convexHull(oval_xy), 255)
    # Erode face mask so hairline pixels (outside shrunken face) stay in ROI.
    ksz = max(5, int(0.012 * min(h, w)))
    if ksz % 2 == 0:
        ksz += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    face_core = cv2.erode(face_fill, ker, iterations=1)

    hair = cv2.bitwise_and(band, cv2.bitwise_not(face_core))
    # Optional: keep only connected components touching the top edge (reduces
    # stray background below chin in odd crops). Skip if it kills all pixels.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
    touching_top = [
        lab
        for lab in range(1, num)
        if any(labels[0, c] == lab for c in range(w))
    ]
    if touching_top:
        keep = np.isin(labels, touching_top).astype(np.uint8) * 255
        if int(np.sum(keep > 0)) > 0.15 * int(np.sum(hair > 0)):
            hair = keep

    # Mid-length / side hair below the crown band: wide ear-adjacent strips.
    y_sl = int(brow_y)
    y_sh = min(h, int(brow_y + int(0.55 * fh)))
    w_side = max(10, int(0.17 * w), int(0.28 * face_w))
    if y_sl < y_sh:
        side = np.zeros((h, w), dtype=np.uint8)
        xl0, xl1 = x0, min(w, x0 + w_side)
        xr0, xr1 = max(0, x1 - w_side), x1
        if xl1 > xl0:
            side[y_sl:y_sh, xl0:xl1] = 255
        if xr1 > xr0:
            side[y_sl:y_sh, xr0:xr1] = 255
        side = cv2.bitwise_and(side, cv2.bitwise_not(face_core))
        hair = cv2.bitwise_or(hair, side)

    if int(np.sum(hair > 0)) < 80:
        raise ValueError("Hair region too small after mesh masking.")

    return hair


def _color_sample_from_mask(
    image_bgr: np.ndarray, mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None
) -> ColorSample:
    work = mask.copy()
    if exclude_mask is not None:
        work = cv2.bitwise_and(work, cv2.bitwise_not(exclude_mask))
    r, g, b = _sample_color_masked(image_bgr, work)
    return ColorSample(hex=rgb_to_hex(r, g, b), rgb=(r, g, b))


def _detect_landmarks(
    image_bgr: np.ndarray,
    landmarker: Any,
) -> Optional[List[Any]]:
    if image_lib is None:
        raise RuntimeError("MediaPipe image module unavailable.")
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def analyze_facial_colors(
    image_bgr: np.ndarray,
    *,
    landmarker: Optional[Any] = None,
    allow_model_download: bool = True,
) -> dict[str, Any]:
    """
    Detect face mesh landmarks and return skin / eye / hair colors.

    ``landmarker``: optional pre-built ``FaceLandmarker`` (for batch use).
    ``allow_model_download``: when True, may download the .task model to the
    user cache once if missing.
    """
    empty = {
        "success": False,
        "error": None,
        "skin_tone": None,
        "eye_color": None,
        "hair_color": None,
    }

    try:
        if landmarker is not None:
            lm_holder = landmarker
        elif allow_model_download:
            lm_holder = get_face_landmarker()
        else:
            if FaceLandmarker is None:
                raise RuntimeError(
                    "mediapipe is not installed or failed to import. "
                    f"Import error: {_MEDIAPIPE_IMPORT_ERROR}"
                )
            model_path = str(resolve_face_landmarker_model_path(allow_download=False))
            lm_holder = FaceLandmarker.create_from_model_path(model_path)

        lm_list = _detect_landmarks(image_bgr, lm_holder)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        return {**empty, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        msg = str(e)
        if "kGpuService" in msg or "NSOpenGL" in msg or "EGL" in msg:
            msg += (
                " MediaPipe Face Landmarker needs a graphics-capable environment on "
                "some platforms (e.g. a normal macOS desktop session, not a restricted "
                "sandbox without OpenGL/EGL)."
            )
        return {**empty, "error": f"Face analysis failed: {msg}"}

    if lm_list is None:
        return {**empty, "error": "No face detected in the image."}

    try:
        skin_mask = _build_skin_mask(image_bgr, lm_list)
        skin_well = _build_skin_well_lit_mask(image_bgr, lm_list)
        iris_mask = _build_iris_mask(image_bgr, lm_list)
        hair_mask = _build_hair_mask(image_bgr, lm_list)

        work_skin = cv2.bitwise_and(skin_well, cv2.bitwise_not(iris_mask))
        if int(np.sum(work_skin > 0)) < 120:
            work_skin = cv2.bitwise_and(skin_mask, cv2.bitwise_not(iris_mask))

        skin_px = image_bgr[work_skin > 0]
        skin_med = (
            np.median(skin_px.reshape(-1, 3), axis=0)
            if skin_px.size >= 90
            else None
        )

        sr, sg, sb = _sample_skin_color_robust(image_bgr, work_skin)
        skin = ColorSample(hex=rgb_to_hex(sr, sg, sb), rgb=(sr, sg, sb))
        er, eg, eb = _sample_eye_color_robust(image_bgr, iris_mask)
        eye = ColorSample(hex=rgb_to_hex(er, eg, eb), rgb=(er, eg, eb))
        hr, hg, hb = _sample_hair_color_robust(
            image_bgr, hair_mask, skin_med, lm_list
        )
        hair = ColorSample(hex=rgb_to_hex(hr, hg, hb), rgb=(hr, hg, hb))
    except ValueError as e:
        return {**empty, "error": f"Could not extract colors: {e}"}

    def pack(cs: ColorSample) -> dict[str, Any]:
        return {"hex": cs.hex, "rgb": list(cs.rgb)}

    return {
        "success": True,
        "error": None,
        "skin_tone": pack(skin),
        "eye_color": pack(eye),
        "hair_color": pack(hair),
    }


def analyze_image_path(
    image_path: Path,
    *,
    allow_model_download: bool = True,
) -> dict[str, Any]:
    try:
        image = load_image_bgr(image_path)
    except (FileNotFoundError, ValueError) as e:
        return {
            "success": False,
            "error": str(e),
            "skin_tone": None,
            "eye_color": None,
            "hair_color": None,
        }
    return analyze_facial_colors(image, allow_model_download=allow_model_download)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract skin, eye, and hair colors from a selfie (StylePalette)."
    )
    parser.add_argument("image", type=Path, help="Path to input image (e.g. selfie JPEG/PNG).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON (stdout if omitted).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download face_landmarker.task; require local model or env path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = analyze_image_path(args.image, allow_model_download=not args.no_download)
    text = json.dumps(result, indent=2)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
