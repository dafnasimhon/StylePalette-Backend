from __future__ import annotations

import asyncio
import re
from typing import Any, Sequence, Tuple, Union

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import Settings, get_settings
from app.cv.facial_color_analysis import analyze_facial_colors, load_image_bgr_from_bytes
from app.domain.enums import EyeColor, HairColor, Season, SkinType
from app.domain.style_palettes import StylePalettes
from app.schemas.analysis import (
    AnalysisResult,
    ColorMeasurement,
    PaletteColor,
    RgbRange,
    SeasonPalette,
    TraitCorrectionRequest,
    TraitEstimate,
)
from app.services.season_analyzer import SeasonAnalyzer

# ---------------------------------------------------------------------------
# Color parsing & trait classification (inlined; no clothing_palette import)
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _parse_hex(hex_str: str) -> Tuple[int, int, int]:
    m = _HEX_RE.match(hex_str.strip())
    if not m:
        raise ValueError(f"Invalid hex color: {hex_str!r}")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _to_rgb(value: Union[str, Sequence[int], dict[str, Any]]) -> Tuple[int, int, int]:
    if isinstance(value, str):
        return _parse_hex(value)
    if isinstance(value, dict):
        if "rgb" in value:
            seq = value["rgb"]
            if not isinstance(seq, (list, tuple)) or len(seq) != 3:
                raise ValueError("'rgb' must be a length-3 sequence")
            return int(seq[0]), int(seq[1]), int(seq[2])
        raise ValueError("Mapping must include 'rgb'")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    raise TypeError("Unsupported color input")


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def lin(c: int) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _undertone_is_warm(skin_rgb: Tuple[int, int, int]) -> bool | None:
    r, g, b = skin_rgb
    delta_rb = r - b
    if delta_rb >= 18:
        return True
    if delta_rb <= -12:
        return False
    return None


def _bgr_hsv1(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Single-pixel HSV in OpenCV ranges (H 0–179, S/V 0–255)."""
    r, g, b_ = rgb
    bgr = np.uint8([[[b_, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def classify_skin_tone_from_rgb(skin_rgb: Tuple[int, int, int]) -> str:
    lum = _relative_luminance(skin_rgb)
    r, g, b = skin_rgb
    # Conservative thresholds: avoid over-lightening darker skin under highlights.
    if lum >= 0.72:
        return "Very Fair"
    if lum >= 0.46:
        return "Fair"
    # Warm indoor light can lower luminance while channels still indicate fair skin.
    if lum >= 0.34 and r >= 145 and g >= 120 and b >= 95:
        return "Fair"
    if lum >= 0.30:
        return "Medium/Tan"
    return "Dark"


def classify_hair_color_from_rgb(hair_rgb: Tuple[int, int, int]) -> str:
    h, s, v = _bgr_hsv1(hair_rgb)
    s_f = s / 255.0
    v_f = v / 255.0
    r, g, b_ = (c / 255.0 for c in hair_rgb)
    total = sum(hair_rgb)

    # Strong blonde signal: bright, warm/yellow-ish, not deeply saturated brown.
    if total >= 420 and v_f >= 0.42 and s_f <= 0.52 and h <= 55:
        return "Blonde"
    if total >= 400 and v_f >= 0.38 and s_f <= 0.45 and h <= 50:
        return "Blonde"

    if v_f < 0.18 and s_f < 0.5:
        return "Black"
    if v_f < 0.12:
        return "Black"

    # Prefer blonde before red/ginger for bright low-saturation hair.
    if v_f > 0.54 and s_f < 0.42 and 10 <= h <= 55:
        return "Blonde"
    if v_f > 0.66 and s_f < 0.46 and 8 <= h <= 60:
        return "Blonde"

    is_red_hue = h <= 10 or h >= 175
    # Keep ginger/red very strict: only classify when hue, saturation and channel
    # dominance all strongly indicate true red/copper hair.
    if is_red_hue and 0.20 <= v_f <= 0.70 and s_f >= 0.50:
        red_dominant = (r >= g * 1.20) and (r >= b_ * 1.28)
        strong_red_delta = (r - g) >= 0.12 and (r - b_) >= 0.16 and r >= 0.32
        if red_dominant or strong_red_delta:
            return "Red/Ginger"

    if v_f > 0.50 and s_f < 0.40 and 12 <= h <= 60:
        return "Blonde"
    if v_f > 0.62 and s_f < 0.36 and 8 <= h <= 50:
        return "Blonde"

    if v_f < 0.25:
        return "Black"

    return "Brown"


def classify_eye_color_from_rgb(eye_rgb: Tuple[int, int, int]) -> str:
    r, g, b_ = eye_rgb
    bgr_px = np.uint8([[[b_, g, r]]])
    lab = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2LAB)[0, 0]
    ll, _a_lab, bb = float(lab[0]), float(lab[1]), float(lab[2])

    h, s, v = _bgr_hsv1(eye_rgb)
    s_f = s / 255.0
    b_f = b_ / 255.0
    g_f = g / 255.0
    r_f = r / 255.0

    # Lab b* (OpenCV 0–255, ~128 neutral): lower ⇒ more blue, higher ⇒ yellow.
    if ll >= 48 and bb <= 128 and v >= 40 and not (52 <= h <= 88):
        if b_ >= r - 6 or bb <= 120:
            return "Blue"

    # Hue-first blue detection to handle desaturated blue irises in studio lighting.
    if 80 <= h <= 138 and v >= 35 and s >= 12:
        return "Blue"
    if 70 <= h <= 150 and b_ >= g and b_ >= r and (b_ - r) >= 6:
        return "Blue"
    if b_f >= r_f + 0.04 and b_f >= g_f + 0.03 and s_f > 0.10:
        return "Blue"
    if b_ > 90 and b_ >= r + 4 and b_ >= g + 3:
        return "Blue"

    if g_f > r_f + 0.05 and g_f > b_f + 0.02 and 35 <= h <= 100 and s_f > 0.15:
        return "Green"
    if g > r and g > b_ and (g - max(r, b_)) > 10 and 40 < h < 100:
        return "Green"

    return "Brown"


class AnalysisService:
    """Wraps the Smart Diagnosis pipeline (MediaPipe + OpenCV facial color analysis)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def analyze_selfie(self, image: UploadFile) -> AnalysisResult:
        data = await image.read()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._analyze_facial_from_bytes, data)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    def _analyze_facial_from_bytes(self, data: bytes) -> AnalysisResult:
        bgr = load_image_bgr_from_bytes(data)
        raw = analyze_facial_colors(
            bgr,
            allow_model_download=self._settings.mediapipe_allow_model_download,
        )
        if not raw.get("success"):
            raise ValueError(raw.get("error") or "Face analysis failed.")

        skin_blob = raw["skin_tone"]
        eye_blob = raw["eye_color"]
        hair_blob = raw["hair_color"]
        if not skin_blob or not eye_blob or not hair_blob:
            raise ValueError("Incomplete color samples from the pipeline.")

        sr, sg, sb = (int(x) for x in skin_blob["rgb"])
        er, eg, eb = (int(x) for x in eye_blob["rgb"])
        hr, hg, hb = (int(x) for x in hair_blob["rgb"])

        skin_label = classify_skin_tone_from_rgb((sr, sg, sb))
        eye_label = classify_eye_color_from_rgb((er, eg, eb))
        hair_label = classify_hair_color_from_rgb((hr, hg, hb))

        skin = self._to_skin_type(skin_label)
        eyes = self._to_eye_color(eye_label)
        hair = self._to_hair_color(hair_label)
        seasonal_palette = self._classify_season(skin=skin, hair=hair, eyes=eyes)
        season_palette = self._get_palette_recommendation(seasonal_palette)

        return AnalysisResult(
            seasonal_palette=seasonal_palette,
            traits=TraitEstimate(
                skin_tone=skin.value,
                eye_color=eyes.value,
                hair_color=hair.value,
                skin_sample=ColorMeasurement(rgb=list(skin_blob["rgb"])),
                eye_sample=ColorMeasurement(rgb=list(eye_blob["rgb"])),
                hair_sample=ColorMeasurement(rgb=list(hair_blob["rgb"])),
            ),
            confidence=0.82,
            palette_recommendation=season_palette,
            notes=(
                "Facial color analysis: MediaPipe Face Landmarker + HSV multi-zone sampling; "
                "traits from measured RGB."
            ),
        )

    def classify_from_traits(self, skin_tone: str, hair_color: str, eye_color: str) -> str:
        skin = self._to_skin_type(skin_tone)
        hair = self._to_hair_color(hair_color)
        eyes = self._to_eye_color(eye_color)
        return self._classify_season(skin=skin, hair=hair, eyes=eyes)

    def build_palette_from_traits(self, body: TraitCorrectionRequest) -> AnalysisResult:
        """Season + palette from user-corrected trait labels (optionally echo measured RGB)."""
        seasonal_palette = self.classify_from_traits(
            body.skin_tone, body.hair_color, body.eye_color
        )
        season_palette = self._get_palette_recommendation(seasonal_palette)
        return AnalysisResult(
            seasonal_palette=seasonal_palette,
            traits=TraitEstimate(
                skin_tone=body.skin_tone,
                eye_color=body.eye_color,
                hair_color=body.hair_color,
                skin_sample=body.skin_sample,
                eye_sample=body.eye_sample,
                hair_sample=body.hair_sample,
            ),
            confidence=None,
            palette_recommendation=season_palette,
            notes="Palette derived from user-confirmed skin, eye, and hair traits.",
        )

    @staticmethod
    def _classify_season(skin: SkinType, hair: HairColor, eyes: EyeColor) -> str:
        analyzer = SeasonAnalyzer(skin=skin, hair=hair, eyes=eyes)
        return analyzer.get_final_season().value

    @staticmethod
    def _get_palette_recommendation(seasonal_palette: str) -> SeasonPalette:
        season_mapping = {
            "Winter": Season.WINTER,
            "Summer": Season.SUMMER,
            "Autumn": Season.AUTUMN,
            "Spring": Season.SPRING,
        }
        season_enum = season_mapping[seasonal_palette]
        palette = StylePalettes.get_palette(season_enum)

        def _pc(item: dict[str, Any]) -> PaletteColor:
            rr = item["rgb_range"]
            return PaletteColor(
                rgb_range=RgbRange(rgb_min=list(rr["rgb_min"]), rgb_max=list(rr["rgb_max"]))
            )

        return SeasonPalette(
            description=palette["description"],
            power_colors=[_pc(item) for item in palette["power_colors"]],
            neutral_colors=[_pc(item) for item in palette["neutral_colors"]],
        )

    @staticmethod
    def _to_skin_type(value: str) -> SkinType:
        normalized = value.strip().lower()
        mapping = {
            "very fair": SkinType.VERY_FAIR,
            "fair": SkinType.FAIR,
            "medium/tan": SkinType.MEDIUM_TAN,
            "medium": SkinType.MEDIUM_TAN,
            "tan": SkinType.MEDIUM_TAN,
            "dark": SkinType.DARK,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported skin tone: {value}")
        return mapping[normalized]

    @staticmethod
    def _to_hair_color(value: str) -> HairColor:
        normalized = value.strip().lower()
        mapping = {
            "black": HairColor.BLACK,
            "dark brown": HairColor.BROWN,
            "brown": HairColor.BROWN,
            "blonde": HairColor.BLONDE,
            "red/ginger": HairColor.RED_GINGER,
            "red": HairColor.RED_GINGER,
            "ginger": HairColor.RED_GINGER,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported hair color: {value}")
        return mapping[normalized]

    @staticmethod
    def _to_eye_color(value: str) -> EyeColor:
        normalized = value.strip().lower()
        mapping = {
            "brown": EyeColor.BROWN,
            "blue": EyeColor.BLUE,
            "green": EyeColor.GREEN,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported eye color: {value}")
        return mapping[normalized]
