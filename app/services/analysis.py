from fastapi import UploadFile

from app.config import Settings, get_settings
from app.domain.enums import EyeColor, HairColor, Season, SkinType
from app.domain.style_palettes import StylePalettes
from app.schemas.analysis import AnalysisResult, PaletteColor, SeasonPalette, TraitEstimate
from app.services.season_analyzer import SeasonAnalyzer


class AnalysisService:
    """Wraps the Smart Diagnosis pipeline. Replace `_mock_analyze` with partner CV integration."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def analyze_selfie(self, image: UploadFile) -> AnalysisResult:
        _ = await image.read()
        if self._settings.use_mock_analysis:
            return self._mock_analyze()
        # Partner integration: call OpenCV/Mediapipe pipeline here and map output to AnalysisResult.
        raise NotImplementedError("Wire CV engine or set USE_MOCK_ANALYSIS=true")

    def _mock_analyze(self) -> AnalysisResult:
        skin = SkinType.FAIR
        hair = HairColor.BROWN
        eyes = EyeColor.BROWN
        seasonal_palette = self._classify_season(skin=skin, hair=hair, eyes=eyes)
        season_palette = self._get_palette_recommendation(seasonal_palette)

        return AnalysisResult(
            seasonal_palette=seasonal_palette,
            traits=TraitEstimate(
                skin_tone=skin.value,
                eye_color=eyes.value,
                hair_color=hair.value,
            ),
            confidence=0.0,
            palette_recommendation=season_palette,
            notes="Mock response; replace with real CV output when integrated.",
        )

    def classify_from_traits(self, skin_tone: str, hair_color: str, eye_color: str) -> str:
        """
        Convert CV trait labels to enums and return the season name.
        Use this once partner output is available.
        """
        skin = self._to_skin_type(skin_tone)
        hair = self._to_hair_color(hair_color)
        eyes = self._to_eye_color(eye_color)
        return self._classify_season(skin=skin, hair=hair, eyes=eyes)

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
        return SeasonPalette(
            description=palette["description"],
            power_colors=[PaletteColor(**item) for item in palette["power_colors"]],
            neutral_colors=[PaletteColor(**item) for item in palette["neutral_colors"]],
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
