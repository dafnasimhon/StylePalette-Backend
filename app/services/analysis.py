from fastapi import UploadFile

from app.config import Settings, get_settings
from app.schemas.analysis import AnalysisResult, TraitEstimate


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
        return AnalysisResult(
            seasonal_palette="Winter",
            traits=TraitEstimate(
                skin_tone="cool",
                eye_color="brown",
                hair_color="dark",
            ),
            confidence=0.0,
            notes="Mock response; replace with real CV output when integrated.",
        )
