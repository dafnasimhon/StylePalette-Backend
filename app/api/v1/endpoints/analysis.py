from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import AnalysisServiceDep
from app.schemas.analysis import AnalysisResult, TraitCorrectionRequest

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post(
    "/selfie",
    response_model=AnalysisResult,
    summary="Analyze a selfie for seasonal palette",
    description=(
        "Accepts an image from the Android client. "
        "When the CV engine is integrated, returns real trait estimates and palette."
    ),
)
async def analyze_selfie(
    service: AnalysisServiceDep,
    image: UploadFile = File(..., description="Front-facing selfie (JPEG/PNG)"),
) -> AnalysisResult:
    return await service.analyze_selfie(image)


@router.post(
    "/palette-from-traits",
    response_model=AnalysisResult,
    summary="Palette from user-confirmed traits",
    description=(
        "After /analysis/selfie, the client may correct skin tone, eye color, and hair color. "
        "Send the chosen labels (same strings as the enum values) and optional measured RGB "
        "samples to receive the seasonal palette recommendation."
    ),
)
def palette_from_traits(
    service: AnalysisServiceDep,
    body: TraitCorrectionRequest,
) -> AnalysisResult:
    try:
        return service.build_palette_from_traits(body)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
