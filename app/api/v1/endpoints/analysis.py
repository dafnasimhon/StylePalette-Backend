from fastapi import APIRouter, File, UploadFile

from app.api.deps import AnalysisServiceDep
from app.schemas.analysis import AnalysisResult

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
