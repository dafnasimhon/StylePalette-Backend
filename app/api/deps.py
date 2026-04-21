from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.services.analysis import AnalysisService


def get_analysis_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisService:
    return AnalysisService(settings=settings)


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
