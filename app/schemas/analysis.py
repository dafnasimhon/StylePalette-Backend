from pydantic import BaseModel, Field


class TraitEstimate(BaseModel):
    skin_tone: str | None = Field(None, description="Estimated skin tone category")
    eye_color: str | None = Field(None, description="Estimated eye color")
    hair_color: str | None = Field(None, description="Estimated hair color")


class AnalysisResult(BaseModel):
    seasonal_palette: str = Field(
        ...,
        description="Seasonal color palette label (e.g. Winter, Spring, Summer, Autumn)",
    )
    traits: TraitEstimate = Field(default_factory=TraitEstimate)
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Model confidence if available"
    )
    notes: str | None = Field(
        None,
        description="Optional message when analysis is mocked or degraded",
    )
