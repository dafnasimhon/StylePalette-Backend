from pydantic import BaseModel, Field


class ColorMeasurement(BaseModel):
    """Measured sample (hex + RGB) from the CV pipeline."""

    hex: str = Field(..., description="CSS-style hex, e.g. #aabbcc")
    rgb: list[int] = Field(
        ..., min_length=3, max_length=3, description="RGB 0–255 as [R, G, B]"
    )


class TraitEstimate(BaseModel):
    skin_tone: str | None = Field(None, description="Estimated skin tone category")
    eye_color: str | None = Field(None, description="Estimated eye color")
    hair_color: str | None = Field(None, description="Estimated hair color")
    skin_sample: ColorMeasurement | None = Field(
        None, description="Raw measured skin color from the image"
    )
    eye_sample: ColorMeasurement | None = Field(
        None, description="Raw measured eye region color from the image"
    )
    hair_sample: ColorMeasurement | None = Field(
        None, description="Raw measured hair color from the image"
    )


class PaletteColor(BaseModel):
    name: str
    hex: str


class SeasonPalette(BaseModel):
    description: str
    power_colors: list[PaletteColor] = Field(default_factory=list)
    neutral_colors: list[PaletteColor] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    seasonal_palette: str = Field(
        ...,
        description="Seasonal color palette label (e.g. Winter, Spring, Summer, Autumn)",
    )
    traits: TraitEstimate = Field(default_factory=TraitEstimate)
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Model confidence if available"
    )
    palette_recommendation: SeasonPalette
    notes: str | None = Field(
        None,
        description="Optional message with analysis context or notes",
    )
