from pydantic import BaseModel, Field, computed_field


class ColorMeasurement(BaseModel):
    """Measured sample from the CV pipeline (single sRGB point)."""

    rgb: list[int] = Field(
        ..., min_length=3, max_length=3, description="RGB 0–255 as [R, G, B]"
    )

    @computed_field
    @property
    def hex(self) -> str:
        """Same color as rgb, lowercase ``#rrggbb``."""
        r, g, b = (int(x) for x in self.rgb)
        return f"#{r:02x}{g:02x}{b:02x}"


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


class RgbRange(BaseModel):
    """Inclusive RGB bounds (0–255) for a recommended palette color."""

    rgb_min: list[int] = Field(
        ..., min_length=3, max_length=3, description="Lower bound [R, G, B]"
    )
    rgb_max: list[int] = Field(
        ..., min_length=3, max_length=3, description="Upper bound [R, G, B]"
    )


class PaletteColor(BaseModel):
    rgb_range: RgbRange = Field(
        ...,
        description="Suggested wearable inclusive RGB bounds",
    )


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


class TraitCorrectionRequest(BaseModel):
    """User-confirmed trait labels after selfie analysis (must match server enum labels)."""

    skin_tone: str = Field(
        ...,
        description='One of: "Very Fair", "Fair", "Medium/Tan", "Dark"',
    )
    eye_color: str = Field(..., description='One of: "Brown", "Blue", "Green"')
    hair_color: str = Field(
        ...,
        description='One of: "Black", "Brown", "Blonde", "Red/Ginger"',
    )
    skin_sample: ColorMeasurement | None = Field(
        None, description="Optional measured RGB from the selfie to echo in the response"
    )
    eye_sample: ColorMeasurement | None = None
    hair_sample: ColorMeasurement | None = None
