from app.domain.enums import Season


def _rgb_range(
    r_min: int,
    r_max: int,
    g_min: int,
    g_max: int,
    b_min: int,
    b_max: int,
) -> dict:
    """Inclusive RGB range. API returns rgb_min and rgb_max only."""
    return {
        "rgb_range": {
            "rgb_min": [
                max(0, r_min),
                max(0, g_min),
                max(0, b_min),
            ],
            "rgb_max": [
                min(255, r_max),
                min(255, g_max),
                min(255, b_max),
            ],
        }
    }


class StylePalettes:
    PALETTES = {
        Season.WINTER: {
            "description": "Bold, icy, and high-contrast colors with a blue base.",
            "power_colors": [
                _rgb_range(20, 80, 30, 120, 120, 255),  # Royal / deep blue
                _rgb_range(100, 160, 0, 45, 20, 90),  # Burgundy
                _rgb_range(55, 110, 0, 70, 110, 210),  # Indigo / purple
                _rgb_range(0, 90, 150, 230, 80, 180),  # Emerald
            ],
            "neutral_colors": [
                _rgb_range(0, 30, 0, 30, 0, 30),  # Black
                _rgb_range(245, 255, 245, 255, 245, 255),  # White
                _rgb_range(10, 45, 15, 60, 70, 140),  # Navy
                _rgb_range(35, 85, 50, 100, 65, 125),  # Cool charcoal
            ],
        },
        Season.SUMMER: {
            "description": "Cool, soft, and muted pastel tones with a blue base.",
            "power_colors": [
                _rgb_range(200, 255, 200, 255, 225, 255),  # Lavender
                _rgb_range(190, 240, 145, 205, 130, 190),  # Dusty rose
                _rgb_range(115, 180, 175, 230, 200, 255),  # Soft sky blue
                _rgb_range(130, 200, 220, 255, 130, 210),  # Soft mint
            ],
            "neutral_colors": [
                _rgb_range(190, 230, 190, 230, 190, 230),  # Light gray
                _rgb_range(0, 55, 0, 65, 90, 165),  # Soft navy
                _rgb_range(75, 125, 100, 155, 125, 185),  # Muted blue-gray
                _rgb_range(240, 255, 240, 255, 235, 255),  # Off white
            ],
        },
        Season.AUTUMN: {
            "description": "Rich, earthy, and deep colors with a gold base.",
            "power_colors": [
                _rgb_range(220, 255, 165, 235, 35, 120),  # Mustard
                _rgb_range(95, 165, 110, 180, 50, 120),  # Olive
                _rgb_range(190, 255, 70, 145, 45, 125),  # Burnt orange
                _rgb_range(0, 90, 95, 180, 0, 90),  # Forest green
            ],
            "neutral_colors": [
                _rgb_range(55, 110, 35, 85, 15, 65),  # Brown
                _rgb_range(170, 225, 130, 185, 75, 145),  # Tan
                _rgb_range(170, 215, 150, 195, 120, 175),  # Beige
                _rgb_range(10, 45, 15, 60, 70, 140),  # Dark navy
            ],
        },
        Season.SPRING: {
            "description": "Bright, warm, and clear colors with a yellow base.",
            "power_colors": [
                _rgb_range(235, 255, 20, 90, 0, 70),  # Bright red-orange
                _rgb_range(235, 255, 95, 170, 50, 130),  # Coral
                _rgb_range(245, 255, 195, 245, 140, 215),  # Peach
                _rgb_range(110, 210, 210, 255, 100, 190),  # Fresh green
            ],
            "neutral_colors": [
                _rgb_range(245, 255, 245, 255, 225, 255),  # Cream
                _rgb_range(175, 215, 150, 205, 90, 155),  # Light tan
                _rgb_range(195, 235, 165, 215, 110, 175),  # Warm beige
                _rgb_range(195, 235, 165, 215, 35, 90),  # Gold
            ],
        },
    }

    @classmethod
    def get_palette(cls, season: Season) -> dict:
        return cls.PALETTES[season]
