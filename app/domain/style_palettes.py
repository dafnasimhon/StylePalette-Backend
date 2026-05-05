from app.domain.enums import Season


def _box(r: int, g: int, b: int, pad: int = 36) -> dict:
    """Inclusive RGB range around a center sRGB point (no hex in API data)."""
    return {
        "rgb_range": {
            "rgb_min": [max(0, r - pad), max(0, g - pad), max(0, b - pad)],
            "rgb_max": [min(255, r + pad), min(255, g + pad), min(255, b + pad)],
        }
    }


class StylePalettes:
    PALETTES = {
        Season.WINTER: {
            "description": "Bold, icy, and high-contrast colors with a blue base.",
            "power_colors": [
                _box(65, 105, 225),
                _box(128, 0, 32),
                _box(75, 0, 130),
                _box(80, 200, 120),
            ],
            "neutral_colors": [
                _box(0, 0, 0),
                _box(255, 255, 255),
                _box(21, 27, 84),
                _box(54, 69, 79),
            ],
        },
        Season.SUMMER: {
            "description": "Cool, soft, and muted pastel tones with a blue base.",
            "power_colors": [
                _box(230, 230, 250),
                _box(220, 174, 150),
                _box(135, 206, 235),
                _box(152, 255, 152),
            ],
            "neutral_colors": [
                _box(211, 211, 211),
                _box(0, 0, 128),
                _box(94, 125, 154),
                _box(250, 249, 246),
            ],
        },
        Season.AUTUMN: {
            "description": "Rich, earthy, and deep colors with a gold base.",
            "power_colors": [
                _box(255, 219, 88),
                _box(138, 154, 91),
                _box(226, 114, 91),
                _box(34, 139, 34),
            ],
            "neutral_colors": [
                _box(75, 54, 33),
                _box(193, 154, 107),
                _box(195, 176, 145),
                _box(21, 27, 84),
            ],
        },
        Season.SPRING: {
            "description": "Bright, warm, and clear colors with a yellow base.",
            "power_colors": [
                _box(255, 36, 0),
                _box(255, 127, 80),
                _box(255, 229, 180),
                _box(144, 238, 144),
            ],
            "neutral_colors": [
                _box(255, 255, 240),
                _box(194, 178, 128),
                _box(210, 180, 140),
                _box(212, 175, 55),
            ],
        },
    }

    @classmethod
    def get_palette(cls, season: Season) -> dict:
        return cls.PALETTES[season]
