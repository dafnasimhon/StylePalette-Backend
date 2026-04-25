from app.domain.enums import Season


class StylePalettes:
    PALETTES = {
        Season.WINTER: {
            "description": "Bold, icy, and high-contrast colors with a blue base.",
            "power_colors": [
                {"name": "Royal Blue", "hex": "#4169E1"},
                {"name": "Wine Red", "hex": "#800020"},
                {"name": "Deep Purple", "hex": "#4B0082"},
                {"name": "Emerald Green", "hex": "#50C878"},
            ],
            "neutral_colors": [
                {"name": "Black", "hex": "#000000"},
                {"name": "Pure White", "hex": "#FFFFFF"},
                {"name": "Dark Denim", "hex": "#151B54"},
                {"name": "Charcoal Gray", "hex": "#36454F"},
            ],
        },
        Season.SUMMER: {
            "description": "Cool, soft, and muted pastel tones with a blue base.",
            "power_colors": [
                {"name": "Lavender", "hex": "#E6E6FA"},
                {"name": "Dusty Rose", "hex": "#DCAE96"},
                {"name": "Sky Blue", "hex": "#87CEEB"},
                {"name": "Mint Green", "hex": "#98FF98"},
            ],
            "neutral_colors": [
                {"name": "Light Gray", "hex": "#D3D3D3"},
                {"name": "Navy Blue", "hex": "#000080"},
                {"name": "Light Denim", "hex": "#5E7D9A"},
                {"name": "Off White", "hex": "#FAF9F6"},
            ],
        },
        Season.AUTUMN: {
            "description": "Rich, earthy, and deep colors with a gold base.",
            "power_colors": [
                {"name": "Mustard Yellow", "hex": "#FFDB58"},
                {"name": "Olive Green", "hex": "#8A9A5B"},
                {"name": "Rust Orange", "hex": "#E2725B"},
                {"name": "Forest Green", "hex": "#228B22"},
            ],
            "neutral_colors": [
                {"name": "Dark Brown", "hex": "#4B3621"},
                {"name": "Camel", "hex": "#C19A6B"},
                {"name": "Khaki", "hex": "#C3B091"},
                {"name": "Dark Denim", "hex": "#151B54"},
            ],
        },
        Season.SPRING: {
            "description": "Bright, warm, and clear colors with a yellow base.",
            "power_colors": [
                {"name": "Bright Red", "hex": "#FF2400"},
                {"name": "Coral", "hex": "#FF7F50"},
                {"name": "Peach", "hex": "#FFE5B4"},
                {"name": "Light Green", "hex": "#90EE90"},
            ],
            "neutral_colors": [
                {"name": "Ivory", "hex": "#FFFFF0"},
                {"name": "Sand", "hex": "#C2B280"},
                {"name": "Light Brown", "hex": "#D2B48C"},
                {"name": "Gold", "hex": "#D4AF37"},
            ],
        },
    }

    @classmethod
    def get_palette(cls, season: Season) -> dict:
        return cls.PALETTES[season]
