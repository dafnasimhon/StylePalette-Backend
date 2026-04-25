from app.domain.enums import EyeColor, HairColor, Season, SkinType


class SeasonAnalyzer:
    def __init__(self, skin: SkinType, hair: HairColor, eyes: EyeColor):
        self.skin = skin
        self.hair = hair
        self.eyes = eyes

    def test_temperature(self) -> str:
        """Test 1: warm (+) vs cool (-)."""
        warmth_map = {
            HairColor.RED_GINGER: 4,
            HairColor.BLONDE: 2,
            HairColor.BROWN: -1,
            HairColor.BLACK: -3,
            SkinType.VERY_FAIR: -3,
            SkinType.FAIR: -1,
            SkinType.MEDIUM_TAN: 2,
            SkinType.DARK: 1,
            EyeColor.BLUE: -2,
            EyeColor.GREEN: 1,
            EyeColor.BROWN: 1,
        }
        score = (
            warmth_map.get(self.skin, 0)
            + warmth_map.get(self.hair, 0)
            + warmth_map.get(self.eyes, 0)
        )
        return "Warm" if score > 0 else "Cool"

    def test_contrast(self) -> str:
        """Test 2: high vs low contrast."""
        if self.skin in [SkinType.VERY_FAIR, SkinType.FAIR] and self.hair in [
            HairColor.BLACK,
            HairColor.BROWN,
        ]:
            return "High"
        if self.skin == SkinType.DARK and self.hair == HairColor.BLONDE:
            return "High"
        return "Low"

    def test_chroma_and_value(self) -> dict[str, str]:
        """Test 3: chroma and value."""
        is_clear = self.eyes in [EyeColor.BLUE, EyeColor.GREEN] or self.hair in [
            HairColor.BLACK,
            HairColor.RED_GINGER,
        ]
        is_light = self.skin in [SkinType.VERY_FAIR, SkinType.FAIR] or self.hair == HairColor.BLONDE
        return {
            "chroma": "Clear" if is_clear else "Muted",
            "value": "Light" if is_light else "Deep",
        }

    def get_final_season(self) -> Season:
        temp = self.test_temperature()
        contrast = self.test_contrast()
        cv = self.test_chroma_and_value()

        if temp == "Cool":
            if contrast == "High" or cv["chroma"] == "Clear":
                return Season.WINTER
            return Season.SUMMER
        if cv["value"] == "Light" and cv["chroma"] == "Clear":
            return Season.SPRING
        return Season.AUTUMN
