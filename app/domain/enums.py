from enum import Enum


class SkinType(Enum):
    VERY_FAIR = "Very Fair"
    FAIR = "Fair"
    MEDIUM_TAN = "Medium/Tan"
    DARK = "Dark"


class HairColor(Enum):
    BLACK = "Black"
    BROWN = "Brown"
    BLONDE = "Blonde"
    RED_GINGER = "Red/Ginger"


class EyeColor(Enum):
    BROWN = "Brown"
    BLUE = "Blue"
    GREEN = "Green"


class Season(Enum):
    WINTER = "Winter"
    SUMMER = "Summer"
    AUTUMN = "Autumn"
    SPRING = "Spring"
