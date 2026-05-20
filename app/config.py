from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    mediapipe_allow_model_download: bool = True
    #: Optional absolute path to face_landmarker.task (env MEDIAPIPE_FACE_LANDMARKER_MODEL).
    mediapipe_face_landmarker_model: str = ""
    #: Optional override download URL (env MEDIAPIPE_FACE_LANDMARKER_URL).
    mediapipe_face_landmarker_url: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
