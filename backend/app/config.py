from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Anime Taste DNA API"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    model_artifact_path: Path = (
        Path(__file__).resolve().parents[2]
        / "data/artifacts/signed-rating-hybrid-v1/signed_rating_hybrid.npz"
    )
    minimum_mapped_ratings: int = 5
    pairwise_preference_weight: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
