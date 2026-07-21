from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MedBrief AI Backend"
    app_env: str = "development"
    database_url: str = "sqlite:///./medbrief.db"
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = Field(default=512, ge=1, le=4096)
    ocr_lang: str = "en"
    nnunet_command: str | None = None
    nnunet_results_dir: Path | None = None
    mri_model_name: str | None = None
    mri_min_confidence: float = Field(default=0.5, ge=0, le=1)
    clinical_intel_enabled: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
