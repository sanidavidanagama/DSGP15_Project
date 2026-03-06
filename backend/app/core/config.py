from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    DATABASE_URL: str
    ALLOWED_ORIGINS: str = ""
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str
    ST_EMBED_MODEL: Optional[str] = None
    RAG_TOP_K: Optional[int] = None
    TF_ENABLE_ONEDNN_OPTS: Optional[int] = None
    PROCESSED_IMAGE_DIR: str = "backend/uploads/processed/"

    @field_validator("ALLOWED_ORIGINS")
    def parse_allowed_origins(cls, v: str) -> List[str]:
        return [origin.strip() for origin in v.split(',')] if v else []

    class Config:
        env_file = '.env'
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()
