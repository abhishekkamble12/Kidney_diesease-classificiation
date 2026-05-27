import os
from pydantic_settings import BaseSettings

from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./monitoring.db"
    GROQ_API_KEY: str = ""
    HF_API_TOKEN: str = ""
    HF_TOKEN: str = ""
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
