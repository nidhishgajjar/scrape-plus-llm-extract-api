import os
from functools import lru_cache
from typing import List

class Settings:
    PROJECT_NAME: str = "Web Scraper API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    BACKEND_CORS_ORIGINS: list = ["*"]
    PORT: int = 8007
    HOST: str = "0.0.0.0"
    AVAILABLE_MODELS: List[str] = ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp"]
    DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "false").lower() == "true"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 