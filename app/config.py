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
    AVAILABLE_MODELS: List[str] = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash", "gemini-2.5-pro", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "claude-3-5-haiku-20241022", "claude-3-5-haiku-latest", "grok-4", "grok-4-latest", "gpt-oss-20b", "gpt-oss-120b"]
    DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "false").lower() == "true"

@lru_cache()
def get_settings() -> Settings:
    return Settings()