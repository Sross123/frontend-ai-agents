# config.py
"""Application configuration and environment validation.

Uses pydantic BaseSettings to load and validate environment variables.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, SecretStr, validator

# Load .env at the project root
load_dotenv()

class Settings(BaseSettings):
    """Settings model for the AI multi‑agent service.

    Environment variables:
        LLM_PROVIDER: currently forced to "gemini" (default).
        GEMINI_API_KEY: required for Gemini access.
    """

    LLM_PROVIDER: Literal["gemini"] = Field("gemini", description="LLM provider – forced to Gemini")
    GEMINI_API_KEY: SecretStr

    @validator("GEMINI_API_KEY", always=True)
    def check_gemini(cls, v):
        if not v:
            raise ValueError("GEMINI_API_KEY must be set for Gemini provider")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """Convenient accessor for a singleton Settings instance."""
    return Settings()
