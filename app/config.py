# config.py
"""Application configuration and environment validation.

Uses pydantic BaseModel to load and validate environment variables.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator

# Load .env at the project root
load_dotenv()

class Settings(BaseModel):
    """Settings model for the AI multi‑agent service.

    Environment variables:
        LLM_PROVIDER: "gemini" or "openrouter".
        GEMINI_API_KEY: required if provider is gemini.
        OPENROUTER_API_KEY: required if provider is openrouter.
    """

    LLM_PROVIDER: Literal["gemini", "openrouter"] = Field("gemini", description="LLM provider – Gemini or OpenRouter")
    GEMINI_API_KEY: SecretStr | None = None
    OPENROUTER_API_KEY: SecretStr | None = None

    @model_validator(mode="after")
    def validate_keys(self) -> "Settings":
        provider = self.LLM_PROVIDER
        
        if provider == "gemini":
            key = self.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
            if not key or not str(key).strip():
                raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER='gemini'")
            if not isinstance(key, SecretStr):
                key = SecretStr(str(key))
            self.GEMINI_API_KEY = key
            
        elif provider == "openrouter":
            key = self.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
            if not key or not str(key).strip():
                raise ValueError("OPENROUTER_API_KEY must be set when LLM_PROVIDER='openrouter'")
            if not isinstance(key, SecretStr):
                key = SecretStr(str(key))
            self.OPENROUTER_API_KEY = key
            
        return self

def get_settings() -> Settings:
    """Convenient accessor for a singleton Settings instance."""
    # Instantiated using environment variables
    return Settings(
        LLM_PROVIDER=os.getenv("LLM_PROVIDER", "gemini"),
        GEMINI_API_KEY=SecretStr(os.getenv("GEMINI_API_KEY", "")) if os.getenv("GEMINI_API_KEY") else None,
        OPENROUTER_API_KEY=SecretStr(os.getenv("OPENROUTER_API_KEY", "")) if os.getenv("OPENROUTER_API_KEY") else None
    )
