# config.py
"""Application configuration and environment validation.

Uses pydantic BaseSettings to load and validate environment variables with security best practices.
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
        LLM_PROVIDER: "gemini" or "openrouter" (default: "gemini").
        GEMINI_API_KEY: Required if provider is gemini.
        OPENROUTER_API_KEY: Required if provider is openrouter.
        CORS_ORIGINS: Comma-separated list of allowed origins (default: "*" for dev).
        LOG_LEVEL: Logging level (default: "INFO").
    """

    LLM_PROVIDER: Literal["gemini", "openrouter"] = Field(
        "gemini",
        description="LLM provider – Gemini or OpenRouter"
    )
    GEMINI_API_KEY: SecretStr | None = None
    OPENROUTER_API_KEY: SecretStr | None = None
    
    # CORS and security
    CORS_ORIGINS: str = Field(
        "*",
        description="Comma-separated CORS origins (default: '*' for dev, restrict in production)"
    )
    
    # Logging
    LOG_LEVEL: str = Field("INFO", description="Logging level")

    @model_validator(mode="after")
    def validate_keys(self) -> "Settings":
        """Validate LLM provider keys."""
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

    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings(
            LLM_PROVIDER=os.getenv("LLM_PROVIDER", "gemini"),
            GEMINI_API_KEY=SecretStr(os.getenv("GEMINI_API_KEY", "")) if os.getenv("GEMINI_API_KEY") else None,
            OPENROUTER_API_KEY=SecretStr(os.getenv("OPENROUTER_API_KEY", "")) if os.getenv("OPENROUTER_API_KEY") else None,
            CORS_ORIGINS=os.getenv("CORS_ORIGINS", "*"),
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        )
    return _settings
