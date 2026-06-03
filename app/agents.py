# agents.py
"""Agent model factory supporting Gemini and OpenRouter LLMs.

Returns two `ChatOpenAI` instances (individually configured for temperature and model)
for the Product Manager and Coder agents depending on settings.
"""

from typing import Tuple

from langchain_openai import ChatOpenAI
from .config import get_settings


def initialize_agent_models() -> Tuple[ChatOpenAI, ChatOpenAI]:
    """Create LLM clients for both agents based on LLM_PROVIDER settings."""
    settings = get_settings()
    provider = settings.LLM_PROVIDER

    if provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY.get_secret_value()
        base_url = "https://openrouter.ai/api/v1"
        pm_model = ChatOpenAI(
            model="anthropic/claude-3-haiku",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.2,
            max_tokens=4096,  # Cap max tokens to bypass excessive pre-billing credit checks
        )
        coder_model = ChatOpenAI(
            model="qwen/qwen3-coder-next",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1,
            max_tokens=8192,  # Cap to bypass pre-billing credit checks while leaving ample space for code
        )
    else:  # gemini
        api_key = settings.GEMINI_API_KEY.get_secret_value()
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        pm_model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.2,
        )
        coder_model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1,
        )

    return pm_model, coder_model
