# agents.py
"""Agent model factory for a single Gemini LLM.

Only the Google Gemini model is supported. The function returns two
`ChatOpenAI` instances (identical configuration) for the Product Manager
and Coder agents.
"""

from typing import Tuple

from langchain_openai import ChatOpenAI
from .config import get_settings


def initialize_agent_models() -> Tuple[ChatOpenAI, ChatOpenAI]:
    """Create Gemini LLM clients for both agents.

    The environment must provide ``GEMINI_API_KEY``. If the key is missing,
    a ``ValueError`` will be raised by the settings validator.
    """
    settings = get_settings()
    # Force Gemini provider – other providers are not used.
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
