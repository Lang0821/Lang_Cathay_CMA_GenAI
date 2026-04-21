from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

load_dotenv(override=True)


def get_llm() -> Optional["ChatGoogleGenerativeAI"]:
    if ChatGoogleGenerativeAI is None:
        return None

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=2,
    )
