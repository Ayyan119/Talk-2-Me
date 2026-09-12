"""Module: llm
Layer: Core/Domain
Purpose: Large Language Model module domain interfaces.
Dependencies: src.llm.base
"""

from src.llm.base import LanguageModel
from src.llm.openai_llm import OpenAILLM

__all__ = ["LanguageModel", "OpenAILLM"]
