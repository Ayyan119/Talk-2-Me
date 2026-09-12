"""Module: base
Layer: Core/Domain
Purpose: Defines the abstract interface contract for Large Language Model (LLM) providers.
Dependencies: abc, collections.abc, src.core.types, src.core.exceptions
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from src.core.types import Message


class LanguageModel(ABC):
    """Abstract base contract for Large Language Model (LLM) inference providers."""

    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs: Any) -> str:
        """Generates a complete textual response given a sequence of conversation messages.

        Args:
            messages: Ordered list of conversation messages representing chat context.
            **kwargs: Provider-specific inference options (e.g., temperature, max_tokens).

        Returns:
            The complete generated response string from the language model.

        Raises:
            LanguageModelError: If model inference fails, times out, or returns a malformed response.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Streams generated response tokens/chunks given conversation history.

        Args:
            messages: Ordered list of conversation messages representing chat context.
            **kwargs: Provider-specific inference options (e.g., temperature, max_tokens).

        Returns:
            An asynchronous iterator yielding textual delta tokens as they are generated.

        Raises:
            LanguageModelError: If streaming fails or connection is prematurely terminated.
        """
        raise NotImplementedError
