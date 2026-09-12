"""Module: base
Layer: Core/Domain
Purpose: Defines the abstract interface contract for Text-to-Speech (TTS) providers.
Dependencies: abc, collections.abc, src.core.types, src.core.exceptions
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from src.core.types import AudioChunk, SynthesisChunk


class TextToSpeech(ABC):
    """Abstract base contract for text-to-speech synthesis providers."""

    @abstractmethod
    async def synthesize(self, text: str, **kwargs: Any) -> AudioChunk:
        """Synthesizes complete audio from an input text string.

        Args:
            text: The text content to synthesize into speech.
            **kwargs: Provider-specific synthesis options (e.g., voice_id, model_id).

        Returns:
            An AudioChunk containing the complete synthesized PCM audio data.

        Raises:
            SynthesisError: If speech synthesis fails, times out, or receives invalid input.
        """
        raise NotImplementedError

    @abstractmethod
    async def synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[SynthesisChunk]:
        """Streams synthesized audio chunks for an input text segment.

        Args:
            text: The text content to synthesize into a streaming audio response.
            **kwargs: Provider-specific synthesis options (e.g., voice_id, model_id).

        Returns:
            An asynchronous iterator yielding discrete SynthesisChunk objects as audio arrives.

        Raises:
            SynthesisError: If stream connection fails or synthesis encounters an error.
        """
        raise NotImplementedError
