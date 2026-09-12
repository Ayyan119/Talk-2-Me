"""Module: base
Layer: Core/Domain
Purpose: Defines the abstract interface contract for Speech-to-Text (STT) providers.
Dependencies: abc, collections.abc, src.core.types, src.core.exceptions
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.core.types import AudioChunk, TranscriptionResult


class SpeechToText(ABC):
    """Abstract base contract for speech-to-text transcription providers."""

    @abstractmethod
    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """Transcribes a discrete audio chunk into text.

        Args:
            audio: The audio data chunk containing raw PCM bytes and sampling metadata.

        Returns:
            TranscriptionResult containing the recognized text transcript and metadata.

        Raises:
            TranscriptionError: If the transcription process fails or times out.
        """
        raise NotImplementedError

    @abstractmethod
    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribes a continuous stream of audio chunks into intermediate transcripts.

        Args:
            audio_stream: An asynchronous iterator yielding incoming audio chunks.

        Returns:
            An asynchronous iterator yielding intermediate or final transcription results.

        Raises:
            TranscriptionError: If the streaming transcription encounters an unrecoverable failure.
        """
        raise NotImplementedError
