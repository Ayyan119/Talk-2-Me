"""Module: base
Layer: Core/Domain
Purpose: Defines the abstract interface contract for Voice Activity Detection (VAD).
Dependencies: abc, src.core.types, src.core.exceptions
"""

from abc import ABC, abstractmethod

from src.core.types import AudioChunk, VADResult


class VoiceActivityDetector(ABC):
    """Abstract base contract for real-time Voice Activity Detection (VAD)."""

    @abstractmethod
    def process_chunk(self, audio: AudioChunk) -> VADResult:
        """Processes an audio chunk and evaluates speech probability and turn boundaries.

        Args:
            audio: An AudioChunk containing raw audio frame bytes and metadata.

        Returns:
            VADResult containing speech probability, presence flag, and silence boundary info.

        Raises:
            VoiceActivityDetectionError: If chunk processing fails or audio format is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def is_speech(self, audio: AudioChunk) -> bool:
        """Determines whether a given audio chunk contains active speech.

        Args:
            audio: An AudioChunk containing raw audio frame bytes.

        Returns:
            True if voice activity is detected in the chunk, False otherwise.

        Raises:
            VoiceActivityDetectionError: If evaluation encounters an error.
        """
        raise NotImplementedError

    @abstractmethod
    def is_end_of_turn(self, audio: AudioChunk) -> bool:
        """Evaluates whether the speaker has finished speaking based on silence duration.

        Args:
            audio: The current incoming AudioChunk frame.

        Returns:
            True if a complete silence duration threshold is reached after active speech.

        Raises:
            VoiceActivityDetectionError: If state tracking encounters an error.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal detector state, buffers, and silence accumulators.

        Raises:
            VoiceActivityDetectionError: If state clearing encounters an error.
        """
        raise NotImplementedError
