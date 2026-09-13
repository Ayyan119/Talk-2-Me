"""Module: audio_io
Layer: Core/Domain
Purpose: Defines abstract interface contract for audio frame capture and playback.
Dependencies: abc, src.core.types, src.core.exceptions
"""

from abc import ABC, abstractmethod

from src.core.types import AudioChunk, SynthesisChunk


class AudioIO(ABC):
    """Abstract contract for microphone capture and speaker playback I/O."""

    @abstractmethod
    async def read_frame(self) -> AudioChunk:
        """Reads a discrete incoming microphone audio frame.

        Returns:
            An AudioChunk containing raw audio frame bytes.

        Raises:
            AudioProcessingError: If capturing or reading audio frames fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def play_chunk(self, chunk: SynthesisChunk) -> None:
        """Plays a synthesized audio chunk out to the speaker device.

        Args:
            chunk: The SynthesisChunk containing PCM audio bytes to play.

        Raises:
            AudioProcessingError: If audio playback encounters a failure.
        """
        raise NotImplementedError

    @abstractmethod
    async def start(self) -> None:
        """Starts audio I/O hardware streams and initializes capture/playback buffers.

        Raises:
            AudioProcessingError: If hardware stream initialization fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Stops active audio I/O streams and cleans up hardware resources."""
        raise NotImplementedError
