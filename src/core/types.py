"""Module: types
Layer: Core/Domain
Purpose: Domain entities, data structures, and value objects for the voice pipeline.
Dependencies: Standard library only (dataclasses, enum, time, typing)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Enumeration of message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """Represents a single message in a conversation.

    Attributes:
        role: The sender role (system, user, or assistant).
        content: The textual content of the message.
        timestamp: Unix epoch timestamp in seconds when the message was created.
        metadata: Optional dictionary for extra contextual data.
    """

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioFormat:
    """Represents audio specifications.

    Attributes:
        sample_rate: Sampling frequency in Hertz (e.g., 16000).
        channels: Number of audio channels (1 for mono, 2 for stereo).
        sample_width_bytes: Number of bytes per sample (e.g., 2 for 16-bit PCM).
    """

    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2


@dataclass(frozen=True)
class AudioChunk:
    """Represents a discrete raw audio buffer chunk.

    Attributes:
        data: Raw PCM audio bytes.
        sample_rate: Sampling rate in Hz.
        channels: Channel count (1 = mono, 2 = stereo).
        duration_seconds: Duration of the audio slice in seconds.
    """

    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class TranscriptionResult:
    """Represents the output of a speech-to-text operation.

    Attributes:
        text: The recognized text transcript.
        language: Detected or configured language code (e.g., 'en').
        confidence: Optional confidence score between 0.0 and 1.0.
        duration_seconds: Duration of the transcribed audio segment.
    """

    text: str
    language: str | None = None
    confidence: float | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class SynthesisChunk:
    """Represents a chunk of synthesized speech audio in a streaming response.

    Attributes:
        audio: Raw PCM audio bytes for this chunk.
        is_final: True if this is the terminating chunk of the synthesis stream.
        sample_rate: Sampling rate of the synthesized audio in Hz.
    """

    audio: bytes
    is_final: bool = False
    sample_rate: int = 24000


@dataclass(frozen=True)
class VADResult:
    """Represents the evaluation output of a voice activity detector.

    Attributes:
        is_speech: True if speech was detected in the processed chunk.
        confidence: Speech probability score between 0.0 and 1.0.
        is_end_of_speech: True if a transition from speech to silence exceeded silence threshold.
    """

    is_speech: bool
    confidence: float = 0.0
    is_end_of_speech: bool = False
