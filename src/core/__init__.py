"""Module: core
Layer: Core/Domain
Purpose: Core domain models, contracts, and shared exception definitions.
Dependencies: Standard library only
"""

from src.core.exceptions import (
    AudioProcessingError,
    ConfigurationError,
    ConversationMemoryError,
    LanguageModelError,
    SpeechToTextError,
    SynthesisError,
    TalkToMeDomainError,
    TranscriptionError,
    VoiceActivityDetectionError,
)
from src.core.types import (
    AudioChunk,
    AudioFormat,
    Message,
    MessageRole,
    SynthesisChunk,
    TranscriptionResult,
    VADResult,
)

__all__ = [
    "AudioChunk",
    "AudioFormat",
    "AudioProcessingError",
    "ConfigurationError",
    "ConversationMemoryError",
    "LanguageModelError",
    "Message",
    "MessageRole",
    "SpeechToTextError",
    "SynthesisChunk",
    "SynthesisError",
    "TalkToMeDomainError",
    "TranscriptionError",
    "TranscriptionResult",
    "VADResult",
    "VoiceActivityDetectionError",
]
