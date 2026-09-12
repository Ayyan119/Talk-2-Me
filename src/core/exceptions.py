"""Module: exceptions
Layer: Core/Domain
Purpose: Defines domain-level hierarchical exceptions for the voice pipeline.
Dependencies: Standard library only
"""


class TalkToMeDomainError(Exception):
    """Base exception for all domain errors within the application.

    Attributes:
        message: Descriptive explanation of the error condition.
    """

    def __init__(self, message: str) -> None:
        """Initializes the base domain error.

        Args:
            message: Descriptive error message.
        """
        super().__init__(message)
        self.message = message


class ConfigurationError(TalkToMeDomainError):
    """Raised when application or environment configuration is invalid or missing."""


class AudioProcessingError(TalkToMeDomainError):
    """Raised when audio capture, encoding, or format conversion fails."""


class SpeechToTextError(TalkToMeDomainError):
    """Base error for speech-to-text recognition failures."""


class TranscriptionError(SpeechToTextError):
    """Raised when audio transcription processing encounters a failure."""


class LanguageModelError(TalkToMeDomainError):
    """Raised when generating a response from an LLM fails or times out."""


class SynthesisError(TalkToMeDomainError):
    """Raised when text-to-speech audio synthesis fails or streaming drops."""


class ConversationMemoryError(TalkToMeDomainError):
    """Raised when conversation history retrieval, storage, or truncation fails."""


class VoiceActivityDetectionError(TalkToMeDomainError):
    """Raised when voice activity evaluation or state transition encounters an error."""
