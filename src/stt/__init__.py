"""Module: stt
Layer: Core/Domain
Purpose: Speech-to-Text module domain interfaces.
Dependencies: src.stt.base
"""

from src.stt.base import SpeechToText
from src.stt.faster_whisper_stt import FasterWhisperSTT

__all__ = ["FasterWhisperSTT", "SpeechToText"]
