"""Module: tts
Layer: Core/Domain
Purpose: Text-to-Speech module domain interfaces.
Dependencies: src.tts.base
"""

from src.tts.base import TextToSpeech
from src.tts.elevenlabs_tts import ElevenLabsTTS

__all__ = ["ElevenLabsTTS", "TextToSpeech"]
