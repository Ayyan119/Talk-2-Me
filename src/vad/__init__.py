"""Module: vad
Layer: Core/Domain
Purpose: Voice Activity Detection module domain interfaces.
Dependencies: src.vad.base
"""

from src.vad.base import VoiceActivityDetector
from src.vad.silero_vad import SileroVoiceActivityDetector

__all__ = ["SileroVoiceActivityDetector", "VoiceActivityDetector"]
