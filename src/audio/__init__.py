"""Module: audio
Layer: Adapter/Implementation
Purpose: Audio capture and playback hardware adapters.
Dependencies: src.audio.sounddevice_io
"""

from src.audio.sounddevice_io import SoundDeviceAudioIO

__all__ = ["SoundDeviceAudioIO"]
