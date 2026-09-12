"""Module: silero_vad
Layer: Adapter/Implementation
Purpose: Concrete Silero VAD implementation for voice activity and turn boundary detection.
Dependencies: numpy, torch, silero_vad, src.core.types, src.core.exceptions, src.vad.base
"""

from typing import Any

import numpy as np
import torch
from silero_vad import load_silero_vad

from src.core.exceptions import VoiceActivityDetectionError
from src.core.types import AudioChunk, AudioFormat, VADResult
from src.vad.base import VoiceActivityDetector


class SileroVoiceActivityDetector(VoiceActivityDetector):
    """Voice Activity Detector implementation using the pre-trained Silero VAD model."""

    def __init__(
        self,
        silence_threshold_ms: float = 600.0,
        speech_threshold: float = 0.5,
        audio_format: AudioFormat | None = None,
        model: Any | None = None,
    ) -> None:
        """Initializes the Silero VAD detector.

        Args:
            silence_threshold_ms: Duration in ms of consecutive silence required to trigger end of turn.
            speech_threshold: Probability threshold (0.0 to 1.0) above which audio is marked as speech.
            audio_format: Expected AudioFormat specifications (must be 8000 or 16000 Hz, mono).
            model: Optional pre-loaded or mock Silero model instance.

        Raises:
            VoiceActivityDetectionError: If parameters are invalid or model loading fails.
        """
        resolved_format = (
            audio_format
            if audio_format is not None
            else AudioFormat(sample_rate=16000, channels=1, sample_width_bytes=2)
        )

        if silence_threshold_ms <= 0:
            raise VoiceActivityDetectionError(
                "silence_threshold_ms must be greater than zero"
            )
        if not (0.0 <= speech_threshold <= 1.0):
            raise VoiceActivityDetectionError(
                "speech_threshold must be between 0.0 and 1.0"
            )
        if resolved_format.sample_rate not in (8000, 16000):
            raise VoiceActivityDetectionError(
                f"Unsupported sample rate: {resolved_format.sample_rate}. Silero VAD supports 8000 and 16000 Hz."
            )
        if resolved_format.channels != 1:
            raise VoiceActivityDetectionError(
                f"Unsupported channel count: {resolved_format.channels}. Silero VAD only supports mono (1 channel)."
            )

        self._silence_threshold_ms = silence_threshold_ms
        self._speech_threshold = speech_threshold
        self._audio_format = resolved_format
        self._window_size = 512 if resolved_format.sample_rate == 16000 else 256

        self._has_speech_started: bool = False
        self._consecutive_silence_ms: float = 0.0

        if model is not None:
            self._model = model
        else:
            try:
                self._model = load_silero_vad()
            except Exception as err:
                raise VoiceActivityDetectionError(
                    f"Failed to load Silero VAD model: {err}"
                ) from err

    @property
    def silence_threshold_ms(self) -> float:
        """Returns the configured silence threshold in milliseconds."""
        return self._silence_threshold_ms

    @property
    def speech_threshold(self) -> float:
        """Returns the configured speech probability threshold."""
        return self._speech_threshold

    @property
    def audio_format(self) -> AudioFormat:
        """Returns the configured audio format."""
        return self._audio_format

    def _validate_chunk(self, audio: AudioChunk) -> None:
        """Validates incoming audio chunk specifications.

        Args:
            audio: AudioChunk to validate.

        Raises:
            VoiceActivityDetectionError: If audio attributes or data are invalid.
        """
        if not isinstance(audio, AudioChunk):
            raise VoiceActivityDetectionError(
                f"Expected AudioChunk, got {type(audio).__name__}"
            )
        if audio.sample_rate != self._audio_format.sample_rate:
            raise VoiceActivityDetectionError(
                f"Sample rate mismatch: expected {self._audio_format.sample_rate}Hz, got {audio.sample_rate}Hz"
            )
        if audio.channels != self._audio_format.channels:
            raise VoiceActivityDetectionError(
                f"Channel mismatch: expected {self._audio_format.channels}, got {audio.channels}"
            )
        if not audio.data:
            raise VoiceActivityDetectionError("Audio chunk contains no data")
        if len(audio.data) % self._audio_format.sample_width_bytes != 0:
            raise VoiceActivityDetectionError(
                "Audio chunk byte length is not aligned with sample width"
            )

    def _evaluate_probability(self, audio: AudioChunk) -> float:
        """Evaluates speech probability from raw audio bytes using the model.

        Args:
            audio: Validated AudioChunk instance.

        Returns:
            Calculated speech probability between 0.0 and 1.0.

        Raises:
            VoiceActivityDetectionError: If tensor conversion or model inference fails.
        """
        try:
            samples = (
                np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            tensor = torch.from_numpy(samples)

            with torch.no_grad():
                if len(tensor) == self._window_size:
                    prob = float(
                        self._model(tensor, self._audio_format.sample_rate).item()
                    )
                elif len(tensor) < self._window_size:
                    padded = torch.zeros(self._window_size, dtype=torch.float32)
                    padded[: len(tensor)] = tensor
                    prob = float(
                        self._model(padded, self._audio_format.sample_rate).item()
                    )
                else:
                    probs: list[float] = []
                    for i in range(
                        0, len(tensor) - self._window_size + 1, self._window_size
                    ):
                        chunk = tensor[i : i + self._window_size]
                        prob_val = float(
                            self._model(chunk, self._audio_format.sample_rate).item()
                        )
                        probs.append(prob_val)
                    prob = max(probs) if probs else 0.0

            return max(0.0, min(1.0, prob))
        except Exception as err:
            raise VoiceActivityDetectionError(
                f"Model evaluation failed: {err}"
            ) from err

    def process_chunk(self, audio: AudioChunk) -> VADResult:
        """Processes an audio chunk and evaluates speech probability and turn boundaries.

        Args:
            audio: An AudioChunk containing raw audio frame bytes and metadata.

        Returns:
            VADResult containing speech probability, presence flag, and silence boundary info.

        Raises:
            VoiceActivityDetectionError: If chunk processing fails or audio format is invalid.
        """
        self._validate_chunk(audio)
        speech_prob = self._evaluate_probability(audio)
        is_speech_present = speech_prob >= self._speech_threshold

        if audio.duration_seconds > 0.0:
            chunk_duration_ms = audio.duration_seconds * 1000.0
        else:
            num_samples = len(audio.data) // self._audio_format.sample_width_bytes
            chunk_duration_ms = (num_samples / self._audio_format.sample_rate) * 1000.0

        is_end = False
        if is_speech_present:
            self._has_speech_started = True
            self._consecutive_silence_ms = 0.0
        elif self._has_speech_started:
            self._consecutive_silence_ms += chunk_duration_ms
            if self._consecutive_silence_ms >= self._silence_threshold_ms:
                is_end = True

        return VADResult(
            is_speech=is_speech_present,
            confidence=speech_prob,
            is_end_of_speech=is_end,
        )

    def is_speech(self, audio: AudioChunk) -> bool:
        """Determines whether a given audio chunk contains active speech.

        Args:
            audio: An AudioChunk containing raw audio frame bytes.

        Returns:
            True if voice activity is detected in the chunk, False otherwise.

        Raises:
            VoiceActivityDetectionError: If evaluation encounters an error.
        """
        return self.process_chunk(audio).is_speech

    def is_end_of_turn(self, audio: AudioChunk) -> bool:
        """Evaluates whether the speaker has finished speaking based on silence duration.

        Args:
            audio: The current incoming AudioChunk frame.

        Returns:
            True if a complete silence duration threshold is reached after active speech.

        Raises:
            VoiceActivityDetectionError: If state tracking encounters an error.
        """
        return self.process_chunk(audio).is_end_of_speech

    def reset(self) -> None:
        """Resets the internal detector state, buffers, and silence accumulators.

        Raises:
            VoiceActivityDetectionError: If state clearing encounters an error.
        """
        try:
            self._has_speech_started = False
            self._consecutive_silence_ms = 0.0
            if hasattr(self._model, "reset_states"):
                self._model.reset_states()
        except Exception as err:
            raise VoiceActivityDetectionError(
                f"Failed to reset VAD state: {err}"
            ) from err
