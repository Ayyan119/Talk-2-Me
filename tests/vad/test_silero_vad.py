"""Module: test_silero_vad
Layer: Testing
Purpose: Comprehensive unit test suite for SileroVoiceActivityDetector.
Dependencies: pytest, unittest.mock, numpy, torch, src.core.types, src.core.exceptions, src.vad.silero_vad
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.core.exceptions import VoiceActivityDetectionError
from src.core.types import AudioChunk, AudioFormat
from src.vad.silero_vad import SileroVoiceActivityDetector


def _create_pcm_chunk(
    sample_count: int = 512,
    sample_rate: int = 16000,
    amplitude: float = 0.0,
    frequency: float = 440.0,
    duration_seconds: float = 0.0,
) -> AudioChunk:
    """Helper to create synthetic AudioChunk instances."""
    if amplitude == 0.0:
        pcm_array = np.zeros(sample_count, dtype=np.int16)
    else:
        time_axis = np.arange(sample_count) / sample_rate
        waveform = amplitude * np.sin(2 * np.pi * frequency * time_axis)
        pcm_array = (waveform * 32767).astype(np.int16)

    data_bytes = pcm_array.tobytes()
    calculated_duration = (
        duration_seconds if duration_seconds > 0.0 else sample_count / sample_rate
    )

    return AudioChunk(
        data=data_bytes,
        sample_rate=sample_rate,
        channels=1,
        duration_seconds=calculated_duration,
    )


class TestSileroVoiceActivityDetector:
    """Unit test suite for the Silero VAD adapter."""

    def test_model_loads_successfully_on_init(self) -> None:
        """Verifies that the pre-trained Silero VAD model loads without errors."""
        detector = SileroVoiceActivityDetector(
            silence_threshold_ms=600.0, speech_threshold=0.5
        )
        assert detector.silence_threshold_ms == 600.0
        assert detector.speech_threshold == 0.5
        assert detector.audio_format.sample_rate == 16000

    def test_model_load_failure_raises_domain_exception(self) -> None:
        """Verifies VoiceActivityDetectionError is raised when model loading fails."""
        with (
            patch(
                "src.vad.silero_vad.load_silero_vad",
                side_effect=RuntimeError("Model file corrupted"),
            ),
            pytest.raises(
                VoiceActivityDetectionError, match="Failed to load Silero VAD model"
            ),
        ):
            SileroVoiceActivityDetector()

    def test_invalid_constructor_parameters(self) -> None:
        """Verifies error handling for invalid initialization arguments."""
        with pytest.raises(
            VoiceActivityDetectionError, match="silence_threshold_ms must be greater"
        ):
            SileroVoiceActivityDetector(silence_threshold_ms=0.0)

        with pytest.raises(
            VoiceActivityDetectionError, match="speech_threshold must be between"
        ):
            SileroVoiceActivityDetector(speech_threshold=1.5)

        with pytest.raises(
            VoiceActivityDetectionError, match="Unsupported sample rate"
        ):
            SileroVoiceActivityDetector(
                audio_format=AudioFormat(sample_rate=44100, channels=1)
            )

        with pytest.raises(
            VoiceActivityDetectionError, match="Unsupported channel count"
        ):
            SileroVoiceActivityDetector(
                audio_format=AudioFormat(sample_rate=16000, channels=2)
            )

    def test_is_speech_detection_with_mock(self) -> None:
        """Verifies is_speech returns True for speech-like signal and False for silence."""
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.85])  # High speech probability

        detector = SileroVoiceActivityDetector(speech_threshold=0.5, model=mock_model)
        speech_chunk = _create_pcm_chunk(
            sample_count=512, amplitude=0.5, frequency=300.0
        )

        assert detector.is_speech(speech_chunk) is True

        # Now simulate silence output from model
        mock_model.return_value = torch.tensor([0.02])
        silence_chunk = _create_pcm_chunk(sample_count=512, amplitude=0.0)

        assert detector.is_speech(silence_chunk) is False

    def test_is_end_of_turn_triggers_only_after_threshold(self) -> None:
        """Verifies end of turn triggers only when accumulated silence >= threshold."""
        mock_model = MagicMock()
        detector = SileroVoiceActivityDetector(
            silence_threshold_ms=600.0,
            speech_threshold=0.5,
            model=mock_model,
        )

        # 1. Silence before any speech -> should NOT trigger end of turn
        mock_model.return_value = torch.tensor([0.05])
        initial_silence = _create_pcm_chunk(
            sample_count=3200, duration_seconds=0.2
        )  # 200ms
        assert detector.is_end_of_turn(initial_silence) is False

        # 2. User starts speaking
        mock_model.return_value = torch.tensor([0.90])
        speech_chunk = _create_pcm_chunk(
            sample_count=3200, amplitude=0.5, duration_seconds=0.2
        )  # 200ms
        result_speech = detector.process_chunk(speech_chunk)
        assert result_speech.is_speech is True
        assert result_speech.is_end_of_speech is False

        # 3. User pauses: 1st silence chunk (200ms total, threshold=600ms)
        mock_model.return_value = torch.tensor([0.05])
        silence_chunk_1 = _create_pcm_chunk(sample_count=3200, duration_seconds=0.2)
        assert detector.is_end_of_turn(silence_chunk_1) is False

        # 4. 2nd silence chunk (400ms total)
        silence_chunk_2 = _create_pcm_chunk(sample_count=3200, duration_seconds=0.2)
        assert detector.is_end_of_turn(silence_chunk_2) is False

        # 5. 3rd silence chunk (600ms total -> threshold reached!)
        silence_chunk_3 = _create_pcm_chunk(sample_count=3200, duration_seconds=0.2)
        assert detector.is_end_of_turn(silence_chunk_3) is True

    def test_reset_clears_silence_and_speech_state(self) -> None:
        """Verifies that reset() clears all internal state and silence tracking."""
        mock_model = MagicMock()
        mock_model.reset_states = MagicMock()
        detector = SileroVoiceActivityDetector(
            silence_threshold_ms=600.0,
            speech_threshold=0.5,
            model=mock_model,
        )

        # Trigger speech
        mock_model.return_value = torch.tensor([0.90])
        speech_chunk = _create_pcm_chunk(sample_count=3200, duration_seconds=0.2)
        detector.process_chunk(speech_chunk)
        assert detector._has_speech_started is True

        # Accumulate some silence (400ms)
        mock_model.return_value = torch.tensor([0.05])
        detector.process_chunk(
            _create_pcm_chunk(sample_count=6400, duration_seconds=0.4)
        )
        assert detector._consecutive_silence_ms == 400.0

        # Reset
        detector.reset()
        assert detector._has_speech_started is False
        assert detector._consecutive_silence_ms == 0.0
        mock_model.reset_states.assert_called_once()

        # Silence after reset should not trigger end of turn
        assert (
            detector.is_end_of_turn(
                _create_pcm_chunk(sample_count=6400, duration_seconds=0.4)
            )
            is False
        )

    def test_chunk_validation_raises_error(self) -> None:
        """Verifies error handling for mismatched or invalid chunk inputs."""
        detector = SileroVoiceActivityDetector(model=MagicMock())

        # Invalid type
        with pytest.raises(VoiceActivityDetectionError, match="Expected AudioChunk"):
            detector.process_chunk("invalid_audio")  # type: ignore[arg-type]

        # Sample rate mismatch
        with pytest.raises(VoiceActivityDetectionError, match="Sample rate mismatch"):
            detector.process_chunk(
                AudioChunk(data=b"\x00" * 1024, sample_rate=8000, channels=1)
            )

        # Channel mismatch
        with pytest.raises(VoiceActivityDetectionError, match="Channel mismatch"):
            detector.process_chunk(
                AudioChunk(data=b"\x00" * 1024, sample_rate=16000, channels=2)
            )

        # Empty data
        with pytest.raises(
            VoiceActivityDetectionError, match="Audio chunk contains no data"
        ):
            detector.process_chunk(AudioChunk(data=b"", sample_rate=16000, channels=1))

        # Unaligned byte length (odd number of bytes for 16-bit PCM)
        with pytest.raises(
            VoiceActivityDetectionError, match="not aligned with sample width"
        ):
            detector.process_chunk(
                AudioChunk(data=b"\x00" * 513, sample_rate=16000, channels=1)
            )

    def test_real_model_inference_with_silence(self) -> None:
        """Verifies real Silero model inference produces near-zero probability for silence."""
        detector = SileroVoiceActivityDetector(speech_threshold=0.5)
        silence_chunk = _create_pcm_chunk(sample_count=512, amplitude=0.0)

        result = detector.process_chunk(silence_chunk)
        assert result.is_speech is False
        assert result.confidence < 0.1
        assert result.is_end_of_speech is False
