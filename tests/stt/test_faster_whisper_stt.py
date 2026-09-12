"""Module: test_faster_whisper_stt
Layer: Testing
Purpose: Comprehensive unit test suite for FasterWhisperSTT adapter.
Dependencies: asyncio, time, unittest.mock, numpy, pytest, src.core.types, src.core.exceptions, src.stt.faster_whisper_stt
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.exceptions import TranscriptionError
from src.core.types import AudioChunk, AudioFormat, TranscriptionResult
from src.stt.faster_whisper_stt import FasterWhisperSTT


@dataclass
class DummySegment:
    """Mock Whisper segment representing recognized text slice."""

    text: str


@dataclass
class DummyTranscriptionInfo:
    """Mock Whisper transcription info metadata."""

    language: str = "en"
    language_probability: float = 0.98


def _generate_synthetic_pcm_chunk(
    sample_count: int = 16000,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency: float = 440.0,
    amplitude: float = 0.2,
) -> AudioChunk:
    """Generates a synthetic sine wave PCM AudioChunk for testing."""
    time_axis = np.arange(sample_count) / sample_rate
    waveform = amplitude * np.sin(2 * np.pi * frequency * time_axis)
    pcm_bytes = (waveform * 32767).astype(np.int16).tobytes()

    return AudioChunk(
        data=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=sample_count / sample_rate,
    )


class TestFasterWhisperSTT:
    """Unit test suite for the FasterWhisperSTT adapter."""

    def test_model_loading_and_attributes(self) -> None:
        """Verifies model loading speed and property initialization using tiny model."""
        start_time = time.perf_counter()
        stt = FasterWhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
        load_time = time.perf_counter() - start_time

        assert stt.model_size == "tiny"
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
        assert stt.audio_format.sample_rate == 16000
        assert load_time < 10.0  # Cached load is sub-second

    def test_model_load_failure_raises_domain_exception(self) -> None:
        """Verifies TranscriptionError is raised if WhisperModel instantiation fails."""
        with (
            patch(
                "src.stt.faster_whisper_stt.WhisperModel",
                side_effect=RuntimeError("Model download failed"),
            ),
            pytest.raises(
                TranscriptionError, match="Failed to load faster-whisper model"
            ),
        ):
            FasterWhisperSTT(model_size="invalid_model")

    def test_invalid_constructor_parameters(self) -> None:
        """Verifies validation errors for invalid constructor arguments."""
        with pytest.raises(
            TranscriptionError, match="stream_buffer_threshold_seconds must be positive"
        ):
            FasterWhisperSTT(
                model_size="tiny",
                stream_buffer_threshold_seconds=0.0,
                model=MagicMock(),
            )

        with pytest.raises(TranscriptionError, match="beam_size must be at least 1"):
            FasterWhisperSTT(model_size="tiny", beam_size=0, model=MagicMock())

        with pytest.raises(TranscriptionError, match="Unsupported sample rate"):
            FasterWhisperSTT(
                model_size="tiny",
                audio_format=AudioFormat(sample_rate=8000),
                model=MagicMock(),
            )

        with pytest.raises(TranscriptionError, match="Unsupported channel count"):
            FasterWhisperSTT(
                model_size="tiny",
                audio_format=AudioFormat(channels=2),
                model=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_transcribe_successful_with_mock(self) -> None:
        """Verifies transcribe populates all expected fields in TranscriptionResult."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [DummySegment("Hello"), DummySegment(" world!")],
            DummyTranscriptionInfo(language="en", language_probability=0.97),
        )

        stt = FasterWhisperSTT(model_size="tiny", model=mock_model)
        chunk = _generate_synthetic_pcm_chunk(sample_count=16000)

        result = await stt.transcribe(chunk)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world!"
        assert result.language == "en"
        assert result.confidence == pytest.approx(0.97)
        assert result.duration_seconds == pytest.approx(1.0)
        mock_model.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_mismatched_sample_rate_raises_error(self) -> None:
        """Verifies wrong sample rate raises TranscriptionError."""
        stt = FasterWhisperSTT(model_size="tiny", model=MagicMock())
        mismatched_chunk = AudioChunk(data=b"\x00" * 1024, sample_rate=8000, channels=1)

        with pytest.raises(TranscriptionError, match="Sample rate mismatch"):
            await stt.transcribe(mismatched_chunk)

    @pytest.mark.asyncio
    async def test_transcribe_mismatched_channels_raises_error(self) -> None:
        """Verifies multi-channel audio chunk raises TranscriptionError."""
        stt = FasterWhisperSTT(model_size="tiny", model=MagicMock())
        multi_channel_chunk = AudioChunk(
            data=b"\x00" * 1024, sample_rate=16000, channels=2
        )

        with pytest.raises(TranscriptionError, match="Channel mismatch"):
            await stt.transcribe(multi_channel_chunk)

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_raises_error(self) -> None:
        """Verifies empty audio chunk data raises TranscriptionError."""
        stt = FasterWhisperSTT(model_size="tiny", model=MagicMock())
        empty_chunk = AudioChunk(data=b"", sample_rate=16000, channels=1)

        with pytest.raises(TranscriptionError, match="Audio chunk contains no data"):
            await stt.transcribe(empty_chunk)

    @pytest.mark.asyncio
    async def test_transcribe_unaligned_bytes_raises_error(self) -> None:
        """Verifies odd number of bytes for 16-bit PCM raises TranscriptionError."""
        stt = FasterWhisperSTT(model_size="tiny", model=MagicMock())
        unaligned_chunk = AudioChunk(data=b"\x00" * 513, sample_rate=16000, channels=1)

        with pytest.raises(TranscriptionError, match="not aligned with sample width"):
            await stt.transcribe(unaligned_chunk)

    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_results(self) -> None:
        """Verifies transcribe_stream yields intermediate and final transcription results."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            (
                [DummySegment("Partial transcript")],
                DummyTranscriptionInfo(language="en", language_probability=0.95),
            ),
            (
                [DummySegment("Partial transcript finished.")],
                DummyTranscriptionInfo(language="en", language_probability=0.98),
            ),
        ]

        stt = FasterWhisperSTT(
            model_size="tiny",
            stream_buffer_threshold_seconds=0.5,
            model=mock_model,
        )

        # Stream 4 chunks of 0.25s each (total 1.0s)
        async def audio_stream() -> AsyncIterator[AudioChunk]:
            for _ in range(4):
                yield _generate_synthetic_pcm_chunk(sample_count=4000)

        results: list[TranscriptionResult] = []
        async for res in stt.transcribe_stream(audio_stream()):
            results.append(res)

        assert len(results) >= 1
        assert "Partial transcript" in results[0].text
        assert results[-1].text == "Partial transcript finished."
        assert results[-1].duration_seconds == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_transcribe_stream_empty_raises_error(self) -> None:
        """Verifies empty stream raises TranscriptionError."""
        stt = FasterWhisperSTT(model_size="tiny", model=MagicMock())

        async def empty_stream() -> AsyncIterator[AudioChunk]:
            if False:
                yield _generate_synthetic_pcm_chunk()

        with pytest.raises(TranscriptionError, match="Stream yielded no audio data"):
            async for _ in stt.transcribe_stream(empty_stream()):
                pass

    @pytest.mark.asyncio
    async def test_real_model_inference_benchmark_cpu(self) -> None:
        """Runs actual inference using real tiny model on CPU and measures latency."""
        stt = FasterWhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
        chunk = _generate_synthetic_pcm_chunk(sample_count=16000)  # 1.0 second audio

        start_time = time.perf_counter()
        result = await stt.transcribe(chunk)
        elapsed_inference_time = time.perf_counter() - start_time

        assert isinstance(result, TranscriptionResult)
        assert result.duration_seconds == pytest.approx(1.0)
        assert elapsed_inference_time < 3.0  # Real CPU inference should be fast
