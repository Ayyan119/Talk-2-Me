"""Module: faster_whisper_stt
Layer: Adapter/Implementation
Purpose: Speech-to-Text adapter implementation using faster-whisper on CPU.
Dependencies: asyncio, collections.abc, faster_whisper, numpy, src.core.types, src.core.exceptions, src.stt.base
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from src.core.exceptions import TranscriptionError
from src.core.types import AudioChunk, AudioFormat, TranscriptionResult
from src.stt.base import SpeechToText


class FasterWhisperSTT(SpeechToText):
    """Speech-to-Text adapter utilizing faster-whisper optimized for CPU execution."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        audio_format: AudioFormat | None = None,
        language: str | None = None,
        beam_size: int = 1,
        stream_buffer_threshold_seconds: float = 1.0,
        model: Any | None = None,
    ) -> None:
        """Initializes the FasterWhisperSTT provider.

        Args:
            model_size: Whisper model size name (e.g. 'tiny', 'base', 'small').
            device: Computing device ('cpu' or 'cuda').
            compute_type: Quantization precision ('int8', 'float32', etc.).
            audio_format: Expected AudioFormat specifications (default 16000Hz, mono, 16-bit PCM).
            language: Optional language code (e.g., 'en') to force transcription language.
            beam_size: Beam search width (default 1 for greedy fast inference).
            stream_buffer_threshold_seconds: Minimum accumulated duration before emitting streaming updates.
            model: Optional pre-loaded or mock WhisperModel instance.

        Raises:
            TranscriptionError: If model initialization or parameter validation fails.
        """
        if stream_buffer_threshold_seconds <= 0.0:
            raise TranscriptionError("stream_buffer_threshold_seconds must be positive")
        if beam_size < 1:
            raise TranscriptionError("beam_size must be at least 1")

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._stream_buffer_threshold_seconds = stream_buffer_threshold_seconds
        self._audio_format = (
            audio_format
            if audio_format is not None
            else AudioFormat(sample_rate=16000, channels=1, sample_width_bytes=2)
        )

        if self._audio_format.sample_rate != 16000:
            raise TranscriptionError(
                f"Unsupported sample rate: {self._audio_format.sample_rate}Hz. "
                "FasterWhisper requires 16000Hz audio."
            )
        if self._audio_format.channels != 1:
            raise TranscriptionError(
                f"Unsupported channel count: {self._audio_format.channels}. "
                "FasterWhisper requires mono (1 channel) audio."
            )

        if model is not None:
            self._model = model
        else:
            try:
                self._model = WhisperModel(
                    model_size_or_path=self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            except Exception as err:
                raise TranscriptionError(
                    f"Failed to load faster-whisper model '{model_size}': {err}"
                ) from err

    @property
    def model_size(self) -> str:
        """Returns the configured model size."""
        return self._model_size

    @property
    def device(self) -> str:
        """Returns the execution device."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Returns the quantization precision compute type."""
        return self._compute_type

    @property
    def audio_format(self) -> AudioFormat:
        """Returns the configured expected audio format."""
        return self._audio_format

    def _validate_and_convert_chunk(self, audio: AudioChunk) -> np.ndarray:
        """Validates an audio chunk and converts PCM bytes to a float32 numpy array.

        Args:
            audio: The AudioChunk instance to validate and convert.

        Returns:
            1D numpy array of float32 samples normalized in range [-1.0, 1.0].

        Raises:
            TranscriptionError: If audio chunk format or byte payload is invalid.
        """
        if not isinstance(audio, AudioChunk):
            raise TranscriptionError(
                f"Expected AudioChunk instance, got {type(audio).__name__}"
            )
        if audio.sample_rate != self._audio_format.sample_rate:
            raise TranscriptionError(
                f"Sample rate mismatch: expected {self._audio_format.sample_rate}Hz, got {audio.sample_rate}Hz"
            )
        if audio.channels != self._audio_format.channels:
            raise TranscriptionError(
                f"Channel mismatch: expected {self._audio_format.channels}, got {audio.channels}"
            )
        if not audio.data:
            raise TranscriptionError("Audio chunk contains no data")
        if len(audio.data) % self._audio_format.sample_width_bytes != 0:
            raise TranscriptionError(
                "Audio chunk byte length is not aligned with sample width"
            )

        try:
            pcm_samples = np.frombuffer(audio.data, dtype=np.int16)
            return pcm_samples.astype(np.float32) / 32768.0
        except Exception as err:
            raise TranscriptionError(
                f"Failed to parse audio chunk bytes: {err}"
            ) from err

    def _transcribe_samples(
        self, samples: np.ndarray, duration_seconds: float
    ) -> TranscriptionResult:
        """Synchronously transcribes float32 samples using the loaded Whisper model.

        Args:
            samples: 1D float32 numpy array of audio samples.
            duration_seconds: Total duration of the audio segment.

        Returns:
            Populated TranscriptionResult object.

        Raises:
            TranscriptionError: If Whisper inference fails.
        """
        try:
            segments, info = self._model.transcribe(
                samples,
                language=self._language,
                beam_size=self._beam_size,
            )
            text_segments = [segment.text for segment in segments]
            full_text = "".join(text_segments).strip()

            detected_lang = getattr(info, "language", None)
            lang_prob = getattr(info, "language_probability", None)
            confidence = float(lang_prob) if lang_prob is not None else None

            return TranscriptionResult(
                text=full_text,
                language=detected_lang,
                confidence=confidence,
                duration_seconds=duration_seconds,
            )
        except Exception as err:
            raise TranscriptionError(f"Whisper inference failed: {err}") from err

    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """Transcribes a discrete audio chunk into text.

        Converts the raw 16-bit PCM bytes into float32 samples and runs faster-whisper
        inference off the main event loop thread.

        Args:
            audio: The audio data chunk containing raw PCM bytes and sampling metadata.

        Returns:
            TranscriptionResult containing the recognized text transcript and metadata.

        Raises:
            TranscriptionError: If audio validation fails or transcription encounters an error.
        """
        samples = self._validate_and_convert_chunk(audio)
        duration = (
            audio.duration_seconds
            if audio.duration_seconds > 0.0
            else len(samples) / self._audio_format.sample_rate
        )

        return await asyncio.to_thread(self._transcribe_samples, samples, duration)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribes a continuous stream of audio chunks into intermediate transcripts.

        Buffering Strategy:
        Accumulates incoming PCM audio chunks in an internal byte buffer. Whenever the
        accumulated buffer exceeds `stream_buffer_threshold_seconds` (default 1.0s),
        an intermediate transcription snapshot is evaluated on the full accumulated audio
        and yielded. When the input stream closes, a final transcription is yielded
        for the entire accumulated audio segment.

        Args:
            audio_stream: An asynchronous iterator yielding incoming audio chunks.

        Returns:
            An asynchronous iterator yielding intermediate or final transcription results.

        Raises:
            TranscriptionError: If streaming fails or audio data format is invalid.
        """
        accumulated_bytes = bytearray()
        bytes_per_second = (
            self._audio_format.sample_rate
            * self._audio_format.channels
            * self._audio_format.sample_width_bytes
        )
        threshold_bytes = int(self._stream_buffer_threshold_seconds * bytes_per_second)
        last_transcribed_len = 0

        try:
            async for chunk in audio_stream:
                self._validate_and_convert_chunk(chunk)
                accumulated_bytes.extend(chunk.data)

                if (len(accumulated_bytes) - last_transcribed_len) >= threshold_bytes:
                    current_samples = (
                        np.frombuffer(accumulated_bytes, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )
                    duration = len(current_samples) / self._audio_format.sample_rate
                    result = await asyncio.to_thread(
                        self._transcribe_samples, current_samples, duration
                    )
                    last_transcribed_len = len(accumulated_bytes)
                    yield result

            if (
                len(accumulated_bytes) > 0
                and len(accumulated_bytes) != last_transcribed_len
            ):
                final_samples = (
                    np.frombuffer(accumulated_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                duration = len(final_samples) / self._audio_format.sample_rate
                final_result = await asyncio.to_thread(
                    self._transcribe_samples, final_samples, duration
                )
                yield final_result
            elif len(accumulated_bytes) == 0:
                raise TranscriptionError("Stream yielded no audio data")
        except TranscriptionError:
            raise
        except Exception as err:
            raise TranscriptionError(f"Streaming transcription failed: {err}") from err
