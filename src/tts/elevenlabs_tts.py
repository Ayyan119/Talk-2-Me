"""Module: elevenlabs_tts
Layer: Adapter/Implementation
Purpose: Concrete Text-to-Speech adapter implementing ElevenLabs low-latency voice synthesis.
Dependencies: asyncio, collections.abc, os, dotenv, elevenlabs, httpx, src.core.types, src.core.exceptions, src.tts.base
"""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from elevenlabs.core.api_error import ApiError

from src.core.exceptions import ConfigurationError, SynthesisError
from src.core.types import AudioChunk, SynthesisChunk
from src.tts.base import TextToSpeech

load_dotenv()


class ElevenLabsTTS(TextToSpeech):
    """ElevenLabs Text-to-Speech synthesis adapter with streaming and retry policies."""

    def __init__(
        self,
        voice_id: str,
        api_key: str | None = None,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "pcm_24000",
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        initial_retry_delay_seconds: float = 0.5,
        client: AsyncElevenLabs | None = None,
    ) -> None:
        """Initializes the ElevenLabs TTS adapter."""
        resolved_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ConfigurationError(
                "ElevenLabs API key is missing. Set ELEVENLABS_API_KEY environment variable or pass api_key."
            )

        if not voice_id or not voice_id.strip():
            raise SynthesisError("voice_id is required and cannot be empty")

        if timeout_seconds <= 0.0:
            raise SynthesisError("timeout_seconds must be positive")
        if max_retries < 0:
            raise SynthesisError("max_retries cannot be negative")

        self._voice_id = voice_id.strip()
        self._model_id = model_id
        self._output_format = output_format
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._initial_retry_delay = initial_retry_delay_seconds
        self._sample_rate = self._parse_sample_rate(output_format)

        if client is not None:
            self._client = client
        else:
            self._client = AsyncElevenLabs(
                api_key=resolved_key,
                timeout=self._timeout_seconds,
            )

    @property
    def voice_id(self) -> str:
        """Returns the configured voice identifier."""
        return self._voice_id

    @property
    def model_id(self) -> str:
        """Returns the configured model identifier."""
        return self._model_id

    @property
    def output_format(self) -> str:
        """Returns the audio output format."""
        return self._output_format

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate in Hertz."""
        return self._sample_rate

    def _parse_sample_rate(self, output_format: str) -> int:
        """Extracts the sample rate frequency in Hz from the format string."""
        for part in output_format.split("_"):
            if part.isdigit() and int(part) in (
                8000,
                16000,
                22050,
                24000,
                32000,
                44100,
                48000,
            ):
                return int(part)
        return 24000

    def _is_transient_error(self, err: Exception) -> bool:
        """Determines whether an exception is transient and eligible for retry."""
        if isinstance(err, ApiError):
            return err.status_code in (429, 500, 502, 503, 504)
        return isinstance(
            err,
            (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
                asyncio.TimeoutError,
            ),
        )

    def _is_fatal_error(self, err: Exception) -> bool:
        """Determines whether an exception is permanent and should fail immediately without retry."""
        if isinstance(err, ApiError) and err.status_code in (
            400,
            401,
            403,
            404,
        ):
            return True
        return isinstance(err, (ValueError, TypeError))

    async def synthesize(self, text: str, **kwargs: Any) -> AudioChunk:
        """Synthesizes complete audio from an input text string."""
        if not text or not text.strip():
            raise SynthesisError("Input text for synthesis cannot be empty")

        voice_id = kwargs.get("voice_id", self._voice_id)
        model_id = kwargs.get("model_id", self._model_id)
        output_format = kwargs.get("output_format", self._output_format)
        sample_rate = self._parse_sample_rate(output_format)

        attempt = 0
        delay = self._initial_retry_delay

        while True:
            attempt += 1
            try:
                raw_bytes_list: list[bytes] = []
                voice_settings = kwargs.get(
                    "voice_settings",
                    VoiceSettings(
                        stability=0.50,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )
                stream_res = self._client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id,
                    output_format=output_format,
                    voice_settings=voice_settings,
                    optimize_streaming_latency=4,
                )
                if asyncio.iscoroutine(stream_res):
                    audio_stream = await asyncio.wait_for(
                        stream_res, timeout=self._timeout_seconds
                    )
                else:
                    audio_stream = stream_res

                async for chunk in audio_stream:
                    if chunk:
                        raw_bytes_list.append(chunk)

                complete_audio = b"".join(raw_bytes_list)
                if not complete_audio:
                    raise SynthesisError("ElevenLabs returned empty audio payload")

                sample_width_bytes = 2
                duration_seconds = len(complete_audio) / (
                    sample_rate * sample_width_bytes
                )

                return AudioChunk(
                    data=complete_audio,
                    sample_rate=sample_rate,
                    channels=1,
                    duration_seconds=duration_seconds,
                )

            except Exception as err:
                if self._is_fatal_error(err):
                    raise SynthesisError(
                        f"ElevenLabs synthesis client error: {err}"
                    ) from err

                if self._is_transient_error(err) and attempt <= self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                    continue

                raise SynthesisError(
                    f"ElevenLabs synthesis failed after {attempt} attempt(s): {err}"
                ) from err

    async def synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[SynthesisChunk]:
        """Streams synthesized audio chunks for an input text segment."""
        if not text or not text.strip():
            raise SynthesisError("Input text for synthesis stream cannot be empty")

        voice_id = kwargs.get("voice_id", self._voice_id)
        model_id = kwargs.get("model_id", self._model_id)
        output_format = kwargs.get("output_format", self._output_format)
        sample_rate = self._parse_sample_rate(output_format)

        attempt = 0
        delay = self._initial_retry_delay
        audio_stream = None

        while True:
            attempt += 1
            try:
                voice_settings = kwargs.get(
                    "voice_settings",
                    VoiceSettings(
                        stability=0.50,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )
                stream_res = self._client.text_to_speech.stream(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id,
                    output_format=output_format,
                    voice_settings=voice_settings,
                    optimize_streaming_latency=4,
                )
                if asyncio.iscoroutine(stream_res):
                    audio_stream = await asyncio.wait_for(
                        stream_res, timeout=self._timeout_seconds
                    )
                else:
                    audio_stream = stream_res
                break

            except Exception as err:
                if self._is_fatal_error(err):
                    raise SynthesisError(
                        f"ElevenLabs stream connection client error: {err}"
                    ) from err

                if self._is_transient_error(err) and attempt <= self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                    continue

                raise SynthesisError(
                    f"ElevenLabs stream initiation failed after {attempt} attempt(s): {err}"
                ) from err

        pending_chunk: bytes | None = None
        try:
            async for chunk in audio_stream:
                if not chunk:
                    continue

                if pending_chunk is not None:
                    yield SynthesisChunk(
                        audio=pending_chunk,
                        is_final=False,
                        sample_rate=sample_rate,
                    )
                pending_chunk = chunk

            if pending_chunk is not None:
                yield SynthesisChunk(
                    audio=pending_chunk,
                    is_final=True,
                    sample_rate=sample_rate,
                )
            else:
                raise SynthesisError("ElevenLabs stream returned no audio data")

        except Exception as err:
            if isinstance(err, SynthesisError):
                raise
            raise SynthesisError(f"ElevenLabs stream reading failed: {err}") from err
