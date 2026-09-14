"""Module: deepgram_stt
Layer: Adapter/Implementation
Purpose: Real-time streaming Speech-to-Text adapter implementing Deepgram WebSocket and REST APIs with interim transcripts, VAD, and endpointing.
Dependencies: asyncio, json, logging, os, urllib.parse, dotenv, httpx, websockets, src.core.types, src.core.exceptions, src.stt.base
"""

import asyncio
import json
import logging
import os
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

import httpx
from dotenv import load_dotenv

from src.core.exceptions import ConfigurationError, TranscriptionError
from src.core.types import AudioChunk, AudioFormat, TranscriptionResult
from src.stt.base import SpeechToText

load_dotenv()
logger = logging.getLogger("talk_to_me.stt.deepgram")


class DeepgramSTT(SpeechToText):
    """Real-time streaming and discrete Speech-to-Text adapter powered by Deepgram (Nova-2 / Nova-3)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-2",
        language: str = "en",
        smart_format: bool = True,
        punctuate: bool = True,
        interim_results: bool = True,
        endpointing_ms: int = 250,
        vad_events: bool = True,
        timeout_seconds: float = 15.0,
        audio_format: AudioFormat | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initializes the Deepgram STT adapter."""
        resolved_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if (
            not resolved_key
            or not resolved_key.strip()
            or resolved_key == "your-deepgram-api-key-here"
        ):
            raise ConfigurationError(
                "Deepgram API key is missing. Set DEEPGRAM_API_KEY environment variable or pass api_key."
            )

        if timeout_seconds <= 0.0:
            raise TranscriptionError("timeout_seconds must be positive")

        self._api_key = resolved_key.strip()
        self._model = model
        self._language = language
        self._smart_format = smart_format
        self._punctuate = punctuate
        self._interim_results = interim_results
        self._endpointing_ms = endpointing_ms
        self._vad_events = vad_events
        self._timeout_seconds = timeout_seconds
        self._audio_format = (
            audio_format
            if audio_format is not None
            else AudioFormat(sample_rate=16000, channels=1, sample_width_bytes=2)
        )

        if http_client is not None:
            self._http_client = http_client
        else:
            try:
                self._http_client = httpx.AsyncClient(
                    timeout=self._timeout_seconds,
                    http2=True,
                )
            except (ImportError, ValueError):
                self._http_client = httpx.AsyncClient(
                    timeout=self._timeout_seconds,
                )

    @property
    def model(self) -> str:
        """Returns the configured model name."""
        return self._model

    @property
    def language(self) -> str:
        """Returns the configured language."""
        return self._language

    @property
    def audio_format(self) -> AudioFormat:
        """Returns the expected audio format."""
        return self._audio_format

    def _validate_chunk(self, audio: AudioChunk) -> None:
        """Validates incoming audio chunk specifications."""
        if not isinstance(audio, AudioChunk):
            raise TranscriptionError(
                f"Expected AudioChunk instance, got {type(audio).__name__}"
            )
        if not audio.data:
            raise TranscriptionError("Audio chunk contains no data")
        if len(audio.data) % self._audio_format.sample_width_bytes != 0:
            raise TranscriptionError(
                "Audio chunk byte length is not aligned with sample width"
            )

    def _get_ws_url(self) -> str:
        """Constructs the Deepgram WebSocket streaming endpoint URL with query parameters."""
        params = {
            "model": self._model,
            "language": self._language,
            "smart_format": "true" if self._smart_format else "false",
            "punctuate": "true" if self._punctuate else "false",
            "interim_results": "true" if self._interim_results else "false",
            "vad_events": "true" if self._vad_events else "false",
            "encoding": "linear16",
            "sample_rate": str(self._audio_format.sample_rate),
            "channels": str(self._audio_format.channels),
        }
        if self._endpointing_ms > 0:
            params["endpointing"] = str(self._endpointing_ms)

        query_str = urllib.parse.urlencode(params)
        return f"wss://api.deepgram.com/v1/listen?{query_str}"

    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """Transcribes a discrete audio chunk into text using Deepgram REST API."""
        self._validate_chunk(audio)
        sample_rate = audio.sample_rate or self._audio_format.sample_rate
        channels = audio.channels or self._audio_format.channels
        duration = (
            audio.duration_seconds
            if audio.duration_seconds > 0.0
            else len(audio.data) / (sample_rate * channels * 2)
        )

        params: dict[str, Any] = {
            "model": self._model,
            "smart_format": "true" if self._smart_format else "false",
            "punctuate": "true" if self._punctuate else "false",
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": str(channels),
        }
        if self._language:
            params["language"] = self._language

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/raw",
        }

        url = f"https://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"
        try:
            response = await self._http_client.post(
                url,
                headers=headers,
                content=audio.data,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            channels_res = data.get("results", {}).get("channels", [])
            transcript = ""
            confidence = 1.0
            if channels_res:
                alts = channels_res[0].get("alternatives", [])
                if alts:
                    transcript = alts[0].get("transcript", "").strip()
                    confidence = float(alts[0].get("confidence", 1.0))

            return TranscriptionResult(
                text=transcript,
                language=self._language,
                confidence=confidence,
                duration_seconds=duration,
                is_final=True,
                speech_final=True,
            )
        except Exception as err:
            logger.error("Deepgram REST transcription failed: %s", err)
            raise TranscriptionError(f"Deepgram transcription failed: {err}") from err

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribes a continuous stream of audio chunks in real-time over Deepgram WebSocket."""
        import websockets

        ws_url = self._get_ws_url()
        headers = {"Authorization": f"Token {self._api_key}"}

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=10,
                ping_timeout=10,
            ) as ws:
                async def sender() -> None:
                    try:
                        async for chunk in audio_stream:
                            self._validate_chunk(chunk)
                            await ws.send(chunk.data)
                        await ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception as send_err:
                        logger.debug("Sender error or stream end: %s", send_err)

                send_task = asyncio.create_task(sender())

                try:
                    async for raw_msg in ws:
                        if isinstance(raw_msg, str):
                            data = json.loads(raw_msg)
                            msg_type = data.get("type", "")

                            if msg_type == "Results":
                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [])
                                if alternatives:
                                    alt = alternatives[0]
                                    text = alt.get("transcript", "").strip()
                                    is_final = bool(data.get("is_final", False))
                                    speech_final = bool(data.get("speech_final", False))
                                    confidence = float(alt.get("confidence", 0.0))

                                    if text:
                                        yield TranscriptionResult(
                                            text=text,
                                            language=self._language,
                                            confidence=confidence,
                                            is_final=is_final,
                                            speech_final=speech_final,
                                        )
                            elif msg_type == "Metadata":
                                logger.debug("Deepgram session metadata: %s", data)
                            elif msg_type == "SpeechStarted":
                                logger.debug("Deepgram VAD: Speech started")
                            elif msg_type == "UtteranceEnd":
                                logger.debug("Deepgram VAD: Utterance ended")
                finally:
                    if not send_task.done():
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass

        except Exception as err:
            logger.error("Deepgram streaming transcription error: %s", err)
            raise TranscriptionError(f"Deepgram stream error: {err}") from err
