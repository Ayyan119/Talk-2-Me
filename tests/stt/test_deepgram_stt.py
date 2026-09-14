"""Module: test_deepgram_stt
Layer: Testing
Purpose: Comprehensive unit test suite for DeepgramSTT real-time streaming and REST adapter.
Dependencies: pytest, unittest.mock, httpx, src.core.types, src.core.exceptions, src.stt.deepgram_stt
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.exceptions import ConfigurationError, TranscriptionError
from src.core.types import AudioChunk, TranscriptionResult
from src.stt.deepgram_stt import DeepgramSTT


class TestDeepgramSTT:
    """Unit test suite for Deepgram STT adapter."""

    def test_missing_api_key_raises_configuration_error(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigurationError, match="Deepgram API key is missing"),
        ):
            DeepgramSTT(api_key="")

    def test_invalid_timeout_raises_error(self) -> None:
        with pytest.raises(
            TranscriptionError, match="timeout_seconds must be positive"
        ):
            DeepgramSTT(api_key="dg-test", timeout_seconds=-1.0)

    @pytest.mark.asyncio
    async def test_transcribe_successful(self) -> None:
        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello, how can I help you today?",
                                "confidence": 0.98,
                            }
                        ]
                    }
                ]
            }
        }
        mock_http.post = AsyncMock(return_value=mock_response)

        stt = DeepgramSTT(
            api_key="dg-test",
            http_client=mock_http,
        )

        chunk = AudioChunk(
            data=b"\x00\x00" * 8000,
            sample_rate=16000,
            channels=1,
            duration_seconds=0.5,
        )

        result = await stt.transcribe(chunk)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, how can I help you today?"
        assert result.confidence == 0.98
        assert result.is_final is True

    @pytest.mark.asyncio
    async def test_transcribe_http_error_raises_transcription_error(self) -> None:
        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401 Unauthorized", request=MagicMock(), response=MagicMock()
            )
        )

        stt = DeepgramSTT(
            api_key="dg-test",
            http_client=mock_http,
        )

        chunk = AudioChunk(
            data=b"\x00\x00" * 8000,
            sample_rate=16000,
            channels=1,
            duration_seconds=0.5,
        )

        with pytest.raises(TranscriptionError, match="Deepgram transcription failed"):
            await stt.transcribe(chunk)

    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_results(self) -> None:
        stt = DeepgramSTT(api_key="dg-test")

        sample_frames = [
            json.dumps(
                {
                    "type": "Results",
                    "is_final": False,
                    "speech_final": False,
                    "channel": {
                        "alternatives": [{"transcript": "hello", "confidence": 0.9}]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "Results",
                    "is_final": True,
                    "speech_final": True,
                    "channel": {
                        "alternatives": [
                            {"transcript": "hello world", "confidence": 0.99}
                        ]
                    },
                }
            ),
        ]

        class MockWS:
            def __init__(self) -> None:
                self.sent: list[Any] = []

            async def send(self, data: Any) -> None:
                self.sent.append(data)

            async def __aiter__(self):
                for frame in sample_frames:
                    yield frame

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        async def dummy_audio_stream():
            yield AudioChunk(
                data=b"\x00\x00" * 1600,
                sample_rate=16000,
                channels=1,
                duration_seconds=0.1,
            )

        with patch("websockets.connect", return_value=MockWS()):
            results = []
            async for res in stt.transcribe_stream(dummy_audio_stream()):
                results.append(res)

            assert len(results) == 2
            assert results[0].text == "hello"
            assert results[0].is_final is False
            assert results[1].text == "hello world"
            assert results[1].is_final is True
            assert results[1].speech_final is True
