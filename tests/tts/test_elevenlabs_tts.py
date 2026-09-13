"""Module: test_elevenlabs_tts
Layer: Testing
Purpose: Unit test suite for ElevenLabsTTS adapter using mocked AsyncElevenLabs client.
Dependencies: asyncio, unittest.mock, pytest, elevenlabs, src.core.types, src.core.exceptions, src.tts.elevenlabs_tts
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from elevenlabs.core.api_error import ApiError

from src.core.exceptions import ConfigurationError, SynthesisError
from src.core.types import AudioChunk, SynthesisChunk
from src.tts.elevenlabs_tts import ElevenLabsTTS


class MockAsyncByteStream:
    """Async iterator simulating ElevenLabs streaming audio bytes."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Initializes the mock stream with a sequence of byte chunks."""
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self) -> "MockAsyncByteStream":
        """Returns the async iterator instance."""
        return self

    async def __anext__(self) -> bytes:
        """Yields the next chunk in the mock stream."""
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class TestElevenLabsTTS:
    """Unit test suite for the ElevenLabsTTS adapter."""

    def test_missing_api_key_raises_configuration_error(self) -> None:
        """Verifies ConfigurationError is raised when API key is missing."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigurationError, match="ElevenLabs API key is missing"),
        ):
            ElevenLabsTTS(voice_id="21m00Tcm4TlvDq8ikWAM", api_key=None)

    def test_invalid_constructor_parameters(self) -> None:
        """Verifies validation of constructor options."""
        mock_client = MagicMock()

        # Missing or empty voice_id
        with pytest.raises(
            SynthesisError, match="voice_id is required and cannot be empty"
        ):
            ElevenLabsTTS(voice_id="", api_key="test_key", client=mock_client)

        # Invalid timeout
        with pytest.raises(SynthesisError, match="timeout_seconds must be positive"):
            ElevenLabsTTS(
                voice_id="valid_id",
                api_key="test_key",
                timeout_seconds=0.0,
                client=mock_client,
            )

        # Negative max_retries
        with pytest.raises(SynthesisError, match="max_retries cannot be negative"):
            ElevenLabsTTS(
                voice_id="valid_id",
                api_key="test_key",
                max_retries=-1,
                client=mock_client,
            )

    @pytest.mark.asyncio
    async def test_synthesize_success(self) -> None:
        """Verifies synthesize returns a complete AudioChunk with expected audio payload."""
        mock_client = MagicMock()
        mock_audio_data = [b"\x00\x01" * 12000]  # 0.5 seconds of 24kHz mono 16-bit PCM
        mock_stream = MockAsyncByteStream(mock_audio_data)

        mock_client.text_to_speech.convert = AsyncMock(return_value=mock_stream)

        tts = ElevenLabsTTS(
            voice_id="test_voice_123",
            api_key="test_key",
            model_id="eleven_flash_v2_5",
            output_format="pcm_24000",
            client=mock_client,
        )

        result = await tts.synthesize("Hello world!")

        assert isinstance(result, AudioChunk)
        assert result.data == b"".join(mock_audio_data)
        assert result.sample_rate == 24000
        assert result.channels == 1
        assert result.duration_seconds == pytest.approx(0.5)

        mock_client.text_to_speech.convert.assert_called_once_with(
            voice_id="test_voice_123",
            text="Hello world!",
            model_id="eleven_flash_v2_5",
            output_format="pcm_24000",
        )

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_raises_error(self) -> None:
        """Verifies synthesize with empty text raises SynthesisError."""
        tts = ElevenLabsTTS(
            voice_id="test_voice", api_key="test_key", client=MagicMock()
        )

        with pytest.raises(
            SynthesisError, match="Input text for synthesis cannot be empty"
        ):
            await tts.synthesize("")

    @pytest.mark.asyncio
    async def test_synthesize_stream_success(self) -> None:
        """Verifies synthesize_stream yields SynthesisChunk sequence with is_final=True only on the last chunk."""
        mock_client = MagicMock()
        chunk_1 = b"\x01\x02" * 100
        chunk_2 = b"\x03\x04" * 100
        chunk_3 = b"\x05\x06" * 100
        mock_stream = MockAsyncByteStream([chunk_1, chunk_2, chunk_3])

        mock_client.text_to_speech.stream = AsyncMock(return_value=mock_stream)

        tts = ElevenLabsTTS(
            voice_id="test_voice",
            api_key="test_key",
            output_format="pcm_24000",
            client=mock_client,
        )

        yielded_chunks: list[SynthesisChunk] = []
        async for chunk in tts.synthesize_stream("Streaming audio response"):
            yielded_chunks.append(chunk)

        assert len(yielded_chunks) == 3
        assert yielded_chunks[0].audio == chunk_1
        assert yielded_chunks[0].is_final is False
        assert yielded_chunks[0].sample_rate == 24000

        assert yielded_chunks[1].audio == chunk_2
        assert yielded_chunks[1].is_final is False

        assert yielded_chunks[2].audio == chunk_3
        assert yielded_chunks[2].is_final is True

    @pytest.mark.asyncio
    async def test_rate_limit_retry_eventual_success(self) -> None:
        """Verifies transient 429 RateLimit error retries and eventually succeeds."""
        mock_client = MagicMock()
        rate_limit_err = ApiError(
            status_code=429,
            body={"detail": {"message": "Rate limit exceeded"}},
        )
        success_stream = MockAsyncByteStream([b"\x01\x02" * 500])

        # Fail twice with 429, then succeed on 3rd attempt
        mock_client.text_to_speech.convert = AsyncMock(
            side_effect=[rate_limit_err, rate_limit_err, success_stream]
        )

        tts = ElevenLabsTTS(
            voice_id="test_voice",
            api_key="test_key",
            max_retries=3,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )

        result = await tts.synthesize("Retry test")

        assert isinstance(result, AudioChunk)
        assert mock_client.text_to_speech.convert.call_count == 3

    @pytest.mark.asyncio
    async def test_authentication_error_fails_immediately_without_retry(
        self,
    ) -> None:
        """Verifies 401/403 unauthorized errors fail immediately without retry."""
        mock_client = MagicMock()
        auth_err = ApiError(
            status_code=401,
            body={"detail": {"message": "Invalid API key"}},
        )
        mock_client.text_to_speech.convert = AsyncMock(side_effect=auth_err)

        tts = ElevenLabsTTS(
            voice_id="test_voice",
            api_key="invalid_key",
            max_retries=3,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )

        with pytest.raises(SynthesisError, match="ElevenLabs synthesis client error"):
            await tts.synthesize("Auth test")

        assert mock_client.text_to_speech.convert.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_voice_id_fails_immediately_without_retry(
        self,
    ) -> None:
        """Verifies 404 voice not found fails immediately without retry."""
        mock_client = MagicMock()
        voice_not_found_err = ApiError(
            status_code=404,
            body={"detail": {"message": "Voice not found"}},
        )
        mock_client.text_to_speech.convert = AsyncMock(side_effect=voice_not_found_err)

        tts = ElevenLabsTTS(
            voice_id="nonexistent_voice",
            api_key="test_key",
            max_retries=3,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )

        with pytest.raises(SynthesisError, match="ElevenLabs synthesis client error"):
            await tts.synthesize("Voice test")

        assert mock_client.text_to_speech.convert.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises_synthesis_error(self) -> None:
        """Verifies exceeding max_retries on server error raises SynthesisError."""
        mock_client = MagicMock()
        server_err = ApiError(
            status_code=500,
            body={"detail": {"message": "Internal Server Error"}},
        )
        mock_client.text_to_speech.convert = AsyncMock(side_effect=server_err)

        tts = ElevenLabsTTS(
            voice_id="test_voice",
            api_key="test_key",
            max_retries=2,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )

        with pytest.raises(
            SynthesisError, match="ElevenLabs synthesis failed after 3 attempt"
        ):
            await tts.synthesize("Server error test")

        assert mock_client.text_to_speech.convert.call_count == 3
