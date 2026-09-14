"""Module: test_websocket_server
Layer: Testing
Purpose: Comprehensive unit test suite for VoiceCallServer and ClientSession.
Dependencies: pytest, unittest.mock, json, src.server.websocket_server, src.core.types
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import AudioChunk, Message, MessageRole, TranscriptionResult
from src.server.websocket_server import (
    ClientSession,
    decode_audio_to_pcm,
    pcm_to_wav,
)


class TestWebSocketServerUtils:
    """Unit tests for WebSocket utility functions."""

    def test_pcm_to_wav_header(self) -> None:
        pcm = b"\x00\x00" * 100
        wav = pcm_to_wav(pcm, sample_rate=16000, channels=1)
        assert wav.startswith(b"RIFF")
        assert b"WAVE" in wav[:16]

    def test_decode_audio_to_pcm_wav_passthrough(self) -> None:
        pcm = b"\x01\x00" * 500
        wav = pcm_to_wav(pcm, sample_rate=16000, channels=1)
        decoded = decode_audio_to_pcm(wav, target_sample_rate=16000)
        assert decoded == pcm

    def test_decode_audio_to_pcm_empty(self) -> None:
        assert decode_audio_to_pcm(b"") == b""


class TestClientSession:
    """Unit tests for ClientSession turn orchestration and barge-in."""

    @pytest.fixture
    def mock_pipeline(self) -> MagicMock:
        pipeline = MagicMock()
        pipeline._memory = MagicMock()
        pipeline._memory.add_message = AsyncMock()
        pipeline._memory.get_context = MagicMock(
            return_value=[Message(role=MessageRole.USER, content="hello")]
        )

        pipeline._llm = MagicMock()

        async def dummy_stream(context):
            for token in ["Hello", " there!", " How", " can", " I", " assist?"]:
                yield token

        pipeline._llm.generate_stream = MagicMock(side_effect=dummy_stream)

        pipeline._tts = MagicMock()
        pipeline._tts.synthesize = AsyncMock(
            return_value=AudioChunk(
                data=b"\x00\x00" * 2400,
                sample_rate=24000,
                channels=1,
                duration_seconds=0.1,
            )
        )
        return pipeline

    @pytest.fixture
    def mock_websocket(self) -> AsyncMock:
        ws = AsyncMock()
        ws.send = AsyncMock()
        return ws

    def test_sentence_extraction(
        self, mock_pipeline: MagicMock, mock_websocket: AsyncMock
    ) -> None:
        session = ClientSession(
            websocket=mock_websocket,
            pipeline=mock_pipeline,
            config={},
        )
        buffer = "Hello world! This is a test. How are you?"
        sentences, rem = session.extract_sentences(buffer)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world!"
        assert sentences[1] == "This is a test."
        assert sentences[2] == "How are you?"
        assert rem == ""

    def test_early_clause_extraction(
        self, mock_pipeline: MagicMock, mock_websocket: AsyncMock
    ) -> None:
        session = ClientSession(
            websocket=mock_websocket,
            pipeline=mock_pipeline,
            config={},
        )
        buffer = "Yes indeed, absolutely"
        sentences, rem = session.extract_sentences(buffer)
        assert len(sentences) >= 1
        assert "Yes indeed," in sentences[0]

    @pytest.mark.asyncio
    async def test_process_user_turn_lifecycle(
        self, mock_pipeline: MagicMock, mock_websocket: AsyncMock
    ) -> None:
        session = ClientSession(
            websocket=mock_websocket,
            pipeline=mock_pipeline,
            config={},
        )

        await session.process_user_turn(
            user_text="Hi there",
            stt_time=0.15,
            t_start=100.0,
            turn_id=0,
        )

        sent_messages = [
            json.loads(call.args[0]) for call in mock_websocket.send.call_args_list
        ]
        msg_types = [m["type"] for m in sent_messages]

        assert "user_transcript" in msg_types
        assert "assistant_start" in msg_types
        assert "assistant_token" in msg_types
        assert "audio_chunk" in msg_types
        assert "turn_complete" in msg_types

    @pytest.mark.asyncio
    async def test_interrupt_barge_in(
        self, mock_pipeline: MagicMock, mock_websocket: AsyncMock
    ) -> None:
        session = ClientSession(
            websocket=mock_websocket,
            pipeline=mock_pipeline,
            config={},
        )

        initial_turn = session.turn_id
        await session.interrupt()
        assert session.turn_id == initial_turn + 1
        assert session.is_assistant_speaking is False

        last_sent = json.loads(mock_websocket.send.call_args.args[0])
        assert last_sent["type"] == "interrupted"
        assert last_sent["turn_id"] == session.turn_id
