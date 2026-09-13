"""Module: test_voice_assistant
Layer: Testing
Purpose: Comprehensive unit test suite for VoiceAssistantPipeline orchestrator using mocked domain interfaces.
Dependencies: asyncio, unittest.mock, pytest, src.core.types, src.core.exceptions, src.pipeline.voice_assistant
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.audio_io import AudioIO
from src.core.exceptions import (
    LanguageModelError,
    TranscriptionError,
)
from src.core.types import (
    AudioChunk,
    Message,
    MessageRole,
    SynthesisChunk,
    TranscriptionResult,
    VADResult,
)
from src.llm.base import LanguageModel
from src.memory.base import ConversationMemory
from src.pipeline.voice_assistant import VoiceAssistantPipeline
from src.stt.base import SpeechToText
from src.tts.base import TextToSpeech
from src.vad.base import VoiceActivityDetector


class MockAudioStream:
    """Async iterator for mock LLM token or TTS chunk streams."""

    def __init__(self, items: list[str | SynthesisChunk]) -> None:
        """Initializes stream with items."""
        self._items = items
        self._idx = 0

    def __aiter__(self) -> "MockAudioStream":
        """Returns self."""
        return self

    async def __anext__(self) -> str | SynthesisChunk:
        """Yields next item or stops."""
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


class TestVoiceAssistantPipeline:
    """Unit test suite for the pipeline orchestration layer."""

    @pytest.mark.asyncio
    async def test_full_conversation_turn_success(self) -> None:
        """Verifies end-to-end processing of a user turn with streaming sentence-level TTS synthesis."""
        mock_vad = MagicMock(spec=VoiceActivityDetector)
        mock_stt = MagicMock(spec=SpeechToText)
        mock_llm = MagicMock(spec=LanguageModel)
        mock_tts = MagicMock(spec=TextToSpeech)
        mock_memory = MagicMock(spec=ConversationMemory)
        mock_audio_io = MagicMock(spec=AudioIO)

        frame_speech = AudioChunk(
            data=b"\x01\x02" * 256,
            sample_rate=16000,
            channels=1,
            duration_seconds=0.032,
        )
        frame_silence = AudioChunk(
            data=b"\x00\x00" * 256,
            sample_rate=16000,
            channels=1,
            duration_seconds=0.032,
        )

        mock_audio_io.start = AsyncMock()
        mock_audio_io.stop = AsyncMock()
        mock_audio_io.play_chunk = AsyncMock()

        frame_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        await frame_queue.put(frame_speech)
        await frame_queue.put(frame_silence)

        pipeline = VoiceAssistantPipeline(
            vad=mock_vad,
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            memory=mock_memory,
            audio_io=mock_audio_io,
        )

        async def mock_read_frame() -> AudioChunk:
            if not frame_queue.empty():
                return await frame_queue.get()
            await pipeline.stop()
            return frame_silence

        mock_audio_io.read_frame = AsyncMock(side_effect=mock_read_frame)

        vad_responses = [
            VADResult(is_speech=True, confidence=0.9, is_end_of_speech=False),
            VADResult(is_speech=False, confidence=0.05, is_end_of_speech=True),
        ]
        vad_call_count = 0

        def mock_vad_process(chunk: AudioChunk) -> VADResult:
            nonlocal vad_call_count
            if vad_call_count < len(vad_responses):
                res = vad_responses[vad_call_count]
                vad_call_count += 1
                return res
            return VADResult(is_speech=False, is_end_of_speech=False)

        mock_vad.process_chunk = MagicMock(side_effect=mock_vad_process)
        mock_vad.reset = MagicMock()

        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="What is the capital of France?",
                language="en",
                confidence=0.98,
                duration_seconds=0.5,
            )
        )

        stored_messages: list[Message] = []

        async def mock_add_message(msg: Message) -> None:
            stored_messages.append(msg)

        async def mock_get_context() -> list[Message]:
            return list(stored_messages)

        mock_memory.add_message = AsyncMock(side_effect=mock_add_message)
        mock_memory.get_context = AsyncMock(side_effect=mock_get_context)

        llm_tokens = [
            "The capital ",
            "of France is ",
            "Paris. ",
            "It is a ",
            "beautiful city.",
        ]
        mock_llm.generate_stream = AsyncMock(return_value=MockAudioStream(llm_tokens))

        synthesized_sentences: list[str] = []

        async def mock_tts_stream(
            sentence: str, **kwargs
        ) -> AsyncIterator[SynthesisChunk]:
            synthesized_sentences.append(sentence)
            chunk = SynthesisChunk(
                audio=b"\x10\x20" * 100, is_final=True, sample_rate=24000
            )
            return MockAudioStream([chunk])

        mock_tts.synthesize_stream = AsyncMock(side_effect=mock_tts_stream)

        await pipeline.run_conversation_loop()

        mock_stt.transcribe.assert_called_once()
        assert len(stored_messages) >= 2
        assert stored_messages[0].role == MessageRole.USER
        assert stored_messages[0].content == "What is the capital of France?"

        assert len(synthesized_sentences) == 2
        assert synthesized_sentences[0] == "The capital of France is Paris."
        assert synthesized_sentences[1] == "It is a beautiful city."

        assert stored_messages[1].role == MessageRole.ASSISTANT
        assert (
            stored_messages[1].content
            == "The capital of France is Paris. It is a beautiful city."
        )

        assert mock_audio_io.play_chunk.call_count == 2
        mock_audio_io.stop.assert_called()

    @pytest.mark.asyncio
    async def test_stt_transcription_error_does_not_crash_loop(self) -> None:
        """Verifies that an error in STT is handled gracefully without crashing the loop."""
        mock_vad = MagicMock(spec=VoiceActivityDetector)
        mock_stt = MagicMock(spec=SpeechToText)
        mock_llm = MagicMock(spec=LanguageModel)
        mock_tts = MagicMock(spec=TextToSpeech)
        mock_memory = MagicMock(spec=ConversationMemory)
        mock_audio_io = MagicMock(spec=AudioIO)

        pipeline = VoiceAssistantPipeline(
            vad=mock_vad,
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            memory=mock_memory,
            audio_io=mock_audio_io,
        )

        frame = AudioChunk(data=b"\x01\x02" * 256, sample_rate=16000, channels=1)
        queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        await queue.put(frame)
        await queue.put(frame)

        async def mock_read_frame() -> AudioChunk:
            if not queue.empty():
                return await queue.get()
            await pipeline.stop()
            return frame

        mock_audio_io.read_frame = AsyncMock(side_effect=mock_read_frame)
        mock_audio_io.start = AsyncMock()
        mock_audio_io.stop = AsyncMock()

        vad_responses = [
            VADResult(is_speech=True, is_end_of_speech=False),
            VADResult(is_speech=False, is_end_of_speech=True),
        ]
        vad_call_count = 0

        def mock_vad_process(chunk: AudioChunk) -> VADResult:
            nonlocal vad_call_count
            if vad_call_count < len(vad_responses):
                res = vad_responses[vad_call_count]
                vad_call_count += 1
                return res
            return VADResult(is_speech=False, is_end_of_speech=False)

        mock_vad.process_chunk = MagicMock(side_effect=mock_vad_process)
        mock_vad.reset = MagicMock()

        mock_stt.transcribe = AsyncMock(
            side_effect=TranscriptionError("Speech recognition timeout")
        )

        await pipeline.run_conversation_loop()

        mock_stt.transcribe.assert_called_once()
        mock_llm.generate_stream.assert_not_called()
        mock_vad.reset.assert_called()

    @pytest.mark.asyncio
    async def test_llm_error_does_not_crash_loop(self) -> None:
        """Verifies that an error in LLM generation is caught and loop recovers."""
        mock_vad = MagicMock(spec=VoiceActivityDetector)
        mock_stt = MagicMock(spec=SpeechToText)
        mock_llm = MagicMock(spec=LanguageModel)
        mock_tts = MagicMock(spec=TextToSpeech)
        mock_memory = MagicMock(spec=ConversationMemory)
        mock_audio_io = MagicMock(spec=AudioIO)

        pipeline = VoiceAssistantPipeline(
            vad=mock_vad,
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            memory=mock_memory,
            audio_io=mock_audio_io,
        )

        frame = AudioChunk(data=b"\x01\x02" * 256, sample_rate=16000, channels=1)
        queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        await queue.put(frame)
        await queue.put(frame)

        async def mock_read_frame() -> AudioChunk:
            if not queue.empty():
                return await queue.get()
            await pipeline.stop()
            return frame

        mock_audio_io.read_frame = AsyncMock(side_effect=mock_read_frame)
        mock_audio_io.start = AsyncMock()
        mock_audio_io.stop = AsyncMock()

        vad_responses = [
            VADResult(is_speech=True, is_end_of_speech=False),
            VADResult(is_speech=False, is_end_of_speech=True),
        ]
        vad_call_count = 0

        def mock_vad_process(chunk: AudioChunk) -> VADResult:
            nonlocal vad_call_count
            if vad_call_count < len(vad_responses):
                res = vad_responses[vad_call_count]
                vad_call_count += 1
                return res
            return VADResult(is_speech=False, is_end_of_speech=False)

        mock_vad.process_chunk = MagicMock(side_effect=mock_vad_process)
        mock_vad.reset = MagicMock()

        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hello", language="en")
        )
        mock_memory.add_message = AsyncMock()
        mock_memory.get_context = AsyncMock(return_value=[])

        mock_llm.generate_stream = AsyncMock(
            side_effect=LanguageModelError("OpenAI API rate limit")
        )

        await pipeline.run_conversation_loop()

        mock_stt.transcribe.assert_called_once()
        mock_llm.generate_stream.assert_called_once()
        mock_tts.synthesize_stream.assert_not_called()
        mock_vad.reset.assert_called()

    @pytest.mark.asyncio
    async def test_stop_method_halts_pipeline(self) -> None:
        """Verifies stop() updates is_running flag and shuts down cleanly."""
        mock_audio_io = MagicMock(spec=AudioIO)
        mock_audio_io.stop = AsyncMock()

        pipeline = VoiceAssistantPipeline(
            vad=MagicMock(),
            stt=MagicMock(),
            llm=MagicMock(),
            tts=MagicMock(),
            memory=MagicMock(),
            audio_io=mock_audio_io,
        )

        pipeline._is_running = True
        await pipeline.stop()

        assert pipeline.is_running is False
        assert pipeline._stop_event.is_set()
        mock_audio_io.stop.assert_called_once()
