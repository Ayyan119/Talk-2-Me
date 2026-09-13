"""Module: voice_assistant
Layer: Orchestration/Use-Case
Purpose: Orchestrates the end-to-end voice pipeline: Mic -> VAD -> STT -> Memory -> LLM -> Sentence Splitter -> TTS -> Audio Playback.
Dependencies: asyncio, logging, re, time, src.core.types, src.core.exceptions, src.core.audio_io, src.vad.base, src.stt.base, src.llm.base, src.tts.base, src.memory.base
"""

import asyncio
import logging
import re
import time

from src.core.audio_io import AudioIO
from src.core.exceptions import (
    AudioProcessingError,
    ConversationMemoryError,
    LanguageModelError,
    SpeechToTextError,
    SynthesisError,
    TalkToMeDomainError,
    TranscriptionError,
    VoiceActivityDetectionError,
)
from src.core.types import AudioChunk, Message, MessageRole
from src.llm.base import LanguageModel
from src.memory.base import ConversationMemory
from src.stt.base import SpeechToText
from src.tts.base import TextToSpeech
from src.vad.base import VoiceActivityDetector

logger = logging.getLogger("talk_to_me.pipeline")


class VoiceAssistantPipeline:
    """Orchestrates the real-time speech conversation loop with low-latency streaming."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        llm: LanguageModel,
        tts: TextToSpeech,
        memory: ConversationMemory,
        audio_io: AudioIO,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """Initializes the voice assistant pipeline with abstract domain interfaces.

        Args:
            vad: Voice Activity Detector for segmenting speech boundaries.
            stt: Speech-to-Text provider for transcribing spoken audio.
            llm: Language Model provider for generating streaming conversational responses.
            tts: Text-to-Speech provider for streaming audio synthesis.
            memory: Conversation memory provider for context management.
            audio_io: Audio I/O interface for microphone capture and speaker playback.
            custom_logger: Optional logger instance for latency telemetry and diagnostics.
        """
        self._vad = vad
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._memory = memory
        self._audio_io = audio_io
        self._logger = custom_logger or logger

        self._is_running = False
        self._stop_event = asyncio.Event()

        # Regex for finding completed sentences ending with ., !, ?, or newlines
        self._sentence_split_regex = re.compile(r"([.?!;]+(?:\s+|\Z)|\n+)")

    @property
    def is_running(self) -> bool:
        """Returns True if the conversation pipeline loop is actively running."""
        return self._is_running

    def _extract_sentences(self, buffer: str) -> tuple[list[str], str]:
        """Extracts complete sentences from a streaming text buffer.

        Args:
            buffer: Current accumulated text buffer.

        Returns:
            A tuple of (list of complete sentences, remaining unpunctuated text).
        """
        sentences: list[str] = []
        while True:
            match = self._sentence_split_regex.search(buffer)
            if not match:
                break
            end_idx = match.end()
            sentence = buffer[:end_idx].strip()
            buffer = buffer[end_idx:]
            if sentence:
                sentences.append(sentence)
        return sentences, buffer

    async def _synthesize_and_play(
        self, sentence: str, turn_metrics: dict[str, float]
    ) -> None:
        """Synthesizes a sentence stream and immediately plays audio chunks as they arrive.

        Args:
            sentence: Sentence text to synthesize.
            turn_metrics: Dictionary tracking latency metrics for the current turn.
        """
        self._logger.debug("Synthesizing sentence: %s", sentence)
        stream = await self._tts.synthesize_stream(sentence)

        first_audio_logged = False
        async for chunk in stream:
            if not chunk or not chunk.audio:
                continue

            if "t_first_tts_audio" not in turn_metrics:
                turn_metrics["t_first_tts_audio"] = time.perf_counter()
                ttfa = (
                    turn_metrics["t_first_tts_audio"] - turn_metrics["t_end_of_speech"]
                )
                self._logger.info(
                    "⚡ Latency Checkpoint [TTFA]: First TTS audio chunk generated in %.3fs",
                    ttfa,
                )

            t_play_start = time.perf_counter()
            await self._audio_io.play_chunk(chunk)

            if not first_audio_logged:
                first_audio_logged = True
                if "t_first_audio_play" not in turn_metrics:
                    turn_metrics["t_first_audio_play"] = t_play_start
                    tt_play = (
                        turn_metrics["t_first_audio_play"]
                        - turn_metrics["t_end_of_speech"]
                    )
                    self._logger.info(
                        "🔊 Latency Checkpoint: First audio played out to speaker in %.3fs",
                        tt_play,
                    )

    async def _process_user_turn(self, audio_data: bytes, sample_rate: int) -> None:
        """Processes a single completed user speech utterance through the full pipeline.

        Args:
            audio_data: Raw concatenated PCM audio bytes of the user utterance.
            sample_rate: Audio sampling frequency in Hz.
        """
        turn_metrics: dict[str, float] = {"t_end_of_speech": time.perf_counter()}
        duration_seconds = len(audio_data) / (sample_rate * 2)
        speech_chunk = AudioChunk(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            duration_seconds=duration_seconds,
        )

        # 1. Speech-To-Text Transcription
        try:
            self._logger.info(
                "🎙️ Transcribing utterance (%.2fs audio)...", duration_seconds
            )
            transcription = await self._stt.transcribe(speech_chunk)
            turn_metrics["t_stt_done"] = time.perf_counter()
            stt_latency = turn_metrics["t_stt_done"] - turn_metrics["t_end_of_speech"]
            self._logger.info(
                "✓ STT Transcribed in %.3fs: '%s'",
                stt_latency,
                transcription.text,
            )
        except (TranscriptionError, SpeechToTextError) as err:
            self._logger.error("STT transcription error: %s", err)
            return

        user_text = transcription.text.strip()
        if not user_text:
            self._logger.info(
                "Empty transcript received; ignoring turn and resuming listening."
            )
            return

        # 2. Update Memory with User Message
        try:
            await self._memory.add_message(
                Message(role=MessageRole.USER, content=user_text)
            )
            context = await self._memory.get_context()
        except ConversationMemoryError as err:
            self._logger.error("Memory retrieval error: %s", err)
            return

        # 3. Stream LLM Response & Synthesize Completed Sentences Early
        full_assistant_response: list[str] = []
        running_text_buffer = ""

        try:
            self._logger.info("🧠 Prompting LLM with conversation context...")
            llm_stream = await self._llm.generate_stream(context)

            async for token in llm_stream:
                if "t_first_llm_token" not in turn_metrics:
                    turn_metrics["t_first_llm_token"] = time.perf_counter()
                    ttft = (
                        turn_metrics["t_first_llm_token"]
                        - turn_metrics["t_end_of_speech"]
                    )
                    self._logger.info(
                        "⚡ Latency Checkpoint [TTFT]: First LLM token arrived in %.3fs",
                        ttft,
                    )

                full_assistant_response.append(token)
                running_text_buffer += token

                completed_sentences, running_text_buffer = self._extract_sentences(
                    running_text_buffer
                )
                for sentence in completed_sentences:
                    await self._synthesize_and_play(sentence, turn_metrics)

            # Synthesize any remaining unpunctuated text at end of stream
            remaining_text = running_text_buffer.strip()
            if remaining_text:
                await self._synthesize_and_play(remaining_text, turn_metrics)

        except (LanguageModelError, SynthesisError, AudioProcessingError) as err:
            self._logger.error("Error during LLM generation or TTS stream: %s", err)
            return

        # 4. Save Final Assistant Response to Memory
        complete_reply = "".join(full_assistant_response).strip()
        if complete_reply:
            try:
                await self._memory.add_message(
                    Message(role=MessageRole.ASSISTANT, content=complete_reply)
                )
            except ConversationMemoryError as err:
                self._logger.error(
                    "Failed to save assistant response to memory: %s", err
                )

        t_total = time.perf_counter() - turn_metrics["t_end_of_speech"]
        self._logger.info(
            "🎉 Turn Completed in %.3fs (Total pipeline turnaround)", t_total
        )

    async def run_conversation_loop(self) -> None:
        """Runs the continuous asynchronous microphone listening and conversation loop."""
        self._is_running = True
        self._stop_event.clear()
        self._logger.info("Starting conversation pipeline loop...")

        try:
            await self._audio_io.start()
        except AudioProcessingError as err:
            self._logger.error("Failed to start audio I/O: %s", err)
            self._is_running = False
            return

        speech_buffer: list[bytes] = []
        is_speaking = False
        sample_rate = 16000

        try:
            while self._is_running and not self._stop_event.is_set():
                try:
                    frame_chunk = await self._audio_io.read_frame()
                except AudioProcessingError as err:
                    self._logger.error("Error reading audio frame: %s", err)
                    await asyncio.sleep(0.05)
                    continue

                sample_rate = frame_chunk.sample_rate

                try:
                    vad_result = self._vad.process_chunk(frame_chunk)
                except VoiceActivityDetectionError as err:
                    self._logger.error("VAD evaluation error: %s", err)
                    self._vad.reset()
                    speech_buffer.clear()
                    is_speaking = False
                    continue

                if vad_result.is_speech:
                    if not is_speaking:
                        is_speaking = True
                        self._logger.debug("Speech activity detected. Buffering...")
                    speech_buffer.append(frame_chunk.data)
                elif is_speaking:
                    speech_buffer.append(frame_chunk.data)

                if vad_result.is_end_of_speech and is_speaking:
                    self._logger.debug("End of user turn detected.")
                    if speech_buffer:
                        full_audio = b"".join(speech_buffer)
                        speech_buffer.clear()
                        is_speaking = False
                        self._vad.reset()

                        try:
                            await self._process_user_turn(full_audio, sample_rate)
                        except (TalkToMeDomainError, RuntimeError) as err:
                            self._logger.error("Error during turn processing: %s", err)
                        finally:
                            self._vad.reset()
                            speech_buffer.clear()
                            is_speaking = False

        finally:
            self._is_running = False
            try:
                await self._audio_io.stop()
            except (AudioProcessingError, OSError, RuntimeError) as err:
                self._logger.warning("Error stopping audio I/O: %s", err)
            self._logger.info("Conversation pipeline loop has shut down.")

    async def stop(self) -> None:
        """Signals the pipeline to stop its conversation loop cleanly."""
        self._logger.info("Stopping VoiceAssistantPipeline...")
        self._is_running = False
        self._stop_event.set()
        try:
            stop_res = self._audio_io.stop()
            if asyncio.iscoroutine(stop_res):
                await stop_res
        except (AudioProcessingError, OSError, RuntimeError):
            pass
