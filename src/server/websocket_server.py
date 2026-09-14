"""Module: websocket_server
Layer: Presentation / Server
Purpose: Low-latency full-duplex WebSocket voice server bridging real-time browser audio streaming to VoiceAssistantPipeline with live STT, streaming LLM, early TTS chunking, and instant interruption (barge-in).
Dependencies: asyncio, base64, io, json, logging, re, time, wave, typing, av, websockets, src.core.types, src.core.exceptions, src.pipeline.factory, src.pipeline.voice_assistant
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
import wave
from typing import Any

import av
import websockets
from websockets.server import WebSocketServerProtocol

from src.core.types import AudioChunk, Message, MessageRole
from src.pipeline.factory import (
    build_voice_assistant_pipeline,
    load_and_validate_config,
)
from src.pipeline.voice_assistant import VoiceAssistantPipeline
from src.stt.deepgram_stt import DeepgramSTT

logger = logging.getLogger("talk_to_me.websocket")


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wraps raw 16-bit PCM bytes into a standard WAV audio container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def decode_audio_to_pcm(raw_bytes: bytes, target_sample_rate: int = 16000) -> bytes:
    """Decodes browser audio blob (WAV/WebM/OGG) or raw PCM to 16kHz mono 16-bit PCM."""
    if not raw_bytes:
        return b""

    # Fast path: check for standard 16-bit mono 16kHz WAV container
    if raw_bytes.startswith(b"RIFF") and b"WAVE" in raw_bytes[:16]:
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
                if (
                    sample_rate == target_sample_rate
                    and channels == 1
                    and sample_width == 2
                ):
                    return frames
        except Exception:
            pass

    # Universal decoder via PyAV
    try:
        buf = io.BytesIO(raw_bytes)
        container = av.open(buf)
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=target_sample_rate
        )
        pcm_chunks = []
        for frame in container.decode(audio=0):
            for r_frame in resampler.resample(frame):
                pcm_chunks.append(r_frame.to_ndarray().tobytes())
        for r_frame in resampler.resample(None):
            pcm_chunks.append(r_frame.to_ndarray().tobytes())
        return b"".join(pcm_chunks)
    except Exception:
        # If decode fails, and payload length is aligned to 16-bit, assume raw PCM
        if len(raw_bytes) % 2 == 0:
            return raw_bytes
        return b""


class ClientSession:
    """Represents an active client voice call session with stateful turn tracking and barge-in control."""

    def __init__(
        self,
        websocket: WebSocketServerProtocol,
        pipeline: VoiceAssistantPipeline,
        config: dict[str, Any],
    ) -> None:
        self.websocket = websocket
        self.pipeline = pipeline
        self.config = config
        self.turn_id = 0
        self.is_assistant_speaking = False
        self.is_user_speaking = False
        self.active_turn_task: asyncio.Task[None] | None = None
        self.audio_buffer: list[bytes] = []
        self.speech_start_time: float = 0.0
        self.last_speech_time: float = 0.0
        self.sentence_regex = re.compile(r"([.?!;]+(?:\s+|\Z)|\n+)")
        self.lock = asyncio.Lock()

        # Streaming STT queue for Deepgram if used
        self.stt_audio_queue: asyncio.Queue[AudioChunk] | None = None
        self.deepgram_task: asyncio.Task[None] | None = None

    async def interrupt(self) -> None:
        """Interrupts and cancels active assistant speech synthesis and generation immediately."""
        async with self.lock:
            self.turn_id += 1
            current_turn = self.turn_id
            self.is_assistant_speaking = False

            if self.active_turn_task and not self.active_turn_task.done():
                self.active_turn_task.cancel()
                self.active_turn_task = None
                logger.info(
                    "⚡ [Barge-In] Interrupted active assistant turn #%d",
                    current_turn - 1,
                )

            try:
                await self.websocket.send(
                    json.dumps({"type": "interrupted", "turn_id": current_turn})
                )
            except Exception:
                pass

    def extract_sentences(self, buffer: str) -> tuple[list[str], str]:
        """Extracts completed sentences or early speech clauses for ultra-low latency playback."""
        sentences: list[str] = []
        while True:
            # 1. Check for standard sentence boundaries (. ? ! ; \n)
            match = self.sentence_regex.search(buffer)
            if match:
                end_idx = match.end()
                sentence = buffer[:end_idx].strip()
                buffer = buffer[end_idx:].strip()
                if sentence:
                    sentences.append(sentence)
                continue

            # 2. Check for early clause delimiters (comma, colon, dash) with at least 2 words
            words = buffer.split()
            found_early_break = False
            if len(words) >= 3:
                for punct in [",", ":", " - ", " — ", " – "]:
                    if punct in buffer:
                        idx = buffer.find(punct) + len(punct)
                        sub = buffer[:idx].strip()
                        if len(sub.split()) >= 2:
                            sentences.append(sub)
                            buffer = buffer[idx:].strip()
                            found_early_break = True
                            break
            if found_early_break:
                continue

            # 3. If buffer has 4+ words without punctuation, emit the first 3-4 words early so TTS starts instantly!
            if len(words) >= 4:
                split_len = len(" ".join(words[:3]))
                sub = buffer[:split_len].strip()
                buffer = buffer[split_len:].strip()
                if sub:
                    sentences.append(sub)
                continue

            break

        return sentences, buffer

    async def process_user_turn(
        self,
        user_text: str,
        stt_time: float,
        t_start: float,
        turn_id: int,
    ) -> None:
        """Processes user input text through LLM and streams synthesized TTS audio chunks."""
        if turn_id != self.turn_id:
            logger.debug(
                "Discarding outdated turn #%d (current turn is #%d)",
                turn_id,
                self.turn_id,
            )
            return

        self.is_assistant_speaking = True

        # 1. Emit user transcript event
        try:
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "user_transcript",
                        "text": user_text,
                        "stt_time": stt_time,
                        "turn_id": turn_id,
                    }
                )
            )
        except Exception:
            return

        # 2. Update Conversation Memory
        try:
            mem_res = self.pipeline._memory.add_message(
                Message(role=MessageRole.USER, content=user_text)
            )
            if asyncio.iscoroutine(mem_res):
                await mem_res
            ctx_res = self.pipeline._memory.get_context()
            context = await ctx_res if asyncio.iscoroutine(ctx_res) else ctx_res
        except Exception as err:
            logger.error("Memory retrieval error: %s", err)
            context = [Message(role=MessageRole.USER, content=user_text)]

        # 3. Stream LLM and Synthesize First Chunk Early with Concurrent Queue
        full_reply_tokens: list[str] = []
        text_buffer = ""
        t_llm_start = time.perf_counter()
        ttft: float | None = None
        ttfa: float | None = None

        try:
            await self.websocket.send(
                json.dumps({"type": "assistant_start", "turn_id": turn_id})
            )
        except Exception:
            return

        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def tts_consumer() -> None:
            nonlocal ttfa
            while True:
                sentence = await tts_queue.get()
                if sentence is None or turn_id != self.turn_id:
                    break
                clean_sentence = sentence.strip()
                if not clean_sentence:
                    continue
                try:
                    audio_chunk = await self.pipeline._tts.synthesize(clean_sentence)
                    if turn_id != self.turn_id:
                        break
                    if audio_chunk and audio_chunk.data:
                        if ttfa is None:
                            ttfa = time.perf_counter() - t_llm_start
                            logger.info(
                                "⚡ Latency Checkpoint [TTFA]: First TTS audio ready in %.3fs",
                                ttfa,
                            )
                            await self.websocket.send(
                                json.dumps(
                                    {
                                        "type": "metric_ttfa",
                                        "ttfa": ttfa,
                                        "turn_id": turn_id,
                                    }
                                )
                            )
                        wav_payload = pcm_to_wav(
                            audio_chunk.data, sample_rate=audio_chunk.sample_rate
                        )
                        b64_audio = base64.b64encode(wav_payload).decode("utf-8")
                        await self.websocket.send(
                            json.dumps(
                                {
                                    "type": "audio_chunk",
                                    "audio": b64_audio,
                                    "turn_id": turn_id,
                                }
                            )
                        )
                except Exception as synth_err:
                    logger.warning(
                        "Error synthesizing sentence '%s': %s",
                        clean_sentence,
                        synth_err,
                    )

        consumer_task = asyncio.create_task(tts_consumer())

        try:
            llm_stream = self.pipeline._llm.generate_stream(context)
            if asyncio.iscoroutine(llm_stream):
                llm_stream = await llm_stream

            async for token in llm_stream:
                if turn_id != self.turn_id:
                    logger.info("Turn #%d was interrupted during LLM stream", turn_id)
                    await tts_queue.put(None)
                    return

                if ttft is None:
                    ttft = time.perf_counter() - t_llm_start
                    logger.info(
                        "⚡ Latency Checkpoint [TTFT]: First LLM token in %.3fs", ttft
                    )
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "metric_ttft",
                                "ttft": ttft,
                                "turn_id": turn_id,
                            }
                        )
                    )

                full_reply_tokens.append(token)
                text_buffer += token
                await self.websocket.send(
                    json.dumps(
                        {
                            "type": "assistant_token",
                            "token": token,
                            "turn_id": turn_id,
                        }
                    )
                )

                # Check for completed phrases/sentences to synthesize immediately
                completed_sentences, text_buffer = self.extract_sentences(text_buffer)
                for sentence in completed_sentences:
                    if turn_id != self.turn_id:
                        break
                    await tts_queue.put(sentence)

            # Flush remaining unpunctuated text buffer
            rem_text = text_buffer.strip()
            if rem_text and turn_id == self.turn_id:
                await tts_queue.put(rem_text)

            # Sentinel to finish TTS consumer
            await tts_queue.put(None)
            await consumer_task

        except asyncio.CancelledError:
            logger.info("Turn #%d processing task cancelled via barge-in", turn_id)
            if not consumer_task.done():
                consumer_task.cancel()
            return
        except Exception as err:
            logger.error("LLM/TTS processing error in turn #%d: %s", turn_id, err)
            if not consumer_task.done():
                consumer_task.cancel()
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Pipeline error: {err}",
                        "turn_id": turn_id,
                    }
                )
            )
            return

        if turn_id != self.turn_id:
            return

        # 4. Save assistant response to memory
        complete_reply = "".join(full_reply_tokens).strip()
        if complete_reply:
            try:
                save_res = self.pipeline._memory.add_message(
                    Message(role=MessageRole.ASSISTANT, content=complete_reply)
                )
                if asyncio.iscoroutine(save_res):
                    await save_res
            except Exception as err:
                logger.debug("Error saving assistant reply: %s", err)

        total_turnaround = time.perf_counter() - t_start
        logger.info(
            "🎉 Turn #%d Completed: STT=%.2fs | TTFT=%.2fs | TTFA=%.2fs | Total=%.2fs",
            turn_id,
            stt_time,
            ttft or 0.0,
            ttfa or 0.0,
            total_turnaround,
        )

        await self.websocket.send(
            json.dumps(
                {
                    "type": "turn_complete",
                    "text": complete_reply,
                    "turn_id": turn_id,
                    "metrics": {
                        "stt_time": stt_time,
                        "ttft": ttft or 0.0,
                        "ttfa": ttfa or 0.0,
                        "total_time": total_turnaround,
                    },
                }
            )
        )


class VoiceCallServer:
    """Asynchronous WebSocket server handling full-duplex real-time streaming voice sessions."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8503,
        pipeline: VoiceAssistantPipeline | None = None,
    ) -> None:
        """Initializes the voice call server."""
        self.host = host
        self.port = port
        self.pipeline = pipeline
        self._server: Any = None
        self._config: dict[str, Any] | None = None

    def _ensure_pipeline(self) -> tuple[VoiceAssistantPipeline, dict[str, Any]]:
        """Ensures that the VoiceAssistantPipeline and config are initialized."""
        if self.pipeline is None or self._config is None:
            self._config = load_and_validate_config()
            self.pipeline, self._config = build_voice_assistant_pipeline(
                config=self._config
            )
        return self.pipeline, self._config

    async def _start_deepgram_stream(self, session: ClientSession) -> None:
        """Connects and manages a live streaming Deepgram transcription session for the client."""
        if not isinstance(session.pipeline._stt, DeepgramSTT):
            return

        deepgram: DeepgramSTT = session.pipeline._stt
        session.stt_audio_queue = asyncio.Queue()

        async def audio_generator() -> AsyncIterator[AudioChunk]:
            while True:
                chunk = await session.stt_audio_queue.get()
                if chunk.data == b"":  # Sentinel to close stream
                    break
                yield chunk

        try:
            async for result in deepgram.transcribe_stream(audio_generator()):
                if result.text.strip():
                    if result.speech_final or result.is_final:
                        logger.info("✓ Deepgram Final: '%s'", result.text)
                        # Trigger LLM turn
                        session.turn_id += 1
                        turn_id = session.turn_id
                        stt_time = 0.20  # Deepgram live streaming latency
                        t_start = time.perf_counter()
                        session.active_turn_task = asyncio.create_task(
                            session.process_user_turn(
                                result.text.strip(), stt_time, t_start, turn_id
                            )
                        )
                    else:
                        # Emit interim live transcript
                        await session.websocket.send(
                            json.dumps(
                                {
                                    "type": "interim_transcript",
                                    "text": result.text.strip(),
                                    "turn_id": session.turn_id,
                                }
                            )
                        )
        except Exception as err:
            logger.debug("Deepgram stream ended or reconnect: %s", err)

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handles incoming messages for an active WebSocket client connection."""
        pipeline, config = self._ensure_pipeline()
        session = ClientSession(websocket=websocket, pipeline=pipeline, config=config)
        logger.info("New client connected to Voice Call server.")

        # Start Deepgram streaming task if STT provider is Deepgram
        if isinstance(pipeline._stt, DeepgramSTT):
            session.deepgram_task = asyncio.create_task(
                self._start_deepgram_stream(session)
            )

        try:
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    await self._handle_audio_chunk(session, raw_message)
                elif isinstance(raw_message, str):
                    try:
                        data = json.loads(raw_message)
                        msg_type = data.get("type")

                        if msg_type == "audio_chunk":
                            audio_b64 = data.get("data", "")
                            pcm_bytes = base64.b64decode(audio_b64)
                            await self._handle_audio_chunk(session, pcm_bytes)

                        elif msg_type == "audio":
                            # Complete utterance or clip
                            audio_b64 = data.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)
                            await self._handle_discrete_audio(session, audio_bytes)

                        elif msg_type == "interrupt":
                            await session.interrupt()

                        elif msg_type == "text_query":
                            text = data.get("text", "").strip()
                            if text:
                                await session.interrupt()
                                session.turn_id += 1
                                turn_id = session.turn_id
                                session.active_turn_task = asyncio.create_task(
                                    session.process_user_turn(
                                        text,
                                        stt_time=0.0,
                                        t_start=time.perf_counter(),
                                        turn_id=turn_id,
                                    )
                                )
                            else:
                                await websocket.send(json.dumps({"type": "empty_turn"}))

                        elif msg_type == "clear_memory":
                            clear_res = pipeline._memory.clear()
                            if asyncio.iscoroutine(clear_res):
                                await clear_res
                            await websocket.send(json.dumps({"type": "memory_cleared"}))

                        elif msg_type == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received: %s", raw_message[:100])

        except websockets.exceptions.ConnectionClosed:
            logger.info("Voice call client disconnected.")
        except Exception as exc:
            logger.error("Unexpected error in WebSocket client handler: %s", exc)
        finally:
            if session.deepgram_task and not session.deepgram_task.done():
                session.deepgram_task.cancel()
            if session.active_turn_task and not session.active_turn_task.done():
                session.active_turn_task.cancel()

    async def _handle_audio_chunk(
        self, session: ClientSession, pcm_data: bytes
    ) -> None:
        """Processes real-time streaming audio chunk from client."""
        if not pcm_data or len(pcm_data) < 32:
            return

        chunk = AudioChunk(
            data=pcm_data,
            sample_rate=16000,
            channels=1,
            duration_seconds=len(pcm_data) / 32000.0,
        )

        # If Deepgram streaming is active, push chunk to Deepgram queue
        if session.stt_audio_queue is not None:
            await session.stt_audio_queue.put(chunk)
            return

        # Fallback to server-side Silero VAD + Whisper STT
        vad = session.pipeline._vad
        vad_result = vad.process_chunk(chunk)

        if vad_result.is_speech:
            if not session.is_user_speaking:
                session.is_user_speaking = True
                session.speech_start_time = time.perf_counter()
                logger.debug("Speech activity detected via VAD")
                # If assistant was speaking, trigger instant barge-in!
                if session.is_assistant_speaking:
                    await session.interrupt()
            session.audio_buffer.append(pcm_data)
        elif session.is_user_speaking:
            session.audio_buffer.append(pcm_data)

        if vad_result.is_end_of_speech and session.is_user_speaking:
            session.is_user_speaking = False
            vad.reset()
            full_pcm = b"".join(session.audio_buffer)
            session.audio_buffer.clear()

            if len(full_pcm) >= 1600:  # > 50ms
                await self._transcribe_and_dispatch(session, full_pcm)

    async def _transcribe_and_dispatch(
        self, session: ClientSession, full_pcm: bytes
    ) -> None:
        """Transcribes completed audio buffer and triggers LLM generation."""
        t_start = time.perf_counter()
        speech_chunk = AudioChunk(
            data=full_pcm,
            sample_rate=16000,
            channels=1,
            duration_seconds=len(full_pcm) / 32000.0,
        )

        try:
            transcription = await session.pipeline._stt.transcribe(speech_chunk)
            stt_time = time.perf_counter() - t_start
            user_text = transcription.text.strip()
            if not user_text:
                await session.websocket.send(json.dumps({"type": "empty_turn"}))
                return

            session.turn_id += 1
            turn_id = session.turn_id
            session.active_turn_task = asyncio.create_task(
                session.process_user_turn(user_text, stt_time, t_start, turn_id)
            )
        except Exception as err:
            logger.error("STT transcription error: %s", err)
            await session.websocket.send(
                json.dumps({"type": "error", "message": f"STT failed: {err}"})
            )

    async def _handle_discrete_audio(
        self, session: ClientSession, raw_audio: bytes
    ) -> None:
        """Handles discrete audio utterance (e.g. from file upload or discrete turn)."""
        pcm_data = decode_audio_to_pcm(raw_audio, target_sample_rate=16000)
        if not pcm_data or len(pcm_data) < 1600:
            await session.websocket.send(json.dumps({"type": "empty_turn"}))
            return
        await self._transcribe_and_dispatch(session, pcm_data)

    async def start(self) -> None:
        """Starts the WebSocket server asynchronously."""
        self._server = await websockets.serve(
            self.handle_client, self.host, self.port, max_size=10 * 1024 * 1024
        )
        logger.info(
            "VoiceCall WebSocket Server running at ws://%s:%d", self.host, self.port
        )

    async def stop(self) -> None:
        """Stops the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("VoiceCall WebSocket Server stopped.")


def run_websocket_server(host: str = "0.0.0.0", port: int = 8503) -> None:
    """Runs the voice call WebSocket server in an asyncio event loop."""
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = VoiceCallServer(host=host, port=port)
    loop.run_until_complete(server.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(server.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    run_websocket_server()
