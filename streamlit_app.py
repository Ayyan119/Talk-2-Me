"""Module: streamlit_app
Layer: Presentation / UI Wrapper
Purpose: Streamlit interactive user interface wrapper for the VoiceAssistantPipeline running on a background worker thread.
Dependencies: asyncio, logging, queue, threading, time, typing, streamlit, src.core.types, src.core.exceptions, src.pipeline.factory, src.pipeline.voice_assistant
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from src.core.types import (
    AudioChunk,
    Message,
    SynthesisChunk,
    TranscriptionResult,
)
from src.llm.base import LanguageModel
from src.pipeline.factory import (
    build_voice_assistant_pipeline,
    load_and_validate_config,
)
from src.pipeline.voice_assistant import VoiceAssistantPipeline
from src.stt.base import SpeechToText
from src.tts.base import TextToSpeech

logger = logging.getLogger("talk_to_me.ui")


@dataclass
class UIEvent:
    """Event container passed from background pipeline thread to the Streamlit UI thread."""

    event_type: str
    payload: Any = field(default=None)


class UIEventSTTProxy(SpeechToText):
    """Proxy decorator for SpeechToText capturing transcription events and latency."""

    def __init__(
        self, target_stt: SpeechToText, event_queue: queue.Queue[UIEvent]
    ) -> None:
        """Initializes the STT proxy.

        Args:
            target_stt: The underlying SpeechToText implementation.
            event_queue: Thread-safe queue for UI event dispatch.
        """
        self._target = target_stt
        self._queue = event_queue

    async def transcribe(self, audio_chunk: AudioChunk) -> TranscriptionResult:
        """Transcribes audio and emits UI status and transcript events."""
        self._queue.put(UIEvent("status", "Processing"))
        t_start = time.perf_counter()
        try:
            result = await self._target.transcribe(audio_chunk)
            latency = time.perf_counter() - t_start
            if result.text.strip():
                self._queue.put(UIEvent("user_message", result.text.strip()))
                self._queue.put(UIEvent("metric_stt", latency))
            return result
        except Exception as err:
            self._queue.put(UIEvent("error", f"STT Error: {err}"))
            raise

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Streams transcriptions from target STT."""
        async for res in self._target.transcribe_stream(audio_stream):
            yield res


class UIEventLLMProxy(LanguageModel):
    """Proxy decorator for LanguageModel capturing token generation events and TTFT latency."""

    def __init__(
        self, target_llm: LanguageModel, event_queue: queue.Queue[UIEvent]
    ) -> None:
        """Initializes the LLM proxy.

        Args:
            target_llm: The underlying LanguageModel implementation.
            event_queue: Thread-safe queue for UI event dispatch.
        """
        self._target = target_llm
        self._queue = event_queue

    async def generate(self, messages: list[Message], **kwargs: Any) -> str:
        """Generates full response text via target LLM."""
        return await self._target.generate(messages, **kwargs)

    async def generate_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Streams tokens from target LLM and records Time-To-First-Token (TTFT)."""
        t_start = time.perf_counter()
        first_token = True
        try:
            stream = self._target.generate_stream(messages, **kwargs)
            if asyncio.iscoroutine(stream):
                stream = await stream

            async for token in stream:
                if first_token:
                    first_token = False
                    ttft = time.perf_counter() - t_start
                    self._queue.put(UIEvent("metric_ttft", ttft))
                    self._queue.put(UIEvent("assistant_start"))
                self._queue.put(UIEvent("assistant_token", token))
                yield token
        except Exception as err:
            self._queue.put(UIEvent("error", f"LLM Error: {err}"))
            raise


class UIEventTTSProxy(TextToSpeech):
    """Proxy decorator for TextToSpeech capturing audio synthesis events and TTFA latency."""

    def __init__(
        self, target_tts: TextToSpeech, event_queue: queue.Queue[UIEvent]
    ) -> None:
        """Initializes the TTS proxy.

        Args:
            target_tts: The underlying TextToSpeech implementation.
            event_queue: Thread-safe queue for UI event dispatch.
        """
        self._target = target_tts
        self._queue = event_queue

    async def synthesize(self, text: str, **kwargs: Any) -> AudioChunk:
        """Synthesizes complete audio via target TTS."""
        return await self._target.synthesize(text, **kwargs)

    async def synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[SynthesisChunk]:
        """Streams synthesized chunks from target TTS and records Time-To-First-Audio (TTFA)."""
        self._queue.put(UIEvent("status", "Speaking"))
        t_start = time.perf_counter()
        first_chunk = True
        try:
            stream = self._target.synthesize_stream(text, **kwargs)
            if asyncio.iscoroutine(stream):
                stream = await stream

            async for chunk in stream:
                if first_chunk:
                    first_chunk = False
                    ttfa = time.perf_counter() - t_start
                    self._queue.put(UIEvent("metric_ttfa", ttfa))
                yield chunk
        except Exception as err:
            self._queue.put(UIEvent("error", f"TTS Error: {err}"))
            raise


class PipelineThreadController:
    """Manages the background execution thread and event loop for VoiceAssistantPipeline."""

    def __init__(self, event_queue: queue.Queue[UIEvent]) -> None:
        """Initializes the thread controller."""
        self.event_queue = event_queue
        self.pipeline: VoiceAssistantPipeline | None = None
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Returns True if the worker thread is actively executing."""
        return self._is_running and self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        """Initializes domain adapters and starts the conversation loop in a worker thread."""
        if self.is_running:
            return

        try:
            config = load_and_validate_config()
            raw_pipeline, _ = build_voice_assistant_pipeline(config=config)

            # Wrap STT, LLM, TTS with event dispatching proxies
            wrapped_stt = UIEventSTTProxy(raw_pipeline._stt, self.event_queue)
            wrapped_llm = UIEventLLMProxy(raw_pipeline._llm, self.event_queue)
            wrapped_tts = UIEventTTSProxy(raw_pipeline._tts, self.event_queue)

            self.pipeline = VoiceAssistantPipeline(
                vad=raw_pipeline._vad,
                stt=wrapped_stt,
                llm=wrapped_llm,
                tts=wrapped_tts,
                memory=raw_pipeline._memory,
                audio_io=raw_pipeline._audio_io,
            )
        except Exception as err:  # noqa: BLE001
            self.event_queue.put(UIEvent("error", f"Initialization Error: {err}"))
            return

        self._is_running = True
        self.event_queue.put(UIEvent("status", "Listening"))

        def _worker() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                if self.pipeline:
                    self.loop.run_until_complete(self.pipeline.run_conversation_loop())
            except Exception as exc:  # noqa: BLE001
                self.event_queue.put(UIEvent("error", f"Pipeline Loop Error: {exc}"))
            finally:
                self._is_running = False
                self.event_queue.put(UIEvent("status", "Idle"))
                self.loop.close()

        self.thread = threading.Thread(
            target=_worker, daemon=True, name="VoiceAssistantThread"
        )
        self.thread.start()

    def stop(self) -> None:
        """Stops the pipeline and terminates the worker thread."""
        if self.pipeline:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.pipeline.stop(), self.loop)
            else:
                try:
                    asyncio.run(self.pipeline.stop())
                except RuntimeError:
                    pass

        self._is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.event_queue.put(UIEvent("status", "Idle"))


def init_session_state() -> None:
    """Initializes persistent Streamlit session state keys."""
    if "event_queue" not in st.session_state:
        st.session_state.event_queue = queue.Queue()
    if "controller" not in st.session_state:
        st.session_state.controller = PipelineThreadController(
            st.session_state.event_queue
        )
    if "status" not in st.session_state:
        st.session_state.status = "Idle"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_metrics" not in st.session_state:
        st.session_state.latest_metrics = {
            "stt_time": 0.0,
            "ttft": 0.0,
            "ttfa": 0.0,
            "total_time": 0.0,
        }
    if "active_assistant_message" not in st.session_state:
        st.session_state.active_assistant_message = ""
    if "last_error" not in st.session_state:
        st.session_state.last_error = None


def drain_event_queue() -> None:
    """Processes pending events from the background thread queue into session state."""
    eq: queue.Queue[UIEvent] = st.session_state.event_queue
    while not eq.empty():
        try:
            event = eq.get_nowait()
        except queue.Empty:
            break

        if event.event_type == "status":
            st.session_state.status = str(event.payload)
        elif event.event_type == "user_message":
            st.session_state.messages.append(
                {"role": "user", "content": str(event.payload)}
            )
            st.session_state.active_assistant_message = ""
        elif event.event_type == "assistant_start":
            st.session_state.active_assistant_message = ""
        elif event.event_type == "assistant_token":
            st.session_state.active_assistant_message += str(event.payload)
            # Update or create the last assistant message
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "assistant"
            ):
                st.session_state.messages[-1][
                    "content"
                ] = st.session_state.active_assistant_message
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": st.session_state.active_assistant_message,
                    }
                )
        elif event.event_type == "metric_stt":
            st.session_state.latest_metrics["stt_time"] = float(event.payload)
        elif event.event_type == "metric_ttft":
            st.session_state.latest_metrics["ttft"] = float(event.payload)
        elif event.event_type == "metric_ttfa":
            st.session_state.latest_metrics["ttfa"] = float(event.payload)
            st.session_state.latest_metrics["total_time"] = (
                st.session_state.latest_metrics["stt_time"]
                + st.session_state.latest_metrics["ttfa"]
            )
        elif event.event_type == "error":
            st.session_state.last_error = str(event.payload)


def main() -> None:
    """Main Streamlit application rendering routine."""
    st.set_page_config(
        page_title="Talk-To-Me: Voice Assistant",
        page_icon="🎙️",
        layout="wide",
    )

    init_session_state()
    drain_event_queue()

    controller: PipelineThreadController = st.session_state.controller

    # Title and subtitle
    st.title("🎙️ Talk-To-Me: Voice Assistant")
    st.caption("Low-Latency Real-Time Voice Conversation with Short-Term Memory")

    # Layout columns: Left = Controls & Metrics, Right = Chat Display
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("Controls & Pipeline Status")

        # Status badge
        status_colors = {
            "Idle": "⚪ **Status**: `Idle`",
            "Listening": "🟢 **Status**: `Listening (Mic Active)`",
            "Processing": "🟠 **Status**: `Processing (STT / LLM)`",
            "Speaking": "🔵 **Status**: `Speaking (TTS Audio Out)`",
        }
        status_text = status_colors.get(
            st.session_state.status, f"⚪ **Status**: `{st.session_state.status}`"
        )
        st.markdown(status_text)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(
                "▶️ Start Listening",
                disabled=controller.is_running,
                use_container_width=True,
            ):
                st.session_state.last_error = None
                controller.start()
                st.rerun()

        with btn_col2:
            if st.button(
                "⏹️ Stop", disabled=not controller.is_running, use_container_width=True
            ):
                controller.stop()
                st.rerun()

        if st.session_state.last_error:
            st.error(f"⚠️ {st.session_state.last_error}")

        st.divider()

        st.subheader("📊 Latency Metrics (Latest Turn)")
        m = st.session_state.latest_metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("STT Transcribe", f"{m['stt_time']:.2f} s")
            st.metric("TTFA (First Audio)", f"{m['ttfa']:.2f} s")
        with m_col2:
            st.metric("TTFT (First Token)", f"{m['ttft']:.2f} s")
            st.metric("Est. Total Turnaround", f"{m['total_time']:.2f} s")

        st.info(
            "💡 **Note**: Microphone and audio playback run via your local hardware devices "
            "(sounddevice). Streamlit runs as the visual presenter.",
            icon="ℹ️",
        )

    with col_right:
        st.subheader("💬 Conversation History")
        chat_container = st.container(height=500)

        with chat_container:
            if not st.session_state.messages:
                st.markdown(
                    "*No conversation turns yet. Click **Start Listening** and speak into your microphone.*"
                )
            else:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

    # Auto-polling loop when active
    if controller.is_running:
        time.sleep(0.15)
        st.rerun()


if __name__ == "__main__":
    main()
