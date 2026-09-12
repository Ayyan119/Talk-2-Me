"""Script: manual_listen_test
Layer: Diagnostic / Tooling
Purpose: Interactive diagnostic CLI tool to test microphone capture, Silero VAD, and FasterWhisper STT end-to-end.
Dependencies: asyncio, time, sounddevice, numpy, src.core.types, src.core.exceptions, src.vad.silero_vad, src.stt.faster_whisper_stt
"""

import asyncio
import signal
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import sounddevice as sd

from src.core.exceptions import (
    AudioProcessingError,
    TalkToMeDomainError,
    TranscriptionError,
    VoiceActivityDetectionError,
)
from src.core.types import AudioChunk, AudioFormat
from src.stt.faster_whisper_stt import FasterWhisperSTT
from src.vad.silero_vad import SileroVoiceActivityDetector

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM (int16)
FRAME_SIZE_SAMPLES = 512  # Silero VAD 16kHz window size (32ms per frame)
FRAME_DURATION_SECONDS = FRAME_SIZE_SAMPLES / SAMPLE_RATE  # 0.032s


async def run_listener() -> None:
    """Runs the real-time microphone listening loop with VAD and STT."""
    print("=" * 60)
    print("  Talk-To-Me: Manual Audio Pipeline Diagnostic Tool")
    print("=" * 60)
    print("Initializing models...")

    try:
        t0 = time.perf_counter()
        vad = SileroVoiceActivityDetector(
            silence_threshold_ms=600.0,
            speech_threshold=0.5,
            audio_format=AudioFormat(
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                sample_width_bytes=SAMPLE_WIDTH_BYTES,
            ),
        )
        vad_load_time = time.perf_counter() - t0
        print(f"✓ Silero VAD loaded ({vad_load_time:.2f}s)")

        t0 = time.perf_counter()
        stt = FasterWhisperSTT(
            model_size="base",
            device="cpu",
            compute_type="int8",
            audio_format=AudioFormat(
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                sample_width_bytes=SAMPLE_WIDTH_BYTES,
            ),
        )
        stt_load_time = time.perf_counter() - t0
        print(f"✓ FasterWhisper STT loaded [base/int8/cpu] ({stt_load_time:.2f}s)")
    except (TalkToMeDomainError, RuntimeError, ValueError) as err:
        print(f"Error during initialization: {err}")
        return

    print("-" * 60)
    print("🎤 Microphone is active. Start speaking...")
    print("💡 Press Ctrl+C to stop listening and exit.")
    print("-" * 60)

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: Mapping[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        """Callback function invoked by sounddevice for each captured audio frame."""
        if status:
            print(f"[Warning] Audio callback status: {status}", file=sys.stderr)
        raw_bytes = indata.tobytes()
        loop.call_soon_threadsafe(audio_queue.put_nowait, raw_bytes)

    stop_event = asyncio.Event()

    def handle_sigint() -> None:
        """Handles SIGINT/Ctrl+C signals."""
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_sigint)
        except NotImplementedError:
            pass

    speech_buffer: list[bytes] = []
    is_speaking = False
    total_vad_time = 0.0
    vad_frame_count = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=FRAME_SIZE_SAMPLES,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    frame_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                chunk = AudioChunk(
                    data=frame_bytes,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    duration_seconds=FRAME_DURATION_SECONDS,
                )

                try:
                    t_vad_start = time.perf_counter()
                    vad_result = vad.process_chunk(chunk)
                    t_vad_elapsed = time.perf_counter() - t_vad_start

                    if vad_result.is_speech:
                        if not is_speaking:
                            is_speaking = True
                            print(
                                "\n[Speech Detected] Listening to utterance...",
                                end="",
                                flush=True,
                            )
                        speech_buffer.append(frame_bytes)
                        total_vad_time += t_vad_elapsed
                        vad_frame_count += 1
                    elif is_speaking:
                        # Continue buffering silence during an ongoing utterance until end of turn
                        speech_buffer.append(frame_bytes)
                        total_vad_time += t_vad_elapsed
                        vad_frame_count += 1

                    if vad_result.is_end_of_speech and is_speaking:
                        print(" [End of Utterance]")
                        if speech_buffer:
                            full_pcm = b"".join(speech_buffer)
                            total_duration = len(full_pcm) / (
                                SAMPLE_RATE * SAMPLE_WIDTH_BYTES
                            )
                            speech_chunk = AudioChunk(
                                data=full_pcm,
                                sample_rate=SAMPLE_RATE,
                                channels=CHANNELS,
                                duration_seconds=total_duration,
                            )

                            try:
                                t_stt_start = time.perf_counter()
                                transcription = await stt.transcribe(speech_chunk)
                                t_stt_elapsed = time.perf_counter() - t_stt_start

                                print(f'👉 You said: "{transcription.text}"')
                                avg_vad_latency = (
                                    (total_vad_time / vad_frame_count * 1000.0)
                                    if vad_frame_count > 0
                                    else 0.0
                                )
                                print(
                                    f"📊 Metrics: Audio={total_duration:.2f}s | "
                                    f"Avg VAD per frame={avg_vad_latency:.2f}ms | "
                                    f"STT Latency={t_stt_elapsed:.2f}s "
                                    f"(Language={transcription.language})"
                                )
                                print("-" * 60)
                            except TranscriptionError as err:
                                print(f"[Error] STT Transcription failed: {err}")

                        vad.reset()
                        speech_buffer.clear()
                        is_speaking = False
                        total_vad_time = 0.0
                        vad_frame_count = 0

                except VoiceActivityDetectionError as err:
                    print(f"\n[Error] VAD processing error: {err}")
                    vad.reset()
                    speech_buffer.clear()
                    is_speaking = False

    except sd.PortAudioError as err:
        print(f"\n[Error] Failed to open microphone stream: {err}")
        print("Please check that an input audio device is connected and accessible.")
    except (AudioProcessingError, OSError, RuntimeError) as err:
        print(f"\n[Error] Audio capture error: {err}")
    finally:
        print("\n\nStopped listening. Goodbye!")


def main() -> None:
    """Entry point for the manual listen test diagnostic script."""
    try:
        asyncio.run(run_listener())
    except KeyboardInterrupt:
        print("\nStopped listening. Goodbye!")


if __name__ == "__main__":
    main()
