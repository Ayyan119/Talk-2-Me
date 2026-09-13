"""Module: main
Layer: Entry-Point
Purpose: Application bootstrap, dependency injection, environment configuration validation, and graceful shutdown handling.
Dependencies: asyncio, logging, os, signal, sys, dotenv, src.core.exceptions, src.memory, src.vad, src.stt, src.llm, src.tts, src.audio, src.pipeline
"""

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

from src.audio.sounddevice_io import SoundDeviceAudioIO
from src.core.exceptions import ConfigurationError, TalkToMeDomainError
from src.llm.openai_llm import OpenAILLM
from src.memory.sliding_window_memory import SlidingWindowMemory
from src.pipeline.voice_assistant import VoiceAssistantPipeline
from src.stt.faster_whisper_stt import FasterWhisperSTT
from src.tts.elevenlabs_tts import ElevenLabsTTS
from src.vad.silero_vad import SileroVoiceActivityDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("talk_to_me.main")


def load_and_validate_config() -> dict[str, str | float | int]:
    """Loads environment configuration and validates mandatory settings.

    Returns:
        Dictionary containing verified application configuration options.

    Raises:
        ConfigurationError: If any required credentials or settings are missing.
    """
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key or openai_key == "your-openai-api-key-here":
        raise ConfigurationError(
            "OPENAI_API_KEY is not set or contains default placeholder. Please configure .env"
        )

    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not elevenlabs_key or elevenlabs_key == "your-elevenlabs-api-key-here":
        raise ConfigurationError(
            "ELEVENLABS_API_KEY is not set or contains default placeholder. Please configure .env"
        )

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip() or "JBFqnCBsd6RMkjVDRZzb"
    whisper_model = os.getenv("WHISPER_MODEL_SIZE", "").strip() or "base"
    openai_model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    silence_raw = os.getenv("VAD_SILENCE_THRESHOLD_MS", "").strip()
    silence_threshold = float(silence_raw) if silence_raw else 600.0
    memory_turns_raw = os.getenv("MAX_MEMORY_TURNS", "").strip()
    max_memory_turns = int(memory_turns_raw) if memory_turns_raw else 10

    return {
        "openai_api_key": openai_key,
        "elevenlabs_api_key": elevenlabs_key,
        "elevenlabs_voice_id": voice_id,
        "whisper_model_size": whisper_model,
        "openai_model": openai_model,
        "silence_threshold_ms": silence_threshold,
        "max_memory_turns": max_memory_turns,
    }


async def main_async() -> None:
    """Asynchronous entry point bootstrapping all layers and starting the conversation loop."""
    print("=" * 65, flush=True)
    print("  Talk-To-Me: Voice-Operated LLM Assistant with Memory", flush=True)
    print("=" * 65, flush=True)
    print("Initializing system configuration and models...", flush=True)

    try:
        config = load_and_validate_config()
    except ConfigurationError as err:
        logger.critical("Configuration Error: %s", err)
        print(f"\n❌ Configuration Error: {err}", flush=True)
        print(
            "Please check your .env file and configure the required API keys.\n",
            flush=True,
        )
        sys.exit(1)

    try:
        logger.info("Loading Silero VAD adapter...")
        vad = SileroVoiceActivityDetector(
            silence_threshold_ms=float(config["silence_threshold_ms"]),
            speech_threshold=0.5,
        )

        logger.info("Loading FasterWhisper STT adapter on CPU...")
        stt = FasterWhisperSTT(
            model_size=str(config["whisper_model_size"]),
            device="cpu",
            compute_type="int8",
        )

        logger.info("Initializing OpenAILLM adapter...")
        llm = OpenAILLM(
            api_key=str(config["openai_api_key"]),
            model=str(config["openai_model"]),
            temperature=0.7,
        )

        logger.info("Initializing ElevenLabs TTS adapter...")
        tts = ElevenLabsTTS(
            api_key=str(config["elevenlabs_api_key"]),
            voice_id=str(config["elevenlabs_voice_id"]),
            model_id="eleven_flash_v2_5",
            output_format="pcm_24000",
        )

        logger.info("Initializing Conversation Memory...")
        memory = SlidingWindowMemory(
            max_turns=int(config["max_memory_turns"]),
            system_prompt=(
                "You are a helpful, friendly, and concise conversational voice assistant. "
                "Keep answers brief, conversational, and direct since they will be read aloud."
            ),
        )

        logger.info("Initializing SoundDevice Audio I/O...")
        audio_io = SoundDeviceAudioIO(
            output_sample_rate=24000,
            frame_size_samples=512,
        )

    except TalkToMeDomainError as err:
        logger.critical("Initialization Failure: %s", err)
        print(f"\n❌ Failed to initialize pipeline components: {err}\n", flush=True)
        sys.exit(1)
    except (OSError, RuntimeError, ValueError) as err:
        logger.critical("Startup Error: %s", err)
        print(f"\n❌ Unexpected error during startup: {err}\n", flush=True)
        sys.exit(1)

    pipeline = VoiceAssistantPipeline(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        memory=memory,
        audio_io=audio_io,
    )

    loop = asyncio.get_running_loop()

    def handle_shutdown() -> None:
        print("\n\nShutting down voice assistant...", flush=True)
        asyncio.create_task(pipeline.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            pass

    print("-" * 65, flush=True)
    print(
        "🎙️ Voice assistant is ready. Start speaking into your microphone!", flush=True
    )
    print("💡 Press Ctrl+C to stop the assistant and exit.", flush=True)
    print("-" * 65, flush=True)

    try:
        await pipeline.run_conversation_loop()
    except asyncio.CancelledError:
        pass
    finally:
        print("Goodbye! Voice assistant has stopped cleanly.", flush=True)


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nShutting down... Goodbye!", flush=True)
    except Exception as exc:
        print(f"\n❌ Unhandled error in main: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
