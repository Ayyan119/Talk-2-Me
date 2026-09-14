"""Module: factory
Layer: Composition/Bootstrap
Purpose: Shared factory functions for assembling and configuring VoiceAssistantPipeline with concrete domain adapters.
Dependencies: logging, os, typing, dotenv, src.audio.sounddevice_io, src.core.audio_io, src.core.exceptions, src.llm.openai_llm, src.memory.sliding_window_memory, src.pipeline.voice_assistant, src.stt.faster_whisper_stt, src.tts.elevenlabs_tts, src.vad.silero_vad
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

from src.core.audio_io import AudioIO
from src.core.exceptions import ConfigurationError
from src.llm.openai_llm import OpenAILLM
from src.memory.sliding_window_memory import SlidingWindowMemory
from src.pipeline.voice_assistant import VoiceAssistantPipeline
from src.stt.deepgram_stt import DeepgramSTT
from src.stt.faster_whisper_stt import FasterWhisperSTT
from src.stt.openai_whisper_stt import OpenAIWhisperSTT
from src.tts.elevenlabs_tts import ElevenLabsTTS
from src.vad.silero_vad import SileroVoiceActivityDetector

logger = logging.getLogger("talk_to_me.factory")


def load_and_validate_config() -> dict[str, str | float | int]:
    """Loads environment configuration and validates mandatory settings."""
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

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip() or "JBFqnCBsd6RMkjVDRZzb"
    whisper_model = os.getenv("WHISPER_MODEL_SIZE", "").strip() or "base"

    default_stt = (
        "deepgram"
        if (deepgram_key and deepgram_key != "your-deepgram-api-key-here")
        else "openai"
    )
    stt_provider = os.getenv("STT_PROVIDER", default_stt).strip().lower()

    openai_model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o"
    silence_raw = os.getenv("VAD_SILENCE_THRESHOLD_MS", "").strip()
    silence_threshold = float(silence_raw) if silence_raw else 600.0
    memory_turns_raw = os.getenv("MAX_MEMORY_TURNS", "").strip()
    max_memory_turns = int(memory_turns_raw) if memory_turns_raw else 10

    return {
        "openai_api_key": openai_key,
        "elevenlabs_api_key": elevenlabs_key,
        "deepgram_api_key": deepgram_key,
        "elevenlabs_voice_id": voice_id,
        "whisper_model_size": whisper_model,
        "stt_provider": stt_provider,
        "openai_model": openai_model,
        "silence_threshold_ms": silence_threshold,
        "max_memory_turns": max_memory_turns,
    }


def build_voice_assistant_pipeline(
    config: dict[str, Any] | None = None,
    audio_io: AudioIO | None = None,
    custom_logger: logging.Logger | None = None,
) -> tuple[VoiceAssistantPipeline, dict[str, Any]]:
    """Constructs and wires all domain adapters into a VoiceAssistantPipeline instance."""
    resolved_config = config if config is not None else load_and_validate_config()

    logger.info("Initializing Silero VAD adapter...")
    vad = SileroVoiceActivityDetector(
        silence_threshold_ms=float(resolved_config["silence_threshold_ms"]),
        speech_threshold=0.5,
    )

    stt_choice = str(resolved_config.get("stt_provider", "openai")).lower()
    deepgram_key = str(resolved_config.get("deepgram_api_key", "")).strip()

    if (
        stt_choice == "deepgram"
        and deepgram_key
        and deepgram_key != "your-deepgram-api-key-here"
    ):
        logger.info("Initializing Deepgram Streaming STT adapter (Nova-2)...")
        stt = DeepgramSTT(
            api_key=deepgram_key,
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
            interim_results=True,
            endpointing_ms=250,
        )
    elif stt_choice == "faster-whisper":
        logger.info("Initializing FasterWhisper STT adapter on CPU...")
        stt = FasterWhisperSTT(
            model_size=str(resolved_config["whisper_model_size"]),
            device="cpu",
            compute_type="int8",
        )
    else:
        logger.info("Initializing OpenAI Whisper STT (whisper-1 cloud GPU)...")
        stt = OpenAIWhisperSTT(
            api_key=str(resolved_config["openai_api_key"]),
            model="whisper-1",
            temperature=0.0,
        )

    logger.info("Initializing OpenAILLM adapter...")
    llm = OpenAILLM(
        api_key=str(resolved_config["openai_api_key"]),
        model=str(resolved_config["openai_model"]),
        temperature=0.7,
    )

    logger.info("Initializing ElevenLabs TTS adapter...")
    tts = ElevenLabsTTS(
        api_key=str(resolved_config["elevenlabs_api_key"]),
        voice_id=str(resolved_config["elevenlabs_voice_id"]),
        model_id="eleven_flash_v2_5",
        output_format="pcm_24000",
    )

    logger.info("Initializing SlidingWindowMemory...")
    memory = SlidingWindowMemory(
        max_turns=int(resolved_config["max_memory_turns"]),
        system_prompt=(
            "You are a friendly, natural human conversational companion on a live voice call. "
            "Speak naturally and warmly in everyday conversational English. "
            "Keep answers concise, engaging, and under two sentences so the conversation flows seamlessly. "
            "Never use bullet points, markdown symbols, asterisks, emojis, or numbered lists since your reply is read aloud."
        ),
    )

    if audio_io is not None:
        resolved_audio_io = audio_io
    else:
        from src.audio.sounddevice_io import SoundDeviceAudioIO

        resolved_audio_io = SoundDeviceAudioIO(
            output_sample_rate=24000,
            frame_size_samples=512,
        )

    pipeline = VoiceAssistantPipeline(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        memory=memory,
        audio_io=resolved_audio_io,
        custom_logger=custom_logger,
    )

    return pipeline, resolved_config
