"""Module: main
Layer: Entry-Point
Purpose: Application bootstrap, dependency injection, environment configuration validation, and graceful shutdown handling.
Dependencies: asyncio, logging, os, signal, sys, dotenv, src.core.exceptions, src.memory, src.vad, src.stt, src.llm, src.tts, src.audio, src.pipeline
"""

import asyncio
import logging
import signal
import sys

from src.core.exceptions import ConfigurationError, TalkToMeDomainError
from src.pipeline.factory import (
    build_voice_assistant_pipeline,
    load_and_validate_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("talk_to_me.main")


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
        pipeline, _ = build_voice_assistant_pipeline(config=config)
    except TalkToMeDomainError as err:
        logger.critical("Initialization Failure: %s", err)
        print(f"\n❌ Failed to initialize pipeline components: {err}\n", flush=True)
        sys.exit(1)
    except (OSError, RuntimeError, ValueError) as err:
        logger.critical("Startup Error: %s", err)
        print(f"\n❌ Unexpected error during startup: {err}\n", flush=True)
        sys.exit(1)

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
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Unhandled error in main: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
