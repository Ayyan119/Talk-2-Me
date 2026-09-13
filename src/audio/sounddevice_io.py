"""Module: sounddevice_io
Layer: Adapter/Implementation
Purpose: Concrete AudioIO adapter for microphone input capture and speaker playback via sounddevice.
Dependencies: asyncio, collections.abc, logging, numpy, sounddevice, src.core.types, src.core.exceptions, src.core.audio_io
"""

import asyncio
import logging
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import sounddevice as sd

from src.core.audio_io import AudioIO
from src.core.exceptions import AudioProcessingError
from src.core.types import AudioChunk, AudioFormat, SynthesisChunk

logger = logging.getLogger("talk_to_me.audio")


class SoundDeviceAudioIO(AudioIO):
    """Audio I/O adapter using sounddevice for live microphone streaming and speaker playback."""

    def __init__(
        self,
        input_format: AudioFormat | None = None,
        output_sample_rate: int = 24000,
        frame_size_samples: int = 512,
    ) -> None:
        """Initializes the SoundDevice Audio I/O adapter.

        Args:
            input_format: Format specification for microphone capture (default 16kHz mono 16-bit).
            output_sample_rate: Output sampling rate for playback (default 24000 Hz).
            frame_size_samples: Samples per captured frame (512 samples = 32ms at 16kHz).
        """
        self._input_format = (
            input_format
            if input_format is not None
            else AudioFormat(sample_rate=16000, channels=1, sample_width_bytes=2)
        )
        self._output_sample_rate = output_sample_rate
        self._frame_size_samples = frame_size_samples
        self._frame_duration_seconds = (
            frame_size_samples / self._input_format.sample_rate
        )

        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._is_running: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_running(self) -> bool:
        """Returns whether audio streams are active."""
        return self._is_running

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Mapping[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        """Callback triggered by sounddevice on every captured audio frame."""
        if status:
            print(
                f"[AudioIO Warning] Input stream status: {status}",
                file=sys.stderr,
            )
        if self._loop and self._is_running:
            raw_bytes = indata.tobytes()
            self._loop.call_soon_threadsafe(self._queue.put_nowait, raw_bytes)

    async def start(self) -> None:
        """Starts input and output sounddevice streams.

        Raises:
            AudioProcessingError: If stream creation or hardware access fails.
        """
        if self._is_running:
            return

        self._loop = asyncio.get_running_loop()
        try:
            self._input_stream = sd.InputStream(
                samplerate=self._input_format.sample_rate,
                channels=self._input_format.channels,
                dtype="int16",
                blocksize=self._frame_size_samples,
                callback=self._audio_callback,
            )
            self._input_stream.start()

            self._output_stream = sd.OutputStream(
                samplerate=self._output_sample_rate,
                channels=1,
                dtype="int16",
            )
            self._output_stream.start()
            self._is_running = True
        except (sd.PortAudioError, OSError, RuntimeError) as err:
            self._is_running = False
            raise AudioProcessingError(
                f"Failed to start sounddevice audio streams: {err}"
            ) from err

    async def read_frame(self) -> AudioChunk:
        """Reads the next available microphone audio frame.

        Returns:
            An AudioChunk containing raw 16-bit PCM audio frame bytes.

        Raises:
            AudioProcessingError: If streams are stopped or reading fails.
        """
        if not self._is_running:
            raise AudioProcessingError(
                "Cannot read audio frame: AudioIO stream is not running"
            )

        try:
            data_bytes = await self._queue.get()
            return AudioChunk(
                data=data_bytes,
                sample_rate=self._input_format.sample_rate,
                channels=self._input_format.channels,
                duration_seconds=self._frame_duration_seconds,
            )
        except (asyncio.CancelledError, RuntimeError) as err:
            raise AudioProcessingError(
                f"Failed to read audio frame from queue: {err}"
            ) from err

    async def play_chunk(self, chunk: SynthesisChunk) -> None:
        """Plays a synthesized audio chunk out to the speaker device.

        Args:
            chunk: SynthesisChunk containing PCM audio bytes.

        Raises:
            AudioProcessingError: If playback encounters a failure.
        """
        if not self._is_running or self._output_stream is None:
            raise AudioProcessingError(
                "Cannot play audio chunk: AudioIO output stream is not running"
            )

        if not chunk.audio:
            return

        try:
            samples = np.frombuffer(chunk.audio, dtype=np.int16)
            await asyncio.to_thread(self._output_stream.write, samples)
        except (sd.PortAudioError, OSError, RuntimeError) as err:
            raise AudioProcessingError(f"Audio playback failed: {err}") from err

    async def stop(self) -> None:
        """Stops active input and output streams and cleans up hardware resources."""
        self._is_running = False

        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except (sd.PortAudioError, OSError, RuntimeError) as err:
                logger.debug("Error stopping input stream: %s", err)
            self._input_stream = None

        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except (sd.PortAudioError, OSError, RuntimeError) as err:
                logger.debug("Error stopping output stream: %s", err)
            self._output_stream = None

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
