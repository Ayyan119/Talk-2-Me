"""Module: fastapi_server
Layer: Presentation / Server
Purpose: Production FastAPI server exposing real-time duplex WebSocket /ws/voice, REST endpoints, and integrated Voice Assistant Web App.
Dependencies: asyncio, base64, logging, typing, fastapi, starlette, uvicorn, src.core.types, src.core.exceptions, src.pipeline.factory, src.server.websocket_server, src.server.call_component
"""

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.core.types import AudioChunk, Message, MessageRole
from src.pipeline.factory import (
    build_voice_assistant_pipeline,
    load_and_validate_config,
)
from src.server.call_component import get_voice_call_html
from src.server.websocket_server import ClientSession, decode_audio_to_pcm

logger = logging.getLogger("talk_to_me.fastapi")

app = FastAPI(
    title="Talk-To-Me: Real-Time Voice Assistant API",
    description="Ultra-low latency streaming voice chatbot API powered by Deepgram/Whisper STT, GPT-4o, and ElevenLabs TTS.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None
_config = None


def get_pipeline():
    """Lazily initializes and returns the singleton VoiceAssistantPipeline instance."""
    global _pipeline, _config
    if _pipeline is None:
        _config = load_and_validate_config()
        _pipeline, _config = build_voice_assistant_pipeline(config=_config)
    return _pipeline, _config


class TextChatRequest(BaseModel):
    text: str


class TextChatResponse(BaseModel):
    reply: str
    metrics: dict[str, float]


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "talk-to-me-voice-assistant"}


@app.get("/api/status")
async def get_system_status() -> dict[str, Any]:
    """Returns runtime pipeline status and active provider configuration."""
    try:
        pipeline, config = get_pipeline()
        return {
            "status": "ready",
            "stt_provider": config.get("stt_provider"),
            "openai_model": config.get("openai_model"),
            "elevenlabs_voice_id": config.get("elevenlabs_voice_id"),
            "vad_silence_threshold_ms": config.get("silence_threshold_ms"),
            "max_memory_turns": config.get("max_memory_turns"),
        }
    except Exception as err:
        return {"status": "error", "error": str(err)}


@app.post("/api/chat", response_model=TextChatResponse)
async def chat_endpoint(req: TextChatRequest) -> TextChatResponse:
    """Processes a text query and returns the assistant response."""
    clean_text = req.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    pipeline, _ = get_pipeline()
    t_start = asyncio.get_event_loop().time()
    reply = await pipeline.process_text_message(clean_text)
    total_time = asyncio.get_event_loop().time() - t_start

    return TextChatResponse(
        reply=reply,
        metrics={"total_time": total_time},
    )


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket) -> None:
    """Full-duplex real-time voice streaming WebSocket endpoint."""
    await websocket.accept()
    pipeline, config = get_pipeline()
    session = ClientSession(
        websocket=websocket,  # type: ignore[arg-type]
        pipeline=pipeline,
        config=config,
    )
    logger.info("FastAPI Client connected to /ws/voice")

    try:
        while True:
            raw_msg = await websocket.receive_text()
            # Feed message into the handler
            import json

            data = json.loads(raw_msg)
            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                audio_b64 = data.get("data", "")
                import base64

                pcm_bytes = base64.b64decode(audio_b64)
                if pcm_bytes:
                    chunk = AudioChunk(
                        data=pcm_bytes,
                        sample_rate=16000,
                        channels=1,
                        duration_seconds=len(pcm_bytes) / 32000.0,
                    )
                    vad_res = pipeline._vad.process_chunk(chunk)
                    if vad_res.is_speech:
                        if not session.is_user_speaking:
                            session.is_user_speaking = True
                            if session.is_assistant_speaking:
                                await session.interrupt()
                        session.audio_buffer.append(pcm_bytes)
                    elif session.is_user_speaking:
                        session.audio_buffer.append(pcm_bytes)

                    if vad_res.is_end_of_speech and session.is_user_speaking:
                        session.is_user_speaking = False
                        pipeline._vad.reset()
                        full_pcm = b"".join(session.audio_buffer)
                        session.audio_buffer.clear()
                        if len(full_pcm) >= 1600:
                            t0 = asyncio.get_event_loop().time()
                            transcription = await pipeline._stt.transcribe(
                                AudioChunk(
                                    data=full_pcm,
                                    sample_rate=16000,
                                    channels=1,
                                    duration_seconds=len(full_pcm) / 32000.0,
                                )
                            )
                            stt_time = asyncio.get_event_loop().time() - t0
                            user_text = transcription.text.strip()
                            if user_text:
                                session.turn_id += 1
                                turn_id = session.turn_id
                                session.active_turn_task = asyncio.create_task(
                                    session.process_user_turn(
                                        user_text,
                                        stt_time,
                                        t_start=t0,
                                        turn_id=turn_id,
                                    )
                                )

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
                            t_start=asyncio.get_event_loop().time(),
                            turn_id=turn_id,
                        )
                    )

            elif msg_type == "clear_memory":
                mem_clear = pipeline._memory.clear()
                if asyncio.iscoroutine(mem_clear):
                    await mem_clear
                await websocket.send_json({"type": "memory_cleared"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("FastAPI voice WebSocket disconnected")
    except Exception as exc:
        logger.error("FastAPI WebSocket error: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def get_web_ui() -> HTMLResponse:
    """Renders the embedded real-time Voice Call Web UI."""
    return HTMLResponse(content=get_voice_call_html(ws_port=8503))
