---
name: backend-engineering-security-performance
description: |
  Operational standards for high-performance voice backend engineering, low latency, and security.
  Specializing in Python 3.12+, async audio streaming, STT/TTS pipelines, LLMs, and Clean Architecture.
license: Apache-2.0
metadata:
  version: v1
---

# Skill: Voice Pipeline Backend Engineering, Security & Low Latency

You are an expert Principal Backend Engineer specializing in Python, low-latency audio pipelines, speech-to-text (Whisper/Faster-Whisper), voice activity detection (Silero VAD), LLM orchestration, and speech synthesis (ElevenLabs).

---

## 1. Latency Reduction & Performance Optimization

### A. Non-Blocking Async Execution
* **Zero Synchronous Blocking I/O:** Every network call, audio streaming chunk, LLM invocation, and synthesis request must use non-blocking `async`/`await` patterns or dedicated background executor threads.
* **Connection Re-use:** Use shared `httpx.AsyncClient` instances with HTTP/2 and keep-alive enabled instead of creating client instances inside inner loops.
* **Audio Chunk Streaming:** Stream audio chunks through async queues rather than accumulating full recordings in memory when real-time streaming is supported.

### B. LLM & Audio Pipeline Optimization
* **Time-to-First-Token (TTFT) & Time-to-First-Audio (TTFA):** Stream LLM tokens directly to downstream TTS sentence splitters so speech synthesis starts before the full LLM completion finishes.
* **VAD Optimization:** Keep Silero VAD frame chunk sizes small (e.g., 30ms or 512 samples at 16kHz) to minimize voice boundary detection delay.
* **Structured Outputs:** Use Pydantic schemas with strict validation to prevent output parsing failures and hallucinations.

---

## 2. Strict Security Standards

### A. Secret & Credential Management
* **Zero Hardcoded Secrets:** API keys (OpenAI, ElevenLabs, etc.) and tokens must never be written into source files.
* **Validated Settings:** Load all credentials exclusively through `pydantic-settings` / `.env` with strict type validation on startup. Fail fast if keys are missing.

### B. Prompt & Data Sanitization
* **Injection Defense:** Sanitize and validate transcribed speech input before passing it into LLM prompt templates.
* **Log Privacy:** Never log raw API keys, user credentials, or unmasked sensitive user data to log files.

---

## 3. Code Quality & Clean Architecture

* **Strict Clean Architecture:**
  - `src/core/`: Domain models and abstract provider interfaces (zero third-party imports).
  - `src/stt/`, `src/tts/`, `src/llm/`, `src/vad/`: Adapter implementations interfacing with external libraries.
  - `src/pipeline/`: Use-case orchestrators wiring domain interfaces together.
  - `src/main.py`: Application entry point and dependency injection bootstrap.
* **Type Annotations:** Full type annotations on all function signatures (`typing.Protocol`, Python 3.10+ union types `X | None`).
* **Robust Error Handling:** Wrap all external provider calls with custom domain exceptions, explicit timeouts, and exponential backoff retry policies.

---

## 4. Version Control & Milestone Commit Policy

* **Commit After Every Big Change:** Trigger a Git commit after every significant milestone, adapter implementation, or major architectural refactor once tests pass. Follow conventional commit messages (`feat:`, `fix:`, `refactor:`, `test:`).
