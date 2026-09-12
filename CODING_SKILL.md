# Coding Standards & Architecture Skill: `talk-to-me`

## Overview & Purpose
This document establishes the mandatory architectural rules, coding standards, reliability guarantees, security practices, testability constraints, and tooling configurations for the `talk-to-me` voice-operated LLM Q&A system. All future implementation code across the codebase must strictly adhere to these specifications.

---

## 1. Architectural Guidelines (Clean Architecture)

The codebase strictly enforces a Clean Architecture structure divided into four distinct concentric layers. Dependencies must only point inward toward the domain core.

```
[Layer 4: Entry Point (main.py)]
           │
           ▼
[Layer 3: Orchestration / Use Cases (Pipeline Logic)]
           │
           ▼
[Layer 2: Adapters / Implementations (APIs, Audio I/O, Hardware)]
           │
           ▼
[Layer 1: Core / Domain Interfaces (Contracts & Entities - Zero External Dependencies)]
```

### 1.1 Layer Definitions & Responsibilities
1. **Layer 1: Core / Domain Layer (`src/core/` or domain protocols)**
   - Contains pure Python domain models, data transfer objects, and abstract interface contracts (e.g., `typing.Protocol` or `abc.ABC`) for:
     - Speech-to-Text (STT)
     - Voice Activity Detection (VAD)
     - Large Language Model (LLM)
     - Text-to-Speech (TTS)
     - Short-Term / Conversation Memory
   - **Strict Rule:** Zero third-party dependencies (no imports of `faster_whisper`, `openai`, `elevenlabs`, `sounddevice`, `torch`, etc.). Standard library only.

2. **Layer 2: Adapters / Implementation Layer (`src/stt/`, `src/llm/`, `src/tts/`, `src/vad/`, `src/audio/`, `src/memory/`)**
   - Contains the concrete classes implementing the domain interfaces.
   - Bridges domain contracts to specific external libraries and third-party APIs (e.g., OpenAI API, ElevenLabs SDK, Faster-Whisper, Silero VAD, SoundDevice).
   - Encapsulates network communication, hardware capture/playback, serialization, and vendor-specific data structures.

3. **Layer 3: Orchestration / Use Cases (`src/pipeline/` or `src/orchestration/`)**
   - Implements business logic workflows (e.g., listening for speech -> detecting end-of-speech -> transcribing -> querying memory -> prompting LLM -> generating TTS stream -> audio output).
   - Interacts strictly with domain interfaces supplied via dependency injection.
   - Never directly instantiates concrete third-party adapters.

4. **Layer 4: Entry Point (`src/main.py`)**
   - Responsible solely for application bootstrapping:
     - Loading and validating environment variables / configuration.
     - Instantiating concrete adapters.
     - Injecting dependencies into use-case orchestrators.
     - Handling CLI arguments, signal trapping (`SIGINT`, `SIGTERM`), and graceful startup/shutdown.

### 1.2 Modularity & Class Responsibilities
- **Single Responsibility Principle:** One primary responsibility per file/class.
- **No God Files / God Classes:** Orchestrators, state machines, and I/O handlers must remain distinct.

---

## 2. Code Style & Conventions

### 2.1 Type Hints
- Full type annotations on **every** function signature without exception (including parameters, default values, return types, and class attributes).
- Use standard Python 3.10+ typing syntax (`list[str]`, `dict[str, Any]`, `X | None`, `typing.Protocol`, `typing.Callable`).
- Avoid unconstrained `Any` where a specific generic, union, or domain model can be defined.

### 2.2 Docstrings & Documentation
- Use the **Google Python Style** docstring format across all public modules, classes, methods, and functions.
- Every docstring must specify:
  - Concise description of purpose.
  - `Args:` Detailed explanation of all parameters and expected types.
  - `Returns:` Explicit description of returned value and format.
  - `Raises:` All expected domain or infrastructure exceptions.

### 2.3 Comments & Naming
- **Inline Comments:** Use exclusively for non-obvious design rationale ("why", never "what"). Code must be self-documenting.
- **Naming Conventions:**
  - `PascalCase` for classes and protocols (`SpeechToTextProvider`, `AudioBuffer`).
  - `snake_case` for functions, methods, modules, and variables (`transcribe_chunk`, `session_id`).
  - `UPPER_SNAKE_CASE` for constants (`DEFAULT_TIMEOUT_SECONDS`).
  - **Abbreviations:** Prohibited except for standard project domain acronyms (`stt`, `tts`, `llm`, `vad`). No arbitrary truncated words (e.g., use `configuration` not `cfg_val`, `conversation` not `conv`).

### 2.4 Function Complexity & Length
- Function length must not exceed **30 to 40 lines**. Any function approaching this threshold must be refactored into smaller, cohesive private helper methods.
- Cyclomatic complexity should remain minimal: avoid deeply nested conditional branches.

### 2.5 File Header Template
Every source file must begin with a standardized module header comment stating its purpose, layer, and expected dependencies:

```python
"""Module: <module_name>
Layer: <Core/Domain | Adapter/Implementation | Orchestration/Use-Case | Entry-Point>
Purpose: <1-2 sentences explaining what this file does>
Dependencies: <List of internal contracts or external libraries expected>
"""
```

---

## 3. Error Handling, Reliability & Logging

### 3.1 Exception Management
- **No Bare Excepts:** Never use `except:` or catch unhandled generic `Exception` without re-raising or wrapping into a domain error.
- **Custom Domain Exceptions:** Create hierarchical domain exceptions (e.g., `AudioCaptureError`, `TranscriptionError`, `LLMProviderError`, `SynthesisError`, `ConfigurationError`).
- Wrap low-level third-party exceptions inside domain-specific error types with clear, actionable context messages.

### 3.2 Network Calls & Timeouts
- **Mandatory Timeouts:** Every external HTTP request, WebSocket connection, or API client call must define explicit connection and read timeouts. Indefinite blocking calls are strictly prohibited.

### 3.3 Retry Policies & Failure Modes
- Network calls and remote API requests must implement exponential backoff with jitter for transient errors (HTTP 429, 503, connection resets).
- Max retries must be capped (e.g., 3 attempts).
- **Fail Loudly:** If retries are exhausted, the operation must raise a descriptive domain exception. Silent failures or swallowed exceptions are forbidden.

### 3.4 Logging Standard
- Use Python's built-in `logging` module.
- Never use `print()` statements for diagnostic, informational, or error reporting.
- Log statements must include operational context (e.g., session ID, stage name, execution time, retry attempt).
- Distinguish log severity levels:
  - `DEBUG`: Internal state transitions, raw timings, payload sizes.
  - `INFO`: Lifecycle milestones (session initialized, recording started, response synthesis begun).
  - `WARNING`: Recoverable retries, non-fatal fallback actions.
  - `ERROR`: Pipeline failures, unrecoverable adapter errors.
  - `CRITICAL`: Missing required configurations, startup failure.

---

## 4. Security & Data Protection

### 4.1 Secret Management
- **Zero Hardcoded Credentials:** API keys, tokens, endpoints, and credentials must never exist in source code or default configuration files.
- Configuration must load exclusively from environment variables / `.env` files via structured settings (e.g., `pydantic-settings`).
- **Fail Fast:** The application must validate all mandatory environment variables during bootstrap and immediately exit with an explicit error if any required configuration is absent.

### 4.2 Prompt & Input Sanitization
- All transcribed user speech must be sanitized and validated (length bounds, character filtering, structured formatting) before ingestion into LLM prompt templates to mitigate prompt injection and malformed inputs.

### 4.3 Log Privacy
- API keys, authorization headers, raw personally identifiable information (PII), and sensitive plaintext messages must never be written to disk logs.
- Sensitive values must be masked (e.g., `sk-...abcd`) if logged during debug cycles.

---

## 5. Testability & Quality Assurance

### 5.1 Unit Isolation & Mocking
- All business logic in Layer 1 and Layer 3 must be testable in total isolation without requiring physical hardware (microphones/speakers) or live API access.
- Every external dependency (SoundDevice, OpenAI, ElevenLabs, Whisper) must be mockable via domain interfaces or test doubles.

### 5.2 Test Scaffold Structure
- The `tests/` directory must mirror the `src/` directory layout:
  - `tests/unit/core/`
  - `tests/unit/stt/`
  - `tests/unit/llm/`
  - `tests/unit/tts/`
  - `tests/unit/vad/`
  - `tests/unit/memory/`
  - `tests/unit/pipeline/`
  - `tests/integration/`
- Every core module must have a corresponding test file scaffolded with structured test suites (`pytest` and `pytest-asyncio`).

---

## 6. Tooling & Enforcement Configuration

### 6.1 Code Formatter (`black`)
- Line length: `88` characters.
- Target Python version: `py312` (compatible with `>=3.10`).

```toml
[tool.black]
line-length = 88
target-version = ['py310', 'py311', 'py312']
include = '\.pyi?$'
```

### 6.2 Linter (`ruff`)
- Strict linting rules enabled:
  - `E` / `W`: Pycodestyle errors and warnings
  - `F`: Pyflakes
  - `I`: isort (import sorting)
  - `B`: flake8-bugbear (common bug prevention)
  - `UP`: pyupgrade (modern Python syntax)
  - `D`: pydocstyle (docstring enforcement using Google style)

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "D"]
ignore = ["D100", "D104"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

### 6.3 Test Runner (`pytest`)
```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## 7. Version Control & Milestone Commit Policy

- **Commit After Every Big Change:** A Git commit must be executed immediately following the completion of every major feature, significant refactor, new adapter/pipeline step, or verified milestone (with all tests passing).
- **Commit Workflow:**
  1. Verify changes with tests and code formatting.
  2. Check status (`git status`).
  3. Stage all modified files (`git add .`).
  4. Commit using standard Conventional Commit messages (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
  5. Verify commit (`git log -1 --oneline`).
- **Atomic Commits:** Keep commits cohesive and scoped to the specific big change completed.
