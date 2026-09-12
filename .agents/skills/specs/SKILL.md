---
name: specs
description: Create a technical spec document for the next voice pipeline component or feature step
argument-hint: "Step number and component name e.g. 01 audio-capture"
---

# Voice Pipeline Feature Spec Skill

You are a Principal Software Architect designing modular components for the `talk-to-me` voice-operated LLM Q&A system.
Always adhere strictly to Clean Architecture and the specifications defined in `CODING_SKILL.md`.

## Workflow

### Step 1: Parse Arguments & Context
From user arguments extract:
- `step_number`: 2-digit zero-padded number (e.g. `01`, `02`)
- `feature_slug`: lowercase kebab-case identifier (e.g. `audio-stream-capture`, `vad-silero-adapter`, `whisper-stt-adapter`)
- `feature_title`: Human-readable title in Title Case

### Step 2: Spec Document Structure
Write a spec file at `specs/<step_number>-<feature_slug>.md` with the following structure:

1. **Overview & Goal:** Purpose of this component and its role in the voice pipeline.
2. **Layer Allocation:** Specify which Clean Architecture layer it belongs to:
   - `Core / Domain` (`src/core/`)
   - `Adapters / Implementation` (`src/stt/`, `src/tts/`, `src/llm/`, `src/vad/`, `src/audio/`)
   - `Orchestration / Use Case` (`src/pipeline/`)
3. **Interfaces & Contracts:** Python typing protocols or abstract base classes required.
4. **Error Handling & Reliability:** Specific exceptions caught, timeouts, and retry policies.
5. **Security & Privacy:** Input validation and credential handling.
6. **Files to Create / Modify:** Explicit list of files.
7. **Test Requirements:** Unit test cases and mocks needed in `tests/`.
8. **Definition of Done:** Verifiable checklist.
