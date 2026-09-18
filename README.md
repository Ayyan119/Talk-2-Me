# Talk-2-Me: Real-Time Voice-Operated Conversational AI

A low-latency, voice-operated LLM conversational system with short-term conversational memory, real-time Voice Activity Detection (VAD), Speech-to-Text (STT), and Text-to-Speech (TTS) synthesis.

---

## Architecture Overview

```mermaid
flowchart LR
    Mic([User Microphone]) --> VAD[Silero VAD]
    VAD -->|Voice Segment| STT[Faster-Whisper / Deepgram]
    STT -->|Transcript| Memory[Short-Term Memory Buffer]
    Memory --> LLM[OpenAI / Groq LLM]
    LLM -->|Text Response| TTS[ElevenLabs TTS]
    TTS --> Speaker([Audio Output / Streamlit UI])
```

---

## Key Features

- **Voice Activity Detection (VAD):** Continuous audio monitoring via **Silero VAD** with real-time speech boundary detection.
- **Dual STT Engine Support:** Local high-throughput transcription via **Faster-Whisper** and cloud streaming transcription via **Deepgram SDK**.
- **Context-Aware Reasoning:** Integrated conversational buffer memory preserving multi-turn context across spoken queries.
- **Low-Latency Neural Speech Synthesis:** Expressive voice generation powered by **ElevenLabs API**.
- **Interactive UI & Backend:** Dual interface with a **Streamlit** control dashboard and asynchronous **FastAPI** service.

---

## Tech Stack

- **Speech & Audio:** Silero VAD, Faster-Whisper, Deepgram SDK, ElevenLabs API, SoundDevice, PyTorch
- **LLM & Orchestration:** OpenAI API, LangChain Core, LangChain OpenAI
- **Backend & Serving:** FastAPI, Uvicorn, Pydantic v2, HTTPX
- **Frontend / UI:** Streamlit

---

## Getting Started

### Prerequisites
- Python 3.10+
- PortAudio installed on your system (e.g. `sudo apt-get install portaudio19-dev`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ayyan119/Talk-2-Me.git
cd Talk-2-Me

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

### Running the Application

**Run via Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

**Run via CLI Pipeline:**
```bash
python -m src.main
```

---

## Repository Structure

```text
Talk-2-Me/
├── src/
│   ├── audio/          # Sound recording and playback interfaces
│   ├── core/           # Configuration management and settings
│   ├── llm/            # LLM invocation and prompt templates
│   ├── memory/         # Conversational buffer & state history
│   ├── pipeline/       # End-to-end voice loop coordinator
│   ├── server/         # FastAPI endpoints and audio streaming routes
│   ├── stt/            # Faster-Whisper and Deepgram STT adapters
│   ├── tts/            # ElevenLabs TTS integration
│   ├── vad/            # Silero VAD speech detector
│   └── main.py         # Entrypoint runner
├── streamlit_app.py    # Interactive Streamlit UI
├── requirements.txt    # Production dependencies
└── pyproject.toml      # Project configuration
```

---

## License

MIT License.
