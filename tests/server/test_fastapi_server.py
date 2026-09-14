"""Module: test_fastapi_server
Layer: Testing
Purpose: Comprehensive unit test suite for FastAPI REST and WebSocket endpoints.
Dependencies: pytest, unittest.mock, fastapi.testclient, src.server.fastapi_server
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.server.fastapi_server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_check_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "talk-to-me-voice-assistant"


def test_system_status_endpoint(client: TestClient) -> None:
    with patch("src.server.fastapi_server.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_config = {
            "stt_provider": "deepgram",
            "openai_model": "gpt-4o",
            "elevenlabs_voice_id": "test-voice",
            "silence_threshold_ms": 600.0,
            "max_memory_turns": 10,
        }
        mock_get_pipeline.return_value = (mock_pipeline, mock_config)

        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["stt_provider"] == "deepgram"
        assert data["openai_model"] == "gpt-4o"


def test_chat_endpoint(client: TestClient) -> None:
    with patch("src.server.fastapi_server.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.process_text_message = AsyncMock(
            return_value="Hello, this is your AI voice assistant!"
        )
        mock_get_pipeline.return_value = (mock_pipeline, {})

        response = client.post("/api/chat", json={"text": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Hello, this is your AI voice assistant!"
        assert "total_time" in data["metrics"]


def test_get_web_ui_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Real-Time Voice Assistant" in response.text
