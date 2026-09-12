"""Module: test_openai_llm
Layer: Testing
Purpose: Unit test suite for OpenAILLM adapter using mocked OpenAI clients.
Dependencies: asyncio, unittest.mock, pytest, openai, src.core.types, src.core.exceptions, src.llm.openai_llm
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from src.core.exceptions import ConfigurationError, LanguageModelError
from src.core.types import Message, MessageRole
from src.llm.openai_llm import OpenAILLM


@dataclass
class MockChoiceMessage:
    """Mock OpenAI response choice message."""

    content: str


@dataclass
class MockChoice:
    """Mock OpenAI response choice."""

    message: MockChoiceMessage


@dataclass
class MockChatCompletionResponse:
    """Mock OpenAI ChatCompletion non-streaming response."""

    choices: list[MockChoice]


@dataclass
class MockDelta:
    """Mock delta chunk in streaming response."""

    content: str | None


@dataclass
class MockStreamChoice:
    """Mock streaming choice."""

    delta: MockDelta


@dataclass
class MockStreamChunk:
    """Mock streaming chunk."""

    choices: list[MockStreamChoice]


class MockAsyncStream:
    """Async iterator simulating OpenAI streaming chunks."""

    def __init__(self, chunks: list[str]) -> None:
        """Initializes the mock stream with a sequence of string chunks."""
        self._chunks = [
            MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content=c))])
            for c in chunks
        ]
        self._idx = 0

    def __aiter__(self) -> "MockAsyncStream":
        """Returns the async iterator instance."""
        return self

    async def __anext__(self) -> MockStreamChunk:
        """Yields the next chunk in the mock stream."""
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class TestOpenAILLM:
    """Unit test suite for the OpenAILLM adapter class."""

    def test_missing_api_key_raises_configuration_error(self) -> None:
        """Verifies ConfigurationError is raised when no API key is provided or found in env."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigurationError, match="OpenAI API key is missing"),
        ):
            OpenAILLM(api_key=None)

    def test_constructor_parameter_validations(self) -> None:
        """Verifies validation of constructor options."""
        mock_client = MagicMock()

        with pytest.raises(LanguageModelError, match="temperature must be between"):
            OpenAILLM(api_key="sk-test", temperature=3.0, client=mock_client)

        with pytest.raises(
            LanguageModelError, match="timeout_seconds must be positive"
        ):
            OpenAILLM(api_key="sk-test", timeout_seconds=0.0, client=mock_client)

        with pytest.raises(LanguageModelError, match="max_retries cannot be negative"):
            OpenAILLM(api_key="sk-test", max_retries=-1, client=mock_client)

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Verifies non-streaming generate returns full text response correctly."""
        mock_client = MagicMock()
        mock_response = MockChatCompletionResponse(
            choices=[
                MockChoice(
                    message=MockChoiceMessage(
                        content="Hello! How can I help you today?"
                    )
                )
            ]
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini", client=mock_client)
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hi!"),
        ]

        response = await llm.generate(messages)

        assert response == "Hello! How can I help you today?"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Hi!"}

    @pytest.mark.asyncio
    async def test_generate_empty_messages_raises_error(self) -> None:
        """Verifies generate with empty message list raises LanguageModelError."""
        llm = OpenAILLM(api_key="sk-test", client=MagicMock())

        with pytest.raises(LanguageModelError, match="Message list cannot be empty"):
            await llm.generate([])

    @pytest.mark.asyncio
    async def test_generate_stream_success(self) -> None:
        """Verifies generate_stream yields delta chunks sequentially."""
        mock_client = MagicMock()
        mock_stream = MockAsyncStream(["Hello", " world", "!"])
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        llm = OpenAILLM(api_key="sk-test", client=mock_client)
        messages = [Message(role=MessageRole.USER, content="Hello")]

        yielded_tokens: list[str] = []
        async for token in llm.generate_stream(messages):
            yielded_tokens.append(token)

        assert yielded_tokens == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_rate_limit_retry_eventual_success(self) -> None:
        """Verifies transient RateLimitError retries and returns upon recovery."""
        mock_client = MagicMock()
        mock_response = MockChatCompletionResponse(
            choices=[
                MockChoice(
                    message=MockChoiceMessage(content="Success after rate limit")
                )
            ]
        )

        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        # Fail twice with 429, then succeed on 3rd attempt
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[rate_limit_err, rate_limit_err, mock_response]
        )

        llm = OpenAILLM(
            api_key="sk-test",
            max_retries=3,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )
        messages = [Message(role=MessageRole.USER, content="Test retry")]

        result = await llm.generate(messages)

        assert result == "Success after rate limit"
        assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_authentication_error_fails_immediately_without_retry(self) -> None:
        """Verifies non-transient AuthenticationError does NOT retry."""
        mock_client = MagicMock()
        auth_err = openai.AuthenticationError(
            message="Incorrect API key provided",
            response=MagicMock(status_code=401),
            body=None,
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=auth_err)

        llm = OpenAILLM(
            api_key="sk-invalid",
            max_retries=3,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )
        messages = [Message(role=MessageRole.USER, content="Test auth failure")]

        with pytest.raises(
            LanguageModelError, match="OpenAI API authentication or client error"
        ):
            await llm.generate(messages)

        # Must only call once (no retries for auth errors)
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises_language_model_error(self) -> None:
        """Verifies that exceeding max_retries raises LanguageModelError."""
        mock_client = MagicMock()
        conn_err = openai.APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create = AsyncMock(side_effect=conn_err)

        llm = OpenAILLM(
            api_key="sk-test",
            max_retries=2,
            initial_retry_delay_seconds=0.01,
            client=mock_client,
        )
        messages = [Message(role=MessageRole.USER, content="Test exhaustion")]

        with pytest.raises(
            LanguageModelError, match="OpenAI completion failed after 3 attempt"
        ):
            await llm.generate(messages)

        assert mock_client.chat.completions.create.call_count == 3
