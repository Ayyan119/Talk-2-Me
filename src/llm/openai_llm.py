"""Module: openai_llm
Layer: Adapter/Implementation
Purpose: Concrete LLM adapter implementing OpenAI chat completions with streaming and retries.
Dependencies: asyncio, collections.abc, os, dotenv, openai, src.core.types, src.core.exceptions, src.llm.base
"""

import asyncio
import os
import random
from collections.abc import AsyncIterator
from typing import Any

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.core.exceptions import ConfigurationError, LanguageModelError
from src.core.types import Message
from src.llm.base import LanguageModel

load_dotenv()


class OpenAILLM(LanguageModel):
    """OpenAI Language Model adapter for chat generation and streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        initial_retry_delay_seconds: float = 0.5,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Initializes the OpenAI LLM adapter."""
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ConfigurationError(
                "OpenAI API key is missing. Set OPENAI_API_KEY environment variable or pass api_key."
            )

        if not (0.0 <= temperature <= 2.0):
            raise LanguageModelError("temperature must be between 0.0 and 2.0")
        if timeout_seconds <= 0.0:
            raise LanguageModelError("timeout_seconds must be positive")
        if max_retries < 0:
            raise LanguageModelError("max_retries cannot be negative")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._initial_retry_delay = initial_retry_delay_seconds

        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(
                api_key=resolved_key,
                timeout=self._timeout_seconds,
            )

    @property
    def model(self) -> str:
        """Returns the configured model name."""
        return self._model

    @property
    def temperature(self) -> float:
        """Returns the sampling temperature."""
        return self._temperature

    @property
    def timeout_seconds(self) -> float:
        """Returns the request timeout in seconds."""
        return self._timeout_seconds

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Converts domain Message objects to OpenAI message dictionaries."""
        if not messages:
            raise LanguageModelError("Message list cannot be empty")

        converted: list[dict[str, str]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise LanguageModelError(
                    f"Expected Message instance at index {idx}, got {type(msg).__name__}"
                )
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            converted.append({"role": role_str, "content": msg.content})

        return converted

    def _is_transient_error(self, err: Exception) -> bool:
        """Determines whether an exception is transient and eligible for retry."""
        return isinstance(
            err,
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                asyncio.TimeoutError,
            ),
        )

    async def generate(self, messages: list[Message], **kwargs: Any) -> str:
        """Generates a complete textual response given a sequence of conversation messages."""
        openai_messages = self._convert_messages(messages)
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        timeout = kwargs.get("timeout", self._timeout_seconds)

        attempt = 0
        delay = self._initial_retry_delay

        while True:
            attempt += 1
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model,
                        messages=openai_messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    ),
                    timeout=timeout,
                )

                if not response.choices:
                    raise LanguageModelError("OpenAI returned no response choices")

                content = response.choices[0].message.content
                return content or ""

            except (
                openai.AuthenticationError,
                openai.BadRequestError,
                openai.PermissionDeniedError,
            ) as err:
                raise LanguageModelError(
                    f"OpenAI API authentication or client error: {err}"
                ) from err

            except Exception as err:
                if self._is_transient_error(err) and attempt <= self._max_retries:
                    await asyncio.sleep(delay)
                    delay = (delay * 2.0) + random.uniform(0.01, 0.1 * delay)
                    continue

                raise LanguageModelError(
                    f"OpenAI completion failed after {attempt} attempt(s): {err}"
                ) from err

    async def generate_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Streams generated response tokens/chunks given conversation history."""
        openai_messages = self._convert_messages(messages)
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        timeout = kwargs.get("timeout", self._timeout_seconds)

        attempt = 0
        delay = self._initial_retry_delay
        stream_response = None

        while True:
            attempt += 1
            try:
                stream_response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model,
                        messages=openai_messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    ),
                    timeout=timeout,
                )
                break
            except (
                openai.AuthenticationError,
                openai.BadRequestError,
                openai.PermissionDeniedError,
            ) as err:
                raise LanguageModelError(
                    f"OpenAI API authentication or client error: {err}"
                ) from err
            except Exception as err:
                if self._is_transient_error(err) and attempt <= self._max_retries:
                    await asyncio.sleep(delay)
                    delay = (delay * 2.0) + random.uniform(0.01, 0.1 * delay)
                    continue
                raise LanguageModelError(
                    f"OpenAI stream initiation failed after {attempt} attempt(s): {err}"
                ) from err

        stream_iter = stream_response.__aiter__()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(), timeout=timeout
                    )
                except StopAsyncIteration:
                    break

                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    yield chunk.choices[0].delta.content

        except (
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.PermissionDeniedError,
        ) as err:
            raise LanguageModelError(
                f"OpenAI API client error during stream: {err}"
            ) from err
        except Exception as err:
            raise LanguageModelError(f"OpenAI stream reading failed: {err}") from err
