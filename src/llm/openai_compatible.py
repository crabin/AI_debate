"""OpenAI-compatible LLM implementation for AI Debate System."""

import logging
import time
from typing import Callable

from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """LLM implementation for any OpenAI-SDK-compatible endpoint.

    Works with OpenAI, DeepSeek, Moonshot, Qwen, and other providers
    that expose an OpenAI-compatible chat completions API.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: int = 60,
        max_retries: int = 1,
        retry_delay: float = 2.0,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send messages and return the assistant's response text.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.

        Returns:
            Assistant response text.

        Raises:
            RuntimeError: When all retry attempts are exhausted.
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)
        raise RuntimeError(
            f"LLM call failed after {self._max_retries + 1} attempts: {last_error}"
        )

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        callback: Callable[[str], None] | None = None,
    ) -> str:
        """Send messages and stream the response via callback.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            callback: Optional callback called once per character of the response.

        Returns:
            Complete response text.
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                full_content = ""
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            content = delta.content
                            full_content += content
                            if callback:
                                for char in content:
                                    callback(char)
                return full_content
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM stream failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        logger.warning("Streaming failed, falling back to non-streaming")
        return self.chat(messages, temperature)
