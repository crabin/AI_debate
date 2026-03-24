import time
import logging
from typing import Callable
from zai import ZhipuAiClient
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class ZhipuLLM(BaseLLM):
    """ZhipuAI (智谱) LLM implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.7",
        timeout_seconds: int = 60,
        max_retries: int = 1,
        retry_delay: float = 2.0,
        base_url: str | None = None,
    ):
        self._client = ZhipuAiClient(api_key=api_key, base_url=base_url)
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                )
                # Non-streaming response always has choices
                content = response.choices[0].message.content  # type: ignore[union-attr]
                return content or ""
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
        raise RuntimeError(f"LLM call failed after {self._max_retries + 1} attempts: {last_error}")

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        callback: Callable[[str], None] | None = None,
    ) -> str:
        """Stream chat response with character-by-character output.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            callback: Optional callback for each character chunk

        Returns:
            Complete response text
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
                        delta = chunk.choices[0].delta  # type: ignore[union-attr]
                        if hasattr(delta, "content") and delta.content:
                            content = delta.content  # type: ignore[union-attr]
                            full_content += content

                            # Stream character by character
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

        # Fallback to non-streaming on error
        logger.warning("Streaming failed, falling back to non-streaming")
        return self.chat(messages, temperature)
