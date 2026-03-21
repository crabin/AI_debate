import time
import logging
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
    ):
        self._client = ZhipuAiClient(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
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
