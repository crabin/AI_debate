from abc import ABC, abstractmethod
from typing import Iterator, Callable


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g., 'glm-4.7', 'gpt-4o')."""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send messages and return the assistant's response text."""

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        callback: Callable[[str], None] | None = None,
    ) -> str:
        """Send messages and stream the response via callback.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            callback: Optional callback function for each chunk

        Returns:
            Complete response text
        """
        # Default implementation: call non-streaming chat
        # Subclasses can override for true streaming
        response = self.chat(messages, temperature)
        if callback:
            callback(response)
        return response
