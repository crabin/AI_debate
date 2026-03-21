from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send chat messages and return response."""
        pass
