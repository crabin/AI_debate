import pytest
from src.llm.base import BaseLLM


class FakeLLM(BaseLLM):
    """Fake LLM that returns canned responses for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self.call_history: list[list[dict]] = []

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        self.call_history.append(messages)
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
        else:
            response = "Fake LLM response"
        self._call_count += 1
        return response


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_llm_factory():
    def factory(responses: list[str] | None = None) -> FakeLLM:
        return FakeLLM(responses)
    return factory
