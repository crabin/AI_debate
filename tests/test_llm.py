import pytest
from unittest.mock import MagicMock, patch
from src.llm.base import BaseLLM
from src.llm import LLM_FACTORY, create_llm
from src.llm.openai_compatible import OpenAICompatibleLLM
from tests.conftest import FakeLLM


def test_fake_llm_satisfies_base_interface():
    llm = FakeLLM(responses=["hello"])
    assert isinstance(llm, BaseLLM)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "hello"


def test_fake_llm_records_call_history():
    llm = FakeLLM(responses=["a", "b"])
    llm.chat([{"role": "user", "content": "q1"}])
    llm.chat([{"role": "user", "content": "q2"}])
    assert len(llm.call_history) == 2


def test_llm_factory_contains_zhipu():
    assert "zhipu" in LLM_FACTORY


def test_create_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm({"provider": "nonexistent"})


def test_fake_llm_has_model_name(fake_llm):
    assert fake_llm.model_name == "fake-model"


# --- OpenAICompatibleLLM tests ---


def _make_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        api_key="test-key",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        timeout_seconds=30,
        max_retries=1,
        retry_delay=0.1,
    )


def test_openai_compatible_model_name():
    llm = _make_llm()
    assert llm.model_name == "gpt-4o"


def test_openai_compatible_chat_returns_string():
    llm = _make_llm()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"

    with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
        result = llm.chat([{"role": "user", "content": "Hi"}])

    assert result == "Hello"


def test_openai_compatible_chat_stream_calls_callback():
    llm = _make_llm()
    chunks = []

    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "Hel"
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "lo"
    mock_stream = iter([chunk1, chunk2])

    with patch.object(llm._client.chat.completions, "create", return_value=mock_stream):
        result = llm.chat_stream(
            [{"role": "user", "content": "Hi"}],
            callback=lambda c: chunks.append(c),
        )

    assert result == "Hello"
    assert chunks == ["H", "e", "l", "l", "o"]


# --- role-based create_llm() tests ---


def test_create_llm_uses_role_prefix_for_provider(monkeypatch):
    """PRO_LLM_PROVIDER overrides LLM_PROVIDER when role='pro'."""
    monkeypatch.setenv("PRO_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("PRO_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("PRO_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("PRO_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "zhipu")  # should be ignored for role=pro

    llm = create_llm(role="pro")
    assert llm.model_name == "gpt-4o"
    from src.llm.openai_compatible import OpenAICompatibleLLM
    assert isinstance(llm, OpenAICompatibleLLM)


def test_create_llm_falls_back_to_global_when_no_role_prefix(monkeypatch):
    """Falls back to global LLM_* vars when role-prefix vars absent."""
    monkeypatch.delenv("CON_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("LLM_MODEL", "glm-4.7")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")

    llm = create_llm(role="con")
    assert llm.model_name == "glm-4.7"
    from src.llm.zhipu import ZhipuLLM
    assert isinstance(llm, ZhipuLLM)


def test_create_llm_no_role_is_backward_compatible(monkeypatch):
    """Calling create_llm() without role still works as before."""
    monkeypatch.setenv("LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("LLM_MODEL", "glm-4.7")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")

    llm = create_llm()
    assert llm.model_name == "glm-4.7"
