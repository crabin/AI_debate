import pytest
from src.llm.base import BaseLLM
from src.llm import LLM_FACTORY, create_llm
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
