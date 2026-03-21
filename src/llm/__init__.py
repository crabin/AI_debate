import os
from src.llm.base import BaseLLM
from src.llm.zhipu import ZhipuLLM

LLM_FACTORY: dict[str, type[BaseLLM]] = {
    "zhipu": ZhipuLLM,
}


def create_llm(config: dict) -> BaseLLM:
    """Create an LLM instance from config dict."""
    provider = config["provider"]

    if provider not in LLM_FACTORY:
        raise ValueError(f"Unknown LLM provider: {provider}")

    cls = LLM_FACTORY[provider]

    if provider == "zhipu":
        api_key = os.environ.get("ZAI_API_KEY", "")
        return cls(
            api_key=api_key,
            model=config.get("model", "glm-4.7"),
            timeout_seconds=config.get("timeout_seconds", 60),
            max_retries=config.get("max_retries", 1),
            retry_delay=config.get("retry_delay", 2.0),
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["BaseLLM", "ZhipuLLM", "create_llm", "LLM_FACTORY"]
