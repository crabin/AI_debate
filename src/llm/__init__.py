import os
from src.llm.base import BaseLLM
from src.llm.zhipu import ZhipuLLM
from src.llm.openai_compatible import OpenAICompatibleLLM

LLM_FACTORY: dict[str, type[BaseLLM]] = {
    "zhipu": ZhipuLLM,
    "openai_compatible": OpenAICompatibleLLM,
}


def create_llm(config: dict | None = None) -> BaseLLM:
    """Create an LLM instance from config dict or environment variables.

    Priority (highest to lowest):
    1. Environment variables (LLM_*)
    2. Config file (YAML)
    3. Default values

    Supports both flat config (for tests) and nested config (from YAML):
    - Flat: {"provider": "zhipu", "model": "glm-4.7", ...}
    - Nested: {"llm": {"provider": "zhipu", "model": "glm-4.7", ...}, ...}
    """
    # Get llm_config from config dict
    if config and "llm" in config:
        llm_config = config["llm"]
    elif config:
        llm_config = config
    else:
        llm_config = {}

    # Environment variables take precedence over config file
    provider = os.environ.get("LLM_PROVIDER", llm_config.get("provider", "zhipu"))
    model = os.environ.get("LLM_MODEL", llm_config.get("model", "glm-4.7"))

    # Parse numeric parameters from env
    def _get_float_env(key: str, config_value: float | None = None, default: float = 0.0) -> float:
        val = os.environ.get(key)
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        return config_value if config_value is not None else default

    def _get_int_env(key: str, config_value: int | None = None, default: int = 0) -> int:
        val = os.environ.get(key)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
        return config_value if config_value is not None else default

    temperature = _get_float_env("LLM_TEMPERATURE", llm_config.get("temperature"), 0.7)
    timeout_seconds = _get_int_env("LLM_TIMEOUT_SECONDS", llm_config.get("timeout_seconds"), 60)
    max_retries = _get_int_env("LLM_MAX_RETRIES", llm_config.get("max_retries"), 1)
    retry_delay = _get_float_env("LLM_RETRY_DELAY", llm_config.get("retry_delay"), 2.0)

    if provider not in LLM_FACTORY:
        raise ValueError(f"Unknown LLM provider: {provider}")

    cls = LLM_FACTORY[provider]

    if provider == "zhipu":
        api_key = os.environ.get("ZAI_API_KEY", "")
        if not api_key:
            raise ValueError("ZAI_API_KEY environment variable is required for zhipu provider")
        base_url = os.environ.get("ZAI_BASE_URL")
        return cls(  # type: ignore[call-arg]
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["BaseLLM", "ZhipuLLM", "OpenAICompatibleLLM", "create_llm", "LLM_FACTORY"]
