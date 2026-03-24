import os
from src.llm.base import BaseLLM
from src.llm.zhipu import ZhipuLLM
from src.llm.openai_compatible import OpenAICompatibleLLM

LLM_FACTORY: dict[str, type[BaseLLM]] = {
    "zhipu": ZhipuLLM,
    "openai_compatible": OpenAICompatibleLLM,
}


def create_llm(config: dict | None = None, role: str | None = None) -> BaseLLM:
    """Create an LLM instance from config dict or environment variables.

    Priority (highest to lowest):
    1. Role-prefixed env vars (PRO_LLM_*, CON_LLM_*, JUDGE_LLM_*)
    2. Global env vars (LLM_*)
    3. Config file (YAML)
    4. Default values

    Args:
        config: Optional config dict (flat or nested under "llm" key)
        role: Optional role ("pro", "con", "judge") — enables per-role env vars
    """
    # Get llm_config from config dict
    if config and "llm" in config:
        llm_config = config["llm"]
    elif config:
        llm_config = config
    else:
        llm_config = {}

    prefix = f"{role.upper()}_" if role else ""

    def _env(key: str, default: str = "") -> str:
        """Read role-prefixed env var, falling back to unprefixed then default."""
        return (
            os.environ.get(f"{prefix}{key}")
            or os.environ.get(key)
            or default
        )

    def _get_float(key: str, config_value: float | None, default: float) -> float:
        val = _env(key)
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        return config_value if config_value is not None else default

    def _get_int(key: str, config_value: int | None, default: int) -> int:
        val = _env(key)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
        return config_value if config_value is not None else default

    provider = _env("LLM_PROVIDER", llm_config.get("provider", "zhipu"))
    model = _env("LLM_MODEL", llm_config.get("model", "glm-4.7"))
    temperature = _get_float("LLM_TEMPERATURE", llm_config.get("temperature"), 0.7)
    timeout_seconds = _get_int("LLM_TIMEOUT_SECONDS", llm_config.get("timeout_seconds"), 60)
    max_retries = _get_int("LLM_MAX_RETRIES", llm_config.get("max_retries"), 1)
    retry_delay = _get_float("LLM_RETRY_DELAY", llm_config.get("retry_delay"), 2.0)

    if provider not in LLM_FACTORY:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(LLM_FACTORY.keys())}")

    if provider == "zhipu":
        api_key = (
            os.environ.get(f"{prefix}ZAI_API_KEY")
            or os.environ.get("ZAI_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                f"ZAI_API_KEY (or {prefix}ZAI_API_KEY) is required for zhipu provider"
            )
        base_url = os.environ.get(f"{prefix}ZAI_BASE_URL") or os.environ.get("ZAI_BASE_URL")
        return ZhipuLLM(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
        )

    elif provider == "openai_compatible":
        api_key = (
            os.environ.get(f"{prefix}LLM_API_KEY")
            or os.environ.get("LLM_API_KEY", "")
        )
        base_url = (
            os.environ.get(f"{prefix}LLM_BASE_URL")
            or os.environ.get("LLM_BASE_URL", "")
        )
        if not api_key:
            raise ValueError(
                f"LLM_API_KEY (or {prefix}LLM_API_KEY) is required for openai_compatible provider"
            )
        if not base_url:
            raise ValueError(
                f"LLM_BASE_URL (or {prefix}LLM_BASE_URL) is required for openai_compatible provider"
            )
        return OpenAICompatibleLLM(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    raise ValueError(
        f"Provider '{provider}' is registered in LLM_FACTORY but has no dispatch branch in create_llm()"
    )


__all__ = ["BaseLLM", "ZhipuLLM", "OpenAICompatibleLLM", "create_llm", "LLM_FACTORY"]
