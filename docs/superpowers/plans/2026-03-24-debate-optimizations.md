# Debate Optimizations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four features to the AI Debate System: multi-model support (per-role LLM config via .env), final verdict + topic conclusion, JSON export, and concurrent free debate with side-by-side streaming display.

**Architecture:** Additive changes layered on the existing StageController / MessagePool / BaseLLM pipeline. Feature 3 (multi-model) is implemented first as a foundation; Features 1, 2, 4 build on it. No existing behavior is removed or renamed.

**Tech Stack:** Python 3.11+, `openai` SDK (for OpenAICompatibleLLM), `rich` (Live/Panel for concurrent display), `threading` (Barrier for sync), `python-dotenv`

**Spec:** `docs/superpowers/specs/2026-03-24-debate-optimizations-design.md`

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `src/llm/base.py` | Modify | Add abstract `model_name` property |
| `src/llm/zhipu.py` | Modify | Implement `model_name`; guard `sys.stdout` in `chat_stream` |
| `src/llm/openai_compatible.py` | **New** | OpenAI-SDK-compatible LLM for any compatible endpoint |
| `src/llm/__init__.py` | Modify | Register new provider; refactor `create_llm()` with `role` param |
| `src/agents/base.py` | Modify | Add `model_name` property delegating to `_llm` |
| `src/agents/judge.py` | Modify | Add `generate_verdict()` method |
| `src/engine/scorer.py` | No change | `export()` already returns cards with totals — use as-is |
| `src/export.py` | **New** | `save_debate_json(results, pool, scorer, path, start_time)` |
| `src/stages/free_debate.py` | Modify | Add `execute_concurrent()` method |
| `src/stages/controller.py` | Modify | Call `generate_verdict()`; return `_scorer`; add `concurrent` routing |
| `src/display/terminal.py` | Modify | Add `verdict_panel()`; update `participants()` for model name; add `concurrent_speech_panels()` |
| `src/cli.py` | Modify | Per-role LLMs; `--output`; `--concurrent`; `start_time` |
| `tests/conftest.py` | Modify | Add `model_name` to `FakeLLM` |
| `tests/test_llm.py` | Modify/New | Tests for OpenAICompatibleLLM and create_llm role param |
| `tests/test_judge.py` | Modify | Tests for `generate_verdict()` |
| `tests/test_export.py` | **New** | Tests for `save_debate_json()` |
| `tests/test_stages.py` | Modify | Tests for `execute_concurrent()` |
| `.env.example` | **New** | Document all new env vars |

---

## Task 1: Add `model_name` to `BaseLLM` and `FakeLLM`

**Files:**
- Modify: `src/llm/base.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write a failing test confirming FakeLLM exposes model_name**

Create `tests/test_llm.py` (not in conftest.py — that file is for fixtures only):

```python
def test_fake_llm_has_model_name(fake_llm):
    assert fake_llm.model_name == "fake-model"
```

Run: `pytest tests/ -k "test_fake_llm_has_model_name" -v`
Expected: FAIL — `AttributeError: 'FakeLLM' object has no attribute 'model_name'`

- [ ] **Step 2: Add abstract `model_name` property to `BaseLLM`**

In `src/llm/base.py`, add after the class docstring, before `chat()`:

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g., 'glm-4.7', 'gpt-4o')."""
```

- [ ] **Step 3: Add `model_name` to `FakeLLM` in `tests/conftest.py`**

Add this property inside the `FakeLLM` class (after `__init__`):

```python
@property
def model_name(self) -> str:
    return "fake-model"
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `pytest tests/ -k "test_fake_llm_has_model_name" -v`
Expected: PASS

- [ ] **Step 5: Run full test suite — expect no regressions**

Run: `pytest tests/ -v`
Expected: All previously passing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm/base.py tests/conftest.py
git commit -m "feat: add abstract model_name property to BaseLLM"
```

---

## Task 2: Implement `model_name` in `ZhipuLLM`; fix `chat_stream` stdout side effects

**Files:**
- Modify: `src/llm/zhipu.py`

The current `chat_stream()` calls `sys.stdout.flush()` and `time.sleep(0.01)` per character even when a `callback` is provided. When run in a background thread under Rich's `Live` context, this corrupts terminal output. Fix: only do those side effects when no callback is present.

- [ ] **Step 1: Add `model_name` property to `ZhipuLLM`**

In `src/llm/zhipu.py`, add after `__init__`:

```python
@property
def model_name(self) -> str:
    return self._model
```

- [ ] **Step 2: Guard `sys.stdout` side effects in `chat_stream`**

Current (lines 89-92 of `src/llm/zhipu.py`):
```python
if callback:
    for char in content:
        callback(char)
        sys.stdout.flush()
        time.sleep(0.01)  # Small delay for visual effect
```

Replace with (remove the stdout side effects from the callback path entirely — the callback is responsible for display):
```python
if callback:
    for char in content:
        callback(char)
# No per-character stdout write in either path.
# The non-callback path already accumulates full_content and returns it silently.
```

- [ ] **Step 3: Run existing tests — expect no regressions**

Run: `pytest tests/ -v`
Expected: All PASS (ZhipuLLM is not instantiated in unit tests — FakeLLM is used)

- [ ] **Step 4: Commit**

```bash
git add src/llm/zhipu.py
git commit -m "feat: add model_name to ZhipuLLM; fix chat_stream stdout in callback mode"
```

---

## Task 3: Create `OpenAICompatibleLLM`

**Files:**
- Create: `src/llm/openai_compatible.py`
- Modify: `src/llm/__init__.py`
- Create/Modify: `tests/test_llm.py`

This class uses the `openai` Python SDK and works with any compatible endpoint (OpenAI, DeepSeek, Moonshot, etc.).

First, install the openai package if not already present:
```bash
pip install openai
```
Add `openai` to `pyproject.toml` dependencies.

- [ ] **Step 1: Write failing tests for OpenAICompatibleLLM**

Create `tests/test_llm.py`:

```python
from unittest.mock import MagicMock, patch
from src.llm.openai_compatible import OpenAICompatibleLLM


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

    # Build mock streaming response
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
```

Run: `pytest tests/test_llm.py -v`
Expected: FAIL — ImportError (file doesn't exist yet)

- [ ] **Step 2: Create `src/llm/openai_compatible.py`**

```python
"""OpenAI-compatible LLM implementation for AI Debate System."""

import logging
import time
from typing import Callable
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """LLM implementation for any OpenAI-SDK-compatible endpoint.

    Works with OpenAI, DeepSeek, Moonshot, Qwen, and other providers
    that expose an OpenAI-compatible chat completions API.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: int = 60,
        max_retries: int = 1,
        retry_delay: float = 2.0,
    ) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
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
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, e)
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)
        raise RuntimeError(f"LLM call failed after {self._max_retries + 1} attempts: {last_error}")

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        callback: Callable[[str], None] | None = None,
    ) -> str:
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
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            content = delta.content
                            full_content += content
                            if callback:
                                for char in content:
                                    callback(char)
                return full_content
            except Exception as e:
                last_error = e
                logger.warning("LLM stream failed (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, e)
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        logger.warning("Streaming failed, falling back to non-streaming")
        return self.chat(messages, temperature)
```

- [ ] **Step 3: Register in `src/llm/__init__.py`**

Add to `LLM_FACTORY`:
```python
from src.llm.openai_compatible import OpenAICompatibleLLM

LLM_FACTORY: dict[str, type[BaseLLM]] = {
    "zhipu": ZhipuLLM,
    "openai_compatible": OpenAICompatibleLLM,
}
```

Also update `__all__`:
```python
__all__ = ["BaseLLM", "ZhipuLLM", "OpenAICompatibleLLM", "create_llm", "LLM_FACTORY"]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_llm.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/openai_compatible.py src/llm/__init__.py tests/test_llm.py
git commit -m "feat: add OpenAICompatibleLLM for any OpenAI-SDK-compatible endpoint"
```

---

## Task 4: Refactor `create_llm()` with `role` parameter

**Files:**
- Modify: `src/llm/__init__.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests for role-prefixed env var resolution**

Add to `tests/test_llm.py`:

```python
import os
from unittest.mock import patch
from src.llm import create_llm


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
```

Run: `pytest tests/test_llm.py -k "role" -v`
Expected: FAIL — `create_llm()` doesn't have `role` param yet

- [ ] **Step 2: Refactor `create_llm()` in `src/llm/__init__.py`**

Replace the full `create_llm()` function with:

```python
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
```

- [ ] **Step 3: Run tests — expect PASS**

Run: `pytest tests/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run full suite — no regressions**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/__init__.py tests/test_llm.py
git commit -m "feat: refactor create_llm() with role param for per-team LLM config"
```

---

## Task 5: Add `model_name` to `BaseAgent`; update `participants()` display

**Files:**
- Modify: `src/agents/base.py`
- Modify: `src/display/terminal.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_agents.py`:

```python
def test_base_agent_exposes_model_name(fake_llm):
    from src.agents.base import BaseAgent
    agent = BaseAgent(
        agent_id="pro_1", name="正方一辩", team="pro",
        role="一辩", llm=fake_llm,
    )
    assert agent.model_name == "fake-model"
```

Run: `pytest tests/test_agents.py -k "model_name" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 2: Add `model_name` property to `BaseAgent`**

In `src/agents/base.py`, add after `self._llm = llm` line, as a property:

```python
@property
def model_name(self) -> str:
    """Return the model identifier from the underlying LLM."""
    return self._llm.model_name
```

- [ ] **Step 3: Run test — expect PASS**

Run: `pytest tests/test_agents.py -k "model_name" -v`
Expected: PASS

- [ ] **Step 4: Update `TerminalDisplay.participants()` to show model names**

In `src/display/terminal.py`, replace the `pro_names` and `con_names` list comprehensions in `participants()`:

```python
# Before:
pro_names = [f"{a.name} ({a.position}辩)" for a in sorted(pro_debaters, key=lambda x: x.position)]
con_names = [f"{a.name} ({a.position}辩)" for a in sorted(con_debaters, key=lambda x: x.position)]

# After:
def _agent_line(a) -> str:
    model = getattr(a, "model_name", "")
    model_tag = f" [{model}]" if model else ""
    return f"{a.name} ({a.position}辩){model_tag}"

pro_names = [_agent_line(a) for a in sorted(pro_debaters, key=lambda x: x.position)]
con_names = [_agent_line(a) for a in sorted(con_debaters, key=lambda x: x.position)]
```

Also update the judge display line:
```python
# Before:
self._console.print(f"[yellow]裁判：{judge.name}[/yellow]")

# After:
judge_model = getattr(judge, "model_name", "")
judge_model_tag = f" [{judge_model}]" if judge_model else ""
self._console.print(f"[yellow]裁判：{judge.name}{judge_model_tag}[/yellow]")
```

- [ ] **Step 5: Run full suite — no regressions**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py src/display/terminal.py tests/test_agents.py
git commit -m "feat: expose model_name on BaseAgent; show model in participants panel"
```

---

## Task 6: Update `cli.py` — per-role LLMs in `create_agents()`

**Files:**
- Modify: `src/cli.py`
- Create: `.env.example`

- [ ] **Step 1: Write failing test**

Add to `tests/test_cli.py`:

```python
def test_create_agents_uses_separate_llms(fake_llm_factory):
    """Each team and judge gets its own LLM instance."""
    from src.cli import create_agents
    pro_llm = fake_llm_factory(["pro response"])
    con_llm = fake_llm_factory(["con response"])
    judge_llm = fake_llm_factory(['{"type":"review","summary":"ok","highlights":[],"suggestions":[]}'])

    topic = {"title": "Test", "pro_stance": "Pro", "con_stance": "Con"}
    personalities = {}
    agents = create_agents({}, topic, personalities, pro_llm=pro_llm, con_llm=con_llm, judge_llm=judge_llm)

    assert agents["pro_1"]._llm is pro_llm
    assert agents["con_1"]._llm is con_llm
    assert agents["judge"]._llm is judge_llm
```

Run: `pytest tests/test_cli.py -k "separate_llms" -v`
Expected: FAIL — `create_agents()` doesn't accept separate LLM args

- [ ] **Step 2: Update `create_agents()` signature in `src/cli.py`**

Change the signature from:
```python
def create_agents(config: dict, topic: dict, personalities: dict, llm, display=None) -> dict[str, BaseAgent]:
```
To:
```python
def create_agents(
    config: dict,
    topic: dict,
    personalities: dict,
    pro_llm: BaseLLM,
    con_llm: BaseLLM,
    judge_llm: BaseLLM,
    display=None,
) -> dict[str, BaseAgent]:
```

Update the body to use `pro_llm` / `con_llm` per team:
```python
for position in range(1, 5):
    for team in ["pro", "con"]:
        personality_key = config.get("default_personality", "logical")
        team_llm = pro_llm if team == "pro" else con_llm
        agent = DebaterAgent.create(
            position=position,
            team=team,
            stance=topic["pro_stance"] if team == "pro" else topic["con_stance"],
            topic=topic["title"],
            personality=personality_key,
            llm=team_llm,
        )
        agents[agent.agent_id] = agent

judge = JudgeAgent.create(
    topic=topic["title"],
    pro_stance=topic["pro_stance"],
    con_stance=topic["con_stance"],
    llm=judge_llm,
    display=display,
)
```

Update `run_debate()` to create three LLMs and pass them:
```python
start_time = time.time()  # ADD THIS as first line of run_debate()

# ... existing config loading ...

pro_llm = create_llm(config, role="pro")
con_llm = create_llm(config, role="con")
judge_llm = create_llm(config, role="judge")

agents = create_agents(config, topic, personalities, pro_llm=pro_llm, con_llm=con_llm, judge_llm=judge_llm, display=display)
```

Also add `topic`, `pro_stance`, `con_stance` to the results dict before calling `save_debate_json()` so they appear in `meta`:
```python
results["topic"] = topic["title"]
results["pro_stance"] = topic["pro_stance"]
results["con_stance"] = topic["con_stance"]
```

Add `import time` and `import os` at the top of `cli.py` if not already present.

- [ ] **Step 3: Run tests — expect PASS**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 4: Create `.env.example`**

Create `.env.example` at the project root:

```env
# AI Debate System — Environment Variables
# Copy this file to .env and fill in your keys

# ─────────────────────────────────────────────────────
# Global fallback (used when PRO_*/CON_*/JUDGE_* are not set)
# ─────────────────────────────────────────────────────
LLM_PROVIDER=zhipu          # zhipu | openai_compatible
LLM_MODEL=glm-4.7

# For zhipu provider:
ZAI_API_KEY=your-zhipu-api-key-here
# ZAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4  # optional

# For openai_compatible provider (global):
# LLM_API_KEY=sk-...
# LLM_BASE_URL=https://api.openai.com/v1

# ─────────────────────────────────────────────────────
# Per-role overrides (optional — override global above)
# ─────────────────────────────────────────────────────

# 正方辩手 (Pro team)
# PRO_LLM_PROVIDER=openai_compatible
# PRO_LLM_MODEL=deepseek-chat
# PRO_LLM_BASE_URL=https://api.deepseek.com/v1
# PRO_LLM_API_KEY=sk-...

# 反方辩手 (Con team)
# CON_LLM_PROVIDER=zhipu
# CON_LLM_MODEL=glm-4.7
# CON_ZAI_API_KEY=...

# 裁判 (Judge)
# JUDGE_LLM_PROVIDER=openai_compatible
# JUDGE_LLM_MODEL=claude-sonnet-4-6
# JUDGE_LLM_BASE_URL=https://api.anthropic.com/v1
# JUDGE_LLM_API_KEY=sk-ant-...

# ─────────────────────────────────────────────────────
# Optional settings
# ─────────────────────────────────────────────────────
# LLM_TEMPERATURE=0.7
# LLM_TIMEOUT_SECONDS=60
# LLM_MAX_RETRIES=1
# LLM_RETRY_DELAY=2.0

# Concurrent free debate (Feature 4)
# DEBATE_CONCURRENT_FREE=true
```

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/cli.py .env.example
git commit -m "feat: create_agents() accepts per-role LLMs; add .env.example"
```

---

## Task 7: `JudgeAgent.generate_verdict()` + verdict display

**Files:**
- Modify: `src/agents/judge.py`
- Modify: `src/stages/controller.py`
- Modify: `src/display/terminal.py`
- Modify: `tests/test_judge.py`

- [ ] **Step 1: Write failing test for `generate_verdict()`**

Add to `tests/test_judge.py`:

```python
import json

VERDICT_JSON = json.dumps({
    "winner_reason": "正方逻辑更严密",
    "topic_conclusion": "综合来看，效率优先",
    "best_debater_reason": "二辩表现最佳",
    "key_moments": ["正方二辩三连问"],
})


def test_generate_verdict_returns_structured_dict(fake_llm_factory):
    from src.agents.judge import JudgeAgent
    from src.engine.message_pool import MessagePool

    llm = fake_llm_factory([VERDICT_JSON])
    judge = JudgeAgent.create(
        topic="Test topic",
        pro_stance="Pro",
        con_stance="Con",
        llm=llm,
    )
    pool = MessagePool()

    verdict = judge.generate_verdict(
        pool=pool,
        winner="pro",
        pro_score=42.5,
        con_score=38.0,
        best_debater=("pro_2", 18.5),
    )

    assert verdict["winner_reason"] == "正方逻辑更严密"
    assert verdict["topic_conclusion"] == "综合来看，效率优先"
    assert verdict["key_moments"] == ["正方二辩三连问"]


def test_generate_verdict_returns_fallback_on_parse_error(fake_llm_factory):
    from src.agents.judge import JudgeAgent
    from src.engine.message_pool import MessagePool

    llm = fake_llm_factory(["not valid json"])
    judge = JudgeAgent.create(
        topic="Test", pro_stance="Pro", con_stance="Con", llm=llm
    )
    verdict = judge.generate_verdict(
        pool=MessagePool(), winner="con",
        pro_score=30.0, con_score=35.0,
        best_debater=("con_3", 12.0),
    )

    # Must have all keys with empty/default values
    assert "winner_reason" in verdict
    assert "topic_conclusion" in verdict
    assert "best_debater_reason" in verdict
    assert "key_moments" in verdict
    assert verdict["key_moments"] == []
```

Run: `pytest tests/test_judge.py -k "verdict" -v`
Expected: FAIL

- [ ] **Step 2: Add `VERDICT_FALLBACK` and `generate_verdict()` to `src/agents/judge.py`**

Add constant after imports. Use a factory function to avoid sharing a mutable list across callers:
```python
def _make_verdict_fallback() -> dict:
    """Return a fresh fallback dict (avoids mutable default sharing)."""
    return {
        "winner_reason": "",
        "topic_conclusion": "",
        "best_debater_reason": "",
        "key_moments": [],
    }

VERDICT_FALLBACK = _make_verdict_fallback  # callable, not a dict
```

Add method to `JudgeAgent` (after `generate_review()`):
```python
def generate_verdict(
    self,
    pool,
    winner: str,
    pro_score: float,
    con_score: float,
    best_debater: tuple[str, float],
    temperature: float = 0.5,
) -> dict:
    """Generate final verdict including topic conclusion.

    Args:
        pool: MessagePool instance (for full transcript context)
        winner: "pro", "con", or "tie"
        pro_score: Final pro team score
        con_score: Final con team score
        best_debater: Tuple of (agent_id, score)
        temperature: Sampling temperature

    Returns:
        Dict with winner_reason, topic_conclusion, best_debater_reason, key_moments.
        Returns VERDICT_FALLBACK on parse failure.
    """
    public_messages = pool.get_messages("public")
    public_context = "\n".join([
        f"{m.speaker}（{m.role}）：{m.content}"
        for m in public_messages
    ]) if public_messages else "暂无发言记录"

    winner_label = {"pro": "正方", "con": "反方", "tie": "平局"}.get(winner, winner)
    best_id, best_score = best_debater if best_debater else ("", 0.0)

    prompt = (
        f"【辩题】{self.topic}\n"
        f"【正方立场】{self.pro_stance}\n"
        f"【反方立场】{self.con_stance}\n"
        f"【比分】正方 {pro_score:.1f} vs 反方 {con_score:.1f}\n"
        f"【获胜方】{winner_label}\n"
        f"【最佳辩手】{best_id} ({best_score:.1f}分)\n\n"
        f"【辩论记录】\n{public_context}\n\n"
        "请输出严格JSON（不要markdown代码块），包含字段：\n"
        "winner_reason（获胜原因，100字以内），\n"
        "topic_conclusion（对辩题的结论，150字以内），\n"
        "best_debater_reason（最佳辩手理由，50字以内），\n"
        "key_moments（关键时刻列表，最多3项）"
    )

    messages = [
        {"role": "system", "content": f"你是辩论赛裁判。辩题：{self.topic}"},
        {"role": "user", "content": prompt},
    ]

    response = self._llm.chat(messages, temperature=temperature)

    try:
        json_text = _extract_json_from_markdown(response)
        result = json.loads(json_text)
        # Ensure all keys present; start from a fresh fallback dict
        return {**VERDICT_FALLBACK(), **result}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse verdict response: %s", response[:200])
        return VERDICT_FALLBACK()
```

- [ ] **Step 3: Run tests — expect PASS**

Run: `pytest tests/test_judge.py -k "verdict" -v`
Expected: Both PASS

- [ ] **Step 4: Update `StageController._calculate_final_results()` to call `generate_verdict()`**

In `src/stages/controller.py`, inside `_calculate_final_results()`, after assembling primitives and before the `return`:

```python
# Assemble primitives
pro_total = scorer.get_team_total("pro")
con_total = scorer.get_team_total("con")

if pro_total > con_total:
    winner = "pro"
    margin = pro_total - con_total
elif con_total > pro_total:
    winner = "con"
    margin = con_total - pro_total
else:
    winner = "tie"
    margin = 0.0

best_debater = scorer.get_best_debater()

# Generate review (existing)
judge_agent = agents.get("judge")
review = None
if judge_agent:
    try:
        review = judge_agent.generate_review(pool)
    except Exception as e:
        logger.warning(f"Failed to generate review: {e}")

# Generate verdict (new)
verdict_data: dict = {}
if judge_agent:
    try:
        verdict_data = judge_agent.generate_verdict(
            pool=pool,
            winner=winner,
            pro_score=pro_total,
            con_score=con_total,
            best_debater=best_debater,
        )
    except Exception as e:
        logger.warning(f"Failed to generate verdict: {e}")

return {
    "pro_score": pro_total,
    "con_score": con_total,
    "winner": winner,
    "margin": margin,
    "best_debater": best_debater,
    "review": review,
    "_scorer": scorer,
    **verdict_data,
}
```

- [ ] **Step 5: Add `verdict_panel()` to `TerminalDisplay`; update `final_results()`**

In `src/display/terminal.py`, add this method:

```python
def verdict_panel(self, results: dict) -> None:
    """Display the final verdict panel with topic conclusion and key moments."""
    from rich.rule import Rule

    topic_conclusion = results.get("topic_conclusion", "")
    winner_reason = results.get("winner_reason", "")
    key_moments = results.get("key_moments", [])
    best_debater_reason = results.get("best_debater_reason", "")

    if not any([topic_conclusion, winner_reason, key_moments]):
        return

    parts = []
    if winner_reason:
        parts.append(f"[bold]获胜原因：[/bold]{winner_reason}")
    if topic_conclusion:
        parts.append(f"\n[bold yellow]辩题结论：[/bold yellow]\n{topic_conclusion}")
    if key_moments:
        parts.append("\n[bold]关键时刻：[/bold]")
        parts.extend(f"  • {m}" for m in key_moments)
    if best_debater_reason:
        parts.append(f"\n[dim]{best_debater_reason}[/dim]")

    self._console.print(
        Panel("\n".join(parts), title="[bold yellow]裁判裁决[/bold yellow]", border_style="yellow")
    )
```

In `final_results()`, add a call to `verdict_panel()` after the scoreboard:

```python
def final_results(self, results: dict) -> None:
    # ... existing code for scoreboard + winner text + best debater ...

    # Add verdict panel (new)
    self.verdict_panel(results)

    # ... existing review panel ...
```

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents/judge.py src/stages/controller.py src/display/terminal.py tests/test_judge.py
git commit -m "feat: add generate_verdict() to JudgeAgent with topic conclusion and verdict panel"
```

---

## Task 8: JSON export (`src/export.py` + `--output` CLI arg)

**Files:**
- Create: `src/export.py`
- Modify: `src/cli.py`
- Create: `tests/test_export.py`

Note: `Scorer.export()` already exists and returns `{"cards": [...], "individual_penalties": {...}, "team_penalties": {...}}` — use it directly; no new Scorer methods needed.

- [ ] **Step 1: Write failing tests**

Create `tests/test_export.py`:

```python
import json
from pathlib import Path
import time
from src.export import save_debate_json
from src.engine.message_pool import MessagePool, Message
from src.engine.scorer import Scorer, ScoreCard


def _make_pool_with_messages() -> MessagePool:
    pool = MessagePool()
    msg = Message(
        speaker="pro_1", role="一辩", team="pro",
        stage="opening", content="正方立论", msg_type="speech",
        timestamp=time.time(), word_count=100, metadata=(),
    )
    pool.publish("public", msg)
    return pool


def _make_scorer_with_card() -> Scorer:
    scorer = Scorer()
    card = ScoreCard(
        speaker="pro_1", stage="opening",
        logic=8, persuasion=7, expression=8, teamwork=7, rule_compliance=9,
        violations=(), comment="不错",
    )
    scorer.record(card)
    return scorer


def test_save_debate_json_creates_valid_file(tmp_path):
    pool = _make_pool_with_messages()
    scorer = _make_scorer_with_card()
    results = {
        "winner": "pro", "pro_score": 12.0, "con_score": 10.0,
        "margin": 2.0, "best_debater": ("pro_1", 7.8),
        "winner_reason": "正方更好", "topic_conclusion": "科技优先",
        "best_debater_reason": "表现突出", "key_moments": [],
        "review": None,
    }
    out = tmp_path / "debate.json"
    save_debate_json(results, pool, scorer, out, start_time=time.time() - 5)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["transcript"]) == 1
    assert data["transcript"][0]["content"] == "正方立论"
    assert len(data["scores"]) == 1
    assert data["scores"][0]["speaker"] == "pro_1"
    assert "total" in data["scores"][0]
    assert data["result"]["winner"] == "pro"


def test_save_debate_json_excludes_private_scorer_key(tmp_path):
    pool = MessagePool()
    scorer = Scorer()
    results = {
        "winner": "tie", "pro_score": 0.0, "con_score": 0.0,
        "margin": 0.0, "best_debater": ("", 0.0),
        "_scorer": scorer,  # should be stripped
    }
    out = tmp_path / "debate.json"
    save_debate_json(results, pool, scorer, out, start_time=time.time())

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "_scorer" not in data
    assert "_scorer" not in data.get("result", {})
```

Run: `pytest tests/test_export.py -v`
Expected: FAIL — ImportError

- [ ] **Step 2: Create `src/export.py`**

```python
"""JSON export for AI Debate System."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from src.engine.message_pool import MessagePool
from src.engine.scorer import Scorer

logger = logging.getLogger(__name__)

__all__ = ["save_debate_json"]

# Keys from results dict that should not appear in the JSON output
_PRIVATE_KEYS = {"_scorer", "review"}


def save_debate_json(
    results: dict,
    pool: MessagePool,
    scorer: Scorer,
    path: Path,
    start_time: float,
) -> None:
    """Serialize full debate results to a JSON file.

    Args:
        results: Dict returned by StageController.run_debate()
        pool: MessagePool containing all messages
        scorer: Scorer instance with all recorded ScoreCards
        path: Output file path
        start_time: Unix timestamp when the debate started (from time.time())
    """
    import time

    duration = time.time() - start_time
    now_iso = datetime.now(timezone.utc).isoformat()

    # Build transcript from public channel
    transcript = [
        {
            "stage": m.stage,
            "speaker": m.speaker,
            "role": m.role,
            "team": m.team,
            "content": m.content,
            "word_count": m.word_count,
            "timestamp": m.timestamp,
            "msg_type": m.msg_type,
        }
        for m in pool.get_messages("public")
    ]

    # Build scores from scorer export (already has total computed)
    scorer_data = scorer.export()
    scores = scorer_data["cards"]  # list of dicts with total

    # Build result (strip private keys)
    result_out = {
        k: v for k, v in results.items()
        if k not in _PRIVATE_KEYS
    }
    # best_debater is a tuple — convert to dict for JSON
    if "best_debater" in result_out and isinstance(result_out["best_debater"], tuple):
        bd = result_out["best_debater"]
        result_out["best_debater"] = {"speaker": bd[0], "score": bd[1]} if bd else None

    # Convert key_moments list (ensure JSON-serializable)
    if "key_moments" not in result_out:
        result_out["key_moments"] = []

    # Extract topic metadata from result if present (put there by run_debate())
    meta = {
        "topic": results.get("topic", ""),
        "pro_stance": results.get("pro_stance", ""),
        "con_stance": results.get("con_stance", ""),
        "timestamp": now_iso,
        "duration_seconds": round(duration, 1),
    }

    payload = {
        "meta": meta,
        "transcript": transcript,
        "scores": scores,
        "result": result_out,
    }

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Debate saved to %s", path)
    except OSError as e:
        logger.error("Failed to save debate JSON to %s: %s", path, e)
        print(f"[警告] 无法保存辩论记录：{e}")
```

- [ ] **Step 3: Run tests — expect PASS**

Run: `pytest tests/test_export.py -v`
Expected: All PASS

- [ ] **Step 4: Add `--output` to `cli.py`**

In `src/cli.py`, update `run_debate()` signature:
```python
def run_debate(
    topic_index: int = 0,
    config_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
```

At the end of `run_debate()`, after `display.final_results(results)`:
```python
# Save JSON export if requested
scorer = results.pop("_scorer", None)
if output_path and scorer is not None:
    from src.export import save_debate_json
    save_debate_json(results, pool, scorer, output_path, start_time)
elif output_path and scorer is None:
    logger.warning("Scorer not available; JSON export skipped")
```

In `main()`, add `--output` arg parsing (before calling `run_debate()`):
```python
output_path = None
if "--output" in sys.argv:
    idx = sys.argv.index("--output")
    if idx + 1 < len(sys.argv):
        output_path = Path(sys.argv[idx + 1])

results = run_debate(topic_index=topic_index, output_path=output_path)
```

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/export.py src/cli.py tests/test_export.py
git commit -m "feat: add JSON export with --output CLI flag"
```

---

## Task 9: Concurrent free debate — `FreeDebateStage.execute_concurrent()`

**Files:**
- Modify: `src/stages/free_debate.py`
- Modify: `src/display/terminal.py`
- Modify: `tests/test_stages.py`

This is the most complex task. Read the entire `FreeDebateStage` class before starting (already done — it's in `src/stages/free_debate.py`).

The concurrent version runs one pro speaker + one con speaker per round simultaneously. Each generates into its own `io.StringIO` buffer via the streaming callback. Rich `Live` shows both panels side by side, refreshed from the buffers. After both finish (synchronized via `threading.Barrier`), both messages are committed to the pool.

- [ ] **Step 1: Write a failing test for `execute_concurrent()`**

Add to `tests/test_stages.py`:

```python
def test_execute_concurrent_produces_messages_from_both_teams(fake_llm_factory):
    from src.stages.free_debate import FreeDebateStage
    from src.engine.message_pool import MessagePool
    from src.display.terminal import TerminalDisplay
    from rich.console import Console
    import io

    llm = fake_llm_factory(["并发发言内容"] * 20)
    display = TerminalDisplay(console=Console(file=io.StringIO()))
    stage = FreeDebateStage.create(display=display)
    pool = MessagePool()

    # Build minimal agents dict
    from tests.conftest import FakeLLM
    from src.agents.debater import DebaterAgent

    agents = {}
    for pos in range(1, 5):
        for team in ["pro", "con"]:
            a = DebaterAgent.create(
                position=pos, team=team,
                stance="test stance", topic="test topic",
                personality="logical", llm=llm,
            )
            agents[a.agent_id] = a

    result = stage.execute_concurrent(pool, agents)

    messages = pool.get_messages("public")
    assert len(messages) >= 2, "At least one round should produce 2 messages"
    teams = {m.team for m in messages}
    assert "pro" in teams
    assert "con" in teams
    assert result["status"] == "completed"
    assert result["concurrent"] is True
```

Run: `pytest tests/test_stages.py -k "concurrent" -v`
Expected: FAIL — `execute_concurrent` not found

- [ ] **Step 2: Add `execute_concurrent()` to `FreeDebateStage`**

Add these imports to `src/stages/free_debate.py`:
```python
import io
import threading
from typing import Callable
```

Add the method after `execute()`:

```python
def execute_concurrent(
    self,
    pool,
    agents: dict,
    penalties: dict | None = None,
) -> dict:
    """Execute free debate with concurrent per-round parallel generation.

    Each round, one pro speaker and one con speaker generate simultaneously
    in separate threads. Both speeches are committed to the pool after the
    round completes (both threads reach the Barrier).

    Args:
        pool: MessagePool instance
        agents: Dictionary of agent_id -> Agent
        penalties: Optional penalty configuration

    Returns:
        Result dictionary with status, turn count, and concurrent=True
    """
    self._display.stage_start(self.name, self.description + " [并发模式]")

    messages_published = 0
    speak_counts: dict[str, int] = {}
    round_count = 0
    max_rounds = 10

    # Per-team time tracking
    pro_timer = Timer(total_seconds=self._TEAM_TIME, chars_per_minute=self._CHARS_PER_MINUTE)
    con_timer = Timer(total_seconds=self._TEAM_TIME, chars_per_minute=self._CHARS_PER_MINUTE)

    # Per-team last-speaker tracking (separate, not shared)
    pro_last: str | None = None
    con_last: str | None = None

    while round_count < max_rounds:
        # Check termination
        if pro_timer.is_expired() and con_timer.is_expired():
            break

        # Select speakers for this round
        pro_speaker_id, _ = self._get_next_speaker(agents, "pro", speak_counts, pro_last)
        con_speaker_id, _ = self._get_next_speaker(agents, "con", speak_counts, con_last)

        if pro_speaker_id is None and con_speaker_id is None:
            break

        # Results accumulated from threads
        pro_content: list[str] = [""]   # mutable container
        con_content: list[str] = [""]
        # Use separate single-element lists so each thread has its own error slot.
        # Passing [thread_error[0]] would create an anonymous new list not visible to caller.
        pro_error: list[Exception | None] = [None]
        con_error: list[Exception | None] = [None]

        # Streaming buffers (one per speaker).
        # Protected by buf_lock — both the writing thread (via callback)
        # and the reading main thread (via concurrent_speech_panels) use this lock.
        pro_buf = io.StringIO()
        con_buf = io.StringIO()
        buf_lock = threading.Lock()

        barrier = threading.Barrier(2)

        def _run_speaker(
            speaker_id: str | None,
            timer,
            buf: io.StringIO,
            content_out: list[str],
            error_slot: list,
            barrier: threading.Barrier,
            lock: threading.Lock,
        ) -> None:
            if speaker_id is None or timer.is_expired():
                with lock:
                    buf.write("⏰ 时间到")
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass
                return
            try:
                speaker = agents[speaker_id]
                recent_context = self._get_recent_context(pool, limit=5)

                def _callback(char: str) -> None:
                    with lock:
                        buf.write(char)

                content = speaker.generate_free_debate_speech(
                    pool,
                    recent_context=recent_context,
                    callback=_callback,
                )
                content_out[0] = content
                barrier.wait()
            except threading.BrokenBarrierError:
                pass  # sibling aborted — our content may be fine but round is discarded
            except Exception as e:
                error_slot[0] = e
                logger.warning("Concurrent speaker %s failed: %s", speaker_id, e)
                try:
                    barrier.abort()
                except Exception:
                    pass

        # Launch threads
        pro_thread = threading.Thread(
            target=_run_speaker,
            args=(pro_speaker_id, pro_timer, pro_buf, pro_content, pro_error, barrier, buf_lock),
            daemon=True,
        )
        con_thread = threading.Thread(
            target=_run_speaker,
            args=(con_speaker_id, con_timer, con_buf, con_content, con_error, barrier, buf_lock),
            daemon=True,
        )
        pro_thread.start()
        con_thread.start()

        # Display side-by-side while threads run.
        # Pass buf_lock so concurrent_speech_panels reads buffers safely.
        self._display.concurrent_speech_panels(
            pro_name=agents[pro_speaker_id].name if pro_speaker_id else "正方",
            con_name=agents[con_speaker_id].name if con_speaker_id else "反方",
            pro_buf=pro_buf,
            con_buf=con_buf,
            pro_thread=pro_thread,
            con_thread=con_thread,
            buf_lock=buf_lock,
        )

        pro_thread.join()
        con_thread.join()

        # Commit results (both or neither — atomic round)
        pro_text = pro_content[0]
        con_text = con_content[0]

        committed = 0
        if pro_text and pro_speaker_id and not pro_timer.is_expired():
            pro_speaker = agents[pro_speaker_id]
            word_count = len(pro_text)
            pro_timer.check(word_count)
            msg = Message(
                speaker=pro_speaker.agent_id,
                role=pro_speaker.role,
                team=pro_speaker.team,
                stage=self.name,
                content=pro_text,
                msg_type="free_speech",
                timestamp=time.time(),
                word_count=word_count,
                metadata=("round", round_count, "concurrent", True),
            )
            pool.publish("public", msg)
            speak_counts[pro_speaker_id] = speak_counts.get(pro_speaker_id, 0) + 1
            pro_last = pro_speaker_id
            committed += 1

        if con_text and con_speaker_id and not con_timer.is_expired():
            con_speaker = agents[con_speaker_id]
            word_count = len(con_text)
            con_timer.check(word_count)
            msg = Message(
                speaker=con_speaker.agent_id,
                role=con_speaker.role,
                team=con_speaker.team,
                stage=self.name,
                content=con_text,
                msg_type="free_speech",
                timestamp=time.time(),
                word_count=word_count,
                metadata=("round", round_count, "concurrent", True),
            )
            pool.publish("public", msg)
            speak_counts[con_speaker_id] = speak_counts.get(con_speaker_id, 0) + 1
            con_last = con_speaker_id
            committed += 1

        messages_published += committed
        round_count += 1

        # Early stop: all have spoken and enough rounds done
        if round_count >= 6 and self._all_have_spoken(speak_counts, agents):
            break

    self._display.stage_end(self.name)

    return {
        "status": "completed",
        "stage": self.name,
        "messages_count": messages_published,
        "rounds": round_count,
        "speak_counts": speak_counts,
        "concurrent": True,
        "pro_time_left": pro_timer.time_left(),
        "con_time_left": con_timer.time_left(),
    }
```

- [ ] **Step 3: Add `concurrent_speech_panels()` to `TerminalDisplay`**

In `src/display/terminal.py`, add after `speech_stream()`:

```python
def concurrent_speech_panels(
    self,
    pro_name: str,
    con_name: str,
    pro_buf: "io.StringIO",
    con_buf: "io.StringIO",
    pro_thread: "threading.Thread",
    con_thread: "threading.Thread",
    buf_lock: "threading.Lock",
    refresh_rate: float = 0.1,
) -> None:
    """Display two speakers' streaming output side-by-side using Rich Live.

    Polls both StringIO buffers until both threads finish, refreshing
    a two-panel table at ~10fps.

    Args:
        pro_name: Display name of the pro speaker
        con_name: Display name of the con speaker
        pro_buf: StringIO buffer written to by the pro thread
        con_buf: StringIO buffer written to by the con thread
        pro_thread: Pro speaker's thread (join signal)
        con_thread: Con speaker's thread (join signal)
        buf_lock: Lock shared with the writer threads — protects buf.write() and buf.getvalue()
        refresh_rate: Seconds between Live refreshes
    """
    import io
    import threading
    import time as _time
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel

    def _make_table() -> Table:
        # Acquire the same lock the writer threads use for buf.write()
        with buf_lock:
            pro_text = pro_buf.getvalue()
            con_text = con_buf.getvalue()

        table = Table.grid(expand=True, padding=0)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            Panel(pro_text or "…", title=f"[bold blue]{pro_name}[/bold blue]", border_style="blue"),
            Panel(con_text or "…", title=f"[bold red]{con_name}[/bold red]", border_style="red"),
        )
        return table

    with Live(_make_table(), console=self._console, refresh_per_second=10) as live:
        while pro_thread.is_alive() or con_thread.is_alive():
            live.update(_make_table())
            _time.sleep(refresh_rate)
        # Final update with complete content
        live.update(_make_table())
```

Add `import io` and `import threading` to the top of `src/display/terminal.py`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_stages.py -k "concurrent" -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/stages/free_debate.py src/display/terminal.py tests/test_stages.py
git commit -m "feat: add concurrent free debate with side-by-side streaming panels"
```

---

## Task 10: Wire `--concurrent` flag through `StageController` and `cli.py`

**Files:**
- Modify: `src/stages/controller.py`
- Modify: `src/cli.py`
- Modify: `tests/test_controller.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_controller.py`:

```python
def test_controller_routes_to_execute_concurrent_when_flag_set(fake_llm_factory):
    from src.stages.controller import StageController
    from src.engine.message_pool import MessagePool
    from src.display.terminal import TerminalDisplay
    from rich.console import Console
    import io
    from unittest.mock import patch

    display = TerminalDisplay(console=Console(file=io.StringIO()))
    controller = StageController.create(display=display, concurrent=True)

    pool = MessagePool()
    agents = {}  # Empty agents — stages will skip gracefully

    # Patch execute_concurrent to verify it's called
    with patch("src.stages.free_debate.FreeDebateStage.execute_concurrent") as mock_concurrent:
        mock_concurrent.return_value = {"status": "completed", "stage": "free_debate",
                                        "messages_count": 0, "rounds": 0,
                                        "speak_counts": {}, "concurrent": True,
                                        "pro_time_left": 240, "con_time_left": 240}
        controller.run_debate(pool, agents)

    mock_concurrent.assert_called_once()
```

Run: `pytest tests/test_controller.py -k "concurrent" -v`
Expected: FAIL

- [ ] **Step 2: Add `concurrent` param to `StageController`**

In `src/stages/controller.py`, update `__init__` and `create()`:

```python
def __init__(
    self,
    display,
    penalties: dict | None = None,
    concurrent: bool = False,
) -> None:
    self._display = display
    self._penalties = penalties or {}
    self._stages: dict[str, BaseStage] = {}
    self._scorer = Scorer()
    self._concurrent = concurrent
```

In `run_debate()`, replace the free_debate stage call:
```python
for stage_name in self._STAGE_ORDER:
    if stage_name not in self._stages:
        continue
    stage = self._stages[stage_name]
    logger.info(f"Starting stage: {stage_name}")

    # Route free_debate to concurrent executor when flag is set
    if stage_name == "free_debate" and self._concurrent:
        result = stage.execute_concurrent(pool, agents, self._penalties)
    else:
        result = stage.execute(pool, agents, self._penalties)

    stage_results.append(result)
    logger.info(f"Completed stage: {stage_name}")
```

Update `create()` factory:
```python
@classmethod
def create(cls, display, penalties: dict | None = None, concurrent: bool = False) -> "StageController":
    return cls(display=display, penalties=penalties, concurrent=concurrent)
```

- [ ] **Step 3: Add `--concurrent` to `cli.py` `main()`**

Make sure `import os` is at the top of `src/cli.py` (add it if missing — it is needed for `os.environ.get()`).

In `main()`, alongside the existing arg parsing:
```python
concurrent = "--concurrent" in sys.argv or os.environ.get("DEBATE_CONCURRENT_FREE", "").lower() == "true"

results = run_debate(topic_index=topic_index, output_path=output_path, concurrent=concurrent)
```

Update `run_debate()` signature:
```python
def run_debate(
    topic_index: int = 0,
    config_path: Path | None = None,
    output_path: Path | None = None,
    concurrent: bool = False,
) -> dict:
```

Update `StageController.create()` call in `run_debate()`:
```python
controller = StageController.create(
    display=display,
    penalties=config.get("penalties", {}),
    concurrent=concurrent,
)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/stages/controller.py src/cli.py tests/test_controller.py
git commit -m "feat: wire --concurrent flag through StageController to execute_concurrent()"
```

---

## Final Verification

- [ ] **Run full test suite with coverage**

```bash
pytest tests/ -v --tb=short
```

Expected: All PASS, no errors.

- [ ] **Smoke test imports**

```bash
python -c "
from src.llm.openai_compatible import OpenAICompatibleLLM
from src.export import save_debate_json
from src.agents.judge import JudgeAgent, VERDICT_FALLBACK
from src.stages.free_debate import FreeDebateStage
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Final commit tag**

```bash
git tag v1.1.0-optimizations
```
