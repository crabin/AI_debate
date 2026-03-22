# Multi-Agent AI Debate System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-based multi-agent AI debate system with 9 independent agents (1 judge + 8 debaters) that follows strict Chinese debate competition rules.

**Architecture:** Hybrid pattern — lightweight StageController orchestrates debate flow, MessagePool handles dual-layer communication (public + team-private channels), independent agents generate speeches via configurable LLM (default: ZhipuAI glm-4.7). Scorer handles three-layer scoring (real-time + rule-engine penalties + aggregation).

**Tech Stack:** Python 3.11+, zai (ZhipuAI SDK), rich (terminal UI), pyyaml, python-dotenv

**Spec:** `docs/superpowers/specs/2026-03-22-ai-debate-system-design.md`

---

## File Structure

```
AI_debate/
├── .env                              # ZAI_API_KEY
├── .gitignore
├── pyproject.toml                    # Project config + dependencies
├── config/
│   ├── default.yaml                  # LLM, timer, scoring config
│   ├── topics.yaml                   # Built-in debate topics
│   └── personalities.yaml            # Debater personality templates
├── src/
│   ├── __init__.py
│   ├── main.py                       # CLI entry point (argparse)
│   ├── config.py                     # Config loader (YAML + env)
│   ├── llm/
│   │   ├── __init__.py               # LLM_FACTORY + create_llm()
│   │   ├── base.py                   # BaseLLM ABC
│   │   └── zhipu.py                  # ZhipuLLM implementation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseAgent class
│   │   ├── debater.py                # DebaterAgent (all 4 positions)
│   │   ├── judge.py                  # JudgeAgent
│   │   └── prompts.py                # All prompt templates as constants
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── message_pool.py           # Message dataclass + MessagePool
│   │   ├── timer.py                  # Timer (char-count based)
│   │   ├── scorer.py                 # ScoreCard + Scorer
│   │   └── stage_controller.py       # StageController orchestrator
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseStage ABC
│   │   ├── opening.py                # OpeningStage (陈词)
│   │   ├── cross_exam.py             # CrossExamStage (攻辩)
│   │   ├── free_debate.py            # FreeDebateStage (自由辩论)
│   │   └── closing.py                # ClosingStage (总结陈词)
│   └── display/
│       ├── __init__.py
│       └── terminal.py               # Rich-based colored terminal output
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures (fake LLM, config)
    ├── test_message_pool.py
    ├── test_timer.py
    ├── test_scorer.py
    ├── test_agents.py
    ├── test_stages.py
    ├── test_display.py
    └── test_config.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`, `.env`, `.gitignore`, `src/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/lpb/workspace/myProjects/AI_debate
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "ai-debate"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "zai",
    "rich>=13.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
dist/
.pytest_cache/
htmlcov/
```

- [ ] **Step 4: Create .env with placeholder**

```
ZAI_API_KEY=your_api_key_here
```

- [ ] **Step 5: Create empty __init__.py files**

Create: `src/__init__.py`, `tests/__init__.py`

- [ ] **Step 6: Create tests/conftest.py with FakeLLM fixture**

```python
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
```

- [ ] **Step 7: Install dependencies and verify**

```bash
cd /Users/lpb/workspace/myProjects/AI_debate
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest --co  # should collect 0 tests, no errors
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env src/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold project with dependencies and test fixtures"
```

---

## Task 2: Config System

**Files:**
- Create: `src/config.py`, `config/default.yaml`, `config/topics.yaml`, `config/personalities.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Create config YAML files**

`config/default.yaml`:
```yaml
llm:
  provider: "zhipu"
  model: "glm-4.7"
  temperature: 0.7
  timeout_seconds: 60
  max_retries: 1
  retry_delay: 2.0

timer:
  chars_per_minute: 250
  warning_threshold: 30

scoring:
  weights:
    logic: 0.25
    persuasion: 0.25
    expression: 0.20
    teamwork: 0.15
    rule_compliance: 0.15
  penalties:
    overtime_team: -3
    overtime_individual: -2
    counter_question: -2
    not_direct_answer: -2
    attacker_answered: -2
    off_topic: -2
    scripted_summary: -2
    personal_attack: -3
    consecutive_speech: -3
```

`config/topics.yaml`:
```yaml
topics:
  - title: "人工智能的发展利大于弊 / 弊大于利"
    pro_stance: "利大于弊"
    con_stance: "弊大于利"
  - title: "网络使人更亲近 / 更疏远"
    pro_stance: "更亲近"
    con_stance: "更疏远"
  - title: "大学教育应以就业为导向 / 以学术为导向"
    pro_stance: "以就业为导向"
    con_stance: "以学术为导向"
  - title: "科技发展应优先考虑效率 / 公平"
    pro_stance: "效率"
    con_stance: "公平"
  - title: "个人隐私比公共安全更重要 / 公共安全比个人隐私更重要"
    pro_stance: "个人隐私更重要"
    con_stance: "公共安全更重要"
```

`config/personalities.yaml`:
```yaml
personalities:
  logical:
    name: "逻辑严密型"
    prompt: "你的辩论风格注重逻辑推理，善用三段论和因果链条，用严密的论证压制对手。"
  emotional:
    name: "情感感染型"
    prompt: "你的辩论风格富有感染力，善于用生动案例和情感共鸣打动听众。"
  data_driven:
    name: "数据实证型"
    prompt: "你的辩论风格依托数据和事实，引用研究报告和统计数字增强说服力。"
  aggressive:
    name: "犀利进攻型"
    prompt: "你的辩论风格锋利直接，善于发现对方逻辑漏洞并穷追不舍。"
  diplomatic:
    name: "稳健防守型"
    prompt: "你的辩论风格沉稳大气，善于化解对方攻势并巧妙转化为己方论据。"
```

- [ ] **Step 2: Write failing tests for config loader**

`tests/test_config.py`:
```python
from pathlib import Path
from src.config import load_config, load_topics, load_personalities


def test_load_config_returns_all_sections():
    cfg = load_config()
    assert "llm" in cfg
    assert "timer" in cfg
    assert "scoring" in cfg
    assert cfg["llm"]["model"] == "glm-4.7"


def test_load_config_scoring_weights_sum_to_1():
    cfg = load_config()
    weights = cfg["scoring"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_load_topics_returns_list():
    topics = load_topics()
    assert len(topics) >= 5
    assert "title" in topics[0]
    assert "pro_stance" in topics[0]
    assert "con_stance" in topics[0]


def test_load_personalities_returns_dict():
    personalities = load_personalities()
    assert "logical" in personalities
    assert "name" in personalities["logical"]
    assert "prompt" in personalities["logical"]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 4: Implement config loader**

`src/config.py`:
```python
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Load default.yaml configuration."""
    with open(config_dir / "default.yaml") as f:
        return yaml.safe_load(f)


def load_topics(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Load debate topics from topics.yaml."""
    with open(config_dir / "topics.yaml") as f:
        data = yaml.safe_load(f)
    return data["topics"]


def load_personalities(config_dir: Path = CONFIG_DIR) -> dict:
    """Load personality templates from personalities.yaml."""
    with open(config_dir / "personalities.yaml") as f:
        data = yaml.safe_load(f)
    return data["personalities"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add config/ src/config.py tests/test_config.py
git commit -m "feat: add config system with YAML loaders for LLM, topics, and personalities"
```

---

## Task 3: LLM Abstraction Layer

**Files:**
- Create: `src/llm/__init__.py`, `src/llm/base.py`, `src/llm/zhipu.py`

- [ ] **Step 1: Create BaseLLM ABC**

`src/llm/base.py`:
```python
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send messages and return the assistant's response text."""
```

- [ ] **Step 2: Create ZhipuLLM implementation**

`src/llm/zhipu.py`:
```python
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
```

- [ ] **Step 3: Create LLM factory**

`src/llm/__init__.py`:
```python
import os
from src.llm.base import BaseLLM
from src.llm.zhipu import ZhipuLLM

LLM_FACTORY: dict[str, type[BaseLLM]] = {
    "zhipu": ZhipuLLM,
}


def create_llm(config: dict) -> BaseLLM:
    """Create an LLM instance from config dict."""
    provider = config["provider"]
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
```

- [ ] **Step 4: Write tests for LLM layer**

`tests/test_llm.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_llm.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/llm/ tests/test_llm.py
git commit -m "feat: add LLM abstraction layer with ZhipuAI implementation and factory"
```

---

## Task 4: MessagePool

**Files:**
- Create: `src/engine/__init__.py`, `src/engine/message_pool.py`
- Test: `tests/test_message_pool.py`

- [ ] **Step 1: Write failing tests**

`tests/test_message_pool.py`:
```python
import time
from src.engine.message_pool import Message, MessagePool


def test_publish_to_public():
    pool = MessagePool()
    msg = Message(
        speaker="pro_debater_1",
        role="一辩",
        team="pro",
        stage="opening",
        content="我方认为...",
        msg_type="speech",
        timestamp=time.time(),
        word_count=5,
        metadata=(),
    )
    pool.publish("public", msg)
    assert len(pool.get_messages("public")) == 1
    assert pool.get_messages("public")[0].content == "我方认为..."


def test_team_channel_isolation():
    pool = MessagePool()
    pro_msg = Message(
        speaker="pro_debater_1", role="一辩", team="pro", stage="opening",
        content="正方策略", msg_type="team_strategy",
        timestamp=time.time(), word_count=4, metadata=(),
    )
    con_msg = Message(
        speaker="con_debater_1", role="一辩", team="con", stage="opening",
        content="反方策略", msg_type="team_strategy",
        timestamp=time.time(), word_count=4, metadata=(),
    )
    pool.publish("team_pro", pro_msg)
    pool.publish("team_con", con_msg)
    assert len(pool.get_messages("team_pro")) == 1
    assert len(pool.get_messages("team_con")) == 1
    assert pool.get_messages("team_pro")[0].content == "正方策略"


def test_get_visible_messages_for_pro():
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("pro_debater_1", "一辩", "pro", "opening", "公开", "speech", ts, 2, ()))
    pool.publish("team_pro", Message("pro_debater_1", "一辩", "pro", "opening", "队内", "team_strategy", ts, 2, ()))
    pool.publish("team_con", Message("con_debater_1", "一辩", "con", "opening", "对方队内", "team_strategy", ts, 4, ()))

    visible = pool.get_visible_messages("pro")
    contents = [m.content for m in visible]
    assert "公开" in contents
    assert "队内" in contents
    assert "对方队内" not in contents


def test_get_visible_messages_for_judge():
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("pro_debater_1", "一辩", "pro", "opening", "公开", "speech", ts, 2, ()))
    pool.publish("judge_notes", Message("judge", "裁判", "judge", "opening", "笔记", "score", ts, 2, ()))
    pool.publish("team_pro", Message("pro_debater_1", "一辩", "pro", "opening", "队内", "team_strategy", ts, 2, ()))

    visible = pool.get_visible_messages("judge")
    contents = [m.content for m in visible]
    assert "公开" in contents
    assert "笔记" in contents
    assert "队内" not in contents


def test_message_is_frozen():
    msg = Message("a", "b", "c", "d", "e", "f", 0.0, 0, ())
    try:
        msg.content = "new"
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass


def test_get_messages_by_stage():
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("a", "一辩", "pro", "opening", "陈词", "speech", ts, 2, ()))
    pool.publish("public", Message("b", "二辩", "pro", "cross_exam", "攻辩", "speech", ts, 2, ()))

    opening_msgs = pool.get_messages("public", stage="opening")
    assert len(opening_msgs) == 1
    assert opening_msgs[0].content == "陈词"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_message_pool.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement MessagePool**

`src/engine/__init__.py`: empty file

`src/engine/message_pool.py`:
```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Message:
    speaker: str
    role: str
    team: str
    stage: str
    content: str
    msg_type: str
    timestamp: float
    word_count: int
    metadata: tuple


# Channel -> team visibility mapping
_VISIBILITY: dict[str, list[str]] = {
    "pro": ["public", "team_pro"],
    "con": ["public", "team_con"],
    "judge": ["public", "judge_notes"],
}

_VALID_CHANNELS = {"public", "team_pro", "team_con", "judge_notes"}


class MessagePool:
    """Dual-layer message pool with channel-based access control."""

    def __init__(self) -> None:
        self._channels: dict[str, list[Message]] = {ch: [] for ch in _VALID_CHANNELS}

    def publish(self, channel: str, message: Message) -> None:
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"Invalid channel: {channel}")
        self._channels[channel].append(message)

    def get_messages(
        self,
        channel: str,
        stage: Optional[str] = None,
    ) -> list[Message]:
        msgs = self._channels.get(channel, [])
        if stage is not None:
            msgs = [m for m in msgs if m.stage == stage]
        return list(msgs)

    def get_visible_messages(
        self,
        team: str,
        stage: Optional[str] = None,
    ) -> list[Message]:
        """Get all messages visible to a team, sorted by timestamp."""
        channels = _VISIBILITY.get(team, ["public"])
        result: list[Message] = []
        for ch in channels:
            result.extend(self.get_messages(ch, stage=stage))
        result.sort(key=lambda m: m.timestamp)
        return result

    def export(self) -> dict:
        """Export all public messages for debate log."""
        return {
            ch: [
                {
                    "speaker": m.speaker,
                    "role": m.role,
                    "team": m.team,
                    "stage": m.stage,
                    "content": m.content,
                    "msg_type": m.msg_type,
                    "word_count": m.word_count,
                }
                for m in msgs
            ]
            for ch, msgs in self._channels.items()
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_message_pool.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/__init__.py src/engine/message_pool.py tests/test_message_pool.py
git commit -m "feat: add MessagePool with dual-layer channel isolation"
```

---

## Task 5: Timer

**Files:**
- Create: `src/engine/timer.py`
- Test: `tests/test_timer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_timer.py`:
```python
from src.engine.timer import Timer


def test_estimate_duration_chinese():
    timer = Timer(chars_per_minute=250)
    text = "你" * 250
    assert timer.estimate_duration(text) == 60.0


def test_check_overtime_within_limit():
    timer = Timer(chars_per_minute=250)
    text = "你" * 200  # 200 chars = 48 seconds
    is_over, excess = timer.check_overtime(text, limit_seconds=60)
    assert is_over is False
    assert excess == 0.0


def test_check_overtime_exceeded():
    timer = Timer(chars_per_minute=250)
    text = "你" * 300  # 300 chars = 72 seconds
    is_over, excess = timer.check_overtime(text, limit_seconds=60)
    assert is_over is True
    assert abs(excess - 12.0) < 0.1


def test_char_limit_for_duration():
    timer = Timer(chars_per_minute=250)
    assert timer.char_limit(seconds=180) == 750  # 3 minutes = 750 chars
    assert timer.char_limit(seconds=30) == 125   # 30 seconds = 125 chars
    assert timer.char_limit(seconds=120) == 500  # 2 minutes = 500 chars


def test_is_warning_zone():
    timer = Timer(chars_per_minute=250, warning_threshold=30)
    # 750 char limit, used 730 chars -> 20 chars remaining < 30 threshold
    assert timer.is_warning_zone(used_chars=730, limit_chars=750) is True
    # used 700 chars -> 50 remaining > 30 threshold
    assert timer.is_warning_zone(used_chars=700, limit_chars=750) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_timer.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement Timer**

`src/engine/timer.py`:
```python
class Timer:
    """Character-count based timer for debate speeches."""

    def __init__(self, chars_per_minute: int = 250, warning_threshold: int = 30):
        self._cpm = chars_per_minute
        self._warning_threshold = warning_threshold

    def estimate_duration(self, text: str) -> float:
        """Estimate speech duration in seconds based on character count."""
        return len(text) / self._cpm * 60

    def check_overtime(self, text: str, limit_seconds: float) -> tuple[bool, float]:
        """Check if text exceeds time limit. Returns (is_overtime, excess_seconds)."""
        duration = self.estimate_duration(text)
        if duration > limit_seconds:
            return True, duration - limit_seconds
        return False, 0.0

    def char_limit(self, seconds: float) -> int:
        """Calculate character limit for a given time duration."""
        return int(self._cpm * seconds / 60)

    def is_warning_zone(self, used_chars: int, limit_chars: int) -> bool:
        """Check if remaining characters are below warning threshold."""
        remaining = limit_chars - used_chars
        return remaining < self._warning_threshold
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_timer.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/timer.py tests/test_timer.py
git commit -m "feat: add character-count based Timer with overtime detection"
```

---

## Task 6: Scorer

**Files:**
- Create: `src/engine/scorer.py`
- Test: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_scorer.py`:
```python
from src.engine.scorer import ScoreCard, Scorer


def test_scorecard_is_frozen():
    card = ScoreCard(
        speaker="pro_debater_1", stage="opening",
        logic=8, persuasion=7, expression=8, teamwork=7,
        rule_compliance=10, violations=(), comment="Good",
    )
    try:
        card.logic = 5
        assert False, "Should raise"
    except AttributeError:
        pass


def test_scorer_compute_speech_score():
    scorer = Scorer()
    card = ScoreCard(
        speaker="pro_debater_1", stage="opening",
        logic=8, persuasion=8, expression=8, teamwork=8,
        rule_compliance=8, violations=(), comment="Good",
    )
    score = scorer.compute_speech_score(card)
    assert score == 8.0  # All 8s with any weights = 8.0


def test_scorer_compute_weighted_score():
    scorer = Scorer()
    card = ScoreCard(
        speaker="pro_debater_1", stage="opening",
        logic=10, persuasion=10, expression=10, teamwork=10,
        rule_compliance=10, violations=(), comment="Perfect",
    )
    assert scorer.compute_speech_score(card) == 10.0


def test_scorer_record_and_get_individual_total():
    scorer = Scorer()
    card1 = ScoreCard("pro_debater_1", "opening", 8, 8, 8, 8, 8, (), "ok")
    card2 = ScoreCard("pro_debater_1", "cross_exam", 6, 6, 6, 6, 6, (), "ok")
    scorer.record(card1)
    scorer.record(card2)
    assert scorer.get_individual_total("pro_debater_1") == 14.0  # 8 + 6


def test_scorer_apply_individual_penalty():
    scorer = Scorer()
    card = ScoreCard("pro_debater_2", "cross_exam", 8, 8, 8, 8, 8, (), "ok")
    scorer.record(card)
    scorer.add_individual_penalty("pro_debater_2", -2, "overtime")
    assert scorer.get_individual_total("pro_debater_2") == 6.0  # 8 - 2


def test_scorer_apply_team_penalty():
    scorer = Scorer()
    scorer.add_team_penalty("pro", -3, "overtime")
    assert scorer.get_team_penalty("pro") == -3


def test_scorer_get_team_total():
    scorer = Scorer()
    card1 = ScoreCard("pro_debater_1", "opening", 8, 8, 8, 8, 8, (), "ok")
    card2 = ScoreCard("pro_debater_2", "cross_exam", 6, 6, 6, 6, 6, (), "ok")
    scorer.record(card1)
    scorer.record(card2)
    scorer.add_team_penalty("pro", -3, "overtime")
    # team total = (8 + 6) + (-3) = 11
    assert scorer.get_team_total("pro") == 11.0


def test_scorer_get_best_debater():
    scorer = Scorer()
    scorer.record(ScoreCard("pro_debater_1", "opening", 10, 10, 10, 10, 10, (), ""))
    scorer.record(ScoreCard("con_debater_1", "opening", 5, 5, 5, 5, 5, (), ""))
    best_id, best_score = scorer.get_best_debater()
    assert best_id == "pro_debater_1"
    assert best_score == 10.0


def test_scorer_get_stage_scores():
    scorer = Scorer()
    scorer.record(ScoreCard("pro_debater_1", "opening", 8, 8, 8, 8, 8, (), ""))
    scorer.record(ScoreCard("con_debater_1", "opening", 6, 6, 6, 6, 6, (), ""))
    scorer.record(ScoreCard("pro_debater_2", "cross_exam", 7, 7, 7, 7, 7, (), ""))

    stage_scores = scorer.get_stage_summary("opening")
    assert stage_scores["pro"] == 8.0
    assert stage_scores["con"] == 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scorer.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement Scorer**

`src/engine/scorer.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreCard:
    speaker: str
    stage: str
    logic: int
    persuasion: int
    expression: int
    teamwork: int
    rule_compliance: int
    violations: tuple[str, ...]
    comment: str


# Speaker ID -> team mapping helper
def _team_of(speaker: str) -> str:
    if speaker.startswith("pro"):
        return "pro"
    if speaker.startswith("con"):
        return "con"
    return "judge"


class Scorer:
    """Three-layer scoring system: real-time + rule-engine + aggregation."""

    DEFAULT_WEIGHTS = {
        "logic": 0.25,
        "persuasion": 0.25,
        "expression": 0.20,
        "teamwork": 0.15,
        "rule_compliance": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights or self.DEFAULT_WEIGHTS
        self._cards: list[ScoreCard] = []
        self._individual_penalties: dict[str, list[tuple[float, str]]] = {}
        self._team_penalties: dict[str, list[tuple[float, str]]] = {}

    def compute_speech_score(self, card: ScoreCard) -> float:
        w = self._weights
        return (
            card.logic * w["logic"]
            + card.persuasion * w["persuasion"]
            + card.expression * w["expression"]
            + card.teamwork * w["teamwork"]
            + card.rule_compliance * w["rule_compliance"]
        )

    def record(self, card: ScoreCard) -> float:
        """Record a score card and return the computed speech score."""
        self._cards.append(card)
        return self.compute_speech_score(card)

    def add_individual_penalty(self, speaker: str, penalty: float, reason: str) -> None:
        self._individual_penalties.setdefault(speaker, []).append((penalty, reason))

    def add_team_penalty(self, team: str, penalty: float, reason: str) -> None:
        self._team_penalties.setdefault(team, []).append((penalty, reason))

    def get_team_penalty(self, team: str) -> float:
        return sum(p for p, _ in self._team_penalties.get(team, []))

    def get_individual_total(self, speaker: str) -> float:
        speech_total = sum(
            self.compute_speech_score(c) for c in self._cards if c.speaker == speaker
        )
        penalty_total = sum(
            p for p, _ in self._individual_penalties.get(speaker, [])
        )
        return speech_total + penalty_total

    def get_team_total(self, team: str) -> float:
        members = {c.speaker for c in self._cards if _team_of(c.speaker) == team}
        for speaker in self._individual_penalties:
            if _team_of(speaker) == team:
                members.add(speaker)
        individual_sum = sum(self.get_individual_total(s) for s in members)
        return individual_sum + self.get_team_penalty(team)

    def get_best_debater(self) -> tuple[str, float]:
        speakers = {c.speaker for c in self._cards}
        best_id = ""
        best_score = float("-inf")
        for s in speakers:
            total = self.get_individual_total(s)
            if total > best_score:
                best_score = total
                best_id = s
        return best_id, best_score

    def get_stage_summary(self, stage: str) -> dict[str, float]:
        """Get total scores per team for a specific stage."""
        result: dict[str, float] = {"pro": 0.0, "con": 0.0}
        for card in self._cards:
            if card.stage == stage:
                team = _team_of(card.speaker)
                if team in result:
                    result[team] += self.compute_speech_score(card)
        return result

    def export(self) -> dict:
        """Export full scoring data for debate log."""
        return {
            "cards": [
                {
                    "speaker": c.speaker, "stage": c.stage,
                    "logic": c.logic, "persuasion": c.persuasion,
                    "expression": c.expression, "teamwork": c.teamwork,
                    "rule_compliance": c.rule_compliance,
                    "violations": list(c.violations), "comment": c.comment,
                    "total": self.compute_speech_score(c),
                }
                for c in self._cards
            ],
            "individual_penalties": {
                k: [(p, r) for p, r in v]
                for k, v in self._individual_penalties.items()
            },
            "team_penalties": {
                k: [(p, r) for p, r in v]
                for k, v in self._team_penalties.items()
            },
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/scorer.py tests/test_scorer.py
git commit -m "feat: add Scorer with weighted scoring, penalties, and stage summaries"
```

---

## Task 7: Terminal Display

**Files:**
- Create: `src/display/__init__.py`, `src/display/terminal.py`
- Test: `tests/test_display.py`

- [ ] **Step 1: Write failing tests**

`tests/test_display.py`:
```python
from io import StringIO
from rich.console import Console
from src.display.terminal import TerminalDisplay


def _make_display() -> tuple[TerminalDisplay, StringIO]:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=60)
    display = TerminalDisplay(console=console)
    return display, buf


def test_display_header():
    display, buf = _make_display()
    display.show_header("AI利大于弊", "利大于弊", "弊大于利")
    output = buf.getvalue()
    assert "利大于弊" in output
    assert "弊大于利" in output


def test_display_stage_banner():
    display, buf = _make_display()
    display.show_stage_banner("第一阶段：陈词")
    output = buf.getvalue()
    assert "陈词" in output


def test_display_speech():
    display, buf = _make_display()
    display.show_speech(
        name="正方一辩", team="pro", content="我方认为...",
        time_used=165.0, time_limit=180.0, char_count=500, char_limit=750,
    )
    output = buf.getvalue()
    assert "正方一辩" in output
    assert "我方认为" in output


def test_display_overtime_warning():
    display, buf = _make_display()
    display.show_overtime(
        name="正方二辩", team="pro",
        excess_seconds=8.0, team_penalty=-3, individual_penalty=-2,
    )
    output = buf.getvalue()
    assert "超时" in output or "8" in output


def test_display_score():
    display, buf = _make_display()
    display.show_score(
        speaker_name="正方一辩",
        logic=8, persuasion=7, expression=8, teamwork=7,
        rule_compliance=10, total=8.05, comment="论证清晰",
    )
    output = buf.getvalue()
    assert "8.05" in output or "8.0" in output


def test_display_scoreboard():
    display, buf = _make_display()
    display.show_scoreboard(pro_score=42.5, con_score=38.2)
    output = buf.getvalue()
    assert "42.5" in output
    assert "38.2" in output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_display.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement TerminalDisplay**

`src/display/__init__.py`: empty file

`src/display/terminal.py`:
```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

_TEAM_COLORS = {"pro": "blue", "con": "red", "judge": "yellow"}


class TerminalDisplay:
    """Rich-based colored terminal display for debate output."""

    def __init__(self, console: Console | None = None):
        self._console = console or Console()

    def show_header(self, topic: str, pro_stance: str, con_stance: str) -> None:
        self._console.print()
        self._console.print(
            Panel(
                f"[bold]辩题：{topic}[/bold]\n"
                f"[blue]正方：{pro_stance}[/blue]    "
                f"[red]反方：{con_stance}[/red]",
                title="[bold]AI 辩论赛[/bold]",
                border_style="bright_white",
            )
        )

    def show_stage_banner(self, stage_name: str) -> None:
        self._console.print()
        self._console.print(Rule(f"[bold]{stage_name}[/bold]", style="bright_white"))
        self._console.print()

    def show_speech(
        self,
        name: str,
        team: str,
        content: str,
        time_used: float,
        time_limit: float,
        char_count: int,
        char_limit: int,
    ) -> None:
        color = _TEAM_COLORS.get(team, "white")
        minutes = int(time_used) // 60
        seconds = int(time_used) % 60
        limit_min = int(time_limit) // 60
        limit_sec = int(time_limit) % 60

        is_overtime = time_used > time_limit
        time_style = "red bold" if is_overtime else "green"
        time_mark = "!" if is_overtime else ""

        header = Text()
        header.append(f"[{name}]", style=f"bold {color}")
        header.append(f"  {char_count}/{char_limit}字", style="dim")
        header.append(
            f"  {minutes}:{seconds:02d}/{limit_min}:{limit_sec:02d}{time_mark}",
            style=time_style,
        )

        self._console.print(header)
        self._console.print(
            Panel(content, border_style=color, padding=(0, 1))
        )

    def show_overtime(
        self,
        name: str,
        team: str,
        excess_seconds: float,
        team_penalty: float,
        individual_penalty: float,
    ) -> None:
        self._console.print(
            f"  [magenta bold]超时 {excess_seconds:.0f}秒 — "
            f"{name} {individual_penalty}分, 队伍 {team_penalty}分[/magenta bold]"
        )

    def show_score(
        self,
        speaker_name: str,
        logic: int,
        persuasion: int,
        expression: int,
        teamwork: int,
        rule_compliance: int,
        total: float,
        comment: str,
    ) -> None:
        self._console.print(
            f"  [yellow][裁判][/yellow] "
            f"逻辑:[green]{logic}[/green] "
            f"说服力:[green]{persuasion}[/green] "
            f"表达:[green]{expression}[/green] "
            f"配合:[green]{teamwork}[/green] "
            f"规则:[green]{rule_compliance}[/green] "
            f"| [bold green]{total:.1f}[/bold green]分"
        )
        if comment:
            self._console.print(f"  [yellow]  \"{comment}\"[/yellow]")

    def show_violation(self, speaker_name: str, violation: str, penalty: float) -> None:
        self._console.print(
            f"  [magenta]违规: {speaker_name} — {violation} ({penalty}分)[/magenta]"
        )

    def show_scoreboard(self, pro_score: float, con_score: float) -> None:
        self._console.print()
        table = Table(title="当前比分", show_header=False, border_style="bright_white")
        table.add_column(justify="right")
        table.add_column(justify="center")
        table.add_column(justify="left")
        table.add_row(
            f"[bold blue]正方[/bold blue]",
            f"[bold green]{pro_score:.1f}[/bold green] : [bold green]{con_score:.1f}[/bold green]",
            f"[bold red]反方[/bold red]",
        )
        self._console.print(table)

    def show_final_result(
        self,
        pro_total: float,
        con_total: float,
        best_debater_name: str,
        best_debater_score: float,
        stage_scores: dict[str, dict[str, float]],
        judge_comment: str,
    ) -> None:
        self._console.print()
        self._console.print(Rule("[bold]辩论赛结果[/bold]", style="bright_white"))

        # Winner
        if pro_total > con_total:
            winner = "[bold blue]正方胜[/bold blue]"
        elif con_total > pro_total:
            winner = "[bold red]反方胜[/bold red]"
        else:
            winner = "[bold]平局[/bold]"

        self._console.print(f"\n  {winner}")
        self._console.print(
            f"  最终比分: [blue]{pro_total:.1f}[/blue] : [red]{con_total:.1f}[/red]"
        )
        self._console.print(
            f"  最佳辩手: [bold green]{best_debater_name}[/bold green] ({best_debater_score:.1f}分)"
        )

        # Stage breakdown table
        stage_table = Table(title="各阶段得分", border_style="bright_white")
        stage_table.add_column("阶段")
        stage_table.add_column("正方", style="blue", justify="right")
        stage_table.add_column("反方", style="red", justify="right")

        stage_names = {"opening": "陈词", "cross_exam": "攻辩", "free_debate": "自由辩论", "closing": "总结陈词"}
        for key, label in stage_names.items():
            if key in stage_scores:
                stage_table.add_row(
                    label,
                    f"{stage_scores[key].get('pro', 0):.1f}",
                    f"{stage_scores[key].get('con', 0):.1f}",
                )

        self._console.print()
        self._console.print(stage_table)

        # Judge comment
        if judge_comment:
            self._console.print(
                Panel(judge_comment, title="[yellow]评委点评[/yellow]", border_style="yellow")
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_display.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/display/ tests/test_display.py
git commit -m "feat: add Rich-based terminal display with colored output and scoreboard"
```

---

## Task 8: Prompt Templates

**Files:**
- Create: `src/agents/__init__.py`, `src/agents/prompts.py`

- [ ] **Step 1: Create prompts.py with all prompt templates**

`src/agents/__init__.py`: empty file

`src/agents/prompts.py` — contains all prompt templates as string constants.
Copy the exact prompt text from the spec (lines 371-551) into Python string constants:

```python
COMMON_RULES = """【辩论赛规则 — 你必须严格遵守以下规则，违反将被扣分】

1. 攻辩阶段：
   - 每次提问只限一个问题
   - 攻方必须提出3个以上问题
   - 辩方必须正面回答问题，不得反问
   - 攻方不得回答问题
   - 攻辩双方必须单独完成本轮，不得中途更替

2. 自由辩论阶段：
   - 正方先发言，之后正反方轮流
   - 每位辩手至少发言一次
   - 一方辩手发言后，同方不得连续发言
   - 可引用书本、报刊摘要加强论据

3. 通用规则：
   - 在规定字数内完成发言（超出将被扣分）
   - 发言必须与辩题相关
   - 尊重对手，不得进行人身攻击"""


DEBATER_1_SYSTEM = """【身份】你是{team}一辩，你的队伍立场是：{stance}
【辩题】{topic}
【性格】{personality_prompt}

【职责】
1.【立论陈词】
  - 明确界定核心概念和辩论边界
  - 构建完整论证框架（2-3个核心论点）
  - 每个论点需有论据支撑
  - 字数控制在750字以内（约3分钟）

2.【攻辩小结】
  - 总结本方在攻辩中的优势和收获
  - 指出对方在攻辩中暴露的漏洞
  - 必须引用攻辩阶段的实际发言内容，严禁背稿
  - 字数控制在500字以内（约2分钟）

{common_rules}"""


DEBATER_2_SYSTEM = """【身份】你是{team}二辩，你的队伍立场是：{stance}
【辩题】{topic}
【性格】{personality_prompt}

【职责】
1.【作为攻方提问】
  - 选择向对方二辩或三辩提问（只能选一人）
  - 先输出你的选择：[选择: 对方X辩]
  - 用编号列出3个以上简短精炼的问题（每个问题约30-40字）
  - 只能提问，不得回答
  - 字数控制在125字以内（约30秒）

2.【作为辩方回答】
  - 必须正面回答，不得反问
  - 回答简洁有力，融入己方论点
  - 字数控制在250字以内（约1分钟）

3.【自由辩论】
  - 至少发言一次
  - 抓住对方漏洞反驳

{common_rules}"""


DEBATER_3_SYSTEM = """【身份】你是{team}三辩，你的队伍立场是：{stance}
【辩题】{topic}
【性格】{personality_prompt}

【职责】
1.【作为攻方提问】
  - 选择向对方二辩或三辩提问（只能选一人）
  - 先输出你的选择：[选择: 对方X辩]
  - 用编号列出3个以上深挖逻辑矛盾的问题（每个问题约30-40字）
  - 只能提问，不得回答
  - 字数控制在125字以内（约30秒）

2.【作为辩方回答】
  - 必须正面回答，不得反问
  - 从更高维度化解对方攻势
  - 字数控制在250字以内（约1分钟）

3.【自由辩论】
  - 至少发言一次
  - 承接队友论点深化，与二辩互补

{common_rules}"""


DEBATER_4_SYSTEM = """【身份】你是{team}四辩，你的队伍立场是：{stance}
【辩题】{topic}
【性格】{personality_prompt}

【职责】
1.【自由辩论】
  - 至少发言一次
  - 把控全局方向，拉回被带偏的议题

2.【总结陈词】
  - 回顾全场辩论，梳理关键交锋点
  - 指出对方未能回应的核心问题
  - 强化己方最有力的论点
  - 做出有感染力的最终收束
  - 字数控制在750字以内（约3分钟）

{common_rules}"""

# Map position number to system prompt template
DEBATER_PROMPTS: dict[int, str] = {
    1: DEBATER_1_SYSTEM,
    2: DEBATER_2_SYSTEM,
    3: DEBATER_3_SYSTEM,
    4: DEBATER_4_SYSTEM,
}


JUDGE_SYSTEM = """【身份】你是本场辩论赛的评委，必须保持公正、客观、专业。
【辩题】{topic}
【正方立场】{pro_stance}
【反方立场】{con_stance}

【评分维度】（1-10分）：
1. 逻辑性(logic)：论证严密性，推理有效性
2. 说服力(persuasion)：论据充分性
3. 语言表达(expression)：措辞精准性，流畅度
4. 团队配合(teamwork)：队友衔接，攻防协调
5. 规则遵守(rule_compliance)：是否遵守辩论规则

【违规检测】标记以下行为：
- "counter_question": 辩方反问
- "not_direct_answer": 未正面回答
- "attacker_answered": 攻方回答问题
- "off_topic": 偏离辩题
- "scripted_summary": 攻辩小结背稿
- "personal_attack": 人身攻击

【评分输出格式】（严格 JSON，不要输出其他内容）：
{{"speaker": "发言人ID", "logic": 8, "persuasion": 7, "expression": 8, "teamwork": 7, "rule_compliance": 10, "violations": [], "comment": "简短点评"}}

【点评输出格式】（严格 JSON，不要输出其他内容）：
{{"type": "review", "summary": "点评内容（200字以内）", "highlights": ["亮点1"], "suggestions": ["建议1"]}}"""


CAPTAIN_SYSTEM = """【任务】你是{team}的队长，负责决定本轮自由辩论由谁发言，并给出回应方向建议。

【队内发言记录】
{team_messages}

【公开辩论记录（最近5条）】
{recent_public_messages}

【各辩手已发言次数】
{speak_counts}

【未发言辩手】
{unspeaking_debaters}

【剩余时间】{time_left}秒

请输出严格 JSON，不要输出其他内容：
{{"speaker": "辩手ID", "direction": "建议回应方向"}}"""
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/__init__.py src/agents/prompts.py
git commit -m "feat: add all prompt templates for debaters, judge, and captain"
```

---

## Task 9: BaseAgent and DebaterAgent

**Files:**
- Create: `src/agents/base.py`, `src/agents/debater.py`
- Test: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests**

`tests/test_agents.py`:
```python
import time
from src.agents.base import BaseAgent
from src.agents.debater import DebaterAgent
from src.agents.prompts import COMMON_RULES
from src.engine.message_pool import Message, MessagePool
from tests.conftest import FakeLLM


def test_base_agent_build_context_pro():
    llm = FakeLLM()
    agent = BaseAgent(
        name="正方一辩", agent_id="pro_debater_1", team="pro",
        system_prompt="你是正方一辩", llm=llm,
    )
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("con_debater_1", "一辩", "con", "opening", "反方陈词", "speech", ts, 4, ()))
    pool.publish("team_pro", Message("pro_debater_2", "二辩", "pro", "opening", "队内策略", "team_strategy", ts, 4, ()))
    pool.publish("team_con", Message("con_debater_2", "二辩", "con", "opening", "对方队内", "team_strategy", ts, 4, ()))

    context = agent.build_context(pool, "opening")
    # System prompt should be first
    assert context[0]["role"] == "system"
    # Should see public and team_pro messages, not team_con
    content_str = str(context)
    assert "反方陈词" in content_str
    assert "队内策略" in content_str
    assert "对方队内" not in content_str


def test_base_agent_speak():
    llm = FakeLLM(responses=["我方认为人工智能利大于弊。"])
    agent = BaseAgent(
        name="正方一辩", agent_id="pro_debater_1", team="pro",
        system_prompt="你是正方一辩", llm=llm,
    )
    pool = MessagePool()
    result = agent.speak(pool, "opening", "请进行立论陈词")
    assert result == "我方认为人工智能利大于弊。"
    assert len(llm.call_history) == 1


def test_debater_agent_creates_system_prompt():
    llm = FakeLLM()
    agent = DebaterAgent.create(
        team="pro", position=1, stance="利大于弊",
        topic="AI发展利弊", personality_prompt="逻辑严密型", llm=llm,
    )
    assert agent.name == "正方一辩"
    assert agent.agent_id == "pro_debater_1"
    assert "利大于弊" in agent.system_prompt
    assert "AI发展利弊" in agent.system_prompt
    assert COMMON_RULES in agent.system_prompt


def test_debater_agent_con_team():
    llm = FakeLLM()
    agent = DebaterAgent.create(
        team="con", position=3, stance="弊大于利",
        topic="AI发展利弊", personality_prompt="犀利进攻型", llm=llm,
    )
    assert agent.name == "反方三辩"
    assert agent.agent_id == "con_debater_3"
    assert agent.team == "con"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agents.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement BaseAgent**

`src/agents/base.py`:
```python
from src.llm.base import BaseLLM
from src.engine.message_pool import MessagePool


class BaseAgent:
    """Base class for all debate agents."""

    def __init__(
        self,
        name: str,
        agent_id: str,
        team: str,
        system_prompt: str,
        llm: BaseLLM,
    ):
        self.name = name
        self.agent_id = agent_id
        self.team = team
        self.system_prompt = system_prompt
        self.llm = llm

    def build_context(
        self,
        pool: MessagePool,
        stage: str,
    ) -> list[dict]:
        """Build LLM message list from visible messages."""
        messages = [{"role": "system", "content": self.system_prompt}]

        visible = pool.get_visible_messages(self.team)
        for msg in visible:
            prefix = f"[{msg.role}({msg.team})]" if msg.team != self.team else f"[{msg.role}]"
            messages.append({
                "role": "assistant" if msg.speaker == self.agent_id else "user",
                "content": f"{prefix} {msg.content}",
            })

        return messages

    def speak(
        self,
        pool: MessagePool,
        stage: str,
        instruction: str,
    ) -> str:
        """Generate a speech given the current context and instruction."""
        context = self.build_context(pool, stage)
        context.append({"role": "user", "content": instruction})
        return self.llm.chat(context)
```

- [ ] **Step 4: Implement DebaterAgent**

`src/agents/debater.py`:
```python
from src.agents.base import BaseAgent
from src.agents.prompts import DEBATER_PROMPTS, COMMON_RULES
from src.llm.base import BaseLLM

_TEAM_NAMES = {"pro": "正方", "con": "反方"}
_POSITION_NAMES = {1: "一辩", 2: "二辩", 3: "三辩", 4: "四辩"}


class DebaterAgent(BaseAgent):
    """Debate agent for any position (1-4), configured via prompts."""

    def __init__(
        self,
        name: str,
        agent_id: str,
        team: str,
        position: int,
        position_name: str,
        system_prompt: str,
        llm: BaseLLM,
    ):
        super().__init__(name, agent_id, team, system_prompt, llm)
        self.position = position
        self.position_name = position_name  # "一辩", "二辩", etc.

    @classmethod
    def create(
        cls,
        team: str,
        position: int,
        stance: str,
        topic: str,
        personality_prompt: str,
        llm: BaseLLM,
    ) -> "DebaterAgent":
        team_cn = _TEAM_NAMES[team]
        pos_cn = _POSITION_NAMES[position]
        name = f"{team_cn}{pos_cn}"
        agent_id = f"{team}_debater_{position}"

        template = DEBATER_PROMPTS[position]
        system_prompt = template.format(
            team=team_cn,
            stance=stance,
            topic=topic,
            personality_prompt=personality_prompt,
            common_rules=COMMON_RULES,
        )

        return cls(
            name=name,
            agent_id=agent_id,
            team=team,
            position=position,
            position_name=pos_cn,
            system_prompt=system_prompt,
            llm=llm,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_agents.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/agents/base.py src/agents/debater.py tests/test_agents.py
git commit -m "feat: add BaseAgent and DebaterAgent with prompt-based role configuration"
```

---

## Task 10: JudgeAgent

**Files:**
- Create: `src/agents/judge.py`
- Extend: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_agents.py`:
```python
import json
from src.agents.judge import JudgeAgent


def test_judge_agent_creates_system_prompt():
    llm = FakeLLM()
    judge = JudgeAgent.create(
        topic="AI发展利弊", pro_stance="利大于弊", con_stance="弊大于利", llm=llm,
    )
    assert judge.name == "裁判"
    assert judge.agent_id == "judge"
    assert "公正" in judge.system_prompt
    assert "利大于弊" in judge.system_prompt


def test_judge_parse_score_valid():
    score_json = json.dumps({
        "speaker": "pro_debater_1",
        "logic": 8, "persuasion": 7, "expression": 8,
        "teamwork": 7, "rule_compliance": 10,
        "violations": [], "comment": "Good",
    })
    llm = FakeLLM(responses=[score_json])
    judge = JudgeAgent.create(
        topic="AI发展利弊", pro_stance="利大于弊", con_stance="弊大于利", llm=llm,
    )
    pool = MessagePool()
    card = judge.score_speech(pool, "opening", "pro_debater_1", "发言内容...")
    assert card is not None
    assert card.logic == 8
    assert card.violations == ()


def test_judge_parse_score_with_violations():
    score_json = json.dumps({
        "speaker": "con_debater_2",
        "logic": 6, "persuasion": 5, "expression": 7,
        "teamwork": 6, "rule_compliance": 3,
        "violations": ["counter_question"], "comment": "反问了",
    })
    llm = FakeLLM(responses=[score_json])
    judge = JudgeAgent.create(
        topic="t", pro_stance="p", con_stance="c", llm=llm,
    )
    pool = MessagePool()
    card = judge.score_speech(pool, "cross_exam", "con_debater_2", "内容")
    assert card is not None
    assert "counter_question" in card.violations


def test_judge_parse_review():
    review_json = json.dumps({
        "type": "review",
        "summary": "双方表现不错",
        "highlights": ["正方立论清晰"],
        "suggestions": ["反方需加强"],
    })
    llm = FakeLLM(responses=[review_json])
    judge = JudgeAgent.create(
        topic="t", pro_stance="p", con_stance="c", llm=llm,
    )
    pool = MessagePool()
    review = judge.give_review(pool, "对全场辩论进行总点评")
    assert review is not None
    assert "双方表现不错" in review["summary"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agents.py -v
```
Expected: New tests FAIL

- [ ] **Step 3: Implement JudgeAgent**

`src/agents/judge.py`:
```python
import json
import logging
from typing import Optional

from src.agents.base import BaseAgent
from src.agents.prompts import JUDGE_SYSTEM
from src.engine.message_pool import MessagePool
from src.engine.scorer import ScoreCard
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """Judge agent that scores speeches and detects violations."""

    @classmethod
    def create(
        cls,
        topic: str,
        pro_stance: str,
        con_stance: str,
        llm: BaseLLM,
    ) -> "JudgeAgent":
        system_prompt = JUDGE_SYSTEM.format(
            topic=topic,
            pro_stance=pro_stance,
            con_stance=con_stance,
        )
        return cls(
            name="裁判",
            agent_id="judge",
            team="judge",
            system_prompt=system_prompt,
            llm=llm,
        )

    def score_speech(
        self,
        pool: MessagePool,
        stage: str,
        speaker_id: str,
        speech_content: str,
    ) -> Optional[ScoreCard]:
        """Score a speech and return a ScoreCard, or None on failure."""
        instruction = (
            f"请对以下发言进行评分。\n"
            f"发言人: {speaker_id}\n"
            f"阶段: {stage}\n"
            f"发言内容:\n{speech_content}\n\n"
            f"请输出严格 JSON 评分。"
        )
        for attempt in range(2):
            raw = self.speak(pool, stage, instruction)
            card = self._parse_score(raw, speaker_id, stage)
            if card is not None:
                return card
            logger.warning("Judge score parse failed (attempt %d), retrying", attempt + 1)
        logger.error("Judge score parse failed after retries for %s", speaker_id)
        return None

    def give_review(
        self,
        pool: MessagePool,
        instruction: str,
    ) -> Optional[dict]:
        """Generate a review comment. Returns parsed dict or None."""
        raw = self.speak(pool, "review", instruction)
        return self._parse_review(raw)

    @staticmethod
    def _parse_score(raw: str, speaker_id: str, stage: str) -> Optional[ScoreCard]:
        try:
            # Try to extract JSON from response
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)
            return ScoreCard(
                speaker=data.get("speaker", speaker_id),
                stage=stage,
                logic=int(data["logic"]),
                persuasion=int(data["persuasion"]),
                expression=int(data["expression"]),
                teamwork=int(data["teamwork"]),
                rule_compliance=int(data["rule_compliance"]),
                violations=tuple(data.get("violations", [])),
                comment=data.get("comment", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse judge score: %s", e)
            return None

    @staticmethod
    def _parse_review(raw: str) -> Optional[dict]:
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse judge review: %s", e)
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agents.py -v
```
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/agents/judge.py tests/test_agents.py
git commit -m "feat: add JudgeAgent with score parsing, violation detection, and review"
```

---

## Task 11: Stage Base Class + Opening Stage

**Files:**
- Create: `src/stages/__init__.py`, `src/stages/base.py`, `src/stages/opening.py`
- Test: `tests/test_stages.py`

- [ ] **Step 1: Write failing tests**

`tests/test_stages.py`:
```python
import json
from src.stages.opening import OpeningStage
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent
from src.display.terminal import TerminalDisplay
from tests.conftest import FakeLLM
from io import StringIO
from rich.console import Console


def _make_agents(speech_llm, judge_llm):
    """Create a full set of agents for testing."""
    agents = {}
    for team, stance in [("pro", "利大于弊"), ("con", "弊大于利")]:
        for pos in range(1, 5):
            agent = DebaterAgent.create(
                team=team, position=pos, stance=stance,
                topic="AI利弊", personality_prompt="逻辑型", llm=speech_llm,
            )
            agents[agent.agent_id] = agent
    judge = JudgeAgent.create(
        topic="AI利弊", pro_stance="利大于弊", con_stance="弊大于利", llm=judge_llm,
    )
    agents["judge"] = judge
    return agents, judge


def test_opening_stage_executes_two_speeches():
    score_json = json.dumps({
        "speaker": "test", "logic": 8, "persuasion": 7,
        "expression": 8, "teamwork": 7, "rule_compliance": 10,
        "violations": [], "comment": "ok",
    })
    speech_llm = FakeLLM(responses=["正方一辩的立论陈词内容。" * 10])
    judge_llm = FakeLLM(responses=[score_json])
    agents, judge = _make_agents(speech_llm, judge_llm)

    pool = MessagePool()
    timer = Timer()
    scorer = Scorer()
    buf = StringIO()
    display = TerminalDisplay(console=Console(file=buf, force_terminal=True, width=80))

    stage = OpeningStage()
    stage.execute(agents, pool, timer, scorer, judge, display)

    # Should have 2 public speeches (pro_debater_1 + con_debater_1)
    public_msgs = pool.get_messages("public", stage="opening")
    speech_msgs = [m for m in public_msgs if m.msg_type == "speech"]
    assert len(speech_msgs) == 2
    assert speech_msgs[0].speaker == "pro_debater_1"
    assert speech_msgs[1].speaker == "con_debater_1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stages.py::test_opening_stage_executes_two_speeches -v
```
Expected: FAIL

- [ ] **Step 3: Implement BaseStage and OpeningStage**

`src/stages/__init__.py`: empty file

`src/stages/base.py`:
```python
import time
import logging
from abc import ABC, abstractmethod

from src.agents.base import BaseAgent
from src.agents.judge import JudgeAgent
from src.engine.message_pool import Message, MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.display.terminal import TerminalDisplay

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Abstract base class for debate stages."""

    stage_name: str = ""

    # Default penalty config — overridden by config values passed to __init__
    DEFAULT_PENALTIES = {
        "overtime_team": -3,
        "overtime_individual": -2,
        "counter_question": -2,
        "not_direct_answer": -2,
        "attacker_answered": -2,
        "off_topic": -2,
        "scripted_summary": -2,
        "personal_attack": -3,
        "consecutive_speech": -3,
    }

    def __init__(self, penalties: dict[str, float] | None = None):
        self._penalties = penalties or self.DEFAULT_PENALTIES

    @abstractmethod
    def execute(
        self,
        agents: dict[str, BaseAgent],
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        judge: JudgeAgent,
        display: TerminalDisplay,
    ) -> None:
        """Execute this stage of the debate."""

    def _do_speech(
        self,
        agent: BaseAgent,
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        judge: JudgeAgent,
        display: TerminalDisplay,
        instruction: str,
        time_limit_seconds: float,
    ) -> str:
        """Have an agent speak, score it, check overtime, display it."""
        # LLM failure handling: retry is inside LLM; if it still fails, debater is silent
        try:
            content = agent.speak(pool, self.stage_name, instruction)
        except Exception as e:
            logger.error("Agent %s failed to speak: %s", agent.agent_id, e)
            content = ""
            scorer.add_individual_penalty(agent.agent_id, -2, "llm_failure_silence")
            display._console.print(
                f"  [magenta]{agent.name} 发言失败（LLM 错误），判为沉默，扣 teamwork 分[/magenta]"
            )

        if not content:
            return ""

        char_limit = timer.char_limit(time_limit_seconds)
        word_count = len(content)
        time_used = timer.estimate_duration(content)

        # Publish to public + team channel
        ts = time.time()
        msg = Message(
            speaker=agent.agent_id,
            role=getattr(agent, "position_name", agent.name),
            team=agent.team,
            stage=self.stage_name,
            content=content,
            msg_type="speech",
            timestamp=ts,
            word_count=word_count,
            metadata=(),
        )
        pool.publish("public", msg)
        team_channel = f"team_{agent.team}" if agent.team in ("pro", "con") else None
        if team_channel:
            pool.publish(team_channel, msg)

        # Display speech
        display.show_speech(
            name=agent.name, team=agent.team, content=content,
            time_used=time_used, time_limit=time_limit_seconds,
            char_count=word_count, char_limit=char_limit,
        )

        # Check overtime (penalties from config)
        is_overtime, excess = timer.check_overtime(content, time_limit_seconds)
        if is_overtime:
            team_pen = self._penalties.get("overtime_team", -3)
            ind_pen = self._penalties.get("overtime_individual", -2)
            scorer.add_team_penalty(agent.team, team_pen, f"overtime_{agent.agent_id}")
            scorer.add_individual_penalty(agent.agent_id, ind_pen, "overtime")
            display.show_overtime(agent.name, agent.team, excess, team_pen, ind_pen)

        # Judge scores
        card = judge.score_speech(pool, self.stage_name, agent.agent_id, content)
        if card is not None:
            total = scorer.record(card)
            display.show_score(
                speaker_name=agent.name,
                logic=card.logic, persuasion=card.persuasion,
                expression=card.expression, teamwork=card.teamwork,
                rule_compliance=card.rule_compliance,
                total=total, comment=card.comment,
            )
            # Process ALL violation types from config
            for v in card.violations:
                penalty = self._penalties.get(v, -2)
                scorer.add_individual_penalty(agent.agent_id, penalty, v)
                display.show_violation(agent.name, v, penalty)

        # Update scoreboard
        display.show_scoreboard(
            scorer.get_team_total("pro"),
            scorer.get_team_total("con"),
        )

        return content
```

`src/stages/opening.py`:
```python
from src.agents.base import BaseAgent
from src.agents.judge import JudgeAgent
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.display.terminal import TerminalDisplay
from src.stages.base import BaseStage


class OpeningStage(BaseStage):
    """Stage 1: Opening statements (陈词阶段)."""

    stage_name = "opening"

    def execute(
        self,
        agents: dict[str, BaseAgent],
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        judge: JudgeAgent,
        display: TerminalDisplay,
    ) -> None:
        display.show_stage_banner("第一阶段：陈词")

        # Pro first debater: 3 minutes / 750 chars
        pro_1 = agents["pro_debater_1"]
        self._do_speech(
            pro_1, pool, timer, scorer, judge, display,
            instruction="请进行立论陈词。明确界定核心概念，构建完整论证框架（2-3个核心论点），每个论点需有论据支撑。字数控制在750字以内。",
            time_limit_seconds=180,
        )

        # Con first debater: 3 minutes / 750 chars
        con_1 = agents["con_debater_1"]
        self._do_speech(
            con_1, pool, timer, scorer, judge, display,
            instruction="请进行立论陈词。明确界定核心概念，构建完整论证框架（2-3个核心论点），每个论点需有论据支撑。字数控制在750字以内。",
            time_limit_seconds=180,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stages.py -v
```
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/stages/ tests/test_stages.py
git commit -m "feat: add BaseStage with speech/score/overtime flow, and OpeningStage"
```

---

## Task 12: CrossExamStage

**Files:**
- Create: `src/stages/cross_exam.py`
- Extend: `tests/test_stages.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_stages.py`:
```python
def test_cross_exam_stage_four_rounds_plus_summaries():
    score_json = json.dumps({
        "speaker": "test", "logic": 7, "persuasion": 7,
        "expression": 7, "teamwork": 7, "rule_compliance": 7,
        "violations": [], "comment": "ok",
    })
    # Cross-exam: 4 rounds (question + answer each) + 2 summaries = 10 speeches
    speech_llm = FakeLLM(responses=["[选择: 反方二辩]\n1.问题一？\n2.问题二？\n3.问题三？", "回答内容" * 5])
    judge_llm = FakeLLM(responses=[score_json])
    agents, judge = _make_agents(speech_llm, judge_llm)

    pool = MessagePool()
    timer = Timer()
    scorer = Scorer()
    buf = StringIO()
    display = TerminalDisplay(console=Console(file=buf, force_terminal=True, width=80))

    from src.stages.cross_exam import CrossExamStage
    stage = CrossExamStage()
    stage.execute(agents, pool, timer, scorer, judge, display)

    public_msgs = pool.get_messages("public", stage="cross_exam")
    speech_msgs = [m for m in public_msgs if m.msg_type == "speech"]
    # 4 questions + 4 answers + 2 summaries = 10
    assert len(speech_msgs) == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_stages.py::test_cross_exam_stage_four_rounds_plus_summaries -v
```
Expected: FAIL

- [ ] **Step 3: Implement CrossExamStage**

`src/stages/cross_exam.py`:
```python
import re
import logging

from src.agents.base import BaseAgent
from src.agents.judge import JudgeAgent
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.display.terminal import TerminalDisplay
from src.stages.base import BaseStage

logger = logging.getLogger(__name__)

# Fixed cross-exam rounds: (attacker_id, defender_team)
_ROUNDS = [
    ("pro_debater_2", "con"),
    ("con_debater_2", "pro"),
    ("pro_debater_3", "con"),
    ("con_debater_3", "pro"),
]


def _parse_target(content: str, defender_team: str) -> str:
    """Parse '[选择: 反方X辩]' from attacker's output. Default to debater 2."""
    match = re.search(r"\[选择:\s*(?:正方|反方)(\S)辩\]", content)
    if match:
        pos_map = {"二": 2, "三": 3}
        pos = pos_map.get(match.group(1), 2)
    else:
        pos = 2
        logger.warning("Could not parse target selection, defaulting to debater 2")
    return f"{defender_team}_debater_{pos}"


class CrossExamStage(BaseStage):
    """Stage 2: Cross-examination (攻辩阶段)."""

    stage_name = "cross_exam"

    def execute(
        self,
        agents: dict[str, BaseAgent],
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        judge: JudgeAgent,
        display: TerminalDisplay,
    ) -> None:
        display.show_stage_banner("第二阶段：攻辩")

        for attacker_id, defender_team in _ROUNDS:
            attacker = agents[attacker_id]
            team_cn = "反方" if defender_team == "con" else "正方"

            # Attacker questions (30 seconds / 125 chars)
            question_content = self._do_speech(
                attacker, pool, timer, scorer, judge, display,
                instruction=(
                    f"你现在作为攻方提问。请选择向{team_cn}二辩或{team_cn}三辩提问（只能选一人）。\n"
                    f"先输出你的选择：[选择: {team_cn}X辩]\n"
                    f"然后用编号列出3个以上简短精炼的问题。字数控制在125字以内。"
                ),
                time_limit_seconds=30,
            )

            # Parse target
            target_id = _parse_target(question_content, defender_team)
            defender = agents.get(target_id, agents[f"{defender_team}_debater_2"])

            # Defender answers (1 minute / 250 chars)
            self._do_speech(
                defender, pool, timer, scorer, judge, display,
                instruction=(
                    f"对方刚刚向你提出了问题，请正面回答。不得反问。字数控制在250字以内。"
                ),
                time_limit_seconds=60,
            )

        # Summaries: pro first debater, then con first debater (2 minutes each)
        display.show_stage_banner("攻辩小结")

        self._do_speech(
            agents["pro_debater_1"], pool, timer, scorer, judge, display,
            instruction=(
                "请进行攻辩小结。总结本方在攻辩中的优势和收获，指出对方在攻辩中暴露的漏洞。"
                "必须引用攻辩阶段的实际发言内容，严禁背稿。字数控制在500字以内。"
            ),
            time_limit_seconds=120,
        )

        self._do_speech(
            agents["con_debater_1"], pool, timer, scorer, judge, display,
            instruction=(
                "请进行攻辩小结。总结本方在攻辩中的优势和收获，指出对方在攻辩中暴露的漏洞。"
                "必须引用攻辩阶段的实际发言内容，严禁背稿。字数控制在500字以内。"
            ),
            time_limit_seconds=120,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stages.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/stages/cross_exam.py tests/test_stages.py
git commit -m "feat: add CrossExamStage with 4 rounds, target parsing, and summaries"
```

---

## Task 13: FreeDebateStage

**Files:**
- Create: `src/stages/free_debate.py`
- Extend: `tests/test_stages.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_stages.py`:
```python
def test_free_debate_stage_alternates_teams():
    score_json = json.dumps({
        "speaker": "test", "logic": 7, "persuasion": 7,
        "expression": 7, "teamwork": 7, "rule_compliance": 7,
        "violations": [], "comment": "ok",
    })
    captain_json = json.dumps({"speaker": "pro_debater_2", "direction": "反驳"})
    # Alternate captain decisions + speeches
    speech_responses = ["自由辩论发言内容" * 5] * 20  # enough for full debate
    captain_responses = [captain_json] * 20

    speech_llm = FakeLLM(responses=speech_responses)
    judge_llm = FakeLLM(responses=[score_json])
    # Captain uses separate LLM calls within the stage
    agents, judge = _make_agents(speech_llm, judge_llm)

    pool = MessagePool()
    timer = Timer()
    scorer = Scorer()
    buf = StringIO()
    display = TerminalDisplay(console=Console(file=buf, force_terminal=True, width=80))

    from src.stages.free_debate import FreeDebateStage
    stage = FreeDebateStage(captain_llm=FakeLLM(responses=[captain_json]))
    stage.execute(agents, pool, timer, scorer, judge, display)

    public_msgs = pool.get_messages("public", stage="free_debate")
    speech_msgs = [m for m in public_msgs if m.msg_type == "speech"]
    # Should have alternating teams
    assert len(speech_msgs) >= 2
    # First speaker is pro
    assert speech_msgs[0].team == "pro"
    # Teams alternate
    for i in range(1, len(speech_msgs)):
        assert speech_msgs[i].team != speech_msgs[i - 1].team
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_stages.py::test_free_debate_stage_alternates_teams -v
```
Expected: FAIL

- [ ] **Step 3: Implement FreeDebateStage**

`src/stages/free_debate.py`:
```python
import json
import logging

from src.agents.base import BaseAgent
from src.agents.judge import JudgeAgent
from src.agents.prompts import CAPTAIN_SYSTEM
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.display.terminal import TerminalDisplay
from src.llm.base import BaseLLM
from src.stages.base import BaseStage

logger = logging.getLogger(__name__)

_TEAM_MEMBERS = {
    "pro": ["pro_debater_1", "pro_debater_2", "pro_debater_3", "pro_debater_4"],
    "con": ["con_debater_1", "con_debater_2", "con_debater_3", "con_debater_4"],
}
_TEAM_CN = {"pro": "正方", "con": "反方"}
_TIME_PER_TEAM = 240  # 4 minutes per team


class FreeDebateStage(BaseStage):
    """Stage 3: Free debate (自由辩论阶段)."""

    stage_name = "free_debate"

    def __init__(self, captain_llm: BaseLLM | None = None, penalties: dict[str, float] | None = None):
        super().__init__(penalties=penalties)
        self._captain_llm = captain_llm

    def execute(
        self,
        agents: dict[str, BaseAgent],
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        judge: JudgeAgent,
        display: TerminalDisplay,
    ) -> None:
        display.show_stage_banner("第三阶段：自由辩论")

        time_left = {"pro": float(_TIME_PER_TEAM), "con": float(_TIME_PER_TEAM)}
        speak_counts: dict[str, int] = {aid: 0 for team in _TEAM_MEMBERS for aid in _TEAM_MEMBERS[team]}
        current_team = "pro"  # Pro starts

        while time_left["pro"] > 0 or time_left["con"] > 0:
            if time_left[current_team] <= 0:
                other = "con" if current_team == "pro" else "pro"
                if time_left[other] <= 0:
                    break
                current_team = other
                continue

            # Captain decides who speaks
            speaker_id, direction = self._captain_decide(
                current_team, agents, pool, timer, speak_counts, time_left[current_team],
            )
            speaker = agents.get(speaker_id)
            if speaker is None:
                speaker = agents[_TEAM_MEMBERS[current_team][0]]
                speaker_id = speaker.agent_id

            # Speaker speaks
            instruction = f"自由辩论环节，请发言。{direction} 字数要简洁有力。"
            content = self._do_speech(
                speaker, pool, timer, scorer, judge, display,
                instruction=instruction,
                time_limit_seconds=time_left[current_team],
            )

            # Deduct time
            duration = timer.estimate_duration(content)
            time_left[current_team] -= duration
            speak_counts[speaker_id] = speak_counts.get(speaker_id, 0) + 1

            # Display remaining time
            display._console.print(
                f"  [dim]剩余时间 — 正方: {max(0, time_left['pro']):.0f}秒  "
                f"反方: {max(0, time_left['con']):.0f}秒[/dim]"
            )

            # Switch team
            current_team = "con" if current_team == "pro" else "pro"

        # Enforce minimum participation — penalize non-speakers
        for team in ("pro", "con"):
            for aid in _TEAM_MEMBERS[team]:
                if speak_counts.get(aid, 0) == 0:
                    logger.warning("%s did not speak in free debate", aid)
                    scorer.add_individual_penalty(aid, -2, "no_speech_in_free_debate")
                    agent_name = agents[aid].name if aid in agents else aid
                    display._console.print(
                        f"  [magenta]{agent_name} 未在自由辩论中发言，扣 teamwork 分[/magenta]"
                    )

    def _captain_decide(
        self,
        team: str,
        agents: dict[str, BaseAgent],
        pool: MessagePool,
        timer: Timer,
        speak_counts: dict[str, int],
        time_left: float,
    ) -> tuple[str, str]:
        """Use captain LLM to decide who speaks next."""
        unspeaking = [aid for aid in _TEAM_MEMBERS[team] if speak_counts.get(aid, 0) == 0]

        # Force unspeaking debaters when time is running low (< 90s = ~1 speech per person)
        if unspeaking and time_left < 90 * len(unspeaking):
            return unspeaking[0], "你还未发言，请务必在本轮发言。"

        llm = self._captain_llm
        if llm is None:
            # Fallback: prefer unspeaking, then round-robin
            if unspeaking:
                return unspeaking[0], ""
            return _TEAM_MEMBERS[team][0], ""

        team_msgs = pool.get_visible_messages(team)
        recent_public = pool.get_messages("public")[-5:]
        unspeaking = [aid for aid in _TEAM_MEMBERS[team] if speak_counts.get(aid, 0) == 0]

        counts_str = ", ".join(f"{aid}: {speak_counts.get(aid, 0)}" for aid in _TEAM_MEMBERS[team])
        team_msgs_str = "\n".join(f"[{m.role}] {m.content[:100]}" for m in team_msgs[-5:]) or "（无）"
        public_msgs_str = "\n".join(f"[{m.role}({m.team})] {m.content[:100]}" for m in recent_public) or "（无）"
        unspeaking_str = ", ".join(unspeaking) or "（全员已发言）"

        prompt = CAPTAIN_SYSTEM.format(
            team=_TEAM_CN[team],
            team_messages=team_msgs_str,
            recent_public_messages=public_msgs_str,
            speak_counts=counts_str,
            unspeaking_debaters=unspeaking_str,
            time_left=f"{time_left:.0f}",
        )

        try:
            raw = llm.chat([{"role": "system", "content": prompt}])
            data = json.loads(raw.strip())
            speaker = data.get("speaker", _TEAM_MEMBERS[team][0])
            direction = data.get("direction", "")
            # Validate speaker is on the right team
            if speaker not in _TEAM_MEMBERS[team]:
                speaker = _TEAM_MEMBERS[team][0]
            return speaker, direction
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Captain decision parse failed: %s", e)
            for aid in _TEAM_MEMBERS[team]:
                if speak_counts.get(aid, 0) == 0:
                    return aid, ""
            return _TEAM_MEMBERS[team][0], ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stages.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/stages/free_debate.py tests/test_stages.py
git commit -m "feat: add FreeDebateStage with captain decision mechanism and time tracking"
```

---

## Task 14: ClosingStage

**Files:**
- Create: `src/stages/closing.py`
- Extend: `tests/test_stages.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_stages.py`:
```python
def test_closing_stage_con_then_pro():
    score_json = json.dumps({
        "speaker": "test", "logic": 8, "persuasion": 8,
        "expression": 8, "teamwork": 8, "rule_compliance": 8,
        "violations": [], "comment": "ok",
    })
    speech_llm = FakeLLM(responses=["总结陈词内容。" * 20])
    judge_llm = FakeLLM(responses=[score_json])
    agents, judge = _make_agents(speech_llm, judge_llm)

    pool = MessagePool()
    timer = Timer()
    scorer = Scorer()
    buf = StringIO()
    display = TerminalDisplay(console=Console(file=buf, force_terminal=True, width=80))

    from src.stages.closing import ClosingStage
    stage = ClosingStage()
    stage.execute(agents, pool, timer, scorer, judge, display)

    public_msgs = pool.get_messages("public", stage="closing")
    speech_msgs = [m for m in public_msgs if m.msg_type == "speech"]
    assert len(speech_msgs) == 2
    # Con goes first in closing
    assert speech_msgs[0].speaker == "con_debater_4"
    assert speech_msgs[1].speaker == "pro_debater_4"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_stages.py::test_closing_stage_con_then_pro -v
```
Expected: FAIL

- [ ] **Step 3: Implement ClosingStage**

`src/stages/closing.py`:
```python
from src.agents.base import BaseAgent
from src.agents.judge import JudgeAgent
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.display.terminal import TerminalDisplay
from src.stages.base import BaseStage


class ClosingStage(BaseStage):
    """Stage 4: Closing statements (总结陈词阶段)."""

    stage_name = "closing"

    def execute(
        self,
        agents: dict[str, BaseAgent],
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        judge: JudgeAgent,
        display: TerminalDisplay,
    ) -> None:
        display.show_stage_banner("第四阶段：总结陈词")

        # Con fourth debater goes first: 3 minutes / 750 chars
        self._do_speech(
            agents["con_debater_4"], pool, timer, scorer, judge, display,
            instruction=(
                "请进行总结陈词。回顾全场辩论，梳理关键交锋点，指出对方未能回应的核心问题，"
                "强化己方最有力的论点，做出有感染力的最终收束。字数控制在750字以内。"
            ),
            time_limit_seconds=180,
        )

        # Pro fourth debater: 3 minutes / 750 chars
        self._do_speech(
            agents["pro_debater_4"], pool, timer, scorer, judge, display,
            instruction=(
                "请进行总结陈词。你是全场最后一个发言人，这是正方的最后机会。"
                "回顾全场辩论，梳理关键交锋点，指出对方未能回应的核心问题，"
                "强化己方最有力的论点，做出有感染力的最终收束。字数控制在750字以内。"
            ),
            time_limit_seconds=180,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stages.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/stages/closing.py tests/test_stages.py
git commit -m "feat: add ClosingStage with con-first-then-pro order"
```

---

## Task 15: StageController

**Files:**
- Create: `src/engine/stage_controller.py`

- [ ] **Step 1: Implement StageController**

`src/engine/stage_controller.py`:
```python
import json
import signal
import logging

from src.agents.base import BaseAgent
from src.agents.judge import JudgeAgent
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.display.terminal import TerminalDisplay
from src.llm.base import BaseLLM
from src.stages.opening import OpeningStage
from src.stages.cross_exam import CrossExamStage
from src.stages.free_debate import FreeDebateStage
from src.stages.closing import ClosingStage

logger = logging.getLogger(__name__)

_STAGE_NAMES = {
    "opening": "陈词",
    "cross_exam": "攻辩",
    "free_debate": "自由辩论",
    "closing": "总结陈词",
}

_AGENT_NAMES = {
    "pro_debater_1": "正方一辩",
    "pro_debater_2": "正方二辩",
    "pro_debater_3": "正方三辩",
    "pro_debater_4": "正方四辩",
    "con_debater_1": "反方一辩",
    "con_debater_2": "反方二辩",
    "con_debater_3": "反方三辩",
    "con_debater_4": "反方四辩",
}


class StageController:
    """Orchestrates the full debate flow across all stages."""

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        judge: JudgeAgent,
        pool: MessagePool,
        timer: Timer,
        scorer: Scorer,
        display: TerminalDisplay,
        captain_llm: BaseLLM | None = None,
        topic: str = "",
        pro_stance: str = "",
        con_stance: str = "",
        penalties: dict[str, float] | None = None,
    ):
        self._agents = agents
        self._judge = judge
        self._pool = pool
        self._timer = timer
        self._scorer = scorer
        self._display = display
        self._captain_llm = captain_llm
        self._topic = topic
        self._pro_stance = pro_stance
        self._con_stance = con_stance
        self._penalties = penalties
        self._interrupted = False

    def run(self) -> dict:
        """Run the full debate. Returns export data."""
        # SIGINT handler
        original_handler = signal.getsignal(signal.SIGINT)

        def _handle_interrupt(signum, frame):
            self._interrupted = True
            logger.info("Debate interrupted by user")

        signal.signal(signal.SIGINT, _handle_interrupt)

        try:
            return self._run_debate()
        finally:
            signal.signal(signal.SIGINT, original_handler)

    def _run_debate(self) -> dict:
        # Opening ceremony
        self._display.show_header(self._topic, self._pro_stance, self._con_stance)

        p = self._penalties
        stages = [
            OpeningStage(penalties=p),
            CrossExamStage(penalties=p),
            FreeDebateStage(captain_llm=self._captain_llm, penalties=p),
            ClosingStage(penalties=p),
        ]

        completed_stages: list[str] = []

        for stage in stages:
            if self._interrupted:
                break
            try:
                stage.execute(
                    self._agents, self._pool, self._timer,
                    self._scorer, self._judge, self._display,
                )
                completed_stages.append(stage.stage_name)
            except Exception as e:
                logger.error("Stage %s failed: %s", stage.stage_name, e)
                break

        # Judge final review
        if not self._interrupted:
            review = self._judge.give_review(
                self._pool,
                "请对全场辩论进行总点评，总结双方表现、亮点和不足。",
            )
            judge_comment = review.get("summary", "") if review else ""
        else:
            judge_comment = "（辩论被中断）"

        # Final results
        pro_total = self._scorer.get_team_total("pro")
        con_total = self._scorer.get_team_total("con")
        best_id, best_score = self._scorer.get_best_debater()
        best_name = _AGENT_NAMES.get(best_id, best_id)

        stage_scores = {}
        for sn in completed_stages:
            stage_scores[sn] = self._scorer.get_stage_summary(sn)

        self._display.show_final_result(
            pro_total=pro_total,
            con_total=con_total,
            best_debater_name=best_name,
            best_debater_score=best_score,
            stage_scores=stage_scores,
            judge_comment=judge_comment,
        )

        # Export data
        return {
            "topic": self._topic,
            "pro_stance": self._pro_stance,
            "con_stance": self._con_stance,
            "messages": self._pool.export(),
            "scores": self._scorer.export(),
            "final": {
                "pro_total": pro_total,
                "con_total": con_total,
                "best_debater": best_name,
                "best_debater_score": best_score,
                "judge_comment": judge_comment,
            },
        }
```

- [ ] **Step 2: Write tests for StageController**

`tests/test_stage_controller.py`:
```python
import json
from io import StringIO
from rich.console import Console
from src.engine.stage_controller import StageController
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent
from src.display.terminal import TerminalDisplay
from tests.conftest import FakeLLM


def _setup():
    score_json = json.dumps({
        "speaker": "test", "logic": 7, "persuasion": 7,
        "expression": 7, "teamwork": 7, "rule_compliance": 7,
        "violations": [], "comment": "ok",
    })
    review_json = json.dumps({
        "type": "review", "summary": "点评",
        "highlights": ["亮点"], "suggestions": ["建议"],
    })
    captain_json = json.dumps({"speaker": "pro_debater_2", "direction": "反驳"})

    speech_llm = FakeLLM(responses=[
        "[选择: 反方二辩]\n1.问题？\n2.问题？\n3.问题？",
        "发言内容" * 20,
    ])
    judge_llm = FakeLLM(responses=[score_json, review_json])
    captain_llm = FakeLLM(responses=[captain_json])

    agents = {}
    for team, stance in [("pro", "利大于弊"), ("con", "弊大于利")]:
        for pos in range(1, 5):
            a = DebaterAgent.create(team=team, position=pos, stance=stance,
                                    topic="AI", personality_prompt="逻辑型", llm=speech_llm)
            agents[a.agent_id] = a
    judge = JudgeAgent.create(topic="AI", pro_stance="利大于弊", con_stance="弊大于利", llm=judge_llm)
    agents["judge"] = judge

    buf = StringIO()
    display = TerminalDisplay(console=Console(file=buf, force_terminal=True, width=80))
    return agents, judge, captain_llm, display, buf


def test_controller_run_returns_export_data():
    agents, judge, captain_llm, display, buf = _setup()
    ctrl = StageController(
        agents=agents, judge=judge, pool=MessagePool(), timer=Timer(),
        scorer=Scorer(), display=display, captain_llm=captain_llm,
        topic="AI", pro_stance="利大于弊", con_stance="弊大于利",
    )
    result = ctrl.run()
    assert "topic" in result
    assert "messages" in result
    assert "scores" in result
    assert "final" in result


def test_controller_interrupted_shows_partial():
    agents, judge, captain_llm, display, buf = _setup()
    ctrl = StageController(
        agents=agents, judge=judge, pool=MessagePool(), timer=Timer(),
        scorer=Scorer(), display=display, captain_llm=captain_llm,
        topic="AI", pro_stance="利大于弊", con_stance="弊大于利",
    )
    ctrl._interrupted = True  # Simulate immediate interrupt
    result = ctrl.run()
    assert "final" in result
    assert result["final"]["judge_comment"] == "（辩论被中断）"
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_stage_controller.py -v
```
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add src/engine/stage_controller.py tests/test_stage_controller.py
git commit -m "feat: add StageController with full debate orchestration and SIGINT handling"
```

---

## Task 16: CLI Entry Point

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Implement main.py**

`src/main.py`:
```python
"""AI Debate System — CLI entry point."""

import argparse
import json
import sys
import logging

from dotenv import load_dotenv

from src.config import load_config, load_topics, load_personalities
from src.llm import create_llm
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent
from src.engine.message_pool import MessagePool
from src.engine.timer import Timer
from src.engine.scorer import Scorer
from src.engine.stage_controller import StageController
from src.display.terminal import TerminalDisplay


def _select_topic(topics: list[dict]) -> dict:
    """Interactive topic selection."""
    print("\n请选择辩题：")
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t['title']}")
    print()
    while True:
        try:
            choice = int(input("请输入编号: "))
            if 1 <= choice <= len(topics):
                return topics[choice - 1]
        except (ValueError, EOFError):
            pass
        print("无效输入，请重试。")


def _parse_personalities(raw: str, personalities: dict) -> list[str]:
    """Parse comma-separated personality keys into prompt strings."""
    keys = [k.strip() for k in raw.split(",")]
    result = []
    for k in keys:
        if k in personalities:
            result.append(personalities[k]["prompt"])
        else:
            result.append(f"你的辩论风格是{k}。")
    # Pad to 4 if needed
    while len(result) < 4:
        result.append("你的辩论风格自由发挥。")
    return result[:4]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI 辩论赛系统")
    parser.add_argument("--topic", type=str, help="自定义辩题")
    parser.add_argument("--pro-stance", type=str, help="正方立场")
    parser.add_argument("--con-stance", type=str, help="反方立场")
    parser.add_argument("--pro-personality", type=str, default="logical,data_driven,aggressive,diplomatic")
    parser.add_argument("--con-personality", type=str, default="emotional,logical,diplomatic,data_driven")
    parser.add_argument("--output", type=str, help="导出辩论记录到 JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    # Load config
    cfg = load_config()
    personalities = load_personalities()

    # Topic
    if args.topic:
        if not args.pro_stance or not args.con_stance:
            print("自定义辩题需要同时指定 --pro-stance 和 --con-stance")
            sys.exit(1)
        topic_data = {"title": args.topic, "pro_stance": args.pro_stance, "con_stance": args.con_stance}
    else:
        topics = load_topics()
        topic_data = _select_topic(topics)

    topic = topic_data["title"]
    pro_stance = topic_data["pro_stance"]
    con_stance = topic_data["con_stance"]

    # LLM
    llm = create_llm(cfg["llm"])
    captain_llm = create_llm(cfg["llm"])  # Separate instance for captain

    # Personalities
    pro_personalities = _parse_personalities(args.pro_personality, personalities)
    con_personalities = _parse_personalities(args.con_personality, personalities)

    # Create agents
    agents: dict = {}
    for pos in range(1, 5):
        pro_agent = DebaterAgent.create(
            team="pro", position=pos, stance=pro_stance,
            topic=topic, personality_prompt=pro_personalities[pos - 1], llm=llm,
        )
        con_agent = DebaterAgent.create(
            team="con", position=pos, stance=con_stance,
            topic=topic, personality_prompt=con_personalities[pos - 1], llm=llm,
        )
        agents[pro_agent.agent_id] = pro_agent
        agents[con_agent.agent_id] = con_agent

    judge = JudgeAgent.create(
        topic=topic, pro_stance=pro_stance, con_stance=con_stance, llm=llm,
    )
    agents["judge"] = judge

    # Engine
    pool = MessagePool()
    timer = Timer(
        chars_per_minute=cfg["timer"]["chars_per_minute"],
        warning_threshold=cfg["timer"]["warning_threshold"],
    )
    scorer = Scorer(weights=cfg["scoring"]["weights"])
    display = TerminalDisplay()

    # Run debate
    controller = StageController(
        agents=agents,
        judge=judge,
        pool=pool,
        timer=timer,
        scorer=scorer,
        display=display,
        captain_llm=captain_llm,
        topic=topic,
        pro_stance=pro_stance,
        con_stance=con_stance,
        penalties=cfg["scoring"].get("penalties"),
    )

    result = controller.run()

    # Export if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n辩论记录已导出到: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write tests for CLI helpers**

`tests/test_main.py`:
```python
from src.main import _parse_personalities


def test_parse_personalities_with_known_keys():
    personalities = {
        "logical": {"name": "逻辑型", "prompt": "逻辑严密"},
        "emotional": {"name": "情感型", "prompt": "情感感染"},
    }
    result = _parse_personalities("logical,emotional", personalities)
    assert result[0] == "逻辑严密"
    assert result[1] == "情感感染"
    assert len(result) == 4  # padded to 4


def test_parse_personalities_unknown_key_fallback():
    result = _parse_personalities("unknown_style", {})
    assert "unknown_style" in result[0]
    assert len(result) == 4


def test_parse_personalities_excess_truncated():
    personalities = {f"p{i}": {"name": f"n{i}", "prompt": f"prompt{i}"} for i in range(6)}
    result = _parse_personalities("p0,p1,p2,p3,p4,p5", personalities)
    assert len(result) == 4  # truncated to 4
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_main.py -v
```
Expected: 3 passed

- [ ] **Step 4: Verify CLI help works**

```bash
cd /Users/lpb/workspace/myProjects/AI_debate
source .venv/bin/activate
python -m src.main --help
```
Expected: Shows argument help text

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: add CLI entry point with topic selection, personality config, and export"
```

---

## Task 17: Integration Test

**Files:**
- Extend: `tests/test_stages.py`

- [ ] **Step 1: Write full-flow integration test**

Append to `tests/test_stages.py`:
```python
from src.engine.stage_controller import StageController


def test_full_debate_flow():
    """Integration test: run all 4 stages end-to-end with fake LLM."""
    score_json = json.dumps({
        "speaker": "test", "logic": 7, "persuasion": 7,
        "expression": 7, "teamwork": 7, "rule_compliance": 7,
        "violations": [], "comment": "ok",
    })
    review_json = json.dumps({
        "type": "review", "summary": "双方表现精彩",
        "highlights": ["正方立论清晰"], "suggestions": ["反方需加强"],
    })
    captain_json = json.dumps({"speaker": "pro_debater_2", "direction": "反驳"})

    speech_llm = FakeLLM(responses=[
        "[选择: 反方二辩]\n1.问题一？\n2.问题二？\n3.问题三？",
        "辩论发言内容" * 20,
    ])
    judge_llm = FakeLLM(responses=[score_json, review_json])
    captain_llm = FakeLLM(responses=[captain_json])

    agents, judge = _make_agents(speech_llm, judge_llm)
    pool = MessagePool()
    timer = Timer()
    scorer = Scorer()
    buf = StringIO()
    display = TerminalDisplay(console=Console(file=buf, force_terminal=True, width=80))

    controller = StageController(
        agents=agents, judge=judge, pool=pool, timer=timer,
        scorer=scorer, display=display, captain_llm=captain_llm,
        topic="AI利弊", pro_stance="利大于弊", con_stance="弊大于利",
    )
    result = controller.run()

    # Verify export structure
    assert "topic" in result
    assert "messages" in result
    assert "scores" in result
    assert "final" in result
    assert result["final"]["pro_total"] != 0 or result["final"]["con_total"] != 0
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass

- [ ] **Step 3: Run coverage check**

```bash
pytest tests/ --cov=src --cov-report=term-missing
```
Expected: 80%+ coverage

- [ ] **Step 4: Commit**

```bash
git add tests/test_stages.py
git commit -m "test: add full debate flow integration test"
```

---

## Task 18: Final Verification

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 2: Verify CLI runs with --help**

```bash
python -m src.main --help
```

- [ ] **Step 3: Verify project structure matches spec**

```bash
find src/ -name "*.py" | sort
find tests/ -name "*.py" | sort
find config/ -type f | sort
```

- [ ] **Step 4: Final commit with any cleanup**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
