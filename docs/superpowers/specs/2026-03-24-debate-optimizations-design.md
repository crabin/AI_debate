# AI Debate System — Optimization Features Design

Date: 2026-03-24
Rev: 2 (post spec-review fixes)

## Overview

Four optimization features added to the existing Multi-Agent AI Debate System. The existing architecture (StageController, MessagePool, BaseLLM, 4-stage pipeline) is preserved; all changes are additive or targeted replacements.

---

## Feature 1: Final Verdict + Topic Conclusion

### Problem

`_calculate_final_results()` returns a winner and scores, but there is no structured "verdict" — no reasoning about *why* the winner won, and no answer to the debate topic itself.

### Design

Add `JudgeAgent.generate_verdict(pool, winner, pro_score, con_score, best_debater)` — a new LLM call at the very end. It receives explicit primitives (not a pre-assembled result dict) so the call site is unambiguous.

**Call site in `StageController._calculate_final_results()`**:

```python
# Assemble primitives first
pro_total = scorer.get_team_total("pro")
con_total = scorer.get_team_total("con")
winner = "pro" if pro_total > con_total else ("con" if con_total > pro_total else "tie")
margin = abs(pro_total - con_total)
best_debater = scorer.get_best_debater()

# Call generate_verdict with assembled primitives
verdict_data = {}
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
        logger.warning(f"generate_verdict failed: {e}")

return {
    "pro_score": pro_total,
    "con_score": con_total,
    "winner": winner,
    "margin": margin,
    "best_debater": best_debater,
    **verdict_data,   # spreads verdict, topic_conclusion, key_moments, winner_reason
}
```

**`generate_verdict()` output (strict JSON)**:
```json
{
  "winner_reason": "正方在攻辩阶段成功揭示了反方论证的核心漏洞，逻辑体系更为严密",
  "topic_conclusion": "综合本场辩论，辩题「…」的答案倾向于：效率，因为…",
  "best_debater_reason": "全场逻辑最严密，攻辩表现突出",
  "key_moments": ["正方二辩在攻辩中提出的三连问令反方无从回避"]
}
```

Note: `winner_team` and `best_debater` are *not* in the LLM output — they are already known and passed in as inputs. This avoids the LLM contradicting the score.

**Fallback (JSON parse failure)**: Return a complete fallback dict with empty/default values so callers never need to null-check individual keys:
```python
VERDICT_FALLBACK = {
    "winner_reason": "",
    "topic_conclusion": "",
    "best_debater_reason": "",
    "key_moments": [],
}
```

**Display**: Rich `Panel` with colored sections — winner banner, topic conclusion box, key moments list, best debater highlight.

---

## Feature 2: JSON Export

### Problem

`run_debate()` returns a Python dict but never persists anything. The design spec mentioned `--output` but it was never implemented.

### Design

**CLI**: Add `--output <path>` argument to `main()` / `run_debate()`. If omitted, no file is written.

**Module**: New `src/export.py` — single public function:
```python
def save_debate_json(results: dict, pool: MessagePool, scorer: Scorer, path: Path, start_time: float) -> None
```

`scorer` is passed in so `save_debate_json()` can call `scorer.compute_speech_score(card)` for each card to populate the computed `total` field. The `Scorer` instance used in `_calculate_final_results()` must be returned or passed out to the caller (currently it's a local variable — it must become an instance attribute or be returned as part of results so `run_debate()` can pass it to `save_debate_json()`).

**Concretely**: Change `StageController._calculate_final_results()` to attach the `scorer` to the returned dict as `"_scorer": scorer` (private key, excluded from JSON export). `run_debate()` in `cli.py` extracts it safely before calling `save_debate_json()`:
```python
scorer = results.pop("_scorer", None)
if output_path and scorer is not None:
    save_debate_json(results, pool, scorer, output_path, start_time)
elif output_path and scorer is None:
    logger.warning("Scorer not available; JSON export skipped")
```
This ensures no `KeyError` if the scorer is absent (e.g., in tests that mock `run_debate()`).

**`start_time`**: `run_debate()` in `cli.py` records `start_time = time.time()` as the first line before any work begins. This is passed to `save_debate_json()` for computing `meta.duration_seconds`.

**JSON schema**:
```json
{
  "meta": {
    "topic": "string",
    "pro_stance": "string",
    "con_stance": "string",
    "timestamp": "ISO-8601",
    "duration_seconds": 0.0
  },
  "transcript": [
    {
      "stage": "opening|cross_exam|free_debate|closing",
      "speaker": "pro_debater_1",
      "role": "一辩",
      "team": "pro|con|judge",
      "content": "string",
      "word_count": 0,
      "timestamp": 0.0,
      "msg_type": "string"
    }
  ],
  "scores": [
    {
      "speaker": "string",
      "stage": "string",
      "logic": 0, "persuasion": 0, "expression": 0,
      "teamwork": 0, "rule_compliance": 0,
      "total": 0.0,
      "violations": [],
      "comment": "string"
    }
  ],
  "result": {
    "winner": "pro|con|tie",
    "pro_score": 0.0,
    "con_score": 0.0,
    "margin": 0.0,
    "best_debater": "string",
    "winner_reason": "string",
    "topic_conclusion": "string",
    "best_debater_reason": "string",
    "key_moments": []
  }
}
```

Notes on schema:
- `transcript.msg_type` is an open string (actual values include `"speech"`, `"free_speech"`, `"score"`, `"team_strategy"` depending on stage — not an enum).
- `transcript` sources: all messages from `pool.get_messages("public")`. Judge score messages from `pool.get_messages("judge_notes")` are serialized into the `scores` array separately.
- `scores` rows: iterate `scorer.get_all_cards()` (new method on `Scorer`) and call `scorer.compute_speech_score(card)` per card for the `total` field.
- The `"_scorer"` private key is stripped before JSON serialization.

---

## Feature 3: Multi-Model Support (Per-Role LLM Configuration)

### Problem

`create_llm()` creates one global LLM instance. All 9 agents share the same model.

### Design

#### New LLM Provider: `OpenAICompatibleLLM`

New file `src/llm/openai_compatible.py`:

```python
class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, base_url: str,
                 timeout_seconds: int, max_retries: int, retry_delay: float): ...
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str: ...
    def chat_stream(self, messages: list[dict], temperature: float = 0.7,
                    callback: Callable[[str], None] | None = None) -> str: ...
```

Uses the `openai` Python SDK. `base_url` is passed to `OpenAI(base_url=base_url, api_key=api_key)`.

Registered in `LLM_FACTORY`:
```python
LLM_FACTORY: dict[str, type[BaseLLM]] = {
    "zhipu": ZhipuLLM,
    "openai_compatible": OpenAICompatibleLLM,
}
```

#### `create_llm()` Refactor

The existing hard-coded `if provider == "zhipu": ... raise ValueError` structure is replaced with a general dispatch. The function signature gains an optional `role` parameter:

```python
def create_llm(config: dict | None = None, role: str | None = None) -> BaseLLM:
```

**Env var resolution with role prefix**:
```python
prefix = f"{role.upper()}_" if role else ""

provider = os.environ.get(f"{prefix}LLM_PROVIDER") \
        or os.environ.get("LLM_PROVIDER") \
        or llm_config.get("provider", "zhipu")

model = os.environ.get(f"{prefix}LLM_MODEL") \
     or os.environ.get("LLM_MODEL") \
     or llm_config.get("model", "glm-4.7")
```

**API key resolution** (provider-specific, role-prefixed):
```python
# For zhipu:
api_key = os.environ.get(f"{prefix}ZAI_API_KEY") or os.environ.get("ZAI_API_KEY", "")

# For openai_compatible:
api_key = os.environ.get(f"{prefix}LLM_API_KEY") or os.environ.get("LLM_API_KEY", "")
base_url = os.environ.get(f"{prefix}LLM_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
```

**Enumerated dispatch** — replaces the hard-coded `if provider == "zhipu"` branch. Each provider has its own validation block; adding a new provider requires adding one `elif` here:
```python
if provider not in LLM_FACTORY:
    raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(LLM_FACTORY.keys())}")

cls = LLM_FACTORY[provider]

if provider == "zhipu":
    if not api_key:
        raise ValueError(f"ZAI_API_KEY (or {prefix}ZAI_API_KEY) is required for zhipu provider")
    return cls(api_key=api_key, model=model, base_url=base_url,
               timeout_seconds=timeout_seconds, max_retries=max_retries, retry_delay=retry_delay)

elif provider == "openai_compatible":
    if not api_key:
        raise ValueError(f"LLM_API_KEY (or {prefix}LLM_API_KEY) is required for openai_compatible provider")
    if not base_url:
        raise ValueError(f"LLM_BASE_URL (or {prefix}LLM_BASE_URL) is required for openai_compatible provider")
    return cls(api_key=api_key, model=model, base_url=base_url,
               timeout_seconds=timeout_seconds, max_retries=max_retries, retry_delay=retry_delay)

raise ValueError(f"Provider '{provider}' is registered in LLM_FACTORY but has no dispatch branch in create_llm()")
```

#### `.env` Configuration

```env
# 正方辩手
PRO_LLM_PROVIDER=openai_compatible
PRO_LLM_MODEL=deepseek-chat
PRO_LLM_BASE_URL=https://api.deepseek.com/v1
PRO_LLM_API_KEY=sk-...

# 反方辩手
CON_LLM_PROVIDER=zhipu
CON_LLM_MODEL=glm-4.7
CON_ZAI_API_KEY=...

# 裁判
JUDGE_LLM_PROVIDER=openai_compatible
JUDGE_LLM_MODEL=claude-sonnet-4-6
JUDGE_LLM_BASE_URL=https://api.anthropic.com/v1
JUDGE_LLM_API_KEY=sk-ant-...

# 全局回退（所有 PRO_/CON_/JUDGE_ 未设置时使用）
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4.7
ZAI_API_KEY=...
```

#### `create_agents()` Update

```python
pro_llm   = create_llm(config, role="pro")
con_llm   = create_llm(config, role="con")
judge_llm = create_llm(config, role="judge")
```

Debater agents receive `pro_llm` or `con_llm` based on their team. Judge receives `judge_llm`.

#### Model Name Display in Participants Panel

`BaseAgent` gains a read-only `model_name: str` property that returns the model identifier string (populated at construction from the LLM instance). `TerminalDisplay.participants()` reads `agent.model_name` and displays it next to each agent's name in the participants table.

This requires `BaseLLM` to expose a `model_name: str` property, which both `ZhipuLLM` and `OpenAICompatibleLLM` implement by returning `self._model`.

---

## Feature 4: Concurrent Free Debate

### Problem

`FreeDebateStage.execute()` is fully sequential. Debaters cannot respond to speeches happening "at the same time."

### Design Choice: Round-Based Parallel

Each round: one pro debater and one con debater generate simultaneously in separate threads. Both stream to their own buffers. Both speeches are committed to the shared pool after the round. The next round's speakers read both before generating.

**Why not reactive/continuous**: Terminal output becomes unreadable with 8 unconstrained concurrent streams; round-based maintains coherence and fairness.

#### Thread Structure Per Round

```
Round N:
  Thread A: pro_debater_X.generate() → writes to pro_buffer (StringIO)
  Thread B: con_debater_Y.generate() → writes to con_buffer (StringIO)
  Barrier  → both threads call barrier.wait(); if one raises, it calls barrier.abort() first
  Main     → commit pro_message then con_message to pool; advance Round N+1
```

**Exception safety in threads**: Each thread wraps its work in try/except. On exception: log error, set result to empty string, call `barrier.abort()` (not `barrier.wait()`) so the sibling thread raises `BrokenBarrierError` and unblocks immediately. The main thread catches `BrokenBarrierError` and treats that round as producing no committed output — **even if the sibling thread had already completed its generation successfully before the barrier**. The rationale: a broken round is treated atomically; committing only one side's speech while the other is silent would create an asymmetric exchange that the next round's speakers cannot meaningfully respond to.

#### Terminal Display

Use Rich `Live` context with a `Table` containing two `Panel`s side-by-side. Each thread appends characters to its own `StringIO` buffer. A `threading.Lock` serializes all Rich `Live` updates. The main thread drives `Live` refresh at ~10 fps by polling both buffers.

`ZhipuLLM.chat_stream()` currently calls `sys.stdout.flush()` and `time.sleep()` directly inside its streaming loop — these must not write to stdout when called from a background thread under a Rich `Live` context. Fix: `ZhipuLLM.chat_stream()` suppresses the `sys.stdout.flush()` call and the `time.sleep(0.01)` delay when a `callback` is provided (the caller controls display timing). Concretely: move `sys.stdout.flush()` and `time.sleep()` inside an `if not callback:` guard. The same pattern applies to `OpenAICompatibleLLM.chat_stream()` from the start.

Display sketch:
```
┌─── 正方二辩 (DeepSeek) ──────┐  ┌─── 反方三辩 (GLM-4.7) ───────┐
│ 根据对方的论点，我方认为…      │  │ 正方的例子恰恰说明了…          │
│ （流式追加中）                │  │ ⏰ 时间到                      │
└──────────────────────────────┘  └──────────────────────────────┘
```

#### Speaker Selection Per Round

Two calls to `_get_next_speaker()`, one per team. Each team maintains its own `last_speaker` tracker (not a single shared `last_speaker`). The two selected speakers are always from different teams by definition.

Priority within each team: speakers who haven't spoken yet; tie-broken by fewest speeches.

If a team's time is expired: that team's thread is not launched; that panel shows "⏰ 时间到"; no message is committed for that team this round.

#### Round Termination Conditions

- Max 10 rounds (20 total speeches max, same ceiling as current `max_turns=20`)
- Both teams' timers expired → stop
- All 8 debaters have spoken ≥ once AND round ≥ 6 → stop (allows 2 extra rounds of back-and-forth after minimum coverage)

#### Message Pool Commit Order

Pro message committed first, then con — preserving the convention that pro leads and ensuring deterministic transcript ordering.

#### Opt-In Flag

`--concurrent` CLI flag (or `DEBATE_CONCURRENT_FREE=true` in `.env`). Default: sequential (existing behavior, existing tests unaffected).

`StageController` gains a `concurrent: bool = False` constructor parameter. `run_debate()` checks it and calls either `stage.execute()` or `stage.execute_concurrent()` for the `free_debate` stage only.

---

## Affected Files

| File | Change |
|------|--------|
| `src/llm/openai_compatible.py` | **New** — OpenAI-compatible LLM implementation |
| `src/llm/__init__.py` | Register new provider; refactor `create_llm()` to generic dispatch; add `role` param |
| `src/llm/base.py` | Add `model_name: str` abstract property |
| `src/llm/zhipu.py` | Implement `model_name` property; suppress `sys.stdout.flush()` / `time.sleep()` inside callback path |
| `src/export.py` | **New** — `save_debate_json()` |
| `src/agents/base.py` | Add `model_name: str` property (delegates to `self._llm.model_name`) |
| `src/agents/judge.py` | Add `generate_verdict()` method |
| `src/engine/scorer.py` | Add `get_all_cards()` method returning all recorded `ScoreCard` instances |
| `src/stages/free_debate.py` | Add `execute_concurrent()` method; keep `execute()` unchanged |
| `src/display/terminal.py` | Add `concurrent_speech_panels()` for side-by-side Live display; update `participants()` to show model name |
| `src/stages/controller.py` | Add `concurrent` param; route free debate to `execute_concurrent()`; call `generate_verdict()`; return `_scorer` in results |
| `src/cli.py` | Add `--output`, `--concurrent` args; `start_time = time.time()` at top of `run_debate()`; create per-role LLMs; call `save_debate_json()` |
| `.env.example` | Document all new env vars |

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Concurrent thread raises exception | Catch in thread; log error; call `barrier.abort()`; that panel shows error text; empty message not committed to pool |
| Sibling thread sees `BrokenBarrierError` | Catch in thread; treat as if own generation returned `""`; commit nothing |
| `generate_verdict()` JSON parse fails | Log warning; return `VERDICT_FALLBACK` dict (all keys present, empty values) |
| `save_debate_json()` write fails | Log error; print warning to terminal; do not crash the debate |
| Role-specific LLM env var missing | Fall back to global `LLM_*` / `ZAI_API_KEY` vars; if those also missing, raise `ValueError` with specific key names in message |
| One team's time expires in concurrent round | That team's thread is not launched; panel shows "⏰ 时间到"; no message committed |
| `generate_verdict()` call itself raises | Log warning; spread `VERDICT_FALLBACK` into results dict |

---

## Non-Goals

- No web UI or API server
- No support for more than 2 teams
- No real-time network multiplayer
- No per-debater (sub-team) LLM configuration (only per-role: pro/con/judge)
