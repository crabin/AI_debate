# Multi-Agent AI 辩论赛系统设计文档

## 概述

多智能体 AI 辩论赛系统，严格遵循中文辩论赛赛制。系统包含 9 个独立 Agent（裁判 + 正反方各 4 位辩手），通过双层消息池实现队内共享与公开发言隔离，CLI 彩色终端实时展示辩论过程与评分。使用自定义 LLM 抽象层（默认智谱 AI），不依赖 LangChain 框架。

## 架构：混合模式（轻量编排器 + 消息池）

- **StageController**：轻量编排器，只管阶段流程推进和发言顺序
- **MessagePool**：双层消息池，处理消息路由与权限隔离
- **独立 Agent**：各自持有 prompt + 从可见频道获取上下文 + 生成发言

## 项目结构

```
AI_debate/
├── .env                          # API keys
├── config/
│   ├── default.yaml              # 默认配置（模型、温度、超时等）
│   ├── topics.yaml               # 内置辩题库
│   └── personalities.yaml        # 辩手性格标签库
├── src/
│   ├── __init__.py
│   ├── main.py                   # CLI 入口
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py               # LLM 抽象接口
│   │   └── zhipu.py              # 智谱 AI 实现
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseAgent 基类
│   │   ├── debater.py            # Debater agent（按辩位配置）
│   │   └── judge.py              # Judge agent
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── stage_controller.py   # 阶段流程控制器
│   │   ├── message_pool.py       # 双层消息池
│   │   ├── timer.py              # 计时器 + 超时判罚
│   │   └── scorer.py             # 评分系统
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── opening.py            # 陈词阶段
│   │   ├── cross_exam.py         # 攻辩阶段
│   │   ├── free_debate.py        # 自由辩论阶段
│   │   └── closing.py            # 总结陈词阶段
│   └── display/
│       ├── __init__.py
│       └── terminal.py           # 彩色终端输出 + 进度条
├── tests/
│   └── ...
└── pyproject.toml
```

**设计原则**：
- `agents/` 只管思考和生成内容，不知道流程细节
- `stages/` 只管当前环节的发言顺序和规则
- `engine/` 是粘合层——消息路由、计时、评分
- `display/` 纯展示层，与逻辑完全解耦

## 消息池（MessagePool）

### 频道

| 频道 | 可读 | 可写 |
|------|------|------|
| `public` | 所有 agent | 所有 agent（通过 controller） |
| `team_pro` | 正方 4 辩手 | 正方 4 辩手 |
| `team_con` | 反方 4 辩手 | 反方 4 辩手 |
| `judge_notes` | 裁判 | 裁判 |

### 消息结构

```python
@dataclass(frozen=True)
class Message:
    speaker: str          # "pro_debater_1" / "con_debater_2" / "judge"
    role: str             # "一辩" / "二辩" / "裁判"
    team: str             # "pro" / "con" / "judge"
    stage: str            # "opening" / "cross_exam" / "free_debate" / "closing"
    content: str          # 发言内容
    msg_type: str         # "speech" / "team_strategy" / "score"
    timestamp: float
    word_count: int
    metadata: tuple       # 扩展字段，使用 tuple of (key, value) pairs 保证不可变
```

### Agent 可见性

| Agent 身份 | 可见消息 |
|-----------|---------|
| 正方辩手 | `public` + `team_pro` |
| 反方辩手 | `public` + `team_con` |
| 裁判 | `public` + `judge_notes` |

### 消息发布流程

辩手发言 → StageController 接收 → 写入 `public` + 所属 `team` 频道 → 触发 Judge 实时评分 → 评分写入 `public` + `judge_notes`

## Agent 设计

### BaseAgent 基类

```python
class BaseAgent:
    name: str              # "正方一辩"
    agent_id: str          # "pro_debater_1"
    team: str              # "pro" / "con" / "judge"
    system_prompt: str     # 核心 prompt
    llm: BaseLLM           # 可配置的 LLM 实例

    def build_context(self, message_pool, stage) -> list[dict]:
        """从消息池中按可见权限构建对话上下文"""

    def speak(self, message_pool, stage, instruction) -> str:
        """接收阶段指令，生成发言"""
```

### DebaterAgent

所有辩手共用 `DebaterAgent` 类，通过配置区分角色。Prompt 结构：

```
[身份层]  你是{team}{position}，你的队伍立场是{stance}
[性格层]  你的辩论风格是{personality}（可配置）
[职责层]  按辩位注入不同职责
[规则层]  辩论赛规则约束
[上下文]  当前阶段 + 可见的历史发言
```

**辩位职责**：

| 辩位 | 核心职责 |
|-----|---------|
| 一辩 | 立论陈词（750字/3分钟）+ 攻辩小结（500字/2分钟） |
| 二辩 | 攻辩提问（125字/30秒）+ 攻辩回答（250字/1分钟）+ 自由辩论 |
| 三辩 | 攻辩提问（125字/30秒）+ 攻辩回答（250字/1分钟）+ 自由辩论 |
| 四辩 | 自由辩论 + 总结陈词（750字/3分钟） |

### JudgeAgent

独立设计，不继承 Debater 的职责层。

**评分维度**（1-10 分）：
- 逻辑性 (logic) — 权重 0.25
- 说服力 (persuasion) — 权重 0.25
- 语言表达 (expression) — 权重 0.20
- 团队配合 (teamwork) — 权重 0.15
- 规则遵守 (rule_compliance) — 权重 0.15

**评分输出格式**：

```python
@dataclass(frozen=True)
class ScoreCard:
    speaker: str
    stage: str
    logic: int             # 1-10
    persuasion: int        # 1-10
    expression: int        # 1-10
    teamwork: int          # 1-10
    rule_compliance: int   # 1-10
    violations: tuple[str, ...]  # 违规标记列表
    comment: str
    # total 和 penalty 由 Scorer 根据 weights 和 violations 计算，不由 Judge 输出
```

**违规检测标记**：
- `counter_question`: 辩方在攻辩中反问
- `not_direct_answer`: 辩方未正面回答
- `attacker_answered`: 攻方回答了问题
- `off_topic`: 严重偏离辩题
- `scripted_summary`: 攻辩小结背稿
- `personal_attack`: 人身攻击

## 阶段控制器

### StageController

```python
class StageController:
    stages = [OpeningStage, CrossExamStage, FreeDebateStage, ClosingStage]

    def run(self):
        announce_debate()
        for stage in self.stages:
            stage.execute(agents, message_pool, timer, judge, display)
        judge.final_review()
        announce_result()
```

### Stage 1：陈词阶段

固定顺序：正方一辩（3分钟/750字）→ 反方一辩（3分钟/750字）

### Stage 2：攻辩阶段

4 轮攻辩：
1. 正方二辩提问 → 选择反方二辩或三辩回答
2. 反方二辩提问 → 选择正方二辩或三辩回答
3. 正方三辩提问 → 选择反方二辩或三辩回答
4. 反方三辩提问 → 选择正方二辩或三辩回答

每轮：提问 30 秒（125字，含3个以上简短问题）+ 回答 1 分钟（250字）

> 注意：30秒内需提出3个以上问题，因此问题应简短精炼（每个问题约30-40字）。
> prompt 中会指导辩手用编号列出问题，确保问题数量达标且简明。

攻辩结束后：正方一辩小结（2分钟/500字）→ 反方一辩小结（2分钟/500字）

**攻方选择对手**：攻方 agent 输出 `[选择: 反方X辩]`，由 controller 解析并路由。

**背稿检测**：Judge 检查小结内容是否引用攻辩阶段具体发言，关联度低则扣分。

### Stage 3：自由辩论阶段

- 正方先发言，之后正反方轮流
- 每方总计 4 分钟（按字数估算）
- 每位辩手至少发言一次
- 同方不得连续发言

**队内协商机制**：

轮到某方发言时，StageController 用单次 LLM 调用模拟"队长"决策：

```
队长 Prompt:
  【任务】决定本轮由谁发言，并给出回应方向建议。
  【队内发言记录】{team_messages}
  【公开辩论记录】{recent_public_messages}
  【各辩手已发言次数】{speak_counts}
  【未发言辩手】{unspeaking_debaters}
  【剩余时间】{time_left}秒

  输出格式（严格 JSON）：
  {"speaker": "pro_debater_2", "direction": "针对对方关于就业的论点反驳"}
```

流程：队长 LLM 调用 → 解析 JSON 获取 speaker + direction → 将 direction 注入选中辩手的 `current_instruction` → 选中辩手发起 speak() 调用 → 生成发言。

优先选择未发言者，确保每位辩手至少发言一次。

### Stage 4：总结陈词阶段

固定顺序：反方四辩（3分钟/750字）→ 正方四辩（3分钟/750字）

每次总结陈词后 Judge 实时评分（与其他阶段一致）。全部阶段结束后 Judge 额外进行最终全场点评。

## 计时器

```python
class Timer:
    def estimate_duration(self, text: str) -> float:
        """按字数估算（中文约 250字/分钟）"""

    def check_overtime(self, text: str, limit_seconds: float) -> tuple[bool, float]:
        """返回 (是否超时, 超出秒数)"""

    def apply_penalty(self, scorer, team, speaker, overtime_seconds):
        """超时扣分：队伍-3分，个人-2分"""
```

## 评分系统

### 三层评分

1. **实时评分**（每次发言后）— Judge agent 打分
2. **规则引擎自动扣分**（硬编码）：
   - 超时：队伍 -3 分，个人 -2 分
   - 攻辩中反问：个人 -2 分
   - 攻辩中攻方回答问题：个人 -2 分
   - 自由辩论中同方连续发言：队伍 -3 分
3. **阶段汇总 + 最终汇总**

### 分数计算公式

```
单次发言得分 = logic * 0.25 + persuasion * 0.25 + expression * 0.20
             + teamwork * 0.15 + rule_compliance * 0.15

个人总分 = sum(该辩手所有发言得分) + 个人扣分（负值）
队伍总分 = sum(队伍4人个人总分) + 队伍扣分（负值）

最终得分不设下限（可为负数，体现严重违规的惩罚）
最佳辩手 = 个人总分最高者
```

## CLI 展示

使用 `rich` 库实现。

**配色方案**：

| 角色 | 颜色 |
|------|------|
| 正方 | 蓝色 (Blue) |
| 反方 | 红色 (Red) |
| 裁判 | 黄色 (Yellow) |
| 系统提示 | 灰色 (Dim) |
| 分数 | 绿色 (Green) |
| 警告/扣分 | 品红 (Magenta) |

**展示内容**：发言带颜色标识、用时与字数、实时评分、比分面板、超时/违规警告、最终计分板。

## LLM 抽象层

```python
class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], temperature: float) -> str:
        """统一聊天接口"""

class ZhipuLLM(BaseLLM):
    """智谱 AI 实现，默认 glm-4.7"""

LLM_FACTORY = {"zhipu": ZhipuLLM, ...}
```

## 配置系统

### default.yaml

```yaml
llm:
  provider: "zhipu"
  model: "glm-4.7"
  temperature: 0.7

timer:
  chars_per_minute: 250
  warning_threshold: 30      # 剩余字数低于此值时，终端显示橙色警告提示

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
```

### topics.yaml

内置经典辩题 + 支持 CLI 自定义输入。

### personalities.yaml

可配置性格标签：逻辑严密型、情感感染型、数据实证型、犀利进攻型、稳健防守型。

## CLI 入口

```bash
# 选择内置辩题
python -m src.main

# 自定义辩题
python -m src.main --topic "科技发展应优先考虑效率/公平" \
                   --pro-stance "效率" --con-stance "公平"

# 指定辩手性格
python -m src.main --pro-personality logical,data_driven,aggressive,diplomatic \
                   --con-personality emotional,logical,diplomatic,data_driven
```

## 核心 Prompt

### 通用规则 Prompt（所有辩手共享）

```
【辩论赛规则 — 你必须严格遵守以下规则，违反将被扣分】

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
   - 尊重对手，不得进行人身攻击
```

### 一辩 Prompt

```
【身份】你是{team}一辩，你的队伍立场是：{stance}
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

{common_rules}
【队内信息】{team_messages}
【辩论记录】{public_messages}
【当前任务】{current_instruction}
请直接输出你的发言内容，不要输出任何元信息。
```

### 二辩 Prompt

```
【身份】你是{team}二辩，你的队伍立场是：{stance}
【辩题】{topic}
【性格】{personality_prompt}

【职责】
1.【作为攻方提问】
  - 选择向对方二辩或三辩提问（只能选一人）
  - 先输出：[选择: 对方X辩]
  - 连续提出3个以上有逻辑关联的问题
  - 只能提问，不得回答
  - 字数控制在125字以内（约30秒）

2.【作为辩方回答】
  - 必须正面回答，不得反问
  - 回答简洁有力，融入己方论点
  - 字数控制在250字以内（约1分钟）

3.【自由辩论】
  - 至少发言一次
  - 抓住对方漏洞反驳

{common_rules}
【队内信息】{team_messages}
【辩论记录】{public_messages}
【当前任务】{current_instruction}
请直接输出你的发言内容，不要输出任何元信息。
```

### 三辩 Prompt

```
【身份】你是{team}三辩，你的队伍立场是：{stance}
【辩题】{topic}
【性格】{personality_prompt}

【职责】
1.【作为攻方提问】
  - 选择向对方二辩或三辩提问（只能选一人）
  - 先输出：[选择: 对方X辩]
  - 连续提出3个以上深挖逻辑矛盾的问题
  - 只能提问，不得回答
  - 字数控制在125字以内（约30秒）

2.【作为辩方回答】
  - 必须正面回答，不得反问
  - 从更高维度化解对方攻势
  - 字数控制在250字以内（约1分钟）

3.【自由辩论】
  - 至少发言一次
  - 承接队友论点深化，与二辩互补

{common_rules}
【队内信息】{team_messages}
【辩论记录】{public_messages}
【当前任务】{current_instruction}
请直接输出你的发言内容，不要输出任何元信息。
```

### 四辩 Prompt

```
【身份】你是{team}四辩，你的队伍立场是：{stance}
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

{common_rules}
【队内信息】{team_messages}
【辩论记录】{public_messages}
【当前任务】{current_instruction}
请直接输出你的发言内容，不要输出任何元信息。
```

### 裁判 Prompt

```
【身份】你是本场辩论赛的评委，必须保持公正、客观、专业。
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

【评分输出格式】（严格 JSON）：
{
  "speaker": "发言人ID",
  "logic": 8, "persuasion": 7, "expression": 8,
  "teamwork": 7, "rule_compliance": 10,
  "violations": [],
  "comment": "简短点评"
}

【点评输出格式】：
{
  "type": "review",
  "summary": "点评内容（200字以内）",
  "highlights": ["亮点1", "亮点2"],
  "suggestions": ["建议1", "建议2"]
}

【辩论记录】{public_messages}
【评分记录】{judge_notes}
【当前任务】{current_instruction}
```

## 错误处理

| 场景 | 处理 |
|------|------|
| LLM 调用失败 | 重试1次（延迟2秒）→ 仍失败则该辩手沉默，Judge 扣 teamwork 分 |
| JSON 评分解析失败 | 要求 Judge 重新输出 → 仍失败则使用阶段平均分 |
| 攻辩未选择对手 | 默认选择二辩，Judge 扣 rule_compliance 分 |
| 用户中断 (Ctrl+C) | 捕获 SIGINT，显示已完成阶段的分数汇总，优雅退出 |

## 技术栈

- Python 3.11+
- 智谱 AI SDK (`zai`) — 默认 LLM
- `rich` — 彩色终端输出
- `pyyaml` — 配置加载
- `python-dotenv` — 环境变量
- `dataclasses` — 不可变数据结构

## LLM 调用配置

```yaml
llm:
  provider: "zhipu"
  model: "glm-4.7"
  temperature: 0.7
  timeout_seconds: 60
  max_retries: 1
  retry_delay: 2.0
```

## 辩论记录导出

支持 `--output` 参数导出完整辩论记录：

```bash
python -m src.main --output debate_log.json
```

导出内容包括：辩题、双方立场、全部发言记录、每次评分、违规记录、最终计分板。
