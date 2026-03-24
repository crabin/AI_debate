# AI Debate — 多智能体 AI 辩论系统

一个基于多 LLM 智能体的中文辩论赛模拟系统。9 个独立 AI 智能体（8 名辩手 + 1 名裁判）按照标准辩论赛规则，在终端中完整演绎四阶段正式辩论。

## 特性

- **四阶段标准辩论** — 立论陈词、攻辩阶段、自由辩论、总结陈词
- **9 个独立智能体** — 正方 4 辩 + 反方 4 辩 + 裁判，各有专属系统提示词
- **多模型支持** — 正方、反方、裁判可分别使用不同 LLM（智谱 GLM、DeepSeek、OpenAI 兼容接口等）
- **实时流式输出** — 基于 Rich 的彩色终端 UI，辩手发言逐字呈现
- **并发自由辩论** — `--concurrent` 模式下正反双方同时生成，左右分栏实时对照
- **三层评分体系** — LLM 实时打分 + 规则引擎罚分 + 团队汇总
- **5 种辩手性格** — 逻辑严密、情感感染、数据实证、犀利进攻、稳健防守
- **JSON 导出** — 完整辩论记录、评分、判决一键保存

## 辩论流程


| 阶段 | 名称     | 参与者             | 说明                   |
| ---- | -------- | ------------------ | ---------------------- |
| 1    | 立论陈词 | 正方一辩、反方一辩 | 各 750 字以内          |
| 2    | 攻辩阶段 | 二辩、三辩         | 提问 + 回答 + 小结     |
| 3    | 自由辩论 | 全体辩手           | 正反交替，支持并发模式 |
| 4    | 总结陈词 | 反方四辩、正方四辩 | 各 750 字以内          |

裁判在每次发言后实时评分（逻辑 25%、说服力 25%、表达 20%、团队配合 15%、规则遵守 15%），最终给出判决和辩题总结。

## 安装

### 环境要求

- Python >= 3.11
- pip 或 uv

### 安装步骤

```bash
# 克隆项目
git clone <repo-url>
cd AI_debate

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

### 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

## 使用

### 基本用法

```bash
# 使用默认辩题（人工智能的发展利大于弊/弊大于利）
python -m src.cli

# 指定辩题（编号 0-4）
python -m src.cli 2

# 导出 JSON 辩论记录
python -m src.cli 0 --output debate_log.json

# 并发自由辩论模式（左右分栏同时生成）
python -m src.cli --concurrent

# 组合使用
python -m src.cli 1 --concurrent --output results.json
```

### CLI 参数


| 参数              | 说明             | 默认值 |
| ----------------- | ---------------- | ------ |
| `topic_index`     | 辩题编号（0-4）  | `0`    |
| `--output <path>` | JSON 导出路径    | 无     |
| `--concurrent`    | 启用并发自由辩论 | 关闭   |

### 内置辩题


| 编号 | 辩题                                                | 正方           | 反方           |
| ---- | --------------------------------------------------- | -------------- | -------------- |
| 0    | 人工智能的发展利大于弊 / 弊大于利                   | 利大于弊       | 弊大于利       |
| 1    | 网络使人更亲近 / 更疏远                             | 更亲近         | 更疏远         |
| 2    | 大学教育应以就业为导向 / 以学术为导向               | 以就业为导向   | 以学术为导向   |
| 3    | 科技发展应优先考虑效率 / 公平                       | 效率           | 公平           |
| 4    | 个人隐私比公共安全更重要 / 公共安全比个人隐私更重要 | 个人隐私更重要 | 公共安全更重要 |

## 配置

### 多模型配置

每个角色（正方、反方、裁判）可使用不同的 LLM 提供商和模型。在 `.env` 中配置：

```bash
# 全局默认
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4.7
ZAI_API_KEY=your-key

# 正方使用 DeepSeek
PRO_LLM_PROVIDER=openai_compatible
PRO_LLM_MODEL=deepseek-chat
PRO_LLM_BASE_URL=https://api.deepseek.com/v1
PRO_LLM_API_KEY=sk-...

# 反方使用智谱
CON_LLM_PROVIDER=zhipu
CON_LLM_MODEL=glm-4.7

# 裁判使用 Claude
JUDGE_LLM_PROVIDER=openai_compatible
JUDGE_LLM_MODEL=claude-sonnet-4-6
JUDGE_LLM_BASE_URL=https://api.anthropic.com/v1
JUDGE_LLM_API_KEY=sk-ant-...
```

优先级：角色前缀环境变量 > 全局环境变量 > 配置文件默认值

### 辩手性格

在 `config/personalities.yaml` 中定义了 5 种辩论风格：


| 风格          | 名称       | 特点                       |
| ------------- | ---------- | -------------------------- |
| `logical`     | 逻辑严密型 | 三段论，因果链条，严密论证 |
| `emotional`   | 情感感染型 | 生动案例，情感共鸣         |
| `data_driven` | 数据实证型 | 研究报告，统计数字         |
| `aggressive`  | 犀利进攻型 | 发现漏洞，穷追不舍         |
| `diplomatic`  | 稳健防守型 | 化解攻势，巧妙转化         |

### 评分与罚分

在 `config/default.yaml` 中配置：

- 评分维度权重（逻辑、说服力、表达、团队配合、规则遵守）
- 超时罚分（个人 -2，团队 -3）
- 违规罚分（-2 至 -3 / 次）

## 项目结构

```
AI_debate/
├── config/
│   ├── default.yaml          # LLM、计时、评分配置
│   ├── topics.yaml           # 内置辩题
│   └── personalities.yaml    # 辩手性格模板
├── src/
│   ├── cli.py                # CLI 入口
│   ├── config.py             # YAML 配置加载
│   ├── export.py             # JSON 导出
│   ├── agents/               # 智能体
│   │   ├── base.py           # 基础智能体
│   │   ├── debater.py        # 辩手智能体
│   │   ├── judge.py          # 裁判智能体
│   │   └── prompts.py        # 系统提示词模板
│   ├── engine/               # 引擎
│   │   ├── message_pool.py   # 消息池（公共/队内/裁判频道）
│   │   ├── scorer.py         # 三层评分引擎
│   │   └── timer.py          # 计时与超时追踪
│   ├── llm/                  # LLM 抽象层
│   │   ├── __init__.py       # LLM 工厂 + 角色环境变量解析
│   │   ├── base.py           # 抽象基类
│   │   ├── zhipu.py          # 智谱 GLM 提供商
│   │   └── openai_compatible.py  # OpenAI 兼容提供商
│   ├── stages/               # 辩论阶段
│   │   ├── controller.py     # 阶段控制器
│   │   ├── opening.py        # 立论陈词
│   │   ├── cross_exam.py     # 攻辩阶段
│   │   ├── free_debate.py    # 自由辩论（顺序 + 并发）
│   │   └── closing.py        # 总结陈词
│   └── display/
│       └── terminal.py       # Rich 终端 UI
├── tests/                    # 测试套件
├── docs/                     # 设计文档
├── .env.example              # 环境变量模板
└── pyproject.toml
```

## 开发

### 运行测试

```bash
pytest
pytest --cov=src  # 带覆盖率
```

## 运行效果

运行

```bash
python -m src.cli
```

效果：

![1774363272915](images/README/1774363272915.png)



### 依赖


| 包                     | 用途                        |
| ---------------------- | --------------------------- |
| `zai`                  | 智谱 GLM SDK                |
| `rich >= 13.0`         | 终端 UI、面板、表格、实时流 |
| `pyyaml >= 6.0`        | YAML 配置加载               |
| `python-dotenv >= 1.0` | .env 环境变量加载           |

## 许可证

MIT License
