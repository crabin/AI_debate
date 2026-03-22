"""Tests for JudgeAgent."""

import pytest
from unittest.mock import Mock, patch
import json

from src.agents.judge import JudgeAgent
from src.engine.message_pool import MessagePool, Message
from src.llm.base import BaseLLM


class TestJudgeAgent:
    """Tests for JudgeAgent."""

    def test_create_returns_correct_instance(self):
        """create factory returns properly configured JudgeAgent."""
        mock_llm = Mock(spec=BaseLLM)
        agent = JudgeAgent.create(
            topic="死刑应该被废除",
            pro_stance="支持死刑",
            con_stance="反对死刑",
            llm=mock_llm,
        )

        assert agent.agent_id == "judge"
        assert agent.name == "评委"
        assert agent.team == "judge"
        assert agent.role == "评委"
        assert agent.topic == "死刑应该被废除"

    def test_judge_agent_is_base_agent_subclass(self):
        """JudgeAgent inherits from BaseAgent."""
        from src.agents.base import BaseAgent

        mock_llm = Mock(spec=BaseLLM)
        agent = JudgeAgent.create(
            topic="测试辩题",
            pro_stance="正方立场",
            con_stance="反方立场",
            llm=mock_llm,
        )

        assert isinstance(agent, BaseAgent)

    def test_score_speaker_returns_parsed_scores(self):
        """score_speaker parses JSON response correctly."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = json.dumps({
            "speaker": "pro_1",
            "logic": 8,
            "persuasion": 7,
            "expression": 8,
            "teamwork": 7,
            "rule_compliance": 10,
            "violations": [],
            "comment": "论证清晰",
        })

        agent = JudgeAgent.create(
            topic="测试辩题",
            pro_stance="正方立场",
            con_stance="反方立场",
            llm=mock_llm,
        )

        pool = MessagePool()
        result = agent.score_speaker(pool, speaker_id="pro_1")

        assert result["speaker"] == "pro_1"
        assert result["logic"] == 8
        assert result["persuasion"] == 7
        assert result["expression"] == 8
        assert result["teamwork"] == 7
        assert result["rule_compliance"] == 10
        assert result["violations"] == []
        assert result["comment"] == "论证清晰"

    def test_score_speaker_includes_context(self):
        """score_speaker includes debate history in context."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = json.dumps({
            "speaker": "pro_1",
            "logic": 8,
            "persuasion": 7,
            "expression": 8,
            "teamwork": 7,
            "rule_compliance": 10,
            "violations": [],
            "comment": "好",
        })

        agent = JudgeAgent.create(
            topic="测试辩题",
            pro_stance="正方立场",
            con_stance="反方立场",
            llm=mock_llm,
        )

        pool = MessagePool()
        msg = Message(
            speaker="pro_1",
            role="一辩",
            team="pro",
            stage="opening",
            content="立论内容",
            msg_type="statement",
            timestamp=1.0,
            word_count=4,
            metadata=(),
        )
        pool.publish("public", msg)

        result = agent.score_speaker(pool, speaker_id="pro_1")

        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        assert "立论内容" in system_content
        assert "pro_1" in system_content

    def test_score_speaker_with_violations(self):
        """score_speaker captures violations correctly."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = json.dumps({
            "speaker": "con_2",
            "logic": 6,
            "persuasion": 5,
            "expression": 7,
            "teamwork": 6,
            "rule_compliance": 4,
            "violations": ["counter_question", "not_direct_answer"],
            "comment": "违反规则",
        })

        agent = JudgeAgent.create(
            topic="测试辩题",
            pro_stance="正方立场",
            con_stance="反方立场",
            llm=mock_llm,
        )

        pool = MessagePool()
        result = agent.score_speaker(pool, speaker_id="con_2")

        assert result["rule_compliance"] == 4
        assert "counter_question" in result["violations"]
        assert "not_direct_answer" in result["violations"]

    def test_generate_review_returns_parsed_review(self):
        """generate_review parses JSON response correctly."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = json.dumps({
            "type": "review",
            "summary": "精彩的对决",
            "highlights": ["正方立论有力"],
            "suggestions": ["反方需加强逻辑"],
        })

        agent = JudgeAgent.create(
            topic="测试辩题",
            pro_stance="正方立场",
            con_stance="反方立场",
            llm=mock_llm,
        )

        pool = MessagePool()
        result = agent.generate_review(pool)

        assert result["type"] == "review"
        assert result["summary"] == "精彩的对决"
        assert "正方立论有力" in result["highlights"]
        assert "反方需加强逻辑" in result["suggestions"]

    def test_generate_review_includes_full_debate_history(self):
        """generate_review includes full debate context."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = json.dumps({
            "type": "review",
            "summary": "总结",
            "highlights": [],
            "suggestions": [],
        })

        agent = JudgeAgent.create(
            topic="测试辩题",
            pro_stance="正方立场",
            con_stance="反方立场",
            llm=mock_llm,
        )

        pool = MessagePool()
        msg = Message(
            speaker="pro_1",
            role="一辩",
            team="pro",
            stage="opening",
            content="正方立论",
            msg_type="statement",
            timestamp=1.0,
            word_count=4,
            metadata=(),
        )
        pool.publish("public", msg)

        agent.generate_review(pool)

        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        assert "正方立论" in system_content
        assert "点评" in system_content

    @patch("src.agents.judge.JUDGE_SYSTEM")
    def test_get_system_prompt_uses_judge_template(self, mock_judge_prompt):
        """_get_system_prompt uses JUDGE_SYSTEM template."""
        mock_judge_prompt.format.return_value = "你是评委"

        mock_llm = Mock(spec=BaseLLM)
        agent = JudgeAgent.create(
            topic="死刑应该被废除",
            pro_stance="支持死刑",
            con_stance="反对死刑",
            llm=mock_llm,
        )

        prompt = agent._get_system_prompt()

        mock_judge_prompt.format.assert_called_once()
        call_kwargs = mock_judge_prompt.format.call_args[1]
        assert call_kwargs["topic"] == "死刑应该被废除"
        assert call_kwargs["pro_stance"] == "支持死刑"
        assert call_kwargs["con_stance"] == "反对死刑"
