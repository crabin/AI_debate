"""Tests for debate stages."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from src.stages.base import BaseStage
from src.stages.opening import OpeningStage
from src.engine.message_pool import MessagePool, Message
from src.display.terminal import TerminalDisplay
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent
from src.llm.base import BaseLLM


# Concrete test implementation of BaseStage
class DummyStage(BaseStage):
    """Test implementation of BaseStage."""

    def execute(self, pool, agents: dict, penalties: dict | None = None) -> dict:
        return {"status": "test_completed"}


class TestBaseStage:
    """Tests for BaseStage."""

    def test_stage_properties(self):
        """DummyStage has correct name and description."""
        mock_display = Mock()
        stage = DummyStage(
            name="test_stage",
            description="Test stage description",
            display=mock_display,
        )

        assert stage.name == "test_stage"
        assert stage.description == "Test stage description"

    def test_execute_returns_result(self):
        """DummyStage.execute returns result."""
        mock_display = Mock()
        stage = DummyStage(
            name="test_stage",
            description="Test stage description",
            display=mock_display,
        )

        pool = MessagePool()
        agents = {}

        result = stage.execute(pool, agents)
        assert result["status"] == "test_completed"


class TestOpeningStage:
    """Tests for OpeningStage."""

    def test_create_returns_configured_stage(self):
        """create factory returns configured OpeningStage."""
        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        assert stage.name == "opening"
        assert "立论陈词" in stage.description

    def test_execute_generates_opening_statements(self):
        """execute generates opening statements from both teams."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "这是我的立论陈词"

        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_1": DebaterAgent.create(
                position=1, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_1": DebaterAgent.create(
                position=1, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
        }

        result = stage.execute(pool, agents)

        assert result["status"] == "completed"
        assert result["messages_count"] == 2
        # Verify LLM was called twice
        assert mock_llm.chat.call_count == 2

    def test_execute_publishes_messages_to_pool(self):
        """execute publishes messages to message pool."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "立论内容"

        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_1": DebaterAgent.create(
                position=1, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_1": DebaterAgent.create(
                position=1, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
        }

        stage.execute(pool, agents)

        messages = pool.get_messages("public")
        assert len(messages) == 2
        assert messages[0].stage == "opening"
        assert messages[0].msg_type == "opening_statement"
        assert messages[0].team == "pro"
        assert messages[1].team == "con"

    def test_execute_displays_progress(self):
        """execute calls display to show progress."""
        mock_llm = Mock()
        mock_llm.chat.return_value = "立论"

        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_1": DebaterAgent.create(
                position=1, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_1": DebaterAgent.create(
                position=1, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
        }

        stage.execute(pool, agents)

        # Verify display was called for each speaker
        assert mock_display.stage_start.call_count == 1
        assert mock_display.speech.call_count == 4  # 2 "generating" + 2 actual speeches
        assert mock_display.stage_end.call_count == 1

    def test_execute_handles_missing_agents(self):
        """execute raises error if required agents missing."""
        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        pool = MessagePool()
        agents = {}  # Missing required agents

        with pytest.raises(ValueError, match="Required agent"):
            stage.execute(pool, agents)

    def test_execute_with_judge_scoring(self):
        """execute includes judge scoring when judge agent provided."""
        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            "正方立论",  # pro_1
            "反方立论",  # con_1
            '{"speaker": "pro_1", "logic": 8, "persuasion": 7, "expression": 8, "teamwork": 7, "rule_compliance": 10, "violations": [], "comment": "好"}',  # judge scores pro_1
            '{"speaker": "con_1", "logic": 7, "persuasion": 7, "expression": 7, "teamwork": 7, "rule_compliance": 10, "violations": [], "comment": "好"}',  # judge scores con_1
        ]

        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_1": DebaterAgent.create(
                position=1, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_1": DebaterAgent.create(
                position=1, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "judge": JudgeAgent.create(
                topic="测试", pro_stance="支持", con_stance="反对",
                llm=mock_llm,
            ),
        }

        result = stage.execute(pool, agents)

        assert result["status"] == "completed"
        # Verify judge was called for scoring
        assert mock_llm.chat.call_count == 4  # 2 speeches + 2 scores

    def test_opening_stage_is_base_stage_subclass(self):
        """OpeningStage inherits from BaseStage."""
        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        assert isinstance(stage, BaseStage)
