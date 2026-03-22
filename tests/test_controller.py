"""Tests for StageController."""

import pytest
from unittest.mock import Mock, patch
import json

from src.stages.controller import StageController
from src.engine.message_pool import MessagePool, Message
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent
from src.engine.scorer import ScoreCard
from src.llm.base import BaseLLM


class TestStageController:
    """Tests for StageController."""

    def test_create_returns_configured_controller(self):
        """create factory returns configured StageController."""
        mock_display = Mock()
        controller = StageController.create(display=mock_display)

        assert controller._display is mock_display
        assert controller._penalties == {}

    def test_create_with_penalties(self):
        """create factory passes through penalty config."""
        mock_display = Mock()
        penalties = {"overtime_team": 3, "overtime_player": 2}
        controller = StageController.create(display=mock_display, penalties=penalties)

        assert controller._penalties == penalties

    def test_create_stages_initializes_all_stages(self):
        """create_stages initializes all four stages."""
        mock_display = Mock()
        controller = StageController.create(display=mock_display)

        controller.create_stages()

        assert "opening" in controller._stages
        assert "cross_exam" in controller._stages
        assert "free_debate" in controller._stages
        assert "closing" in controller._stages

    def test_calculate_final_results_with_scores(self):
        """_calculate_final_results aggregates scores correctly."""
        mock_display = Mock()
        controller = StageController.create(display=mock_display)

        pool = MessagePool()
        # Add mock judge scores
        for speaker, logic, persuasion, team in [
            ("pro_1", 8, 8, "pro"),
            ("con_1", 7, 7, "con"),
            ("pro_2", 8, 8, "pro"),
            ("con_2", 6, 6, "con"),
        ]:
            score_data = {
                "speaker": speaker,
                "logic": logic,
                "persuasion": persuasion,
                "expression": 7,
                "teamwork": 7,
                "rule_compliance": 10,
                "violations": [],
                "comment": "",
            }
            msg = Message(
                speaker="judge",
                role="评委",
                team="judge",
                stage="opening",
                content=json.dumps(score_data),
                msg_type="score",
                timestamp=0,
                word_count=0,
                metadata=(),
            )
            pool.publish("judge_notes", msg)

        agents = {}

        result = controller._calculate_final_results(pool, agents)

        # Pro should win (8+8 > 7+6)
        assert result["winner"] == "pro"
        assert result["pro_score"] > result["con_score"]
        assert result["margin"] > 0

    def test_calculate_final_results_without_scores(self):
        """_calculate_final_results returns zero when no scores."""
        mock_display = Mock()
        controller = StageController.create(display=mock_display)

        pool = MessagePool()
        agents = {}

        result = controller._calculate_final_results(pool, agents)

        assert result["pro_score"] == 0
        assert result["con_score"] == 0
        assert result["winner"] == "tie"

    def test_calculate_final_results_with_judge_review(self):
        """_calculate_final_results includes judge review."""
        mock_llm = Mock()
        mock_llm.chat.return_value = json.dumps({
            "type": "review",
            "summary": "精彩辩论",
            "highlights": [],
            "suggestions": [],
        })

        mock_display = Mock()
        controller = StageController.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "judge": JudgeAgent.create(
                topic="测试", pro_stance="支持", con_stance="反对",
                llm=mock_llm,
            ),
        }

        result = controller._calculate_final_results(pool, agents)

        assert "review" in result
        assert result["review"] is not None
        assert result["review"]["type"] == "review"

    def test_calculate_final_results_tie(self):
        """_calculate_final_results handles tie correctly."""
        mock_display = Mock()
        controller = StageController.create(display=mock_display)

        pool = MessagePool()
        # Add equal scores
        for speaker, team in [("pro_1", "pro"), ("con_1", "con")]:
            score_data = {
                "speaker": speaker,
                "logic": 7,
                "persuasion": 7,
                "expression": 7,
                "teamwork": 7,
                "rule_compliance": 10,
                "violations": [],
                "comment": "",
            }
            msg = Message(
                speaker="judge",
                role="评委",
                team="judge",
                stage="opening",
                content=json.dumps(score_data),
                msg_type="score",
                timestamp=0,
                word_count=0,
                metadata=(),
            )
            pool.publish("judge_notes", msg)

        agents = {}

        result = controller._calculate_final_results(pool, agents)

        assert result["winner"] == "tie"
        assert result["margin"] == 0
