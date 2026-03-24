"""Tests for debate stages."""

import pytest
from unittest.mock import Mock

from src.stages.base import BaseStage
from src.stages.opening import OpeningStage
from src.stages.cross_exam import CrossExamStage
from src.stages.free_debate import FreeDebateStage
from src.stages.closing import ClosingStage
from src.engine.message_pool import MessagePool
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent


def create_mock_llm(response: str = "Mock LLM response") -> Mock:
    """Create a Mock LLM with both chat and chat_stream configured.

    This helper ensures that streaming methods work correctly in tests.
    """
    mock_llm = Mock()
    mock_llm.chat.return_value = response
    mock_llm.chat_stream.return_value = response
    return mock_llm


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
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "这是我的立论陈词"
        mock_llm.chat_stream.return_value = "这是我的立论陈词"

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
        # Verify LLM was called twice (speeches use chat_stream)
        assert mock_llm.chat_stream.call_count == 2

    def test_execute_publishes_messages_to_pool(self):
        """execute publishes messages to message pool."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "立论内容"
        mock_llm.chat_stream.return_value = "立论内容"

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
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "立论"
        mock_llm.chat_stream.return_value = "立论"

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
        mock_llm = create_mock_llm()
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
        # Verify LLM was called correctly (speeches use chat_stream, judge uses chat)
        total_calls = mock_llm.chat.call_count + mock_llm.chat_stream.call_count
        assert total_calls == 4  # 2 speeches + 2 scores

    def test_opening_stage_is_base_stage_subclass(self):
        """OpeningStage inherits from BaseStage."""
        mock_display = Mock()
        stage = OpeningStage.create(display=mock_display)

        assert isinstance(stage, BaseStage)


class TestCrossExamStage:
    """Tests for CrossExamStage."""

    def test_create_returns_configured_stage(self):
        """create factory returns configured CrossExamStage."""
        mock_display = Mock()
        stage = CrossExamStage.create(display=mock_display)

        assert stage.name == "cross_exam"
        assert "攻辩" in stage.description

    def test_execute_generates_cross_exam_rounds(self):
        """execute generates 4 rounds of cross-exam."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "这是我的问题/回答"
        mock_llm.chat_stream.return_value = "这是我的问题/回答"

        mock_display = Mock()
        stage = CrossExamStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_2": DebaterAgent.create(
                position=2, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "pro_3": DebaterAgent.create(
                position=3, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_2": DebaterAgent.create(
                position=2, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_3": DebaterAgent.create(
                position=3, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
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
        # 4 rounds x 2 messages (Q + A) + 2 summaries = 10 messages
        assert result["messages_count"] == 10

    def test_execute_publishes_question_answer_pairs(self):
        """execute publishes question and answer messages."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "内容"
        mock_llm.chat_stream.return_value = "内容"

        mock_display = Mock()
        stage = CrossExamStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_2": DebaterAgent.create(
                position=2, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_2": DebaterAgent.create(
                position=2, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
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

        messages = pool.get_messages("public", stage="cross_exam")
        # At least first round Q + A + 2 summaries
        assert len(messages) >= 4

        # Check message types
        msg_types = [m.msg_type for m in messages]
        assert "question" in msg_types
        assert "answer" in msg_types
        assert "summary" in msg_types

    def test_execute_handles_missing_agents(self):
        """execute skips missing agents gracefully."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "内容"
        mock_llm.chat_stream.return_value = "内容"

        mock_display = Mock()
        stage = CrossExamStage.create(display=mock_display)

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

        # Should complete with just summaries (cross-exam rounds skipped)
        result = stage.execute(pool, agents)
        assert result["status"] == "completed"
        # 2 summaries only
        assert result["messages_count"] == 2

    def test_cross_exam_stage_is_base_stage_subclass(self):
        """CrossExamStage inherits from BaseStage."""
        mock_display = Mock()
        stage = CrossExamStage.create(display=mock_display)

        assert isinstance(stage, BaseStage)


class TestFreeDebateStage:
    """Tests for FreeDebateStage."""

    def test_create_returns_configured_stage(self):
        """create factory returns configured FreeDebateStage."""
        mock_display = Mock()
        stage = FreeDebateStage.create(display=mock_display)

        assert stage.name == "free_debate"
        assert "自由辩论" in stage.description

    def test_execute_alternates_teams(self):
        """execute alternates between pro and con teams."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "我的观点是..."
        mock_llm.chat_stream.return_value = "我的观点是..."

        mock_display = Mock()
        stage = FreeDebateStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            f"{team}_{i}": DebaterAgent.create(
                position=i, team=team, stance=f"{team}_立场", topic="测试",
                personality="logical", llm=mock_llm,
            )
            for team in ["pro", "con"]
            for i in range(1, 5)
        }

        result = stage.execute(pool, agents)

        assert result["status"] == "completed"
        assert result["messages_count"] > 0
        assert result["turns"] > 0

    def test_execute_tracks_speaker_counts(self):
        """execute tracks how many times each speaker spoke."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "发言"
        mock_llm.chat_stream.return_value = "发言"

        mock_display = Mock()
        stage = FreeDebateStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            f"{team}_{i}": DebaterAgent.create(
                position=i, team=team, stance=f"{team}_立场", topic="测试",
                personality="logical", llm=mock_llm,
            )
            for team in ["pro", "con"]
            for i in range(1, 5)
        }

        result = stage.execute(pool, agents)

        speak_counts = result["speak_counts"]
        assert isinstance(speak_counts, dict)
        # At least some speakers should have spoken
        assert sum(speak_counts.values()) > 0

    def test_execute_prevents_consecutive_same_team(self):
        """execute prevents same team from speaking twice in a row."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "发言"
        mock_llm.chat_stream.return_value = "发言"

        mock_display = Mock()
        stage = FreeDebateStage.create(display=mock_display)

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

        messages = pool.get_messages("public", stage="free_debate")
        if len(messages) >= 2:
            # Check that teams alternate
            teams = [m.team for m in messages]
            for i in range(len(teams) - 1):
                if i > 0:  # Allow same team to start
                    assert teams[i] != teams[i + 1], f"Same team spoke twice: {teams[i]} at {i}"

    def test_execute_with_timer(self):
        """execute uses timer to track time."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "短发言"
        mock_llm.chat_stream.return_value = "短发言"

        mock_display = Mock()
        from src.engine.timer import Timer
        timer = Timer(total_seconds=240, chars_per_minute=250)
        stage = FreeDebateStage.create(display=mock_display, timer=timer)

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

        assert "pro_time_left" in result
        assert "con_time_left" in result

    def test_free_debate_stage_is_base_stage_subclass(self):
        """FreeDebateStage inherits from BaseStage."""
        mock_display = Mock()
        stage = FreeDebateStage.create(display=mock_display)

        assert isinstance(stage, BaseStage)


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


class TestClosingStage:
    """Tests for ClosingStage."""

    def test_create_returns_configured_stage(self):
        """create factory returns configured ClosingStage."""
        mock_display = Mock()
        stage = ClosingStage.create(display=mock_display)

        assert stage.name == "closing"
        assert "总结陈词" in stage.description

    def test_execute_generates_closing_statements(self):
        """execute generates closing statements from both teams."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "这是我的总结陈词"
        mock_llm.chat_stream.return_value = "这是我的总结陈词"

        mock_display = Mock()
        stage = ClosingStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_4": DebaterAgent.create(
                position=4, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_4": DebaterAgent.create(
                position=4, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
        }

        result = stage.execute(pool, agents)

        assert result["status"] == "completed"
        assert result["messages_count"] == 2
        # Verify LLM was called twice (speeches use chat_stream)
        assert mock_llm.chat_stream.call_count == 2

    def test_execute_con_goes_first(self):
        """execute has con team speak before pro team."""
        mock_llm = create_mock_llm()
        mock_llm.chat.return_value = "总结"
        mock_llm.chat_stream.return_value = "总结"

        mock_display = Mock()
        stage = ClosingStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_4": DebaterAgent.create(
                position=4, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_4": DebaterAgent.create(
                position=4, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
        }

        stage.execute(pool, agents)

        messages = pool.get_messages("public", stage="closing")
        assert len(messages) == 2
        # Con should go first
        assert messages[0].team == "con"
        assert messages[1].team == "pro"

    def test_execute_handles_missing_agents(self):
        """execute raises error if required agents missing."""
        mock_display = Mock()
        stage = ClosingStage.create(display=mock_display)

        pool = MessagePool()
        agents = {}  # Missing required agents

        with pytest.raises(ValueError, match="Required agent"):
            stage.execute(pool, agents)

    def test_execute_with_judge_scoring(self):
        """execute includes judge scoring when judge agent provided."""
        mock_llm = create_mock_llm()
        mock_llm.chat.side_effect = [
            "反方总结",  # con_4
            "正方总结",  # pro_4
            '{"speaker": "con_4", "logic": 8, "persuasion": 7, "expression": 8, "teamwork": 7, "rule_compliance": 10, "violations": [], "comment": "好"}',  # judge scores con_4
            '{"speaker": "pro_4", "logic": 7, "persuasion": 7, "expression": 7, "teamwork": 7, "rule_compliance": 10, "violations": [], "comment": "好"}',  # judge scores pro_4
        ]

        mock_display = Mock()
        stage = ClosingStage.create(display=mock_display)

        pool = MessagePool()
        agents = {
            "pro_4": DebaterAgent.create(
                position=4, team="pro", stance="支持", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "con_4": DebaterAgent.create(
                position=4, team="con", stance="反对", topic="测试",
                personality="logical", llm=mock_llm,
            ),
            "judge": JudgeAgent.create(
                topic="测试", pro_stance="支持", con_stance="反对",
                llm=mock_llm,
            ),
        }

        result = stage.execute(pool, agents)

        assert result["status"] == "completed"
        # Verify LLM was called correctly (speeches use chat_stream, judge uses chat)
        total_calls = mock_llm.chat.call_count + mock_llm.chat_stream.call_count
        assert total_calls == 4  # 2 speeches + 2 scores

    def test_closing_stage_is_base_stage_subclass(self):
        """ClosingStage inherits from BaseStage."""
        mock_display = Mock()
        stage = ClosingStage.create(display=mock_display)

        assert isinstance(stage, BaseStage)



