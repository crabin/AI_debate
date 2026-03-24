"""Tests for agent classes."""

from unittest.mock import Mock, patch

from src.agents.base import BaseAgent
from src.agents.debater import DebaterAgent
from src.engine.message_pool import MessagePool, Message
from src.llm.base import BaseLLM


class TestBaseAgent:
    """Tests for BaseAgent."""

    def test_build_context_with_empty_pool(self):
        """build_context returns formatted string with empty messages."""
        mock_llm = Mock(spec=BaseLLM)
        agent = BaseAgent(
            agent_id="test_agent",
            name="Test Agent",
            team="pro",
            role="一辩",
            llm=mock_llm,
        )

        pool = MessagePool()
        result = agent.build_context(pool, stage="opening")

        assert isinstance(result, str)
        assert "【辩论记录】" in result
        assert "暂无发言" in result

    def test_build_context_with_messages(self):
        """build_context formats messages into readable context."""
        mock_llm = Mock(spec=BaseLLM)
        agent = BaseAgent(
            agent_id="test_agent",
            name="Test Agent",
            team="pro",
            role="一辩",
            llm=mock_llm,
        )

        pool = MessagePool()
        msg1 = Message(
            speaker="正方一辩",
            role="一辩",
            team="pro",
            stage="opening",
            content="正方立论",
            msg_type="statement",
            timestamp=1.0,
            word_count=4,
            metadata=(),
        )
        msg2 = Message(
            speaker="反方一辩",
            role="一辩",
            team="con",
            stage="opening",
            content="反方立论",
            msg_type="statement",
            timestamp=2.0,
            word_count=4,
            metadata=(),
        )
        pool.publish("public", msg1)
        pool.publish("public", msg2)

        result = agent.build_context(pool, stage="opening")

        assert "正方一辩" in result
        assert "正方立论" in result
        assert "反方一辩" in result
        assert "反方立论" in result

    def test_build_context_filters_by_stage(self):
        """build_context only includes messages from specified stage."""
        mock_llm = Mock(spec=BaseLLM)
        agent = BaseAgent(
            agent_id="test_agent",
            name="Test Agent",
            team="pro",
            role="一辩",
            llm=mock_llm,
        )

        pool = MessagePool()
        msg1 = Message(
            speaker="正方一辩",
            role="一辩",
            team="pro",
            stage="opening",
            content="立论内容",
            msg_type="statement",
            timestamp=1.0,
            word_count=4,
            metadata=(),
        )
        msg2 = Message(
            speaker="反方一辩",
            role="一辩",
            team="con",
            stage="cross_exam",
            content="攻辩内容",
            msg_type="question",
            timestamp=2.0,
            word_count=4,
            metadata=(),
        )
        pool.publish("public", msg1)
        pool.publish("public", msg2)

        result = agent.build_context(pool, stage="opening")

        assert "立论内容" in result
        assert "攻辩内容" not in result

    def test_speak_returns_llm_response(self):
        """speak method calls LLM and returns response."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = "这是我的发言内容"

        agent = BaseAgent(
            agent_id="test_agent",
            name="Test Agent",
            team="pro",
            role="一辩",
            llm=mock_llm,
        )

        result = agent.speak(
            system_prompt="你是辩手",
            context="辩论历史",
            instruction="请发言",
        )

        assert result == "这是我的发言内容"
        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "你是辩手" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_agent_properties(self):
        """Agent has correct id, name, team, and role."""
        mock_llm = Mock(spec=BaseLLM)
        agent = BaseAgent(
            agent_id="pro_1",
            name="正方一辩",
            team="pro",
            role="一辩",
            llm=mock_llm,
        )

        assert agent.agent_id == "pro_1"
        assert agent.name == "正方一辩"
        assert agent.team == "pro"
        assert agent.role == "一辩"


class TestDebaterAgent:
    """Tests for DebaterAgent."""

    def test_create_returns_correct_instance(self):
        """create factory returns properly configured DebaterAgent."""
        mock_llm = Mock(spec=BaseLLM)
        agent = DebaterAgent.create(
            position=1,
            team="pro",
            stance="支持死刑",
            topic="死刑应该被废除",
            personality="aggressive",
            llm=mock_llm,
        )

        assert agent.agent_id == "pro_1"
        assert agent.team == "pro"
        assert agent.role == "一辩"
        assert agent.stance == "支持死刑"
        assert agent.topic == "死刑应该被废除"
        assert agent.personality == "aggressive"

    def test_create_pro_team_second_debater(self):
        """create factory configures second debater correctly."""
        mock_llm = Mock(spec=BaseLLM)
        agent = DebaterAgent.create(
            position=2,
            team="pro",
            stance="支持死刑",
            topic="死刑应该被废除",
            personality="logical",
            llm=mock_llm,
        )

        assert agent.agent_id == "pro_2"
        assert agent.role == "二辩"

    def test_create_con_team_fourth_debater(self):
        """create factory configures con team fourth debater."""
        mock_llm = Mock(spec=BaseLLM)
        agent = DebaterAgent.create(
            position=4,
            team="con",
            stance="反对死刑",
            topic="死刑应该被废除",
            personality="emotional",
            llm=mock_llm,
        )

        assert agent.agent_id == "con_4"
        assert agent.role == "四辩"
        assert agent.team == "con"

    @patch("src.agents.debater.DEBATER_PROMPTS")
    def test_generate_opening_statement_uses_correct_prompt(self, mock_prompts):
        """generate_opening_statement uses correct position prompt."""
        mock_prompts.__getitem__.return_value = "你是一辩，{team}立场"
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = "立论陈词内容"

        agent = DebaterAgent.create(
            position=1,
            team="pro",
            stance="支持死刑",
            topic="死刑应该被废除",
            personality="logical",
            llm=mock_llm,
        )

        pool = MessagePool()
        result = agent.generate_opening_statement(pool)

        assert result == "立论陈词内容"
        mock_llm.chat.assert_called_once()

    def test_generate_opening_statement_includes_context(self):
        """generate_opening_statement includes debate context."""
        mock_llm = Mock(spec=BaseLLM)
        mock_llm.chat.return_value = "我的立论"

        agent = DebaterAgent.create(
            position=1,
            team="pro",
            stance="支持死刑",
            topic="死刑应该被废除",
            personality="logical",
            llm=mock_llm,
        )

        pool = MessagePool()
        msg = Message(
            speaker="反方一辩",
            role="一辩",
            team="con",
            stage="opening",
            content="反方立论",
            msg_type="statement",
            timestamp=1.0,
            word_count=4,
            metadata=(),
        )
        pool.publish("public", msg)

        agent.generate_opening_statement(pool)

        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        user_content = messages[1]["content"]
        # Should include context even for opening (to see opponent's opening)
        assert "【辩论记录】" in user_content or "当前辩论记录" in user_content

    def test_get_system_prompt_for_position(self):
        """_get_system_prompt returns correct prompt for each position."""
        mock_llm = Mock(spec=BaseLLM)

        for pos in range(1, 5):
            agent = DebaterAgent.create(
                position=pos,
                team="pro",
                stance="支持死刑",
                topic="测试辩题",
                personality="logical",
                llm=mock_llm,
            )

            prompt = agent._get_system_prompt()
            assert prompt is not None
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_debater_agent_is_base_agent_subclass(self):
        """DebaterAgent inherits from BaseAgent."""
        mock_llm = Mock(spec=BaseLLM)
        agent = DebaterAgent.create(
            position=1,
            team="pro",
            stance="支持死刑",
            topic="死刑应该被废除",
            personality="logical",
            llm=mock_llm,
        )

        assert isinstance(agent, BaseAgent)


def test_base_agent_exposes_model_name(fake_llm):
    from src.agents.base import BaseAgent
    agent = BaseAgent(
        agent_id="pro_1", name="正方一辩", team="pro",
        role="一辩", llm=fake_llm,
    )
    assert agent.model_name == "fake-model"
