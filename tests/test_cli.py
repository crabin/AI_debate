"""Tests for CLI module."""

import sys
import pytest
from unittest.mock import Mock, patch

from src.cli import setup_logging, create_agents, run_debate


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_setup_logging_info_level(self):
        """setup_logging configures INFO level."""
        setup_logging("INFO")
        # Just verify no exception raised

    def test_setup_logging_debug_level(self):
        """setup_logging configures DEBUG level."""
        setup_logging("DEBUG")


class TestCreateAgents:
    """Tests for create_agents."""

    def test_create_agents_returns_all_debaters_and_judge(self):
        """create_agents creates 8 debaters + 1 judge."""
        mock_llm = Mock()

        config = {"default_personality": "logical"}
        topic = {
            "title": "测试辩题",
            "pro_stance": "正方立场",
            "con_stance": "反方立场",
        }
        personalities = {
            "logical": "逻辑严密"
        }

        agents = create_agents(config, topic, personalities, pro_llm=mock_llm, con_llm=mock_llm, judge_llm=mock_llm)

        # Should have 8 debaters + 1 judge = 9 agents
        assert len(agents) == 9
        assert "judge" in agents
        assert "pro_1" in agents
        assert "con_4" in agents

    def test_create_agents_uses_topic_stances(self):
        """create_agents passes correct stances to agents."""
        mock_llm = Mock()

        config = {"default_personality": "logical"}
        topic = {
            "title": "死刑应该被废除",
            "pro_stance": "支持死刑",
            "con_stance": "反对死刑",
        }
        personalities = {"logical": "逻辑严密"}

        agents = create_agents(config, topic, personalities, pro_llm=mock_llm, con_llm=mock_llm, judge_llm=mock_llm)

        assert agents["pro_1"].stance == "支持死刑"
        assert agents["con_1"].stance == "反对死刑"


class TestRunDebate:
    """Tests for run_debate."""

    @patch("src.cli.create_llm")
    @patch("src.cli.TerminalDisplay")
    @patch("src.cli.load_config")
    @patch("src.cli.load_topics")
    @patch("src.cli.load_personalities")
    @patch("src.cli.StageController")
    @patch("src.cli.MessagePool")
    @patch("src.cli.create_agents")
    def test_run_debate_with_valid_topic(
        self,
        mock_create_agents,
        mock_pool_class,
        mock_controller_class,
        mock_load_personalities,
        mock_load_topics,
        mock_load_config,
        mock_display_class,
        mock_create_llm,
    ):
        """run_debate completes successfully with valid topic."""
        # Setup mocks
        mock_load_config.return_value = {"default_personality": "logical"}
        mock_load_topics.return_value = [
            {"title": "辩题1", "pro_stance": "正方", "con_stance": "反方"}
        ]
        mock_load_personalities.return_value = {"logical": "逻辑"}
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm

        mock_display = Mock()
        mock_display_class.return_value = mock_display

        mock_agents = {"judge": Mock()}
        mock_create_agents.return_value = mock_agents

        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_controller = Mock()
        mock_controller.run_debate.return_value = {"status": "completed"}
        mock_controller_class.create.return_value = mock_controller

        # Run
        results = run_debate(topic_index=0)

        # Verify
        assert results["status"] == "completed"
        mock_display.header.assert_called_once()
        mock_display.participants.assert_called_once_with(mock_agents)
        mock_controller.run_debate.assert_called_once_with(mock_pool, mock_agents)

    def test_run_debate_with_invalid_topic_index(self):
        """run_debate raises ValueError for invalid topic index."""
        # Patch all config loading to avoid file operations
        with patch("src.cli.load_topics") as mock_load_topics, \
             patch("src.cli.load_config") as mock_load_config, \
             patch("src.cli.load_personalities") as mock_load_personalities:
            mock_load_config.return_value = {"default_personality": "logical"}
            mock_load_personalities.return_value = {"logical": "逻辑"}
            mock_load_topics.return_value = [
                {"title": "辩题1", "pro_stance": "正方", "con_stance": "反方"}
            ]

            with pytest.raises(ValueError, match="out of range"):
                run_debate(topic_index=5)


class TestCreateAgentsSeparateLLMs:
    """Tests for create_agents with per-role LLMs."""

    def test_create_agents_uses_separate_llms(self, fake_llm_factory):
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


class TestMain:
    """Tests for main function."""

    @patch("src.cli.run_debate")
    @patch("src.cli.setup_logging")
    @patch("src.cli.load_dotenv")
    def test_main_returns_zero_on_success(self, mock_dotenv, mock_setup_logging, mock_run_debate):
        """main returns 0 when debate completes successfully."""
        mock_run_debate.return_value = {"status": "completed"}

        original_argv = sys.argv
        try:
            sys.argv = ["cli.py", "0"]
            from src.cli import main
            result = main()
        finally:
            sys.argv = original_argv

        assert result == 0

    @patch("src.cli.run_debate")
    @patch("src.cli.setup_logging")
    @patch("src.cli.load_dotenv")
    def test_main_returns_one_on_failure(self, mock_dotenv, mock_setup_logging, mock_run_debate):
        """main returns 1 when debate fails."""
        mock_run_debate.return_value = {"status": "failed"}

        original_argv = sys.argv
        try:
            sys.argv = ["cli.py", "0"]
            from src.cli import main
            result = main()
        finally:
            sys.argv = original_argv

        assert result == 1

    @patch("src.cli.run_debate")
    @patch("src.cli.setup_logging")
    @patch("src.cli.load_dotenv")
    def test_main_parses_topic_index_from_argv(self, mock_dotenv, mock_setup_logging, mock_run_debate):
        """main parses topic index from command line args."""
        mock_run_debate.return_value = {"status": "completed"}

        original_argv = sys.argv
        try:
            sys.argv = ["cli.py", "2"]
            from src.cli import main
            main()
        finally:
            sys.argv = original_argv

        mock_run_debate.assert_called_once_with(topic_index=2, output_path=None, concurrent=False)

    @patch("src.cli.setup_logging")
    @patch("src.cli.load_dotenv")
    @patch("src.cli.run_debate")
    def test_main_returns_one_on_invalid_topic_index(self, mock_run_debate, mock_dotenv, mock_setup_logging):
        """main returns 1 for invalid topic index."""
        original_argv = sys.argv
        try:
            sys.argv = ["cli.py", "invalid"]
            from src.cli import main
            result = main()
        finally:
            sys.argv = original_argv

        assert result == 1
