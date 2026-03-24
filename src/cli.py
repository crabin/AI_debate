"""CLI entry point for AI Debate System."""

import sys
import time
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config, load_topics, load_personalities
from src.llm import create_llm
from src.llm.base import BaseLLM
from src.agents.base import BaseAgent
from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent
from src.display.terminal import TerminalDisplay
from src.stages.controller import StageController
from src.engine.message_pool import MessagePool

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_agents(
    config: dict,
    topic: dict,
    personalities: dict,
    pro_llm: BaseLLM,
    con_llm: BaseLLM,
    judge_llm: BaseLLM,
    display=None,
) -> dict[str, BaseAgent]:
    """Create all debate agents.

    Args:
        config: Configuration dictionary
        topic: Topic dictionary with title, pro_stance, con_stance
        personalities: Personality templates dictionary
        pro_llm: LLM instance for the pro team
        con_llm: LLM instance for the con team
        judge_llm: LLM instance for the judge
        display: Optional TerminalDisplay instance for judge output

    Returns:
        Dictionary of agent_id -> Agent
    """
    agents: dict[str, BaseAgent] = {}

    # Create debaters for both teams
    for position in range(1, 5):
        for team in ["pro", "con"]:
            personality_key = config.get("default_personality", "logical")
            team_llm = pro_llm if team == "pro" else con_llm

            agent = DebaterAgent.create(
                position=position,
                team=team,
                stance=topic["pro_stance"] if team == "pro" else topic["con_stance"],
                topic=topic["title"],
                personality=personality_key,
                llm=team_llm,
            )
            agents[agent.agent_id] = agent

    # Create judge
    judge = JudgeAgent.create(
        topic=topic["title"],
        pro_stance=topic["pro_stance"],
        con_stance=topic["con_stance"],
        llm=judge_llm,
        display=display,
    )
    agents["judge"] = judge

    return agents


def run_debate(
    topic_index: int = 0,
    config_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run a complete debate.

    Args:
        topic_index: Index of topic to use (0-based)
        config_path: Optional path to config directory
        output_path: Optional path to save JSON export

    Returns:
        Final results dictionary
    """
    start_time = time.time()

    # Load configuration
    config = load_config(config_path)
    topics = load_topics(config_path)
    personalities = load_personalities(config_path)

    if topic_index >= len(topics):
        raise ValueError(f"Topic index {topic_index} out of range (0-{len(topics)-1})")

    topic = topics[topic_index]

    # Setup per-role LLMs
    pro_llm = create_llm(config, role="pro")
    con_llm = create_llm(config, role="con")
    judge_llm = create_llm(config, role="judge")

    # Create display
    display = TerminalDisplay()

    # Show debate header
    display.header(
        title=topic["title"],
        pro_stance=topic["pro_stance"],
        con_stance=topic["con_stance"],
    )

    # Create agents
    agents = create_agents(
        config, topic, personalities,
        pro_llm=pro_llm, con_llm=con_llm, judge_llm=judge_llm,
        display=display,
    )

    # Show participants
    display.participants(agents)

    # Create message pool and controller
    pool = MessagePool()
    controller = StageController.create(
        display=display,
        penalties=config.get("penalties", {}),
    )

    # Run debate
    results = controller.run_debate(pool, agents)

    # Add topic metadata to results
    results["topic"] = topic["title"]
    results["pro_stance"] = topic["pro_stance"]
    results["con_stance"] = topic["con_stance"]

    # Show final results
    display.final_results(results)

    # Save JSON export if requested
    scorer = results.pop("_scorer", None)
    if output_path and scorer is not None:
        from src.export import save_debate_json
        save_debate_json(results, pool, scorer, output_path, start_time)
    elif output_path and scorer is None:
        logger.warning("Scorer not available; JSON export skipped")

    return results


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    load_dotenv()

    try:
        # Parse simple CLI args
        topic_index = 0
        if len(sys.argv) > 1:
            try:
                topic_index = int(sys.argv[1])
            except ValueError:
                print(f"Invalid topic index: {sys.argv[1]}")
                print("Usage: python -m src.cli [topic_index]")
                return 1

        # Setup logging
        setup_logging("INFO")

        # Parse --output flag
        output_path = None
        if "--output" in sys.argv:
            idx = sys.argv.index("--output")
            if idx + 1 < len(sys.argv):
                output_path = Path(sys.argv[idx + 1])

        # Run debate
        results = run_debate(topic_index=topic_index, output_path=output_path)

        # Exit with appropriate code
        return 0 if results["status"] == "completed" else 1

    except KeyboardInterrupt:
        print("\n辩论被中断")
        return 130
    except Exception as e:
        logger.exception("Debate failed with error")
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
