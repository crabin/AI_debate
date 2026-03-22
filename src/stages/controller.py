"""Stage controller for AI Debate System."""

import logging
from src.stages.base import BaseStage
from src.stages.opening import OpeningStage
from src.stages.cross_exam import CrossExamStage
from src.stages.free_debate import FreeDebateStage
from src.stages.closing import ClosingStage
from src.engine.message_pool import MessagePool
from src.engine.scorer import Scorer, ScoreCard
from src.agents.judge import JudgeAgent

logger = logging.getLogger(__name__)


class StageController:
    """Orchestrates all debate stages in sequence.

    Manages the flow:
    1. Opening statements
    2. Cross-examination
    3. Free debate
    4. Closing statements
    5. Final scoring and results
    """

    _STAGE_ORDER = ["opening", "cross_exam", "free_debate", "closing"]

    def __init__(
        self,
        display,
        penalties: dict | None = None,
    ) -> None:
        """Initialize stage controller.

        Args:
            display: TerminalDisplay instance
            penalties: Optional penalty configuration
        """
        self._display = display
        self._penalties = penalties or {}
        self._stages: dict[str, BaseStage] = {}
        self._scorer = Scorer()

    def create_stages(self) -> None:
        """Create all stage instances."""
        self._stages = {
            "opening": OpeningStage.create(display=self._display),
            "cross_exam": CrossExamStage.create(display=self._display),
            "free_debate": FreeDebateStage.create(display=self._display),
            "closing": ClosingStage.create(display=self._display),
        }

    def run_debate(
        self,
        pool: MessagePool,
        agents: dict,
    ) -> dict:
        """Run complete debate from start to finish.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent

        Returns:
            Result dictionary with final scores and winner
        """
        if not self._stages:
            self.create_stages()

        stage_results = []

        for stage_name in self._STAGE_ORDER:
            if stage_name not in self._stages:
                logger.warning(f"Stage {stage_name} not available, skipping")
                continue

            stage = self._stages[stage_name]

            logger.info(f"Starting stage: {stage_name}")
            result = stage.execute(pool, agents, self._penalties)
            stage_results.append(result)

            logger.info(f"Completed stage: {stage_name}")

        # Calculate final scores
        final_results = self._calculate_final_results(pool, agents)

        return {
            "status": "completed",
            "stage_results": stage_results,
            **final_results,
        }

    def _calculate_final_results(
        self,
        pool: MessagePool,
        agents: dict,
    ) -> dict:
        """Calculate final scores and determine winner.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent

        Returns:
            Dictionary with scores and winner
        """
        # Get all judge scores from judge_notes channel
        judge_notes = pool.get_messages("judge_notes")

        # Create fresh scorer for aggregation
        scorer = Scorer()

        for note in judge_notes:
            if note.msg_type == "score":
                # Parse score from content
                try:
                    import json
                    score_data = json.loads(note.content)

                    # Create ScoreCard from judge response
                    card = ScoreCard(
                        speaker=score_data.get("speaker", ""),
                        stage=note.stage,
                        logic=score_data.get("logic", 5),
                        persuasion=score_data.get("persuasion", 5),
                        expression=score_data.get("expression", 5),
                        teamwork=score_data.get("teamwork", 5),
                        rule_compliance=score_data.get("rule_compliance", 5),
                        violations=tuple(score_data.get("violations", [])),
                        comment=score_data.get("comment", ""),
                    )
                    scorer.record(card)
                except (json.JSONDecodeError, TypeError, KeyError):
                    logger.warning(f"Failed to parse score: {note.content}")

        # Calculate team totals using Scorer
        pro_total = scorer.get_team_total("pro")
        con_total = scorer.get_team_total("con")

        # Determine winner
        if pro_total > con_total:
            winner = "pro"
            margin = pro_total - con_total
        elif con_total > pro_total:
            winner = "con"
            margin = con_total - pro_total
        else:
            winner = "tie"
            margin = 0

        # Get judge review if available
        judge_agent = agents.get("judge")
        review = None
        if judge_agent:
            try:
                review = judge_agent.generate_review(pool)
            except Exception as e:
                logger.warning(f"Failed to generate review: {e}")

        return {
            "pro_score": pro_total,
            "con_score": con_total,
            "winner": winner,
            "margin": margin,
            "review": review,
            "best_debater": scorer.get_best_debater(),
        }

    @classmethod
    def create(cls, display, penalties: dict | None = None) -> "StageController":
        """Factory method to create a StageController.

        Args:
            display: TerminalDisplay instance
            penalties: Optional penalty configuration

        Returns:
            Configured StageController instance
        """
        return cls(display=display, penalties=penalties)
