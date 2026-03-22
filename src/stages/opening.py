"""Opening stage for AI Debate System."""

import time
import logging
from src.stages.base import BaseStage
from src.engine.message_pool import Message

logger = logging.getLogger(__name__)


class OpeningStage(BaseStage):
    """Opening statement stage (立论陈词).

    Both teams' first debaters present their arguments:
    - Pro team first debater: 3 minutes (750 chars)
    - Con team first debater: 3 minutes (750 chars)
    """

    _OPENING_SPEAKERS = ["pro_1", "con_1"]
    _EXPECTED_CHARS = 750  # ~3 minutes at 250 chars/min

    def __init__(self, display) -> None:
        """Initialize opening stage.

        Args:
            display: TerminalDisplay instance for output
        """
        super().__init__(
            name="opening",
            description="立论陈词阶段：正反方一辩各3分钟",
            display=display,
        )

    @classmethod
    def create(cls, display) -> "OpeningStage":
        """Factory method to create an OpeningStage.

        Args:
            display: TerminalDisplay instance for output

        Returns:
            Configured OpeningStage instance
        """
        return cls(display=display)

    def execute(
        self,
        pool,
        agents: dict,
        penalties: dict | None = None,
    ) -> dict:
        """Execute opening statement stage.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent
            penalties: Optional penalty configuration

        Returns:
            Result dictionary with status and message count
        """
        # Validate required agents
        missing = [s for s in self._OPENING_SPEAKERS if s not in agents]
        if missing:
            raise ValueError(f"Required agent(s) missing: {missing}")

        self._display.stage_start(self.name, self.description)

        messages_published = 0
        judge_agent = agents.get("judge")

        for speaker_id in self._OPENING_SPEAKERS:
            agent = agents[speaker_id]

            # Generate opening statement
            self._display.speech(
                speaker=agent.name,
                content="正在生成...",
                word_count=0,
                expected=self._EXPECTED_CHARS,
            )

            content = agent.generate_opening_statement(pool)
            word_count = len(content)

            # Publish message
            message = Message(
                speaker=agent.agent_id,
                role=agent.role,
                team=agent.team,
                stage=self.name,
                content=content,
                msg_type="opening_statement",
                timestamp=time.time(),
                word_count=word_count,
                metadata=(),
            )
            pool.publish("public", message)
            messages_published += 1

            # Display the actual speech
            self._display.speech(
                speaker=agent.name,
                content=content,
                word_count=word_count,
                expected=self._EXPECTED_CHARS,
            )

            # Judge scoring if judge agent available
            if judge_agent:
                try:
                    score = judge_agent.score_speaker(pool, speaker_id)
                    # Publish judge's score to judge_notes channel
                    judge_msg = Message(
                        speaker="judge",
                        role="评委",
                        team="judge",
                        stage=self.name,
                        content=str(score),
                        msg_type="score",
                        timestamp=time.time(),
                        word_count=0,
                        metadata=("score", speaker_id),
                    )
                    pool.publish("judge_notes", judge_msg)
                except Exception as e:
                    logger.warning(f"Judge scoring failed for {speaker_id}: {e}")

        self._display.stage_end(self.name)

        return {
            "status": "completed",
            "stage": self.name,
            "messages_count": messages_published,
        }
