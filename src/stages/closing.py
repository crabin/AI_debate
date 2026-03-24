"""Closing stage for AI Debate System."""

import time
import logging
from src.stages.base import BaseStage
from src.engine.message_pool import Message

logger = logging.getLogger(__name__)


class ClosingStage(BaseStage):
    """Closing statement stage (总结陈词).

    Both teams' fourth debaters give their closing statements:
    - Con team fourth debater: 3 minutes (750 chars)
    - Pro team fourth debater: 3 minutes (750 chars)

    Note: Con goes first, then Pro (reverse of opening).
    """

    _CLOSING_SPEAKERS = ["con_4", "pro_4"]  # Con first!
    _EXPECTED_CHARS = 750  # ~3 minutes at 250 chars/min

    def __init__(self, display) -> None:
        """Initialize closing stage.

        Args:
            display: TerminalDisplay instance for output
        """
        super().__init__(
            name="closing",
            description="总结陈词阶段：反方、正方四辩各3分钟",
            display=display,
        )

    @classmethod
    def create(cls, display) -> "ClosingStage":
        """Factory method to create a ClosingStage.

        Args:
            display: TerminalDisplay instance for output

        Returns:
            Configured ClosingStage instance
        """
        return cls(display=display)

    def execute(
        self,
        pool,
        agents: dict,
        penalties: dict | None = None,
    ) -> dict:
        """Execute closing statement stage.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent
            penalties: Optional penalty configuration

        Returns:
            Result dictionary with status and message count
        """
        # Validate required agents
        missing = [s for s in self._CLOSING_SPEAKERS if s not in agents]
        if missing:
            raise ValueError(f"Required agent(s) missing: {missing}")

        self._display.stage_start(self.name, self.description)

        messages_published = 0
        judge_agent = agents.get("judge")

        for speaker_id in self._CLOSING_SPEAKERS:
            agent = agents[speaker_id]

            # Generate closing statement with streaming
            import sys

            self._display.speech(
                speaker=agent.name,
                content="正在生成总结...",
                word_count=0,
                expected=self._EXPECTED_CHARS,
            )

            # Callback for streaming output
            content_buffer = []
            def stream_callback(char: str) -> None:
                content_buffer.append(char)

            content = agent.generate_closing_statement(pool, callback=stream_callback)
            word_count = len(content)

            # Publish message
            message = Message(
                speaker=agent.agent_id,
                role=agent.role,
                team=agent.team,
                stage=self.name,
                content=content,
                msg_type="closing_statement",
                timestamp=time.time(),
                word_count=word_count,
                metadata=(),
            )
            pool.publish("public", message)
            messages_published += 1

            # Display the actual speech with streaming
            # Calculate time used: chars / (250 chars/min) * 60 = seconds
            time_used = word_count / 250 * 60
            time_limit = self._EXPECTED_CHARS / 250 * 60  # 180 seconds

            # Use streaming display if content was generated with streaming
            if content_buffer:
                self._display.speech_stream(
                    speaker=agent.name,
                    content=content,
                    word_count=word_count,
                    expected=self._EXPECTED_CHARS,
                    time_used=time_used,
                    time_limit=time_limit,
                )
            else:
                self._display.speech(
                    speaker=agent.name,
                    content=content,
                    word_count=word_count,
                    expected=self._EXPECTED_CHARS,
                    time_used=time_used,
                    time_limit=time_limit,
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
