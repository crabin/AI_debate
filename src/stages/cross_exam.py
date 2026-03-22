"""Cross-examination stage for AI Debate System."""

import time
import logging
from src.stages.base import BaseStage
from src.engine.message_pool import Message

logger = logging.getLogger(__name__)


class CrossExamStage(BaseStage):
    """Cross-examination stage (攻辩阶段).

    Four rounds of Q&A:
    1. Pro 2nd -> Con 2nd or 3rd
    2. Con 2nd -> Pro 2nd or 3rd
    3. Pro 3rd -> Con 2nd or 3rd
    4. Con 3rd -> Pro 2nd or 3rd

    Then summary from Pro 1st and Con 1st.

    Question: 30 seconds (125 chars)
    Answer: 1 minute (250 chars)
    Summary: 2 minutes (500 chars)
    """

    # Cross-exam pairs: (attacker, defender_team, expected_chars)
    _CROSS_EXAM_ROUNDS = [
        ("pro_2", "con", 125),  # Pro 2nd asks, Con answers
        ("con_2", "pro", 125),  # Con 2nd asks, Pro answers
        ("pro_3", "con", 125),  # Pro 3rd asks, Con answers
        ("con_3", "pro", 125),  # Con 3rd asks, Pro answers
    ]

    _SUMMARY_ROUNDS = [
        ("pro_1", 500),
        ("con_1", 500),
    ]

    # Available targets for cross-exam by team
    _VALID_TARGETS = {
        "con": ["con_2", "con_3"],
        "pro": ["pro_2", "pro_3"],
    }

    def __init__(self, display) -> None:
        """Initialize cross-exam stage.

        Args:
            display: TerminalDisplay instance for output
        """
        super().__init__(
            name="cross_exam",
            description="攻辩阶段：四轮攻辩 + 双方一辩小结",
            display=display,
        )

    @classmethod
    def create(cls, display) -> "CrossExamStage":
        """Factory method to create a CrossExamStage.

        Args:
            display: TerminalDisplay instance for output

        Returns:
            Configured CrossExamStage instance
        """
        return cls(display=display)

    def _select_target(self, agents: dict, defender_team: str) -> str:
        """Select target for cross-exam.

        Args:
            agents: Dictionary of agent_id -> Agent
            defender_team: Team being attacked ("pro" or "con")

        Returns:
            Target agent ID
        """
        valid_targets = self._VALID_TARGETS[defender_team]
        available = [t for t in valid_targets if t in agents]

        if not available:
            raise ValueError(f"No valid targets for {defender_team}")

        # For simplicity, select the first available
        # In production, the attacker would choose
        return available[0]

    def _extract_target_from_response(self, response: str, default_target: str) -> str:
        """Extract target choice from attacker's response.

        Args:
            response: Attacker's response which may contain target choice
            default_target: Default if no choice found

        Returns:
            Selected target agent ID
        """
        # Check for [选择: 对方X辩] pattern
        if "[选择:" in response:
            try:
                start = response.index("[选择:")
                end = response.index("]", start)
                choice = response[start + 4:end].strip()
                # Map Chinese to agent IDs
                if "对方二辩" in choice or "二辩" in choice:
                    return "con_2" if "con" in default_target else "pro_2"
                elif "对方三辩" in choice or "三辩" in choice:
                    return "con_3" if "con" in default_target else "pro_3"
            except (ValueError, IndexError):
                pass
        return default_target

    def execute(
        self,
        pool,
        agents: dict,
        penalties: dict | None = None,
    ) -> dict:
        """Execute cross-examination stage.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent
            penalties: Optional penalty configuration

        Returns:
            Result dictionary with status and round count
        """
        self._display.stage_start(self.name, self.description)

        messages_published = 0
        judge_agent = agents.get("judge")

        # Four rounds of cross-exam
        for attacker_id, defender_team, expected_chars in self._CROSS_EXAM_ROUNDS:
            if attacker_id not in agents:
                logger.warning(f"Attacker {attacker_id} not available, skipping")
                continue

            attacker = agents[attacker_id]
            target_id = self._select_target(agents, defender_team)
            defender = agents[target_id]

            # Attacker asks questions
            self._display.speech(
                speaker=attacker.name,
                content=f"向{defender.name}提问...",
                word_count=0,
                expected=expected_chars,
            )

            questions = attacker.generate_cross_exam_question(
                pool,
                target_opponent=defender.agent_id,
            )

            # Extract target choice if present
            target_id = self._extract_target_from_response(questions, target_id)
            defender = agents.get(target_id, defender)

            # Publish question
            question_msg = Message(
                speaker=attacker.agent_id,
                role=attacker.role,
                team=attacker.team,
                stage=self.name,
                content=questions,
                msg_type="question",
                timestamp=time.time(),
                word_count=len(questions),
                metadata=("target", target_id),
            )
            pool.publish("public", question_msg)
            messages_published += 1

            self._display.speech(
                speaker=attacker.name,
                content=questions,
                word_count=len(questions),
                expected=expected_chars,
            )

            # Defender answers
            answer_expected = 250  # 1 minute
            self._display.speech(
                speaker=defender.name,
                content="正在回答...",
                word_count=0,
                expected=answer_expected,
            )

            answer = defender.generate_cross_exam_answer(pool, questions)

            # Publish answer
            answer_msg = Message(
                speaker=defender.agent_id,
                role=defender.role,
                team=defender.team,
                stage=self.name,
                content=answer,
                msg_type="answer",
                timestamp=time.time(),
                word_count=len(answer),
                metadata=("in_response_to", attacker.agent_id),
            )
            pool.publish("public", answer_msg)
            messages_published += 1

            self._display.speech(
                speaker=defender.name,
                content=answer,
                word_count=len(answer),
                expected=answer_expected,
            )

            # Judge scoring if available
            if judge_agent:
                try:
                    score = judge_agent.score_speaker(pool, attacker.agent_id)
                    judge_msg = Message(
                        speaker="judge",
                        role="评委",
                        team="judge",
                        stage=self.name,
                        content=str(score),
                        msg_type="score",
                        timestamp=time.time(),
                        word_count=0,
                        metadata=("score", attacker.agent_id),
                    )
                    pool.publish("judge_notes", judge_msg)
                except Exception as e:
                    logger.warning(f"Judge scoring failed: {e}")

        # Summary rounds
        for summarizer_id, expected_chars in self._SUMMARY_ROUNDS:
            if summarizer_id not in agents:
                logger.warning(f"Summarizer {summarizer_id} not available, skipping")
                continue

            summarizer = agents[summarizer_id]

            self._display.speech(
                speaker=summarizer.name,
                content="攻辩小结...",
                word_count=0,
                expected=expected_chars,
            )

            # Generate summary based on cross-exam messages
            summary = summarizer.generate_cross_exam_summary(pool)

            # Publish summary
            summary_msg = Message(
                speaker=summarizer.agent_id,
                role=summarizer.role,
                team=summarizer.team,
                stage=self.name,
                content=summary,
                msg_type="summary",
                timestamp=time.time(),
                word_count=len(summary),
                metadata=(),
            )
            pool.publish("public", summary_msg)
            messages_published += 1

            self._display.speech(
                speaker=summarizer.name,
                content=summary,
                word_count=len(summary),
                expected=expected_chars,
            )

        self._display.stage_end(self.name)

        return {
            "status": "completed",
            "stage": self.name,
            "messages_count": messages_published,
        }
