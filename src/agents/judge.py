"""Judge agent for AI Debate System."""

import json
import logging
from src.llm.base import BaseLLM
from src.agents.base import BaseAgent
from src.agents.prompts import JUDGE_SYSTEM

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """Judge agent that scores and reviews debate performances.

    Responsible for:
    - Scoring each speaker across 5 dimensions
    - Detecting rule violations
    - Providing end-of-debate review
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        team: str,
        role: str,
        topic: str,
        pro_stance: str,
        con_stance: str,
        llm: BaseLLM,
    ) -> None:
        """Initialize judge agent.

        Args:
            agent_id: Unique identifier ("judge")
            name: Display name ("评委")
            team: Team identifier ("judge")
            role: Role name ("评委")
            topic: Debate topic
            pro_stance: Pro team's stance
            con_stance: Con team's stance
            llm: LLM instance for generating responses
        """
        super().__init__(agent_id, name, team, role, llm)
        self.topic = topic
        self.pro_stance = pro_stance
        self.con_stance = con_stance

    @classmethod
    def create(
        cls,
        topic: str,
        pro_stance: str,
        con_stance: str,
        llm: BaseLLM,
    ) -> "JudgeAgent":
        """Factory method to create a judge agent.

        Args:
            topic: Debate topic
            pro_stance: Pro team's stance
            con_stance: Con team's stance
            llm: LLM instance for generating responses

        Returns:
            Configured JudgeAgent instance
        """
        return cls(
            agent_id="judge",
            name="评委",
            team="judge",
            role="评委",
            topic=topic,
            pro_stance=pro_stance,
            con_stance=con_stance,
            llm=llm,
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for judge."""
        return JUDGE_SYSTEM.format(
            topic=self.topic,
            pro_stance=self.pro_stance,
            con_stance=self.con_stance,
            public_messages="",
            judge_notes="",
            current_instruction="",
        )

    def score_speaker(
        self,
        pool,
        speaker_id: str,
        temperature: float = 0.5,
    ) -> dict:
        """Score a speaker's performance.

        Args:
            pool: MessagePool instance
            speaker_id: ID of speaker to score
            temperature: Sampling temperature

        Returns:
            Dictionary with scores and violations:
            {
                "speaker": str,
                "logic": int (1-10),
                "persuasion": int (1-10),
                "expression": int (1-10),
                "teamwork": int (1-10),
                "rule_compliance": int (1-10),
                "violations": list[str],
                "comment": str,
            }
        """
        public_messages = pool.get_messages("public")
        judge_notes = pool.get_messages("judge_notes")

        public_context = "\n".join([
            f"{m.speaker}（{m.role}）：{m.content}"
            for m in public_messages
        ]) if public_messages else "暂无发言记录"

        notes_context = "\n".join([
            f"{m.speaker}：{m.content}"
            for m in judge_notes
        ]) if judge_notes else "暂无评分记录"

        system_prompt = JUDGE_SYSTEM.format(
            topic=self.topic,
            pro_stance=self.pro_stance,
            con_stance=self.con_stance,
            public_messages=public_context,
            judge_notes=notes_context,
            current_instruction=f"请为 {speaker_id} 的发言评分。",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请为 {speaker_id} 的发言评分。"},
        ]

        response = self._llm.chat(messages, temperature=temperature)

        # Parse JSON response
        try:
            result = json.loads(response.strip())
            # Ensure required fields exist
            if "speaker" not in result:
                result["speaker"] = speaker_id
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response: {response}")
            # Return default scores on parse error
            return {
                "speaker": speaker_id,
                "logic": 5,
                "persuasion": 5,
                "expression": 5,
                "teamwork": 5,
                "rule_compliance": 5,
                "violations": ["parse_error"],
                "comment": "评分解析失败",
            }

    def generate_review(
        self,
        pool,
        temperature: float = 0.7,
    ) -> dict:
        """Generate end-of-debate review.

        Args:
            pool: MessagePool instance
            temperature: Sampling temperature

        Returns:
            Dictionary with review:
            {
                "type": "review",
                "summary": str,
                "highlights": list[str],
                "suggestions": list[str],
            }
        """
        public_messages = pool.get_messages("public")

        public_context = "\n".join([
            f"{m.speaker}（{m.role}）：{m.content}"
            for m in public_messages
        ]) if public_messages else "暂无发言记录"

        system_prompt = JUDGE_SYSTEM.format(
            topic=self.topic,
            pro_stance=self.pro_stance,
            con_stance=self.con_stance,
            public_messages=public_context,
            judge_notes="",
            current_instruction="请对本场辩论进行点评。",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请对本场辩论进行点评，输出JSON格式。"},
        ]

        response = self._llm.chat(messages, temperature=temperature)

        # Parse JSON response
        try:
            result = json.loads(response.strip())
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse review response: {response}")
            # Return default review on parse error
            return {
                "type": "review",
                "summary": "点评生成失败",
                "highlights": [],
                "suggestions": [],
            }
