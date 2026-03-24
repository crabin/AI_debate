"""Judge agent for AI Debate System."""

import json
import logging
import re
from src.llm.base import BaseLLM
from src.agents.base import BaseAgent
from src.agents.prompts import JUDGE_SYSTEM

logger = logging.getLogger(__name__)


def _make_verdict_fallback() -> dict:
    """Return a fresh fallback dict (avoids mutable default sharing)."""
    return {
        "winner_reason": "",
        "topic_conclusion": "",
        "best_debater_reason": "",
        "key_moments": [],
    }

VERDICT_FALLBACK = _make_verdict_fallback  # callable, not a dict


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON content from markdown code blocks.

    Args:
        text: Text that may contain JSON in markdown code blocks

    Returns:
        Extracted JSON string, or original text if no code blocks found
    """
    # Try to extract JSON from markdown code blocks (```json...```)
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


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
        display=None,
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
            display: Optional TerminalDisplay instance for visual output
        """
        super().__init__(agent_id, name, team, role, llm)
        self.topic = topic
        self.pro_stance = pro_stance
        self.con_stance = con_stance
        self._display = display

    @classmethod
    def create(
        cls,
        topic: str,
        pro_stance: str,
        con_stance: str,
        llm: BaseLLM,
        display=None,
    ) -> "JudgeAgent":
        """Factory method to create a judge agent.

        Args:
            topic: Debate topic
            pro_stance: Pro team's stance
            con_stance: Con team's stance
            llm: LLM instance for generating responses
            display: Optional TerminalDisplay instance for visual output

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
            display=display,
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
        # Show thinking indicator if display available
        if self._display:
            self._display.show_judge_thinking(speaker_id)

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
            # Extract JSON from markdown code blocks if present
            json_text = _extract_json_from_markdown(response)
            result = json.loads(json_text)
            # Ensure required fields exist
            if "speaker" not in result:
                result["speaker"] = speaker_id

            # Display score if display available
            if self._display:
                speaker_name = result.get("speaker", speaker_id)
                total = (
                    result.get("logic", 5) +
                    result.get("persuasion", 5) +
                    result.get("expression", 5) +
                    result.get("teamwork", 5) +
                    result.get("rule_compliance", 5)
                ) / 2  # Convert to 50-point scale

                self._display.show_judge_score(
                    speaker_name=speaker_name,
                    logic=result.get("logic", 5),
                    persuasion=result.get("persuasion", 5),
                    expression=result.get("expression", 5),
                    teamwork=result.get("teamwork", 5),
                    rule_compliance=result.get("rule_compliance", 5),
                    total=total,
                    comment=result.get("comment", ""),
                )

            return result
        except json.JSONDecodeError:
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
        # Show thinking indicator if display available
        if self._display:
            self._display.show_judge_review_start()

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
            # Extract JSON from markdown code blocks if present
            json_text = _extract_json_from_markdown(response)
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse review response: {response}")
            # Return default review on parse error
            return {
                "type": "review",
                "summary": "点评生成失败",
                "highlights": [],
                "suggestions": [],
            }

    def generate_verdict(
        self,
        pool,
        winner: str,
        pro_score: float,
        con_score: float,
        best_debater: tuple,
        temperature: float = 0.5,
    ) -> dict:
        """Generate final verdict including topic conclusion.

        Args:
            pool: MessagePool instance (for full transcript context)
            winner: "pro", "con", or "tie"
            pro_score: Final pro team score
            con_score: Final con team score
            best_debater: Tuple of (agent_id, score)
            temperature: Sampling temperature

        Returns:
            Dict with winner_reason, topic_conclusion, best_debater_reason, key_moments.
            Returns VERDICT_FALLBACK on parse failure.
        """
        public_messages = pool.get_messages("public")
        public_context = "\n".join([
            f"{m.speaker}（{m.role}）：{m.content}"
            for m in public_messages
        ]) if public_messages else "暂无发言记录"

        winner_label = {"pro": "正方", "con": "反方", "tie": "平局"}.get(winner, winner)
        best_id, best_score = best_debater if best_debater else ("", 0.0)

        prompt = (
            f"【辩题】{self.topic}\n"
            f"【正方立场】{self.pro_stance}\n"
            f"【反方立场】{self.con_stance}\n"
            f"【比分】正方 {pro_score:.1f} vs 反方 {con_score:.1f}\n"
            f"【获胜方】{winner_label}\n"
            f"【最佳辩手】{best_id} ({best_score:.1f}分)\n\n"
            f"【辩论记录】\n{public_context}\n\n"
            "请输出严格JSON（不要markdown代码块），包含字段：\n"
            "winner_reason（获胜原因，100字以内），\n"
            "topic_conclusion（对辩题的结论，150字以内），\n"
            "best_debater_reason（最佳辩手理由，50字以内），\n"
            "key_moments（关键时刻列表，最多3项）"
        )

        messages = [
            {"role": "system", "content": f"你是辩论赛裁判。辩题：{self.topic}"},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages, temperature=temperature)

        try:
            json_text = _extract_json_from_markdown(response)
            result = json.loads(json_text)
            # Ensure all keys present; start from a fresh fallback dict
            return {**VERDICT_FALLBACK(), **result}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse verdict response: %s", response[:200])
            return VERDICT_FALLBACK()
