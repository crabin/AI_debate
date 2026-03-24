"""Debater agent for AI Debate System."""

import logging
from dataclasses import dataclass
from src.llm.base import BaseLLM
from src.agents.base import BaseAgent
from src.agents.prompts import DEBATER_PROMPTS, COMMON_RULES

logger = logging.getLogger(__name__)


# Personality templates mapped to keys
_PERSONALITY_PROMPTS: dict[str, str] = {
    "logical": "你以逻辑严密著称，善于构建完整的论证框架，用数据和事实说话。",
    "aggressive": "你言辞犀利，善于抓住对方漏洞进行猛烈攻击，攻势凌厉。",
    "emotional": "你擅长情感共鸣，用感人的故事和生动的比喻打动人心。",
    "moderate": "你稳健理性，善于在不同观点之间寻找平衡点。",
    "humorous": "你幽默风趣，善于用轻松诙谐的方式化解尖锐对立。",
}


@dataclass(frozen=True)
class DebaterConfig:
    """Configuration for creating a debater agent."""

    position: int  # 1-4
    team: str  # "pro" or "con"
    stance: str  # Team's stance on the topic
    topic: str  # Debate topic
    personality: str  # Personality key


# Chinese number to int mapping
_CHINESE_NUMBERS: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
}


class DebaterAgent(BaseAgent):
    """Debater agent that participates in debate stages.

    Created via factory method `create()` with frozen config.
    """

    _TEAM_NAMES: dict[str, str] = {
        "pro": "正方",
        "con": "反方",
    }

    def __init__(
        self,
        agent_id: str,
        name: str,
        team: str,
        role: str,
        stance: str,
        topic: str,
        personality: str,
        position: int,
        llm: BaseLLM,
    ) -> None:
        """Initialize debater agent.

        Args:
            agent_id: Unique identifier (e.g., "pro_1")
            name: Display name (e.g., "正方一辩")
            team: Team identifier ("pro" or "con")
            role: Role name ("一辩", "二辩", "三辩", "四辩")
            stance: Team's stance on the topic
            topic: Debate topic
            personality: Personality template key
            position: Position number (1-4)
            llm: LLM instance for generating responses
        """
        super().__init__(agent_id, name, team, role, llm)
        self.stance = stance
        self.topic = topic
        self.personality = personality
        self._position = position

    @property
    def position(self) -> int:
        """Return the debater's position number (1-4)."""
        return self._position

    @classmethod
    def create(
        cls,
        position: int,
        team: str,
        stance: str,
        topic: str,
        personality: str,
        llm: BaseLLM,
    ) -> "DebaterAgent":
        """Factory method to create a debater agent.

        Args:
            position: Debater position (1-4)
            team: Team identifier ("pro" or "con")
            stance: Team's stance on the topic
            topic: Debate topic
            personality: Personality template key
            llm: LLM instance for generating responses

        Returns:
            Configured DebaterAgent instance
        """
        if position not in range(1, 5):
            raise ValueError(f"Invalid position: {position}. Must be 1-4.")

        if team not in cls._TEAM_NAMES:
            raise ValueError(f"Invalid team: {team}. Must be 'pro' or 'con'.")

        role = cls._POSITION_NAMES[position]
        team_name = cls._TEAM_NAMES[team]
        agent_id = f"{team}_{position}"
        name = f"{team_name}{role}"

        return cls(
            agent_id=agent_id,
            name=name,
            team=team,
            role=role,
            stance=stance,
            topic=topic,
            personality=personality,
            position=position,
            llm=llm,
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for this debater's position."""
        template = DEBATER_PROMPTS[self._position]
        personality_prompt = _PERSONALITY_PROMPTS.get(
            self.personality,
            _PERSONALITY_PROMPTS["logical"],
        )

        return template.format(
            team=self._TEAM_NAMES[self.team],
            stance=self.stance,
            topic=self.topic,
            personality_prompt=personality_prompt,
            common_rules=COMMON_RULES,
        )

    def generate_opening_statement(self, pool, temperature: float = 0.7, callback=None) -> str:
        """Generate opening statement for position 1 debater.

        Args:
            pool: MessagePool instance
            temperature: Sampling temperature
            callback: Optional callback for streaming output

        Returns:
            Generated opening statement
        """
        system_prompt = self._get_system_prompt()
        context = self.build_context(pool, stage="opening")
        instruction = "请进行立论陈词。"

        if callback:
            return self.speak_stream(system_prompt, context, instruction, temperature, callback)
        return self.speak(system_prompt, context, instruction, temperature)

    def generate_cross_exam_question(
        self,
        pool,
        target_opponent: str,
        temperature: float = 0.8,
        callback=None,
    ) -> str:
        """Generate cross-examination questions.

        Args:
            pool: MessagePool instance
            target_opponent: Target opponent ID (e.g., "con_2")
            temperature: Sampling temperature
            callback: Optional callback for streaming output

        Returns:
            Generated questions
        """
        system_prompt = self._get_system_prompt()
        context = self.build_context(pool, stage="cross_exam")
        instruction = f"请向{target_opponent}提问。"

        if callback:
            return self.speak_stream(system_prompt, context, instruction, temperature, callback)
        return self.speak(system_prompt, context, instruction, temperature)

    def generate_cross_exam_answer(
        self,
        pool,
        question: str,
        temperature: float = 0.7,
        callback=None,
    ) -> str:
        """Generate answer to cross-examination question.

        Args:
            pool: MessagePool instance
            question: The question being answered
            temperature: Sampling temperature
            callback: Optional callback for streaming output

        Returns:
            Generated answer
        """
        system_prompt = self._get_system_prompt()
        context = self.build_context(pool, stage="cross_exam")
        instruction = f"请回答以下问题：\n{question}"

        if callback:
            return self.speak_stream(system_prompt, context, instruction, temperature, callback)
        return self.speak(system_prompt, context, instruction, temperature)

    def generate_free_debate_speech(
        self,
        pool,
        recent_context: str,
        temperature: float = 0.8,
        callback=None,
    ) -> str:
        """Generate free debate speech.

        Args:
            pool: MessagePool instance
            recent_context: Recent debate context
            temperature: Sampling temperature
            callback: Optional callback for streaming output

        Returns:
            Generated speech
        """
        system_prompt = self._get_system_prompt()
        context = self.build_context(pool, stage="free_debate")
        full_context = f"{context}\n{recent_context}"
        instruction = "请进行自由辩论发言。"

        if callback:
            return self.speak_stream(system_prompt, full_context, instruction, temperature, callback)
        return self.speak(system_prompt, full_context, instruction, temperature)

    def generate_cross_exam_summary(
        self,
        pool,
        temperature: float = 0.7,
        callback=None,
    ) -> str:
        """Generate cross-examination summary (for position 1 debaters).

        Args:
            pool: MessagePool instance
            temperature: Sampling temperature
            callback: Optional callback for streaming output

        Returns:
            Generated summary
        """
        system_prompt = self._get_system_prompt()
        # Get only cross_exam messages for context
        context = self.build_context(pool, stage="cross_exam")
        instruction = "请进行攻辩小结，必须引用攻辩阶段的实际发言内容。"

        if callback:
            return self.speak_stream(system_prompt, context, instruction, temperature, callback)
        return self.speak(system_prompt, context, instruction, temperature)

    def generate_closing_statement(
        self,
        pool,
        temperature: float = 0.7,
        callback=None,
    ) -> str:
        """Generate closing statement for position 4 debater.

        Args:
            pool: MessagePool instance
            temperature: Sampling temperature
            callback: Optional callback for streaming output

        Returns:
            Generated closing statement
        """
        system_prompt = self._get_system_prompt()
        context = self.build_context(pool)
        instruction = "请进行总结陈词。"

        if callback:
            return self.speak_stream(system_prompt, context, instruction, temperature, callback)
        return self.speak(system_prompt, context, instruction, temperature)
