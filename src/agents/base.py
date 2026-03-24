"""Base agent class for AI Debate System."""

import logging
import sys
import time
from typing import Callable
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all debate agents.

    Provides common functionality for building context and speaking.
    """

    _POSITION_NAMES: dict[int, str] = {
        1: "一辩",
        2: "二辩",
        3: "三辩",
        4: "四辩",
    }

    def __init__(
        self,
        agent_id: str,
        name: str,
        team: str,
        role: str,
        llm: BaseLLM,
    ) -> None:
        """Initialize base agent.

        Args:
            agent_id: Unique identifier (e.g., "pro_1", "judge")
            name: Display name (e.g., "正方一辩")
            team: Team identifier ("pro", "con", "judge")
            role: Role name (e.g., "一辩", "评委")
            llm: LLM instance for generating responses
        """
        self.agent_id = agent_id
        self.name = name
        self.team = team
        self.role = role
        self._llm = llm

    @property
    def model_name(self) -> str:
        """Return the model identifier from the underlying LLM."""
        return self._llm.model_name

    def build_context(
        self,
        pool,
        stage: str | None = None,
    ) -> str:
        """Build formatted context string from message pool.

        Args:
            pool: MessagePool instance
            stage: Filter messages by stage (optional)

        Returns:
            Formatted context string with debate history
        """
        messages = pool.get_visible_messages(self.team, stage=stage)

        if not messages:
            return "【辩论记录】\n暂无发言\n"

        context_parts = ["【辩论记录】"]
        for msg in messages:
            context_parts.append(f"{msg.speaker}（{msg.role}）：{msg.content}")

        return "\n".join(context_parts) + "\n"

    def speak(
        self,
        system_prompt: str,
        context: str,
        instruction: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate speech using LLM.

        Args:
            system_prompt: System prompt for the agent
            context: Debate context/history
            instruction: Specific instruction for this speech
            temperature: Sampling temperature

        Returns:
            Generated speech content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n【当前任务】{instruction}"},
        ]

        response = self._llm.chat(messages, temperature=temperature)
        return response.strip()

    def speak_stream(
        self,
        system_prompt: str,
        context: str,
        instruction: str,
        temperature: float = 0.7,
        callback: Callable[[str], None] | None = None,
    ) -> str:
        """Generate speech using LLM with streaming output.

        Args:
            system_prompt: System prompt for the agent
            context: Debate context/history
            instruction: Specific instruction for this speech
            temperature: Sampling temperature
            callback: Optional callback function for each character chunk

        Returns:
            Generated speech content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n【当前任务】{instruction}"},
        ]

        response = self._llm.chat_stream(messages, temperature=temperature, callback=callback)
        return response.strip()
