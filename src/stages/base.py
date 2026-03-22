"""Base stage class for AI Debate System."""

from abc import ABC, abstractmethod


class BaseStage(ABC):
    """Abstract base class for all debate stages.

    Each stage is responsible for:
    - Executing its debate logic
    - Publishing messages to the pool
    - Returning execution results
    """

    def __init__(self, name: str, description: str, display) -> None:
        """Initialize base stage.

        Args:
            name: Stage identifier (e.g., "opening", "cross_exam")
            description: Human-readable description
            display: TerminalDisplay instance for output
        """
        self.name = name
        self.description = description
        self._display = display

    @abstractmethod
    def execute(self, pool, agents: dict, penalties: dict | None = None) -> dict:
        """Execute the stage.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent
            penalties: Optional penalty configuration

        Returns:
            Result dictionary with status and metadata
        """
        raise NotImplementedError
