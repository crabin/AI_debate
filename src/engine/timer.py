class Timer:
    """Character-count based timer for debate speeches."""

    def __init__(
        self,
        chars_per_minute: int = 250,
        warning_threshold: int = 30,
        total_seconds: float = 0,
    ):
        """Initialize timer.

        Args:
            chars_per_minute: Speaking rate (characters per minute)
            warning_threshold: Warning threshold in seconds
            total_seconds: Total time budget in seconds (for stateful tracking)
        """
        self._cpm = chars_per_minute
        self._warning_threshold = warning_threshold
        self._total_seconds = total_seconds
        self._used_seconds = 0.0

    def estimate_duration(self, text: str) -> float:
        """Estimate speech duration in seconds based on character count."""
        return len(text) / self._cpm * 60

    def check_overtime(self, text: str, limit_seconds: float) -> tuple[bool, float]:
        """Check if text exceeds time limit. Returns (is_overtime, excess_seconds)."""
        duration = self.estimate_duration(text)
        if duration > limit_seconds:
            return True, duration - limit_seconds
        return False, 0.0

    def char_limit(self, seconds: float) -> int:
        """Calculate character limit for a given time duration."""
        return int(self._cpm * seconds / 60)

    def is_warning_zone(self, used_chars: int, limit_chars: int) -> bool:
        """Check if remaining characters are below warning threshold."""
        remaining = limit_chars - used_chars
        return remaining < self._warning_threshold

    def check(self, char_count: int) -> tuple[bool, float]:
        """Check if adding chars would exceed time budget.

        Args:
            char_count: Number of characters to add

        Returns:
            Tuple of (is_overtime, elapsed_seconds)
        """
        elapsed = self.estimate_duration(" " * char_count)
        self._used_seconds += elapsed
        return (self._used_seconds > self._total_seconds, self._used_seconds)

    def time_left(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self._total_seconds - self._used_seconds)

    def is_expired(self) -> bool:
        """Check if time budget is exhausted."""
        return self._used_seconds >= self._total_seconds

    def reset(self) -> None:
        """Reset timer to initial state."""
        self._used_seconds = 0.0
