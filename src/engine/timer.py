class Timer:
    """Character-count based timer for debate speeches."""

    def __init__(self, chars_per_minute: int = 250, warning_threshold: int = 30):
        self._cpm = chars_per_minute
        self._warning_threshold = warning_threshold

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
