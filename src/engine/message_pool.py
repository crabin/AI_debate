from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Message:
    speaker: str
    role: str
    team: str
    stage: str
    content: str
    msg_type: str
    timestamp: float
    word_count: int
    metadata: tuple


# Channel -> team visibility mapping
_VISIBILITY: dict[str, list[str]] = {
    "pro": ["public", "team_pro"],
    "con": ["public", "team_con"],
    "judge": ["public", "judge_notes"],
}

_VALID_CHANNELS = {"public", "team_pro", "team_con", "judge_notes"}


class MessagePool:
    """Dual-layer message pool with channel-based access control."""

    def __init__(self) -> None:
        self._channels: dict[str, list[Message]] = {ch: [] for ch in _VALID_CHANNELS}

    def publish(self, channel: str, message: Message) -> None:
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"Invalid channel: {channel}")
        self._channels[channel].append(message)

    def get_messages(
        self,
        channel: str,
        stage: Optional[str] = None,
    ) -> list[Message]:
        msgs = self._channels.get(channel, [])
        if stage is not None:
            msgs = [m for m in msgs if m.stage == stage]
        return list(msgs)

    def get_visible_messages(
        self,
        team: str,
        stage: Optional[str] = None,
    ) -> list[Message]:
        """Get all messages visible to a team, sorted by timestamp."""
        channels = _VISIBILITY.get(team, ["public"])
        result: list[Message] = []
        for ch in channels:
            result.extend(self.get_messages(ch, stage=stage))
        result.sort(key=lambda m: m.timestamp)
        return result

    def export(self) -> dict:
        """Export all public messages for debate log."""
        return {
            ch: [
                {
                    "speaker": m.speaker,
                    "role": m.role,
                    "team": m.team,
                    "stage": m.stage,
                    "content": m.content,
                    "msg_type": m.msg_type,
                    "word_count": m.word_count,
                }
                for m in msgs
            ]
            for ch, msgs in self._channels.items()
        }
