from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreCard:
    speaker: str
    stage: str
    logic: int
    persuasion: int
    expression: int
    teamwork: int
    rule_compliance: int
    violations: tuple[str, ...]
    comment: str
    # total 和 penalty 由 Scorer 根据 weights 和 violations 计算，不由 Judge 输出


def _team_of(speaker: str) -> str:
    if speaker.startswith("pro"):
        return "pro"
    if speaker.startswith("con"):
        return "con"
    return "judge"


class Scorer:
    """Three-layer scoring system: real-time + rule-engine + aggregation."""

    DEFAULT_WEIGHTS = {
        "logic": 0.25,
        "persuasion": 0.25,
        "expression": 0.20,
        "teamwork": 0.15,
        "rule_compliance": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights or self.DEFAULT_WEIGHTS
        self._cards: list[ScoreCard] = []
        self._individual_penalties: dict[str, list[tuple[float, str]]] = {}
        self._team_penalties: dict[str, list[tuple[float, str]]] = {}

    def compute_speech_score(self, card: ScoreCard) -> float:
        w = self._weights
        return (
            card.logic * w["logic"]
            + card.persuasion * w["persuasion"]
            + card.expression * w["expression"]
            + card.teamwork * w["teamwork"]
            + card.rule_compliance * w["rule_compliance"]
        )

    def record(self, card: ScoreCard) -> float:
        """Record a score card and return the computed speech score."""
        self._cards.append(card)
        return self.compute_speech_score(card)

    def add_individual_penalty(self, speaker: str, penalty: float, reason: str) -> None:
        self._individual_penalties.setdefault(speaker, []).append((penalty, reason))

    def add_team_penalty(self, team: str, penalty: float, reason: str) -> None:
        self._team_penalties.setdefault(team, []).append((penalty, reason))

    def get_team_penalty(self, team: str) -> float:
        return sum(p for p, _ in self._team_penalties.get(team, []))

    def get_individual_total(self, speaker: str) -> float:
        speech_total = sum(
            self.compute_speech_score(c) for c in self._cards if c.speaker == speaker
        )
        penalty_total = sum(
            p for p, _ in self._individual_penalties.get(speaker, [])
        )
        return speech_total + penalty_total

    def get_team_total(self, team: str) -> float:
        members = {c.speaker for c in self._cards if _team_of(c.speaker) == team}
        for speaker in self._individual_penalties:
            if _team_of(speaker) == team:
                members.add(speaker)
        individual_sum = sum(self.get_individual_total(s) for s in members)
        return individual_sum + self.get_team_penalty(team)

    def get_best_debater(self) -> tuple[str, float]:
        speakers = {c.speaker for c in self._cards}
        best_id = ""
        best_score = float("-inf")
        for s in speakers:
            total = self.get_individual_total(s)
            if total > best_score:
                best_score = total
                best_id = s
        return best_id, best_score

    def get_stage_summary(self, stage: str) -> dict[str, float]:
        """Get total scores per team for a specific stage."""
        result: dict[str, float] = {"pro": 0.0, "con": 0.0}
        for card in self._cards:
            if card.stage == stage:
                team = _team_of(card.speaker)
                if team in result:
                    result[team] += self.compute_speech_score(card)
        return result

    def export(self) -> dict:
        """Export full scoring data for debate log."""
        return {
            "cards": [
                {
                    "speaker": c.speaker, "stage": c.stage,
                    "logic": c.logic, "persuasion": c.persuasion,
                    "expression": c.expression, "teamwork": c.teamwork,
                    "rule_compliance": c.rule_compliance,
                    "violations": list(c.violations), "comment": c.comment,
                    "total": self.compute_speech_score(c),
                }
                for c in self._cards
            ],
            "individual_penalties": {
                k: [(p, r) for p, r in v]
                for k, v in self._individual_penalties.items()
            },
            "team_penalties": {
                k: [(p, r) for p, r in v]
                for k, v in self._team_penalties.items()
            },
        }
