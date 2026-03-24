import json
from pathlib import Path
import time
from src.export import save_debate_json
from src.engine.message_pool import MessagePool, Message
from src.engine.scorer import Scorer, ScoreCard


def _make_pool_with_messages() -> MessagePool:
    pool = MessagePool()
    msg = Message(
        speaker="pro_1", role="一辩", team="pro",
        stage="opening", content="正方立论", msg_type="speech",
        timestamp=time.time(), word_count=100, metadata=(),
    )
    pool.publish("public", msg)
    return pool


def _make_scorer_with_card() -> Scorer:
    scorer = Scorer()
    card = ScoreCard(
        speaker="pro_1", stage="opening",
        logic=8, persuasion=7, expression=8, teamwork=7, rule_compliance=9,
        violations=(), comment="不错",
    )
    scorer.record(card)
    return scorer


def test_save_debate_json_creates_valid_file(tmp_path):
    pool = _make_pool_with_messages()
    scorer = _make_scorer_with_card()
    results = {
        "winner": "pro", "pro_score": 12.0, "con_score": 10.0,
        "margin": 2.0, "best_debater": ("pro_1", 7.8),
        "winner_reason": "正方更好", "topic_conclusion": "科技优先",
        "best_debater_reason": "表现突出", "key_moments": [],
        "review": None,
    }
    out = tmp_path / "debate.json"
    save_debate_json(results, pool, scorer, out, start_time=time.time() - 5)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["transcript"]) == 1
    assert data["transcript"][0]["content"] == "正方立论"
    assert len(data["scores"]) == 1
    assert data["scores"][0]["speaker"] == "pro_1"
    assert "total" in data["scores"][0]
    assert data["result"]["winner"] == "pro"


def test_save_debate_json_excludes_private_scorer_key(tmp_path):
    pool = MessagePool()
    scorer = Scorer()
    results = {
        "winner": "tie", "pro_score": 0.0, "con_score": 0.0,
        "margin": 0.0, "best_debater": ("", 0.0),
        "_scorer": scorer,  # should be stripped
    }
    out = tmp_path / "debate.json"
    save_debate_json(results, pool, scorer, out, start_time=time.time())

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "_scorer" not in data
    assert "_scorer" not in data.get("result", {})
