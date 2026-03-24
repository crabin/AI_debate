"""JSON export for AI Debate System."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from src.engine.message_pool import MessagePool
from src.engine.scorer import Scorer

logger = logging.getLogger(__name__)

__all__ = ["save_debate_json"]

# Keys from results dict that should not appear in the JSON output
_PRIVATE_KEYS = {"_scorer", "review"}


def save_debate_json(
    results: dict,
    pool: MessagePool,
    scorer: Scorer,
    path: Path,
    start_time: float,
) -> None:
    """Serialize full debate results to a JSON file.

    Args:
        results: Dict returned by StageController.run_debate()
        pool: MessagePool containing all messages
        scorer: Scorer instance with all recorded ScoreCards
        path: Output file path
        start_time: Unix timestamp when the debate started (from time.time())
    """
    import time

    duration = time.time() - start_time
    now_iso = datetime.now(timezone.utc).isoformat()

    # Build transcript from public channel
    transcript = [
        {
            "stage": m.stage,
            "speaker": m.speaker,
            "role": m.role,
            "team": m.team,
            "content": m.content,
            "word_count": m.word_count,
            "timestamp": m.timestamp,
            "msg_type": m.msg_type,
        }
        for m in pool.get_messages("public")
    ]

    # Build scores from scorer export (already has total computed)
    scorer_data = scorer.export()
    scores = scorer_data["cards"]  # list of dicts with total

    # Build result (strip private keys)
    result_out = {
        k: v for k, v in results.items()
        if k not in _PRIVATE_KEYS
    }
    # best_debater is a tuple — convert to dict for JSON
    if "best_debater" in result_out and isinstance(result_out["best_debater"], tuple):
        bd = result_out["best_debater"]
        result_out["best_debater"] = {"speaker": bd[0], "score": bd[1]} if bd else None

    # Convert key_moments list (ensure JSON-serializable)
    if "key_moments" not in result_out:
        result_out["key_moments"] = []

    # Extract topic metadata from result if present (put there by run_debate())
    meta = {
        "topic": results.get("topic", ""),
        "pro_stance": results.get("pro_stance", ""),
        "con_stance": results.get("con_stance", ""),
        "timestamp": now_iso,
        "duration_seconds": round(duration, 1),
    }

    payload = {
        "meta": meta,
        "transcript": transcript,
        "scores": scores,
        "result": result_out,
    }

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Debate saved to %s", path)
    except OSError as e:
        logger.error("Failed to save debate JSON to %s: %s", path, e)
        print(f"[警告] 无法保存辩论记录：{e}")
