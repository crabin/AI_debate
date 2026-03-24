from src.config import load_config, load_topics, load_personalities


def test_load_config_returns_all_sections():
    cfg = load_config()
    assert "llm" in cfg
    assert "timer" in cfg
    assert "scoring" in cfg
    assert cfg["llm"]["model"] == "glm-4.7"


def test_load_config_scoring_weights_sum_to_1():
    cfg = load_config()
    weights = cfg["scoring"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_load_topics_returns_list():
    topics = load_topics()
    assert len(topics) >= 5
    assert "title" in topics[0]
    assert "pro_stance" in topics[0]
    assert "con_stance" in topics[0]


def test_load_personalities_returns_dict():
    personalities = load_personalities()
    assert "logical" in personalities
    assert "name" in personalities["logical"]
    assert "prompt" in personalities["logical"]
