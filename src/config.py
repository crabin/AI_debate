from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Load default.yaml configuration."""
    with open(config_dir / "default.yaml") as f:
        return yaml.safe_load(f)


def load_topics(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Load debate topics from topics.yaml."""
    with open(config_dir / "topics.yaml") as f:
        data = yaml.safe_load(f)
    return data["topics"]


def load_personalities(config_dir: Path = CONFIG_DIR) -> dict:
    """Load personality templates from personalities.yaml."""
    with open(config_dir / "personalities.yaml") as f:
        data = yaml.safe_load(f)
    return data["personalities"]
