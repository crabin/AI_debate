from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

__all__ = ["load_config", "load_topics", "load_personalities", "CONFIG_DIR"]

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Load default.yaml configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is malformed.
    """
    config_path = config_dir / "default.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {config_path}")
        raise ValueError(f"Invalid YAML: {e}") from e


def load_topics(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Load debate topics from topics.yaml.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is malformed.
    """
    topics_path = config_dir / "topics.yaml"
    try:
        with open(topics_path) as f:
            data = yaml.safe_load(f)
        return data["topics"]
    except FileNotFoundError:
        logger.error(f"Topics file not found: {topics_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in topics file: {topics_path}")
        raise ValueError(f"Invalid YAML: {e}") from e


def load_personalities(config_dir: Path = CONFIG_DIR) -> dict:
    """Load personality templates from personalities.yaml.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is malformed.
    """
    personalities_path = config_dir / "personalities.yaml"
    try:
        with open(personalities_path) as f:
            data = yaml.safe_load(f)
        return data["personalities"]
    except FileNotFoundError:
        logger.error(f"Personalities file not found: {personalities_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in personalities file: {personalities_path}")
        raise ValueError(f"Invalid YAML: {e}") from e
