"""
Centralised config loader.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/base.yaml", "configs/data.yaml")
    print(cfg.seed)
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(*config_paths: str | Path) -> DictConfig:
    """Merge one or more YAML config files into a single OmegaConf DictConfig.

    Later files override keys from earlier files.

    Args:
        *config_paths: Paths to YAML config files.

    Returns:
        Merged DictConfig.
    """
    configs = [OmegaConf.load(p) for p in config_paths]
    return OmegaConf.merge(*configs)


def load_all_configs(config_dir: str | Path = "configs") -> DictConfig:
    """Load and merge every YAML file in config_dir in sorted order.

    Provides a single convenience object containing all project settings.
    """
    config_dir = Path(config_dir)
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML files found in {config_dir}")
    return load_config(*paths)
