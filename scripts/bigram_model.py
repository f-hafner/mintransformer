from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from omegaconf import DictConfig
from omegaconf import OmegaConf
from mintransformer.dataloading.utils import DataConfig
from mintransformer.dataloading.utils import write_data_to_arrow
from mintransformer.main import main

if TYPE_CHECKING:
    from pathlib import Path


logging.basicConfig(level=logging.INFO)


def load_cfg(cfg_path: str | Path) -> DictConfig:
    """Load cfg and make sure it's the right type."""
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig):
        msg = "cfg must be DictConfig"
        raise TypeError(msg)
    return cfg


if __name__ == "__main__":
    cfg = load_cfg("scripts/bigram_cfg.yaml")
    data_config = DataConfig(**cfg["data_config"])
    vocab_size, _ = write_data_to_arrow(data_config)
    main(cfg=cfg, vocab_size=vocab_size)
