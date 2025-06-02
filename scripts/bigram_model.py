from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.multiprocessing.spawn import spawn
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
    world_size = cfg.world_size
    if world_size == 1:
        main(cfg=cfg, rank=1, world_size=world_size)
    else:
        torch.set_num_interop_threads(4)
        spawn(main, args=(cfg, world_size), nprocs=world_size)
