from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
import torch
from torch.distributed import destroy_process_group
from .dataloading import DataConfig
from .dataloading import load_data
from .distributed import ddp_setup
from .models.bigram import BigramLanguageModel
from .models.bigram import BigramModelConfig
from .trainer import Trainer
from .trainer import TrainerConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

torch.manual_seed(1337)


def main(cfg: DictConfig) -> None:
    """Set up and train model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_size = int(os.environ["RANK"])
    data_config = DataConfig(**cfg["data_config"])

    if world_size > 1:
        backend = "nccl" if device == "cuda" else "gloo"
        ddp_setup(backend=backend)

    trainer_cfg = TrainerConfig(world_size=world_size, **cfg["trainer_config"])
    train_dataset, test_dataset, vocab_size, _ = load_data(data_config)

    model_config = BigramModelConfig(**cfg["model_config"], block_size=data_config.block_size, vocab_size=vocab_size)
    model = BigramLanguageModel(model_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer_config.learning_rate)

    trainer = Trainer(trainer_cfg, train_dataset, test_dataset, model, optimizer)
    trainer.train()

    if world_size > 1:
        destroy_process_group()
