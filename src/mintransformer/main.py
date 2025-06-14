from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
import torch
from torch.distributed import destroy_process_group
from .dataloading.utils import load_datasets
from .distributed import ddp_setup
from .models.bigram import BigramLanguageModel
from .models.bigram import BigramModelConfig
from .trainer import Trainer
from .trainer import TrainerConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

torch.manual_seed(1337)


def main(cfg: DictConfig, vocab_size: int) -> None:
    """Set up and train model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.debug("World size: %d", local_world_size)
    logger.debug("Local rank: %d", local_rank)

    if local_world_size > 1:
        backend = "nccl" if device == "cuda" else "gloo"
        ddp_setup(backend=backend)

    trainer_cfg = TrainerConfig(world_size=local_world_size, **cfg["trainer_config"])
    train_dataset, test_dataset = load_datasets()
    sample = next(iter(train_dataset))
    block_size = sample[0].shape[0]

    model_config = BigramModelConfig(**cfg["model_config"], block_size=block_size, vocab_size=vocab_size)
    model = BigramLanguageModel(model_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer_config.learning_rate)

    trainer = Trainer(trainer_cfg, train_dataset, test_dataset, model, optimizer)
    trainer.train()

    if local_world_size > 1:
        destroy_process_group()
