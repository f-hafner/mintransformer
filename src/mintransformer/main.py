from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import torch
from .dataloading import load_data
from .models.bigram import BigramLanguageModel
from .trainer import Trainer
from .trainer import TrainerConfig

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

block_size = 8 # what is the maximum context length for predictions?
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

def main(input_path: Path) -> None:
    """Set up and train model."""
    # TODO: some are unused, also put all params into one location
    trainer_cfg = TrainerConfig(
            max_iter=1000,
            batch_size=32,
            data_loader_workers=1,
            eval_interval=300,
            save_every=0,
            grad_norm_clip=0.5)

    train_dataset, test_dataset, vocab_size, _ = load_data(input_path, block_size)

    model = BigramLanguageModel(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device)
    model = model.to(device) # important for GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(trainer_cfg, train_dataset, test_dataset, model, optimizer)
    trainer.train()
