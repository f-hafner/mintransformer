from __future__ import annotations
import logging
from pathlib import Path
import torch
from torch.distributed import destroy_process_group
from .dataloading import load_data
from .distributed import ddp_setup
from .models.bigram import BigramLanguageModel
from .trainer import Trainer
from .trainer import TrainerConfig

logger = logging.getLogger(__name__)

block_size = 8  # context length
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)


def main(rank: int, world_size: int, input_path: Path) -> None:
    """Set up and train model."""
    if world_size > 1:
        backend = "nccl" if device == "cuda" else "gloo"
        ddp_setup(rank, world_size, backend=backend)

    # TODO: some are unused, also put all params into one location
    trainer_cfg = TrainerConfig(
        max_epochs=3,
        iter_per_epoch=1000,
        batch_size=32,
        data_loader_workers=1,
        save_every=1,
        grad_norm_clip=0.5,  # unused atm
        world_size=world_size,
        snapshot_path=Path("snapshots/"),
    )

    train_dataset, test_dataset, vocab_size, _ = load_data(input_path, block_size)

    model = BigramLanguageModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, device=device)
    model = model.to(device)  # important for GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(trainer_cfg, train_dataset, test_dataset, model, optimizer, rank)
    trainer.train()
    if world_size > 1:
        destroy_process_group()
