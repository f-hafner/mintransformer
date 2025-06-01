from __future__ import annotations
import logging
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Basic parameters for trainer."""
    max_epochs: int
    iter_per_epoch: int
    batch_size: int
    data_loader_workers: int
    grad_norm_clip: float
    save_every: int
    snapshot_path: Path = Path("./")

@dataclass
class Snapshot:
    """Snapshot for model and optimizer states."""
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    finished_epoch: int


class Trainer:
    """Class to train models."""
    def __init__(
            self,
            trainer_config: TrainerConfig,
            train_dataset: Dataset,
            test_dataset: Dataset,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            ):
        self.config = trainer_config
        # Data
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset)
        # other
        self.model = model
        self.optimizer = optimizer


    def _prepare_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.config.batch_size)


    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor, train: bool = True) -> float:
        with torch.set_grad_enabled(train):
            _, loss = self.model(source, targets)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss.item()


    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True) -> None:
        step_type = "Train" if train else "Test"
        for it, batch in tqdm(enumerate(dataloader)):
            source, targets = batch
            loss = self._run_batch(source, targets, train=train)
            if it % 100 == 0:
                logger.info("Epoch %d | Iter %d | %s Loss %.5f", epoch, it, step_type, loss)


    def _save_snapshot(self, epoch: int) -> None:
        model = self.model
        # If a model is wrapped by DDP, it does not have a "module" attribute
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
                model_state=raw_model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                finished_epoch=epoch,
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, Path(self.config.snapshot_path / f"epoch_{epoch}.ckpt"))
        logger.info("Snapshot saved at epoch %s", epoch)



    def train(self) -> None:
        """Train model by iterating over training batches."""
        for epoch in range(self.config.max_epochs):
            self._run_epoch(epoch, self.train_loader, train=True)

            if epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)







