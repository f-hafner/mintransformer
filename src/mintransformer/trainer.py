from __future__ import annotations
import logging
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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
    snapshot_path: str = ""

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
        for it, batch in enumerate(dataloader):
            source, targets = batch
            loss = self._run_batch(source, targets, train=train)
            if it % 100 == 0:
                msg = f"Epoch {epoch} | Iter {it} | {step_type} Loss {loss:.5f}"
                logger.info(msg)



    def train(self) -> None:
        """Train model by iterating over training batches."""
        for epoch in range(self.config.max_epochs):
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)







