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
    max_iter: int
    batch_size: int
    data_loader_workers: int
    grad_norm_clip: float
    save_every: int
    eval_interval: int
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
        self.test_loader = iter(self._prepare_dataloader(test_dataset)) # TODO: clean up
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


    def train(self) -> None:
        """Train model."""
        for it, batch in enumerate(self.train_loader):
            if it > self.config.max_iter:
                break

            source, targets = batch
            train_loss = self._run_batch(source, targets, train=True)
            if it % self.config.eval_interval == 0:
                test_source, test_target = next(self.test_loader)
                test_loss = self._run_batch(test_source, test_target, train=False)
                msg = f"Step {it}, train loss: {train_loss}, test loss: {test_loss}"
                logger.info(msg)







