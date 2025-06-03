"""Trainer for single-node, multi-gpu training with DDP."""

from __future__ import annotations
import logging
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
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
    world_size: int
    snapshot_path: Path

    def __post_init__(self):
        if isinstance(self.snapshot_path, str):
            self.snapshot_path = Path(self.snapshot_path)


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
        # DDP
        self.rank_id = int(os.environ["LOCAL_RANK"])
        # initialize train states
        self.optimizer = optimizer
        self.epochs_run = 0

        if self.has_cuda:
            self.model = model.to(self.rank_id)
        else:
            self.model = model

        self._load_snapshot()

        if self.config.world_size > 1:
            logger.debug("Using DDP")
            if self.has_cuda:
                logger.debug("Putting model to %d", self.rank_id)
                self.model = DistributedDataParallel(self.model, device_ids=[self.rank_id])
            else:
                self.model = DistributedDataParallel(self.model, device_ids=None)
        # Data
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset)

        logger.debug(
            "Worker %d initiated; is_head is %s, is_distributed is %s", self.rank_id, self.is_head, self.is_distributed
        )

    @property
    def has_cuda(self) -> bool:
        """Check if cuda is available."""
        return torch.cuda.is_available()

    @property
    def is_distributed(self) -> bool:
        """Check of model is wrapped by DDP."""
        return isinstance(self.model, DistributedDataParallel)

    @property
    def is_head(self) -> bool:
        """Check if current process is the head process."""
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        return (local_world_size > 1 and self.rank_id == 0) or local_world_size == 1

    def _load_snapshot(self) -> None:
        try:
            with self.config.snapshot_path as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            parent_path = self.config.snapshot_path.parent
            if self.is_head and not parent_path.exists():
                logger.info("Creating snapshot directory at %s/.", parent_path)
                parent_path.mkdir(parents=True)
            logger.info("Worker %d | Snapshot not found. Training from scratch.", self.rank_id)
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.epochs_run = snapshot.finished_epoch
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        logger.info("Resuming training from snapshot at epoch %d", self.epochs_run)

    def _save_snapshot(self, epoch: int) -> None:
        """Save model snapshot.

        Args:
            epoch: current counter of epochs.

        Note that running epochs are 0-based indexed at the start
        of each epoch, but upon saving (in this function),
        the counter for "epochs run" is incremented by 1 since
        the current epoch is finished.
        """
        model = self.model
        # If a model is wrapped by DDP, it does not have a "module" attribute
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch + 1,
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)
        logger.info("rank: %d | Snapshot saved at epoch %s", self.rank_id, epoch)

    def _prepare_dataloader(self, dataset: Dataset) -> DataLoader:
        if self.config.world_size > 1:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=DistributedSampler(dataset),
                shuffle=False,
                pin_memory=self.has_cuda,
            )
        else:
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=self.has_cuda)

        return dataloader

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor, train: bool = True) -> float:
        if self.has_cuda:
            # DDP-wrapped model has device attribute, but plain torch model does not
            model_device = "cuda:0" if not hasattr(self.model, "device") else self.model.device
            logger.debug(
                "Worker %s, Devices (model, source, targets): %s, %s, %s",
                self.rank_id,
                model_device,
                source.device,
                targets.device,
            )
        with torch.set_grad_enabled(train):
            _, loss = self.model(source, targets)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True) -> None:
        step_type = "Train" if train else "Test"
        if train and self.is_distributed:
            dataloader.sampler.set_epoch(epoch)

        losses = []
        for it, batch in tqdm(enumerate(dataloader)):
            source, targets = batch
            if self.has_cuda:
                logger.debug("Putting data to %d", self.rank_id)
                source = source.to(self.rank_id)
                targets = targets.to(self.rank_id)
            loss = self._run_batch(source, targets, train=train)
            if not train:
                losses.append(loss)

            if train and it % 100 == 0:
                logger.info("Epoch %d | rank %d | Iter %d | %s Loss %.5f", epoch, self.rank_id, it, step_type, loss)

        if not train:
            avg_loss = sum(losses) / len(losses)
            logger.info("Epoch %d | rank %d | %s Loss %.5f", epoch, self.rank_id, step_type, avg_loss)

    def train(self) -> None:
        """Train model by iterating over training batches."""
        for epoch in range(self.epochs_run, self.config.max_epochs):
            self._run_epoch(epoch, self.train_loader, train=True)

            if self.is_head and epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
