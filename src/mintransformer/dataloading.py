from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


@dataclass
class DataConfig:
    """Container for character dataset configuration.

    Args:
        block_size (int): Context length to consider in a sequence.
        train_split (float): Fraction to use for train split (vs test split).
        input_path (Path | str): Path to the data. Strings are
        converted to Path post-initialization.
    """

    block_size: int
    train_split: float
    input_file: Path

    def __post_init__(self):
        if isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)


def load_data(cfg: DataConfig) -> tuple[Dataset, Dataset, int, Callable]:
    """Load dataset.

    Returns:
        A tuple of train dataset, test dataset, vocabulary size, and
        function to decode integers (useful for generating new sequences.)

    """
    data_dict, vocab_size, decode_fct = read_and_prepare_data(cfg)

    train_dataset = TensorDataset(data_dict["x_train"], data_dict["y_train"])
    test_dataset = TensorDataset(data_dict["x_test"], data_dict["y_test"])

    return train_dataset, test_dataset, vocab_size, decode_fct


def read_and_prepare_data(cfg: DataConfig) -> tuple[dict[str, torch.Tensor], int, Callable[[list[int]], str]]:
    """Load data from file and prepare for training.

    Read data from file, create mapping from characters to integers.
    Splits into train and test, and reshape into (n_batches, block_size).

    Returns:
        A tuple consistent of a dict with train and test Xs and Ys, the vocabulary
        size, and the decode function.
    """
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with cfg.input_file.open(encoding="utf-8") as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(set(text))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = dict(enumerate(chars))

    def encode(s: str) -> list[int]:
        """Take a string, output a list of integers."""
        return [stoi[c] for c in s]

    def decode(int_list: list[int]) -> str:
        """Take a list of integers, output a string."""
        return "".join([itos[i] for i in int_list])

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(cfg.train_split * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    test_data = data[n:]

    x_train = train_data[:-1]
    y_train = train_data[1:]

    x_test = test_data[:-1]
    y_test = test_data[1:]

    data_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }
    for k, v in data_dict.items():
        data_dict[k] = reshape_to_batches(v, cfg.block_size)

    return data_dict, vocab_size, decode


def create_splits(data: torch.Tensor, train_split: float = 0.9) -> dict[str, torch.Tensor]:
    """Split data set into train/test, create labels for training."""
    n = int(train_split * len(data))
    train_data = data[:n]
    test_data = data[n:]

    x_train = train_data[:-1]
    y_train = train_data[1:]

    x_test = test_data[:-1]
    y_test = test_data[1:]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def reshape_to_batches(data: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape an array from 1d to 2d of shape (n_batches, block_size)."""
    n = len(data) - len(data) % block_size
    data = data[:n]
    n_batches = int(len(data) / block_size)
    return torch.reshape(data, (n_batches, block_size))
