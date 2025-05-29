from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from typing import Callable
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


def load_data(
        input_path: Path,
        block_size: int,
    ) -> tuple[Dataset, Dataset, int, Callable]:
    """Load dataset.

    Args:
        input_path (Path): path to the data.
        block_size (int): Context length to consider in a sequence.

    Returns:
        A tuple of train dataset, test dataset, vocabulary size, and
        function to decode integers (useful for generating new sequences.)

    """
    data_dict, vocab_size, decode_fct = read_and_prepare_data(
            input_path=input_path, block_size=block_size)

    train_dataset = TensorDataset(data_dict["x_train"], data_dict["y_train"])
    test_dataset = TensorDataset(data_dict["x_test"], data_dict["y_test"])

    return train_dataset, test_dataset, vocab_size, decode_fct


def read_and_prepare_data(
        input_path: Path,
        block_size: int) -> tuple[dict[str, torch.Tensor], int, Callable[[list[int]], str]]:
    """Load data from file and prepare for training.

    Read data from file, create mapping from characters to integers.
    Splits into train and test, and reshape into (n_batches, block_size).

    Args:
        input_path (Path): path to the data.
        block_size (int): Context length to consider in a sequence.

    Returns:
        A tuple consistent of a dict with train and test Xs and Ys, the vocabulary
        size, and the decode function.
    """
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with input_path.open(encoding="utf-8") as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(set(text))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = dict(enumerate(chars))

    def encode(s: str) -> list[int]:
        """Take a string, output a list of integers."""
        return [stoi[c] for c in s]

    def decode(int_list: list[int]) -> str:
        """Take a list of integers, output a string."""
        return "".join([itos[i] for i in int_list])

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
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
        data_dict[k] = reshape_to_batches(v, block_size)

    return data_dict, vocab_size, decode


def reshape_to_batches(data: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape an array from 1d to 2d of shape (n_batches, block_size)."""
    n = len(data) - len(data) % block_size
    data = data[:n]
    n_batches = int(len(data) / block_size)
    return torch.reshape(data, (n_batches, block_size))
