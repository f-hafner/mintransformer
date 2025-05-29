from __future__ import annotations
from pathlib import Path
import torch


def load_data(input_path: Path = Path("data/input.txt")) -> tuple[torch.Tensor, torch.Tensor]:
    """Load data."""
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with input_path.open(encoding="utf-8") as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(set(text))
    #vocab_size = len(chars)

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
    val_data = data[n:]
    return train_data, val_data

def get_batch(
        block_size: int,
        batch_size: int,
        device: str,
        split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch from data."""
    # generate a small batch of data of inputs x and targets y
    train_data, val_data = load_data() # TODO: do not load each time
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # NOTE: important for using GPUs
    return x, y
