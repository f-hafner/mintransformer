
import logging
from pathlib import Path
import torch
from torch.multiprocessing.spawn import spawn
from mintransformer.main import main

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    device = 0
    world_size = 4
    input_path = Path("data/names.txt")
    torch.set_num_interop_threads(4)
    spawn(main, args=(world_size, input_path), nprocs=world_size)
