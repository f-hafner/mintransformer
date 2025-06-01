
import logging
from pathlib import Path
import torch
from torch.multiprocessing.spawn import spawn
from mintransformer.main import main

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    world_size = 1
    input_path = Path("data/names.txt")
    if world_size == 1:
        main(rank=1, world_size=world_size, input_path=input_path)
    else:
        torch.set_num_interop_threads(4)
        spawn(main, args=(world_size, input_path), nprocs=world_size)
