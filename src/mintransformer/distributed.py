
import os
from torch.distributed import init_process_group


def ddp_setup(rank: int, world_size: int, backend: str = "gloo") -> None:
    """Set up DDP.

    Args:
        rank: Unique identifier for each process.
        world_size: Total number of processes.
        backend: backend to use; mostly "gloo" or "nccl".
    """
    os.environ["MASTER_ADDR"] = "localhost" # for single machine
    os.environ["MASTER_PORT"] = "12355" # "any free port" -- how do I see them? TODO -- use correct port
    init_process_group(backend=dist_backend, rank=rank, world_size=world_size)

