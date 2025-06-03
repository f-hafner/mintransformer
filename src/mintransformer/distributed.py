from torch.distributed import init_process_group


def ddp_setup(backend: str = "gloo") -> None:
    """Set up DDP.

    Args:
        backend: backend to use; mostly "gloo" or "nccl".
    """
    init_process_group(backend=backend)
