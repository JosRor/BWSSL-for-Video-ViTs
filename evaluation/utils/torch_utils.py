import torch

def move_to_device(batch, device):
    """
    Recursively moves every Tensor in `batch` to `device`.
    Keeps non-tensor objects unchanged.
    """
    if torch.is_tensor(batch):
        # non_blocking lets the copy overlap with host-to-device traffic
        return batch.to(device, non_blocking=True)

    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
