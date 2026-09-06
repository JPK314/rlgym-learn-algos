import numpy as np
import torch


def torch_to_numpy_dtype(dt: torch.dtype) -> np.dtype:
    return torch.empty(0, dtype=dt).numpy().dtype
