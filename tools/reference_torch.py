"""The torch-side conversions the reference generators share, with torch and NumPy alone, so a
generator in any reference venv imports it."""

from collections.abc import Mapping

import numpy as np
import torch


def host_arrays(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """A state dict as host float32 arrays that own their memory, as `safetensors.numpy.save_file` takes."""
    return {name: tensor.detach().cpu().float().numpy().copy() for name, tensor in state.items()}
