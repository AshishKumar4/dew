"""A torch transformer's parameters in Dew's layout and its gradients back, for the parity tests that
load a torch reference's weights and compare gradients in its own names: kernels as [in, out], a fused
qkv projection split per head into Dew's to_q, to_k and to_v, the output projection read per head."""

import numpy as np

QKV = ("to_q", "to_k", "to_v")


def split_qkv(value: np.ndarray, heads: int) -> np.ndarray:
    """A fused [3 * width, in] qkv weight as Dew's three [in, heads, head] kernels, or its
    [3 * width] bias as three [heads, head] biases, stacked."""
    width = value.shape[-1] if value.ndim == 2 else value.shape[0] // 3
    parts = value.reshape(3, heads, width // heads, *value.shape[1:])
    return parts.transpose(0, 3, 1, 2) if value.ndim == 2 else parts


def join_qkv(parts: list[np.ndarray]) -> np.ndarray:
    """`split_qkv`'s inverse: three Dew kernels or biases back as the fused torch weight or bias."""
    if parts[0].ndim == 3:
        return np.stack(parts).transpose(0, 2, 3, 1).reshape(-1, parts[0].shape[0])
    return np.concatenate([part.ravel() for part in parts])


def output_heads(value: np.ndarray, heads: int) -> np.ndarray:
    """A torch [out, heads * head] output projection as Dew's [heads, head, out] kernel."""
    return value.T.reshape(heads, -1, value.shape[0])


def output_merged(value: np.ndarray) -> np.ndarray:
    """`output_heads`'s inverse."""
    return value.reshape(-1, value.shape[-1]).T


def dew_kernel(value: np.ndarray) -> np.ndarray:
    """A torch linear weight [out, in] as [in, out], or a conv's [out, in, h, w] as [h, w, in, out]."""
    return value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T


def torch_kernel(value: np.ndarray) -> np.ndarray:
    """`dew_kernel`'s inverse."""
    return value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T


def dew_leaf(path: tuple[str, ...], value: np.ndarray, heads: int) -> np.ndarray:
    """A torch tensor other than a fused qkv in Dew's layout at Dew `path`: the output projection per
    head, any other kernel as [in, out], the rest as stored."""
    if path[-2:] == ("to_out_0", "kernel"):
        return output_heads(value, heads)
    return dew_kernel(value) if path[-1] == "kernel" else value


def torch_leaf(path: tuple[str, ...], value: np.ndarray) -> np.ndarray:
    """`dew_leaf`'s inverse."""
    if path[-2:] == ("to_out_0", "kernel"):
        return output_merged(value)
    return torch_kernel(value) if path[-1] == "kernel" else value


def case_arrays(arrays, case: str):
    """A reference fixture's arrays of one case, written as `<case>/<name>`: a reader of them by
    `name`, and the case's parameter names (`<case>/param.<name>`) in the file's order."""
    prefix = f"{case}/param."
    return (lambda name: arrays[f"{case}/{name}"]), [name.removeprefix(prefix) for name in arrays
                                                     if name.startswith(prefix)]


def leaf(tree, path: tuple[str, ...]) -> np.ndarray:
    """The array at `path` in a nested tree."""
    for key in path:
        tree = tree[key]
    return np.asarray(tree)
