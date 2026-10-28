"""The grouped product's float64 oracle, in NumPy alone, independent of every kernel and of JAX: rows
sorted by group, each group's leading rows counted by `sizes`, the rows past the routed ones and an empty
group contributing nothing. The tests' and benchmarks' one reference for the forward, the input gradient
and the kernel gradient (tests/test_grouped_gemm.py, tests/test_moe_precision.py,
tools/benchmark_kernels.py)."""

import numpy as np


def _spans(sizes) -> list[tuple[int, int]]:
    bounds = np.concatenate([[0], np.cumsum(np.asarray(sizes, np.int64))])
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))


def forward(x, w, sizes) -> np.ndarray:
    """Each group's rows of `x` `[rows, depth]` times its matrix of `w` `[groups, depth, width]`."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    out = np.zeros((x.shape[0], w.shape[2]))
    for group, (start, end) in enumerate(_spans(sizes)):
        out[start:end] = x[start:end] @ w[group]
    return out


def input_gradient(dy, w, sizes) -> np.ndarray:
    """Each group's rows of `dy` `[rows, width]` times its matrix of `w` transposed."""
    return forward(dy, np.swapaxes(np.asarray(w, np.float64), 1, 2), sizes)


def kernel_gradient(x, dy, sizes) -> np.ndarray:
    """Each group's rows of `x` transposed times its rows of `dy`: `[groups, depth, width]`."""
    x, dy = np.asarray(x, np.float64), np.asarray(dy, np.float64)
    spans = _spans(sizes)
    out = np.zeros((len(spans), x.shape[1], dy.shape[1]))
    for group, (start, end) in enumerate(spans):
        out[group] = x[start:end].T @ dy[start:end]
    return out


def products(x, w, dy, sizes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The forward, input gradient and kernel gradient of one grouped projection."""
    return forward(x, w, sizes), input_gradient(dy, w, sizes), kernel_gradient(x, dy, sizes)
