"""Patch grids as token sequences: the 2D sincos position signal, and the
raster, hilbert and zigzag orders a sequence can run in.

A scan order is a permutation of the row-major patch index. It depends on the
grid alone, so it is built on the host as numpy and rides into a trace as a
constant; a `jnp` permutation would be a tracer under jit and could not index
the numpy position table.
"""

import math

import einops
import jax.numpy as jnp
import numpy as np


def build_2d_sincos_pos_embed(emb_dim: int, H_P: int, W_P: int) -> np.ndarray:
    """Fixed MAE-style 2D sin-cos positional embedding, row-major [H_P * W_P, emb_dim].
    Half the channels encode the row, half the column. For another scan order,
    index the result by that order's permutation.
    """
    assert emb_dim % 4 == 0, f"emb_dim must be divisible by 4 for 2D sincos, got {emb_dim}"
    half = emb_dim // 2
    quarter = half // 2

    omega = np.arange(quarter, dtype=np.float32) / quarter
    omega = 1.0 / (10000.0 ** omega)  # [quarter]

    rows = np.arange(H_P, dtype=np.float32)
    cols = np.arange(W_P, dtype=np.float32)

    row_emb = np.einsum('h,d->hd', rows, omega)
    col_emb = np.einsum('w,d->wd', cols, omega)

    pos = np.zeros((H_P, W_P, emb_dim), dtype=np.float32)
    pos[..., 0:quarter] = np.sin(row_emb)[:, None, :]
    pos[..., quarter:half] = np.cos(row_emb)[:, None, :]
    pos[..., half:half + quarter] = np.sin(col_emb)[None, :, :]
    pos[..., half + quarter:] = np.cos(col_emb)[None, :, :]
    return pos.reshape(H_P * W_P, emb_dim)


def _d2xy(n: int, d: int) -> tuple[int, int]:
    """(column, row) of index `d` on the Hilbert curve over an n x n grid, n a
    power of two; the d2xy of Wikipedia's Hilbert curve article."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        if ry == 0:
            if rx == 1:
                x = (s - 1) - x
                y = (s - 1) - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y


def hilbert_indices(H_P: int, W_P: int) -> np.ndarray:
    """The hilbert order of an H_P x W_P patch grid: result[i] is the row-major
    index of the i-th patch along the curve.

    The curve runs over the smallest power-of-two square that holds the grid
    and the points outside the grid are skipped, so a rectangular grid gets
    the square's curve with gaps closed up.
    """
    size = max(H_P, W_P)
    order = math.ceil(math.log2(size)) if size > 0 else 0
    n = 1 << order

    indices = []
    for d in range(n * n):
        x, y = _d2xy(n, d)
        if x < W_P and y < H_P:
            indices.append(y * W_P + x)
            if len(indices) == H_P * W_P:
                break
    return np.asarray(indices, dtype=np.int32)


def zigzag_indices(H_P: int, W_P: int) -> np.ndarray:
    """The zigzag (serpentine) order of an H_P x W_P patch grid, as in ZigMa:
    even rows left to right, odd rows right to left. result[i] is the
    row-major index of the i-th patch along the scan."""
    grid = np.arange(H_P * W_P, dtype=np.int32).reshape(H_P, W_P)
    grid[1::2] = grid[1::2, ::-1]
    return grid.reshape(-1)


def inverse_permutation(idx: np.ndarray) -> np.ndarray:
    """`inv` with inv[idx[i]] = i: the row-major index into the scan order."""
    inv = np.empty_like(idx)
    inv[idx] = np.arange(idx.shape[0], dtype=idx.dtype)
    return inv


def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """`[B, H, W, C]` to row-major patches `[B, (H/p) * (W/p), p * p * C]`."""
    B, H, W, C = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})")
    return einops.rearrange(
        x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)


def unpatchify(x: jnp.ndarray, patch_size: int, H: int, W: int, C: int) -> jnp.ndarray:
    """Row-major patches `[B, (H/p) * (W/p), p * p * C]` back to `[B, H, W, C]`."""
    H_P, W_P = H // patch_size, W // patch_size
    assert x.shape[1] == H_P * W_P, \
        f"Number of patches ({x.shape[1]}) does not match expected ({H_P * W_P}) for H={H}, W={W}, patch_size={patch_size}"
    return einops.rearrange(
        x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=H_P, w=W_P, p1=patch_size, p2=patch_size, c=C)


def _ordered_patchify(x: jnp.ndarray, patch_size: int, idx: np.ndarray):
    return patchify(x, patch_size)[:, idx, :], inverse_permutation(idx)


def hilbert_patchify(x: jnp.ndarray, patch_size: int) -> tuple[jnp.ndarray, np.ndarray]:
    """`(patches in hilbert order, inv_idx)`; `hilbert_unpatchify` takes the
    pair back to the image."""
    B, H, W, C = x.shape
    return _ordered_patchify(x, patch_size, hilbert_indices(H // patch_size, W // patch_size))


def zigzag_patchify(x: jnp.ndarray, patch_size: int) -> tuple[jnp.ndarray, np.ndarray]:
    """`(patches in zigzag order, inv_idx)`, the contract of `hilbert_patchify`."""
    B, H, W, C = x.shape
    return _ordered_patchify(x, patch_size, zigzag_indices(H // patch_size, W // patch_size))


def hilbert_unpatchify(x: jnp.ndarray, inv_idx: np.ndarray, patch_size: int,
                       H: int, W: int, C: int) -> jnp.ndarray:
    """Scan-ordered patches `[B, N, p * p * C]` back to the image `[B, H, W, C]`
    through the `inv_idx` their patchify returned, whichever order it was."""
    return unpatchify(x[:, inv_idx, :], patch_size, H, W, C)
