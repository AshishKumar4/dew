"""Patch grids as token sequences: the 2D sincos position signal, and the
raster, hilbert and zigzag orders a sequence can run in.

A scan order is a permutation of the row-major patch index. It depends on the
grid alone, so it is built on the host as numpy and rides into a trace as a
constant; a `jnp` permutation would be a tracer under jit and could not index
the numpy position table.
"""

import math

import einops
import jax
import jax.numpy as jnp
import numpy as np

from .sharding import rows_spec


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


def inverse_permutation(order: np.ndarray) -> np.ndarray:
    """`inv` with inv[idx[i]] = i: the row-major index into the scan order."""
    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0], dtype=order.dtype)
    return inv


def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """`[B, H, W, C]` to row-major patches `[B, (H/p) * (W/p), p * p * C]`."""
    return einops.rearrange(
        x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)


def unpatchify(x: jnp.ndarray, patch_size: int, H: int, W: int, C: int) -> jnp.ndarray:
    """Row-major patches `[B, (H/p) * (W/p), p * p * C]` back to `[B, H, W, C]`,
    as `einops.rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c')`, the
    rows placed as `x`'s are (an Explicit axis splitting them is kept)."""
    H_P, W_P = H // patch_size, W // patch_size
    grid = (x.shape[0], H_P, W_P, patch_size, patch_size, C)
    patches = jax.lax.reshape(x, grid, out_sharding=rows_spec(x, len(grid)))
    image = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    return jax.lax.reshape(image, (x.shape[0], H, W, C), out_sharding=rows_spec(x, 4))


def pixel_unshuffle(x: jnp.ndarray) -> jnp.ndarray:
    """torch's `pixel_unshuffle` by 2, channels last: `[B, H, W, C]` to
    `[B, H/2, W/2, 4C]`, output channel `4c + 2 * row + column`. Channel-major,
    where `patchify` is pixel-major."""
    return einops.rearrange(x, 'b (h p1) (w p2) c -> b h w (c p1 p2)', p1=2, p2=2)


def pixel_shuffle(x: jnp.ndarray) -> jnp.ndarray:
    """torch's `pixel_shuffle` by 2, channels last, the inverse of `pixel_unshuffle`."""
    return einops.rearrange(x, 'b h w (c p1 p2) -> b (h p1) (w p2) c', p1=2, p2=2)


def _ordered_patchify(x: jnp.ndarray, patch_size: int, order: np.ndarray):
    return patchify(x, patch_size)[:, order, :], inverse_permutation(order)


def hilbert_patchify(x: jnp.ndarray, patch_size: int) -> tuple[jnp.ndarray, np.ndarray]:
    """`(patches in hilbert order, inv_idx)`; `hilbert_unpatchify` takes the
    pair back to the image."""
    _, H, W, _ = x.shape
    return _ordered_patchify(x, patch_size, hilbert_indices(H // patch_size, W // patch_size))


def zigzag_patchify(x: jnp.ndarray, patch_size: int) -> tuple[jnp.ndarray, np.ndarray]:
    """`(patches in zigzag order, inv_idx)`, the contract of `hilbert_patchify`."""
    _, H, W, _ = x.shape
    return _ordered_patchify(x, patch_size, zigzag_indices(H // patch_size, W // patch_size))


def hilbert_unpatchify(x: jnp.ndarray, inv_idx: np.ndarray, patch_size: int,
                       H: int, W: int, C: int) -> jnp.ndarray:
    """Scan-ordered patches `[B, N, p * p * C]` back to the image `[B, H, W, C]`
    through the `inv_idx` their patchify returned, whichever order it was."""
    return unpatchify(x[:, inv_idx, :], patch_size, H, W, C)
