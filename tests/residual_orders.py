"""Exact rounding draws of a Dew model: the units of its residual stream permuted.

Every matmul that reads or writes the residual stream sums over its units, and
every norm averages over them, so reordering the units, consistently in every
parameter that holds the residual width, changes the order of those sums and
nothing else. In exact arithmetic the model computes the same function; in
fp32 it rounds differently. A run under each of K orders is one rounding draw
of the same computation (tests/reference_error.py's K-order rule).

The residual axes are the ones a module declares 'embed' (dew.nn.sharding)
and, in an undeclared parameter, its one axis of the residual's width: the
norms' scales, and Kimi K3's depth-attention queries ([width, 1]). That
it is a symmetry is checked, not assumed: the fixture tool that draws the
orders evaluates the float64 truth under a permuted order and holds it to the
identity's within float64 rounding.
"""

import jax
import numpy as np

from dew.nn.sharding import declared_axes


def _residual_axis(path, leaf, width: int) -> int | None:
    axes = declared_axes(path, leaf.ndim)
    if axes is None:
        sized = [axis for axis, size in enumerate(leaf.shape) if size == width]
        return sized[0] if len(sized) == 1 else None
    if "embed" not in axes or leaf.shape[axes.index("embed")] != width:
        return None
    return axes.index("embed")


def residual_width(variables) -> int:
    """The residual stream's width: the 'embed' side of the token embedding
    table. A wrapper's vision tower declares its own, which is not this one
    unless the widths happen to agree; a text-only run never reads it."""
    (width,) = {leaf.shape[1] for path, leaf in jax.tree_util.tree_flatten_with_path(variables["params"])[0]
                if declared_axes(path, leaf.ndim) == ("vocab", "embed")}
    return width


def permuted(variables, order: np.ndarray):
    """`variables` with the residual stream's units in `order`."""
    width = len(order)

    def reorder(path, leaf):
        axis = _residual_axis(path, leaf, width)
        return leaf if axis is None else np.take(np.asarray(leaf), order, axis=axis)

    return {**variables, "params": jax.tree_util.tree_map_with_path(reorder, variables["params"])}


def orders(width: int, count: int, seed: int, group: int = 1) -> np.ndarray:
    """`count` orders of `width` units, the first the identity. With `group`,
    each order moves whole groups of that many consecutive units and permutes
    within each, so a checkpoint quantized in groups along the residual
    (MXFP4's 32) requantizes to the same codes in the new order."""
    rng = np.random.default_rng(seed)

    def draw():
        blocks = rng.permutation(width // group)
        return np.concatenate([block * group + rng.permutation(group) for block in blocks])

    return np.stack([np.arange(width), *(draw() for _ in range(count - 1))])
