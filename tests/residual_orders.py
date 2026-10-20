"""Exact rounding draws of a Dew model: the units of its residual stream permuted.

Every matmul that reads or writes the residual stream sums over its units, and
every norm averages over them, so reordering the units, consistently in every
parameter that holds the residual width, changes the order of those sums and
nothing else. In exact arithmetic the model computes the same function; in
fp32 it rounds differently. A run under each of K orders is one rounding draw
of the same computation (tests/reference_error.py's K-order rule).

The residual axes are the ones a module declares 'embed' (dew.nn.sharding)
and, among undeclared parameters, only those named here: the residual
stream's norms' scales and biases (RESIDUAL_NORMS, the last axis), and Kimi
K3's depth-attention scales and queries (DEPTH_ATTENTION, [width] and
[width, 1]). No parameter is taken for a residual one by its size: a head's
or a convolution's width can equal the residual's (Kimi Delta Attention's
2 heads of 16 in a 32-wide model), and permuting it would break the
symmetry. Composite axes repeat the same order within each residual stream:
Engram's declared embed axis and mHC's flattened stream inputs. The fixture
tool that draws the orders evaluates the float64 truth under them and holds
it to the identity's within float64 rounding.
"""

import re

import jax
import numpy as np

from dew.nn.sharding import declarations, declared_axes, parameter_path

# mHC's small mappings stay replicated; a production 'embed' declaration
# would change their placement. Their columns concatenate residual streams.
COMPOSITE_RESIDUAL_AXES = {("attn_hc", "fn"): 1, ("ffn_hc", "fn"): 1, ("hc_head", "hc_fn"): 1}

RESIDUAL_NORMS = frozenset({
    "input_layernorm", "post_attention_layernorm", "attention_output_norm", "mlp_output_norm",
    "post_per_layer_input_norm", "embedding_layernorm", "head_norm", "enorm", "hnorm"})
"""The norms over the residual stream a Dew decoder names (backbones/decoder_block.py,
causal_transformer.py), beside the final `norm` outside every layer; a mixer's own `norm`
(Mamba-2's, a gated delta rule's) and the query and key norms are over other widths."""

LAYER = re.compile(r"layers(_\d+)?")
"""A decoder layer's module, under which a `norm` is a mixer's own, not the final norm."""

DEPTH_ATTENTION = frozenset({"attention_res", "mlp_res", "output_res"})
"""Kimi K3's depth-attention modules (attention_residuals.py `DepthAttention`): a `scale` [width]
and a query `kernel` [width, 1]."""


def _residual_axis(path, leaf, width: int, declared) -> int | None:
    axes = declared_axes(path, leaf.ndim, declared)
    if axes is None:
        names = parameter_path(path)
        if names[-2:] in COMPOSITE_RESIDUAL_AXES:
            return COMPOSITE_RESIDUAL_AXES[names[-2:]]
        module, name = names[-2:] if len(names) > 1 else ("", names[-1])
        final = module == "norm" and not any(LAYER.fullmatch(outer) for outer in names[:-2])
        if name in ("scale", "bias") and (module in RESIDUAL_NORMS or final):
            axis = leaf.ndim - 1
        elif module in DEPTH_ATTENTION and name in ("scale", "kernel"):
            axis = leaf.ndim - (1 if name == "scale" else 2)
        else:
            return None
        if leaf.shape[axis] != width:
            raise ValueError(f"{'/'.join(names)} is named a residual parameter but its axis {axis} is "
                             f"{leaf.shape[axis]} wide, the residual {width}")
        return axis
    if "embed" not in axes or leaf.shape[axes.index("embed")] % width:
        return None
    return axes.index("embed")


def residual_width(variables) -> int:
    """The residual stream's width: the 'embed' side of the token embedding
    table. A wrapper's vision tower declares its own, which is not this one
    unless the widths happen to agree; a text-only run never reads it."""
    declared = declarations(variables["params"])
    (width,) = {leaf.shape[1] for path, leaf in jax.tree_util.tree_flatten_with_path(variables["params"])[0]
                if declared_axes(path, leaf.ndim, declared) == ("vocab", "embed")}
    return width


def permuted(variables, order: np.ndarray):
    """`variables` with the residual stream's units in `order`."""
    width = len(order)
    declared = declarations(variables["params"])

    def reorder(path, leaf):
        axis = _residual_axis(path, leaf, width, declared)
        if axis is None:
            return leaf
        grouped = (np.arange(leaf.shape[axis] // width)[:, None] * width + order).reshape(-1)
        return np.take(np.asarray(leaf), grouped, axis=axis)

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
