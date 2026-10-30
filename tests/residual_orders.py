"""Exact rounding draws of a Dew model: the units of its residual stream permuted.

Every matmul that reads or writes the residual stream sums over its units, and
every norm averages over them, so reordering the units, consistently in every
parameter that holds the residual width, changes the order of those sums and
nothing else. In exact arithmetic the model computes the same function; in
fp32 it rounds differently. A run under each of K orders is one rounding draw
of the same computation (tests/reference_error.py's K-order rule).

The residual axes are the ones a module declares 'embed' (dew.nn.sharding)
and, among undeclared parameters, only those named here: the residual
stream's norms' scales and biases (RESIDUAL_NORMS, the last axis), Gemma
4's router scale (ROUTER_SCALE), Kimi K3's depth-attention scales and
queries (DEPTH_ATTENTION, [width] and [width, 1]), and Gemma 3n's AltUp:
its corrected output's scale [width] and the square maps its copies enter
and leave by (ALTUP_SQUARE, both axes). No
parameter is taken for a residual one by its size: a head's or a
convolution's width can equal the residual's (Kimi Delta Attention's 2 heads
of 16 in a 32-wide model), and permuting it would break the symmetry.
Composite axes repeat the same order within each residual stream: Engram's
declared embed axis and mHC's flattened stream inputs. The fixture tool that
draws the orders evaluates the float64 truth under them and holds it to the
identity's within float64 rounding, as tests/test_architectures.py holds
each language cell's.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.sharding import declarations, declared_axes, parameter_path

# mHC's small mappings stay replicated; a production 'embed' declaration
# would change their placement. Their columns concatenate residual streams.
COMPOSITE_RESIDUAL_AXES = {("attn_hc", "fn"): 1, ("ffn_hc", "fn"): 1, ("hc_head", "hc_fn"): 1}

RESIDUAL_NORMS = frozenset({
    "input_layernorm", "post_attention_layernorm", "attention_output_norm", "mlp_output_norm",
    "post_per_layer_input_norm", "embedding_layernorm", "head_norm", "enorm", "hnorm",
    "final_norm", "router_norm", "post_laurel_norm", "experts_input_norm", "mlp_branch_norm",
    "experts_output_norm"})
"""The norms over the residual stream a Dew decoder names (backbones/decoder_block.py,
causal_transformer.py; an MTP head's `final_norm`), beside the final `norm` outside every layer;
a mixer's own `norm` (Mamba-2's, a gated delta rule's) and the query and key norms are over
other widths. AltUp's `router_norm` norms the active copy, and LAuReL's `post_laurel_norm` its
output, both over the residual; Gemma 4's mixture norms its experts' input, its dense branch's
output and its experts' output (gemma4_moe.py), each residual-wide."""

ALTUP_SQUARE = re.compile(r"altup_(unembed_)?projections_\d+")
"""Gemma 3n's maps by which AltUp's copies past the first enter and leave the residual
(causal_transformer.py `altup_projections`, `altup_unembed_projections`): [width, width], both
axes the residual's."""

LAYER = re.compile(r"layers(_\d+)?")
"""A decoder layer's module, under which a `norm` is a mixer's own, not the final norm."""

ROUTER_SCALE = ("router", "scale")
"""Gemma 4's router's scale (gemma4_moe.py `Gemma4TextRouter`), [width] over the residual it routes:
named here ahead of its declaration, which GPT-OSS's router's ("embed", "exp") reaches by its module's
name (gpt_oss.py `GptOssMLP`)."""

DEPTH_ATTENTION = frozenset({"attention_res", "mlp_res", "output_res"})
"""Kimi K3's depth-attention modules (attention_residuals.py `DepthAttention`): a `scale` [width]
and a query `kernel` [width, 1]."""


def _residual_axes(path, leaf, width: int, declared) -> tuple[int, ...]:
    """The axes of a parameter that hold the residual stream's units, none for one that holds none."""
    names = parameter_path(path)
    axes = None if names[-2:] == ROUTER_SCALE else declared_axes(path, leaf.ndim, declared)
    if axes is None:
        if names[-2:] in COMPOSITE_RESIDUAL_AXES:
            return (COMPOSITE_RESIDUAL_AXES[names[-2:]],)
        module, name = names[-2:] if len(names) > 1 else ("", names[-1])
        final = module == "norm" and not any(LAYER.fullmatch(outer) for outer in names[:-2])
        if (name in ("scale", "bias") and (module in RESIDUAL_NORMS or final)) or names[-2:] == ROUTER_SCALE:
            found = (leaf.ndim - 1,)
        elif module in DEPTH_ATTENTION and name in ("scale", "kernel"):
            found = (leaf.ndim - (1 if name == "scale" else 2),)
        elif module == "altup" and name == "correct_output_scale":
            found = (leaf.ndim - 1,)
        elif ALTUP_SQUARE.fullmatch(module) and name == "kernel":
            found = (0, 1)
        else:
            return ()
        if any(leaf.shape[axis] != width for axis in found):
            raise ValueError(f"{'/'.join(names)} is named a residual parameter but its axes {found} are "
                             f"{[leaf.shape[axis] for axis in found]} wide, the residual {width}")
        return found
    if "embed" not in axes or leaf.shape[axes.index("embed")] % width:
        return ()
    return (axes.index("embed"),)


def residual_width(variables) -> int:
    """The residual stream's width: the 'embed' side of the token embedding
    table. A wrapper's vision tower declares its own, which is not this one
    unless the widths happen to agree; a text-only run never reads it."""
    declared = declarations(variables["params"])
    (width,) = {leaf.shape[1] for path, leaf in jax.tree_util.tree_flatten_with_path(variables["params"])[0]
                if declared_axes(path, leaf.ndim, declared) == ("vocab", "embed")}
    return width


def permuted(variables, order: np.ndarray):
    """`variables` with the residual stream's units in `order`, each leaf where it is: a device
    array reordered on its device, a host array on the host."""
    width = len(order)
    declared = declarations(variables["params"])

    def reorder(path, leaf):
        for axis in _residual_axes(path, leaf, width, declared):
            grouped = (np.arange(leaf.shape[axis] // width)[:, None] * width + order).reshape(-1)
            if isinstance(leaf, jax.Array):
                leaf = jnp.take(leaf, jnp.asarray(grouped), axis=axis)
            else:
                leaf = np.take(np.asarray(leaf), grouped, axis=axis)
        return leaf

    return {**variables, "params": jax.tree_util.tree_map_with_path(reorder, variables["params"])}


def _grouped(rng, width: int, group: int) -> np.ndarray:
    """One order of `width` units by `rng`, uniform over those that move whole groups of `group`
    consecutive units and permute within each."""
    if group < 1 or width % group:
        raise ValueError(f"an order moves groups of {group} whole, which {width} units do not divide into")
    blocks = rng.permutation(width // group)
    return np.concatenate([block * group + rng.permutation(group) for block in blocks])


def order(width: int, seed: int, group: int = 1) -> np.ndarray:
    """One order of `width` units drawn by `seed`, each seed its own draw of the law `orders` draws
    from (`_grouped`): check 1's draw over an orbit (`orbit`), its seeds declared and sized as any
    site's (`seeds_for`)."""
    return _grouped(np.random.default_rng([0x0BB17, width, group, seed]), width, group)


def orbit(variables, *rest, group: int = 1):
    """Check 1's draw over the residual orbit of `variables` (`assert_as_exact`'s `draw`): seed s gives
    `variables` with the residual stream in `order(width, s, group)`, and the rest of the arguments as
    they are.
    The truth of each draw is the reference's own float64 run on it, so nothing is permuted back."""
    width = residual_width(variables)

    def draw(seed):
        return (permuted(variables, order(width, seed, group)), *rest)

    return draw


def orders(width: int, count: int, seed: int, group: int = 1) -> np.ndarray:
    """`count` orders of `width` units, the first the identity. With `group`,
    each order moves whole groups of that many consecutive units and permutes
    within each, so a checkpoint quantized in groups along the residual
    (MXFP4's 32) requantizes to the same codes in the new order."""
    rng = np.random.default_rng(seed)
    return np.stack([np.arange(width), *(_grouped(rng, width, group) for _ in range(count - 1))])
