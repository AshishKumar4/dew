"""Differential extension (DEX, arXiv 2505.16333): a learned differential on chosen heads' outputs.

On each chosen head `h` of a softmax attention layer, the head's output `O_h` (its attention-weighted
values, before the output projection) becomes

    O'_h = O_h - lambda(t) * O_h W_D,h                                          (paper Eq. 3, 5)

with `W_D,h` a learned `[Dv, Dv]` projection of the head's own, and the other heads unchanged. `lambda` is
annealed over the first `anneal_steps` optimizer updates `t` from zero to a learned scalar (Eq. 4):

    lambda(t) = (1 - a) * (t / T) * lambda_init + a * lambda_learn,   a = min(t / T, 1)

so a pretrained model computes its own function at the start (`lambda(0) = 0`) and the extension enters
as it trains. `lambda_init` defaults to Differential Transformer's depth schedule,
`0.8 - 0.6 exp(-0.3 (l - 1))` for the layer's 1-based depth `l` (arXiv 2410.05258, which DEX adopts).
Outside training (no step, `dew.nn.progress`) a layer reads `lambda_learn`, the anneal's end.

The heads are chosen per layer (`heads`), as the paper picks the half of each layer's heads of least
importance or most attention entropy; a run picks them from a model's own statistics before it attaches
the extension. The paper's recipe trains the attention's key, value and output projections with the
extension and freezes the rest (`dew.optim.ParamGroup`).
"""

import dataclasses
import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

__all__ = ["DEX_PARAMETERS", "DifferentialExtension", "LayerDex", "annealed", "diff_lambda_init",
           "differential"]

TRAINED_PROJECTIONS = ("k_proj", "v_proj", "o_proj")
"""The attention projections the paper's recipe trains beside the extension, the feed-forward and the
rest frozen."""

DEX_PARAMETERS = ("dex_projection", "dex_lambda")
"""The parameters an attention layer adds for DEX, by name: its chosen heads' projections and its learned
lambda (`dew.nn.mixers.attention.CausalSelfAttention`)."""


def diff_lambda_init(depth: int) -> float:
    """Differential Transformer's `lambda_init` for a layer of 1-based `depth`:
    `0.8 - 0.6 exp(-0.3 (l - 1))`."""
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))


@dataclasses.dataclass(frozen=True)
class LayerDex:
    """One layer's differential extension: the heads it acts on, ascending, its `lambda_init` and the
    updates its lambda anneals over (`DifferentialExtension`)."""

    heads: tuple[int, ...]
    lambda_init: float
    anneal_steps: int


@dataclasses.dataclass(frozen=True)
class DifferentialExtension:
    """Which heads of which layers carry the differential extension (this module's doc), as
    `(layer, heads)` pairs, each layer once; `lambda_init` for every chosen layer, or None for
    Differential Transformer's depth schedule (`diff_lambda_init`); and the optimizer updates
    lambda anneals over."""

    layers: tuple[tuple[int, tuple[int, ...]], ...]
    lambda_init: float | None = None
    anneal_steps: int = 1000
    trains: Literal["attention", "all"] = "attention"
    """What of the model trains beside the extension: the paper's recipe, every attention layer's key,
    value and output projections, the rest frozen (`trainable`); or all of it."""

    def __post_init__(self):
        layers = tuple((int(layer), tuple(sorted({int(head) for head in heads})))
                       for layer, heads in self.layers)
        object.__setattr__(self, "layers", layers)
        indices = [layer for layer, _ in layers]
        if not layers or len(set(indices)) != len(indices):
            raise ValueError(f"a differential extension names one layer or more, each once, got {indices}")
        if any(not heads or heads[0] < 0 for _, heads in layers):
            raise ValueError("each layer of a differential extension names one head or more, none negative")
        if self.anneal_steps < 1:
            raise ValueError(f"lambda anneals over a positive number of updates, not {self.anneal_steps}")

    def trainable(self, path: tuple[str, ...], decoder: tuple[str, ...],
                  attention: Callable[[str], bool]) -> bool:
        """Whether the parameter at `path` (`("params", ...)`) trains under the extension's recipe
        (`trains`): every parameter with "all"; else the extension's own, and the key, value and output
        projections of the decoder's softmax attention layers, whose modules' names `attention` takes
        (`layers_3`, or a scanned run's `layers_3_7`), under the decoder's parameters (`decoder`,
        `("params",)` for a bundle's own decoder, deeper in a wrapper's)."""
        if self.trains == "all" or path[-1] in DEX_PARAMETERS:
            return True
        depth = len(decoder)
        return (path[:depth] == decoder and len(path) > depth + 2 and attention(path[depth])
                and path[depth + 1] == "self_attn" and path[depth + 2] in TRAINED_PROJECTIONS)

    def layer(self, index: int) -> LayerDex | None:
        """Layer `index`'s extension, or None where it carries none."""
        heads = dict(self.layers).get(index)
        if heads is None:
            return None
        start = diff_lambda_init(index + 1) if self.lambda_init is None else self.lambda_init
        return LayerDex(heads, start, self.anneal_steps)


def annealed(updates: jax.Array | None, layer: LayerDex, learned: jax.Array) -> jax.Array:
    """`lambda(t)` after `updates` optimizer updates (this module's Eq. 4); `learned` past the anneal,
    and outside training (`updates` None)."""
    if updates is None:
        return learned
    fraction = jnp.minimum(jnp.asarray(updates, jnp.float32) / layer.anneal_steps, 1.0)
    start = fraction * layer.lambda_init
    return (1 - fraction) * start.astype(learned.dtype) + fraction.astype(learned.dtype) * learned


def differential(attention: jax.Array, heads: tuple[int, ...], projection: jax.Array,
                 strength: jax.Array, precision=None) -> jax.Array:
    """The heads' outputs `attention` `[B, S, H, Dv]` with each of `heads` less `strength` times its own
    projection of itself (`projection` `[len(heads), Dv, Dv]`, in `heads`' order), the other heads as
    they are: one product over the chosen heads, scattered back."""
    chosen = jnp.asarray(heads)
    taken = attention[:, :, chosen]
    moved = jnp.einsum("bshd,hde->bshe", taken, projection.astype(attention.dtype), precision=precision)
    return attention.at[:, :, chosen].add(-(strength.astype(attention.dtype) * moved))
