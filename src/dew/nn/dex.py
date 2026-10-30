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
the extension. A layer holds its chosen heads and its `lambda_init` as variables of its own
(`DEX_CONSTANTS`, `DexStart`), so layers that choose as many heads share one spec (`LayerDex`) and a
scanned run or a pipeline stage holds them as it holds any layer's weights. The paper's recipe trains
the attention's key, value and output projections with the extension and freezes the rest
(`dew.optim.ParamGroup`).
"""

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import unflatten_dict

from dew.batch import leaves_by_path

__all__ = ["DEX_COLLECTION", "DEX_CONSTANTS", "DEX_PARAMETERS", "DexStart", "DifferentialExtension",
           "HeadOutputs", "LayerDex", "annealed", "diff_lambda_init", "differential", "folded"]

TRAINED_PROJECTIONS = ("k_proj", "v_proj", "o_proj")
"""The attention projections the paper's recipe trains beside the extension, the feed-forward and the
rest frozen."""

DEX_PARAMETERS = ("dex_projection", "dex_lambda")
"""The parameters an attention layer adds for DEX, by name: its chosen heads' projections and its learned
lambda (`dew.nn.mixers.attention.CausalSelfAttention`)."""

DEX_COLLECTION = "constants"
"""The collection an attention layer holds DEX's start in (`DEX_CONSTANTS`), which never trains."""

DEX_CONSTANTS = ("dex_heads", "dex_lambda_init")
"""The variables an attention layer holds for DEX under `DEX_COLLECTION`, by name: the heads it acts on and
its `lambda_init` (`DexStart`)."""


@runtime_checkable
class HeadOutputs(Protocol):
    """A mixer whose heads' outputs DEX extends, before its output projection: softmax attention
    (`dew.nn.mixers.attention.CausalSelfAttention`)."""

    def head_outputs(self) -> tuple[int, int]:
        """Its heads' count and the width of each one's output."""
        ...

    def dex_fold_refusal(self) -> str | None:
        """Why DEX on its heads does not fold into its output projection (`folded`): what its heads'
        outputs pass through between the two; None where nothing does."""
        ...


def diff_lambda_init(depth: int) -> float:
    """Differential Transformer's `lambda_init` for a layer of 1-based `depth`:
    `0.8 - 0.6 exp(-0.3 (l - 1))`."""
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))


@dataclasses.dataclass(frozen=True)
class LayerDex:
    """One layer's differential extension as every layer that chooses as many heads shares it: how many
    of its heads it acts on and the updates its lambda anneals over (`DifferentialExtension`). Which
    heads, and lambda's start, are the layer's own variables (`DexStart`)."""

    count: int
    anneal_steps: int


@dataclasses.dataclass(frozen=True)
class DexStart:
    """The values one layer's extension holds from its start (`DEX_CONSTANTS`): the heads it acts on,
    ascending, and its `lambda_init`."""

    heads: tuple[int, ...]
    lambda_init: float


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
        """Layer `index`'s extension as its spec holds it, or None where it carries none."""
        heads = dict(self.layers).get(index)
        return None if heads is None else LayerDex(len(heads), self.anneal_steps)

    def start(self, index: int) -> DexStart | None:
        """The values layer `index`'s extension starts from, or None where it carries none."""
        heads = dict(self.layers).get(index)
        if heads is None:
            return None
        return DexStart(heads, diff_lambda_init(index + 1) if self.lambda_init is None else self.lambda_init)


def annealed(updates: jax.Array | None, layer: LayerDex, start: jax.Array, learned: jax.Array) -> jax.Array:
    """`lambda(t)` after `updates` optimizer updates (this module's Eq. 4), from the layer's
    `lambda_init` `start`; `learned` past the anneal, and outside training (`updates` None)."""
    if updates is None:
        return learned
    fraction = jnp.minimum(jnp.asarray(updates, jnp.float32) / layer.anneal_steps, 1.0)
    began = fraction * jnp.asarray(start, jnp.float32)
    return (1 - fraction) * began.astype(learned.dtype) + fraction.astype(learned.dtype) * learned


def differential(attention: jax.Array, heads: jax.Array | tuple[int, ...], projection: jax.Array,
                 strength: jax.Array, precision=None) -> jax.Array:
    """The heads' outputs `attention` `[B, S, H, Dv]` with each of `heads` less `strength` times its own
    projection of itself (`projection` `[len(heads), Dv, Dv]`, in `heads`' order), the other heads as
    they are: one product over the chosen heads, scattered back."""
    chosen = jnp.asarray(heads, jnp.int32)
    taken = attention[:, :, chosen]
    moved = jnp.einsum("bshd,hde->bshe", taken, projection.astype(attention.dtype), precision=precision)
    return attention.at[:, :, chosen].add(-(strength.astype(attention.dtype) * moved))


def folded(variables: Mapping[str, Mapping]) -> dict[str, Mapping]:
    """`variables` with each DEX layer's differential at its learned lambda folded into its output
    projection, the extension's parameters and constants dropped: a layer's `o_proj` kernel
    `[H Dv, D]` has each chosen head `h`'s rows `K_h` made `(I - lambda_learn W_h) K_h`, in float64
    and rounded once to the kernel's dtype, so the source's own layer computes what the extended one
    does outside a run (`annealed`), `O'_h K_h = O_h (I - lambda W_h) K_h`. A layer whose heads'
    outputs pass through more before `o_proj` does not fold (`HeadOutputs.dex_fold_refusal`)."""
    params: dict[tuple[str, ...], jax.Array | np.ndarray] = dict(leaves_by_path(variables["params"]))
    held = leaves_by_path(variables.get(DEX_COLLECTION, {}))
    for prefix in [key[:-1] for key in params if key[-1] == "dex_projection"]:
        projection = np.asarray(params.pop((*prefix, "dex_projection")), np.float64)
        strength = np.asarray(params.pop((*prefix, "dex_lambda")), np.float64)
        heads = np.asarray(held.pop((*prefix, "dex_heads")))
        del held[(*prefix, "dex_lambda_init")]
        kernel = params[(*prefix, "o_proj", "kernel")]
        wide = np.array(kernel, np.float64)
        width = projection.shape[-1]
        for slot, head in enumerate(heads):
            rows = slice(int(head) * width, (int(head) + 1) * width)
            wide[rows] -= strength * projection[slot] @ wide[rows]
        params[(*prefix, "o_proj", "kernel")] = wide.astype(np.asarray(kernel).dtype)
    rest = {name: tree for name, tree in variables.items() if name not in ("params", DEX_COLLECTION)}
    kept = {DEX_COLLECTION: unflatten_dict(held)} if held else {}
    return {**rest, "params": unflatten_dict(params), **kept}
