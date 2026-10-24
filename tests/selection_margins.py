"""Discontinuous selections and the margin a draw clears them by: a lightning indexer's top-k, Gemma 3n's
gaussian top-k and a mixture's routing, read from one compiled forward that captures them (`capture`),
each margin relative to the scale of what it selects (`margin`). A check-1 site whose model selects draws
inputs whose margins clear MARGIN, so its float32 runs and its float64 run select alike: an input within
rounding of a boundary picks other keys, units or experts in other runs, a difference no precision
bounds (docs/design/testing.md asks a stated margin of such a draw)."""

import jax
import numpy as np
from flax import linen as nn

MARGIN = 1e-5
"""The least selection margin a draw admits (`margin`), relative to the scale of what is selected: above
the distance float32 rounding moves a selection's inputs from their float64 values, so every float32
run and the float64 run select alike. Gemma 3n's gate in tests/test_architectures.py's layout cell,
whose inputs carry the rounding of the attention and AltUp before it, moved at most 3.7e-6 of its row's
deviation over the 1,200 forwards of 320 draws, a light tail: each draw's largest was 1.6e-6 at the
median and 2.5e-6 at the 99th percentile."""


def routes(config) -> bool:
    """Whether the model's mixture routes by its router's scores: hash routing selects by the token
    alone, which no rounding moves."""
    mixture = config.get("mixture")
    return mixture is not None and not mixture.get("hash_layers")


def selects(config) -> bool:
    """Whether a decoder built from `config` selects discontinuously: a lightning indexer's top-k, Gemma
    3n's gaussian top-k or a mixture's routing."""
    fields = config.get("mixer", {}).get("fields", {})
    sparsities = config.get("activation_sparsity_pattern", ())
    return bool(fields.get("index_topk") or any(sparsities) or routes(config))


def capture(model):
    """`model`'s forward capturing its discontinuous selections, compiled once: the lightning indexer's
    scores from its `scores` method, Gemma 3n's gate from its gated MLP's Dense, a Router's selection
    scores, group scores and group mask from its own methods, and gpt-oss's and Gemma 4's router logits
    from their Dense. One compiled forward rather than an op at a time: XLA:CPU's JIT failed to
    materialize a gather fusion among the many small programs an unjitted capture compiles."""
    from dew.nn.gemma4_moe import Gemma4TextRouter
    from dew.nn.gpt_oss import GptOssMLP
    from dew.nn.mla import SparseIndexer
    from dew.nn.mlp import GatedMLP
    from dew.nn.moe import Router

    def selecting(module, method):
        if isinstance(module, SparseIndexer):
            return method == "scores"
        if isinstance(module, Router):
            return method in ("selection", "group_scores", "group_mask")
        if isinstance(module, nn.Dense) and method == "__call__" and (
                (isinstance(module.parent, GptOssMLP) and module.name == "router")
                or (isinstance(module.parent, Gemma4TextRouter) and module.name == "proj")):
            return True
        parent = module.parent
        gated = isinstance(parent, GatedMLP) and bool(parent.activation_sparsity)
        return method == "__call__" and isinstance(module, nn.Dense) and module.name == "gate_proj" and gated

    return jax.jit(lambda variables, tokens: model.apply(
        variables, tokens, capture_intermediates=selecting, mutable=["intermediates"])[1])


def gate_margin(values: np.ndarray, sparsity: float) -> float:
    """The smallest margin of Gemma 3n's gaussian top-k over gate rows `values`: each unit's distance
    from its row's cutoff, the mean plus the sparsity's normal quantile times the population
    deviation, as `dew.nn.mlp.gaussian_topk` forms it, over that deviation; the least kept unit's
    height over the cutoff and the cutoff's over the greatest dropped one."""
    from statistics import NormalDist

    values = np.asarray(values, np.float64)
    deviation = values.std(-1, keepdims=True)
    cutoff = values.mean(-1, keepdims=True) + NormalDist().inv_cdf(sparsity) * deviation
    kept = np.min(np.where(values > cutoff, values, np.inf), axis=-1, keepdims=True)
    dropped = np.max(np.where(values > cutoff, -np.inf, values), axis=-1, keepdims=True)
    return float(np.min(np.minimum(kept - cutoff, cutoff - dropped) / deviation))


def top_margin(values: np.ndarray, kept: int) -> np.ndarray:
    """Each row's gap between its `kept`-th largest value and the next, over its largest's size: how
    far a top-`kept` selection over the row's last axis is from flipping. Infinite where the row
    holds no more than `kept` finite values, which decide nothing."""
    ranked = -np.sort(-values, axis=-1)
    chosen, next_best = ranked[..., kept - 1], ranked[..., kept]
    scale = np.maximum(np.abs(ranked[..., 0]), np.finfo(np.float32).tiny)
    return np.where(np.isfinite(next_best), (chosen - next_best) / scale, np.inf)


def margin(config, captured, label: str) -> float:
    """The smallest margin of the selections `captured` (`capture`'s output) of a decoder built from
    `config`, relative to their scale: the lightning indexer's top-k (the gap between the k-th and the
    next visible key's score, over the query's largest), Gemma 3n's gaussian top-k (`gate_margin`) and
    a mixture's routing, read from the router's own operands: a Router's best groups by its group
    scores and its experts by its selection scores within them (`top_margin` of each), gpt-oss's and
    Gemma 4's routers, which rank their logits or their softmax by a plain top-k, by their logits'. A
    query that sees no more keys than its budget, and an exact zero tie of relu'd scores, ordered
    alike by index in every run, decide nothing."""
    fields = config.get("mixer", {}).get("fields", {})
    sparsities = config.get("activation_sparsity_pattern", ())
    smallest, seen = np.inf, 0
    mixture = config.get("mixture") or {}
    routers: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    for path, values in jax.tree_util.tree_flatten_with_path(captured.get("intermediates", {}))[0]:
        seen += 1
        names = [str(getattr(part, "key", part)) for part in path]
        if names[-2] in ("selection", "group_scores", "group_mask"):
            routers.setdefault(tuple(names[:-2]), {})[names[-2]] = np.asarray(values)
            continue
        values = np.asarray(values, np.float64)
        if "gate_proj" in names:
            layer = int(next(name for name in names if name.startswith("layers_")).split("_")[1])
            smallest = min(smallest, gate_margin(values, sparsities[layer]))
            continue
        if names[-3] in ("router", "proj") and names[-2] == "__call__":
            # gpt-oss ranks its router's logits, Gemma 4's router their softmax: one order, by logit.
            smallest = min(smallest, float(np.min(top_margin(values, mixture.get("top_k", 2)))))
            continue
        budget, (queries, keys) = fields["index_topk"], values.shape[-2:]
        if keys <= budget:
            continue
        visible = np.tril(np.ones((queries, keys), bool), keys - queries)
        ranked = -np.sort(-np.where(visible, values, -np.inf), axis=-1)
        kept, rejected = ranked[..., budget - 1], ranked[..., budget]
        decided = np.isfinite(rejected) & ~((kept == 0) & (rejected == 0))
        scale = np.maximum(np.abs(ranked[..., 0]), np.finfo(np.float32).tiny)
        smallest = min(smallest, float(np.min(np.where(decided, (kept - rejected) / scale, np.inf))))
    # A model that selects shows its selections here, or the draw would pass unfiltered.
    for operands in routers.values():
        # A Router chooses its best groups by their scores, then its experts among the groups kept.
        selection = np.asarray(operands["selection"], np.float64)
        if "group_scores" in operands:
            groups = np.asarray(operands["group_scores"], np.float64)
            smallest = min(smallest, float(np.min(top_margin(groups, mixture.get("groups_per_token", 1)))))
            selection = np.where(operands["group_mask"], selection, -np.inf)
        smallest = min(smallest, float(np.min(top_margin(selection, mixture.get("top_k", 2)))))
    assert seen, f"{label}: no selection was captured"
    return smallest


def clearing(draw, clears, tries: int = 64):
    """`draw(key)` at the first key from a seed whose inputs `clears`: the seed itself, then
    `1_000_003 * (attempt + 1) + seed`, apart from every seed of the windows a site draws (a verdict's
    from 0, a sizing's from SIZING_FROM, 2^20), which differ by no multiple of 1_000_003, so a redrawn
    seed takes inputs no other seed of them draws. A seed none of `tries` keys clears fails its site."""
    def drawn(seed):
        for attempt in range(tries):
            inputs = draw(seed if attempt == 0 else 1_000_003 * (attempt + 1) + seed)
            if clears(inputs):
                return inputs
        raise AssertionError(f"no draw of seed {seed} clears its selection margin")
    return drawn
