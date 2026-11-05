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
    """Whether the model's mixture routes any of its layers by its router's scores: hash routing selects
    by the token alone, which no rounding moves, and DeepSeek V4 hashes only its first layers."""
    mixture = config.get("mixture")
    if mixture is None:
        return False
    hashed = set(mixture.get("hash_layers") or ())
    # A mixture that names no layers is every decoder layer's (`CausalTransformer.sparse_layers`).
    layers = mixture.get("layers")
    return bool(set(range(config["num_layers"]) if layers is None else layers) - hashed)


def selects(config) -> bool:
    """Whether a decoder built from `config` selects discontinuously: a lightning indexer's top-k, the
    model's own or a layer kind's (DeepSeek V4's compressed sparse attention), Gemma 3n's gaussian top-k
    or a mixture's routing."""
    mixers = [config.get("mixer", {}), *(kind.get("mixer", {}) for kind in config.get("kinds", {}).values())]
    sparsities = config.get("activation_sparsity_pattern", ())
    indexed = any((mixer or {}).get("fields", {}).get("index_topk") for mixer in mixers)
    return bool(indexed or any(sparsities) or routes(config))


def capturing(model, variables, *arguments, method=None, mutable=(), **options):
    """`model.apply` of `variables` and `arguments` (its forward, or its `method`) capturing its
    discontinuous selections, within a caller's own program: its output and the collections it
    mutated, the selections' (`capture`) and those of `mutable`."""
    from dew.nn.mla import SELECTIONS

    return model.apply(variables, *arguments, method=method, capture_intermediates=_selecting,
                       mutable=["intermediates", SELECTIONS, *mutable], **options)


def _selecting(module, method) -> bool:
    """Whether `capturing` keeps `module`'s call of `method`: a Router's selection scores, group scores
    and group mask, gpt-oss's and Gemma 4's router logits from their Dense, and Gemma 3n's gate from its
    gated MLP's Dense."""
    from dew.nn.gemma4_moe import Gemma4TextRouter
    from dew.nn.gpt_oss import GptOssMLP
    from dew.nn.mlp import GatedMLP
    from dew.nn.moe import Router

    if isinstance(module, Router):
        return method in ("selection", "group_scores", "group_mask")
    if isinstance(module, nn.Dense) and method == "__call__" and (
            (isinstance(module.parent, GptOssMLP) and module.name == "router")
            or (isinstance(module.parent, Gemma4TextRouter) and module.name == "proj")):
        return True
    parent = module.parent
    gated = isinstance(parent, GatedMLP) and bool(parent.activation_sparsity)
    return method == "__call__" and isinstance(module, nn.Dense) and module.name == "gate_proj" and gated


def capture(model, method=None):
    """`model`'s forward of its tokens (or its `method` of the arguments given, a decoder's prediction
    depths over its trunk's states) capturing its discontinuous selections (`capturing`), compiled
    once: the lightning indexer's scores and its heads' agreements before their relu, which it sows
    under `SELECTIONS` for a caller that opens that collection, Gemma 3n's gate from its gated MLP's
    Dense, a Router's selection scores, group scores and group mask from its own methods, and gpt-oss's
    and Gemma 4's router logits from their Dense. One compiled forward rather than an op at a time:
    XLA:CPU's JIT failed to materialize a gather fusion among the many small programs an unjitted
    capture compiles."""
    return jax.jit(lambda variables, *arguments: capturing(model, variables, *arguments, method=method)[1])


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


def scale_of(ranked: np.ndarray) -> np.ndarray:
    """Each row's size, the largest magnitude of its finite values: of scores of either sign, which a
    rounding of that size moves (a row of negative scores is as near a flip as its gap over the
    largest of them)."""
    finite = np.where(np.isfinite(ranked), np.abs(ranked), 0.0)
    return np.maximum(finite.max(axis=-1), np.finfo(np.float32).tiny)


def top_margin(values: np.ndarray, kept: int) -> np.ndarray:
    """Each row's gap between its `kept`-th largest value and the next, over its size (`scale_of`): how
    far a top-`kept` selection over the row's last axis is from flipping. Infinite where the row
    holds no more than `kept` finite values, which decide nothing."""
    ranked = -np.sort(-values, axis=-1)
    chosen, next_best = ranked[..., kept - 1], ranked[..., kept]
    with np.errstate(invalid="ignore"):
        gap = chosen - next_best
    return np.where(np.isfinite(next_best), gap / scale_of(ranked), np.inf)


def indexer_margins(scores: np.ndarray, agreements: np.ndarray, candidates: np.ndarray, budget: int, *,
                    relative: bool = True, stable: bool = True) -> np.ndarray:
    """Each query's margin `[..., S]` of a lightning indexer's top-`budget` over the keys `candidates`
    `[..., S, T]` lets it score (`scores` `[..., S, T]`): the gap between the kept and the next
    candidate's score, and where both are an exact zero tie of keys every head disagrees with
    (`agreements` `[..., S, H, T]`, its heads' before their relu), the least distance of a tied key's
    best agreement from zero: a key every head disagrees with by a rounding's width scores zero in one
    run and a little above or below it in another, entering or leaving a tie that index order breaks.
    `relative`, each over the query's largest score or agreement; else as they are, for a site that
    states its margin in the scores' own units. Where the selections compared are not all `stable`
    (torch's `topk` promises no order among equal scores), a tie has no margin at all. A query with no
    more candidates than `budget` decides nothing."""
    tiny = np.finfo(np.float32).tiny
    ranked = -np.sort(-np.where(candidates, scores, -np.inf), axis=-1)
    if ranked.shape[-1] <= budget:
        return np.full(ranked.shape[:-1], np.inf)
    kept, rejected = ranked[..., budget - 1], ranked[..., budget]
    tied = (kept == 0) & (rejected == 0)
    with np.errstate(invalid="ignore"):
        gap = kept - rejected
    best = np.min(np.where(candidates & (scores == 0), np.abs(agreements.max(axis=-2)), np.inf), axis=-1)
    if relative:
        gap = gap / scale_of(ranked)
        largest = np.max(np.where(candidates, np.abs(agreements).max(axis=-2), 0), axis=-1)
        best = best / np.maximum(largest, tiny)
    return np.where(tied, best if stable else 0.0, np.where(np.isfinite(rejected), gap, np.inf))


def pool_margins(scores: np.ndarray, visible: np.ndarray, blocks: int, block_size: int) -> np.ndarray:
    """Each query's margin `[..., S]` of a candidate pool's block choice (`dew.nn.sparse_selection.
    candidate_pool`) over its `scores` of the entries `visible` to it: the pool always holds the block of
    the query's newest entry and takes the other `blocks - 1` by their best score, so the margin is
    `top_margin` of the other blocks' best scores at `blocks - 1`; infinite where no more of them are
    reachable than it takes, which decides nothing."""
    total = scores.shape[-1]
    count = -(-total // block_size)
    ranked = np.where(visible, scores, -np.inf)
    ranked = np.pad(ranked, [(0, 0)] * (ranked.ndim - 1) + [(0, count * block_size - total)],
                    constant_values=-np.inf)
    best = ranked.reshape(*ranked.shape[:-1], count, block_size).max(axis=-1)
    newest = (visible.sum(axis=-1) - 1) // block_size
    others = np.where(np.arange(count) == newest[..., None], -np.inf, best)
    if blocks - 1 < 1 or count <= blocks - 1:
        return np.full(others.shape[:-1], np.inf)
    return top_margin(others, blocks - 1)


def causal(scores: np.ndarray) -> np.ndarray:
    """The keys each of `scores`' queries sees `[S, T]`: those at or before it, the queries the last."""
    queries, keys = scores.shape[-2:]
    return np.tril(np.ones((queries, keys), bool), keys - queries)


def margin(config, captured, label: str, *, stable: bool = True) -> float:
    """The smallest margin of the selections `captured` (`capture`'s output) of a decoder built from
    `config`, relative to their scale: the lightning indexer's top-k (the gap between the k-th and the
    next visible key's score, over the query's largest), Gemma 3n's gaussian top-k (`gate_margin`) and
    a mixture's routing, read from the router's own operands: a Router's best groups by its group
    scores and its experts by its selection scores within them (`top_margin` of each), gpt-oss's and
    Gemma 4's routers, which rank their logits or their softmax by a plain top-k, by their logits'. A
    query that sees no more keys than its budget decides nothing. Where its budget ends in a tie of
    keys every head disagrees with, scored zero and ordered by index, the tie holds only while each
    such key's heads stay below zero: its margin is the least distance of those keys' best agreement
    from zero, over the query's largest agreement (`indexer_margins`), where every selection compared
    breaks ties by index (`stable`); where one is torch's `topk`, which promises no tie order, a tie has
    none."""
    fields = config.get("mixer", {}).get("fields", {})
    sparsities = config.get("activation_sparsity_pattern", ())
    smallest, seen = np.inf, 0
    mixture = config.get("mixture") or {}
    routers: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    indexers: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    held = {"intermediates": captured.get("intermediates", {}), "selections": captured.get("selections", {})}
    for path, values in jax.tree_util.tree_flatten_with_path(held)[0]:
        seen += 1
        names = [str(getattr(part, "key", getattr(part, "idx", part))) for part in path]
        if names[0] == "selections" and len(names) >= 3 and names[-3] == "selection":
            # A selection sown whole (`dew.nn.mla.sow_selection`), keyed by its module and its call.
            indexers.setdefault(tuple(names[:-1]), {})[names[-1]] = np.asarray(values)
            continue
        if names[-2] in ("selection", "group_scores", "group_mask"):
            routers.setdefault(tuple(names[:-2]), {})[names[-2]] = np.asarray(values)
            continue
        if names[-2] in ("scores", "agreements"):
            # A call's scores and its heads' agreements, keyed by the indexer and the call, whichever
            # collection holds them.
            indexers.setdefault((*names[1:-2], names[-1]), {})[names[-2]] = np.asarray(values, np.float64)
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
        raise AssertionError(f"{label}: a selection {names} was captured that no margin reads")
    for operands in indexers.values():
        # A selection sown whole states the keys it may take and its budget; V3.2's sows its scores
        # alone, over the causal keys, at its model's budget.
        scores = np.asarray(operands["scores"], np.float64)
        visible = operands["visible"].astype(bool) if "visible" in operands else causal(scores)
        budget = int(operands["budget"]) if "budget" in operands else fields["index_topk"]
        agreements = np.asarray(operands["agreements"], np.float64)
        smallest = min(smallest, float(np.min(indexer_margins(scores, agreements, visible, budget,
                                                              stable=stable))))
        if "blocks" in operands:
            smallest = min(smallest, float(np.min(pool_margins(scores, visible, int(operands["blocks"]),
                                                               int(operands["block_size"])))))
    # A model that selects shows its selections here, or the draw would pass unfiltered.
    for operands in routers.values():
        # A Router chooses its best groups by their scores, then its experts among the groups kept.
        selection = np.asarray(operands["selection"], np.float64)
        if "group_scores" in operands:
            groups = np.asarray(operands["group_scores"], np.float64)
            smallest = min(smallest, float(np.min(top_margin(groups, mixture.get("groups_per_token", 1)))))
            # An expert outside the best groups is chosen by the score its router fills in.
            fill = mixture.get("excluded_group_score")
            selection = np.where(operands["group_mask"], selection, -np.inf if fill is None else fill)
        smallest = min(smallest, float(np.min(top_margin(selection, mixture.get("top_k", 2)))))
    assert seen, f"{label}: no selection was captured"
    return smallest


def routing_margin(sown, moe, kept: int) -> float:
    """The smallest margin of the routers' selections a forward sowed in its 'router' collection
    (`sown`; an objective's training forward returns it, trunk and prediction depths, so the margin
    is of the routes the step takes, its dropout included): each router's scores with the balancing
    bias the decoder's 'moe' collection holds for it, where it holds one (`Router.selection` of a text
    token), by `top_margin` at `kept`. A stacked router's bias stacks its layers as its scores do."""
    smallest, seen = np.inf, 0
    for path, scores in jax.tree_util.tree_flatten_with_path(sown)[0]:
        names = [str(getattr(part, "key", getattr(part, "idx", part))) for part in path]
        if "scores" not in names:
            continue
        seen += 1
        held = moe
        for name in names[:names.index("scores")]:
            held = held.get(name, {}) if isinstance(held, dict) else {}
        values = np.asarray(scores, np.float64)
        if isinstance(held, dict) and "e_score_correction_bias" in held:
            bias = np.asarray(held["e_score_correction_bias"], np.float64)
            between = (1,) * (values.ndim - bias.ndim)
            values = values + bias.reshape(bias.shape[:-1] + between + bias.shape[-1:])
        smallest = min(smallest, float(np.min(top_margin(values, kept))))
    # A routed forward sows its routers' scores, or the draw would pass unfiltered.
    if not seen:
        raise AssertionError("the forward sowed no router scores; open its 'router' collection")
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
