"""Activation statistics a run watches (row 2317): while a sampled training step traces its loss, each
module whose class name a watched kind's glob matches sows its output's summary into one collection
(`INSPECTED`), which an objective's `apply` opens beside the others it reads
(`dew.objectives.base.Step.sown`).
Sown values travel as any collection does: a scanned stack returns one summary a layer, stacked on a first
axis (`nn.scan`'s `variable_axes`), and a rematerialized one its forward's, so they leave the module tree
as the outer trace's values, never an inner transform's.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import fnmatch
import re
from collections.abc import Iterator, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import zero_from_primal

from dew.batch import PROBES
from dew.nn.derivatives import Surrogate, contracted, declared_derivative
from dew.nn.sharding import in_pipeline_stage

INSPECTED = "inspected"
"""The collection watched modules sow their outputs' summaries into."""

HISTOGRAM_EDGES = tuple(float(edge) for edge in range(-8, 5))
"""The base-10 exponents bounding an opt-in histogram's bins of magnitudes: a value below 1e-8 in the
first, above 1e4 in the last, one bin a decade between."""


def statistics(values: jax.Array, *, histogram: bool = False,
               where: jax.Array | None = None) -> dict[str, jax.Array]:
    """A tensor's summary in float32 over its global array (`summarized`): mean, standard deviation, root
    mean square, the share of its entries that are not finite, which makes the others non-finite as they
    are of the values, the share that is exactly zero, and its largest magnitude; with `histogram`, the
    share of its magnitudes in each decade (`HISTOGRAM_EDGES`), one scalar a bin, `histogram_<bin>`.
    `where`, given, holds the entries summarized, its axes the values' leading ones, each true entry's
    every trailing one in. Its moments are taken about the entries' own mean, so a deviation far from
    zero is float32's to resolve, not the gap between two squares it cannot tell apart."""
    values = values.astype(jnp.float32)
    keep = _kept(values, where)
    count = jnp.sum(keep, dtype=jnp.float32)
    center = jnp.where(count > 0, jnp.sum(jnp.where(keep, values, 0)) / jnp.maximum(count, 1), 0)
    magnitudes = jnp.where(keep, jnp.abs(values), 0)
    return {**summarized(sufficient(values, where, histogram, center), histogram=histogram, center=center),
            "abs_max": magnitudes.max(initial=0.0)}


def _kept(values: jax.Array, where: jax.Array | None) -> jax.Array:
    """`where` over every entry of `values`, its axes their leading ones; every entry where None."""
    if where is None:
        return jnp.ones(values.shape, bool)
    return jnp.broadcast_to(where.reshape(where.shape + (1,) * (values.ndim - where.ndim)), values.shape)


SOWN = "__summary__"
"""The name a watched module's summary is sown under, one no module of Dew's names a child or a
variable, so it never collides with a module's own (a DiT's `output` head)."""


_POSITIONS: contextvars.ContextVar[jax.Array | None] = contextvars.ContextVar("watched_positions",
                                                                              default=None)


@contextlib.contextmanager
def watched_positions(positions: jax.Array | None) -> Iterator[None]:
    """Within the block, a watched module's summary of an output whose leading axes are `positions`'
    reads its true entries alone: the `[rows, tokens]` real positions of the input the model runs on, a
    scorer's as it prepared them (`dew.objectives.lm.TokenScorer.token_scores`) and a pipeline stage's
    microbatch of them (`dew.nn.backbones.decoder_stack.run_pipeline`). None reads every position."""
    token = _POSITIONS.set(positions)
    try:
        yield
    finally:
        _POSITIONS.reset(token)


def current_positions() -> jax.Array | None:
    """The positions a watched summary reads where the call runs (`watched_positions`), None for every."""
    return _POSITIONS.get()


PROBE = "__probe__"
"""The name a watched module's probe is sown under while a step discovers its probes, and read under."""

STAGED_PROBE = "__staged_probe__"
"""The name a pipeline stage's watched module sows its probe under, a microbatch axis ahead of it: the
stage runs each microbatch, which the pipeline lays out one by one as it does a summary
(`dew.nn.backbones.decoder_stack.run_pipeline`), while it reads its variables, the probe among them, by
layer. The probe is that leaf less its microbatch axis (`probes_of`), and its gradient sums the stage
layer's every microbatch."""

COTANGENT_TERMS = ("count", "sum", "sum_squares", "non_finite", "zeros")
"""A probe's gradient's leading entries: the entries its cotangent counts, their sum and sum of squares,
and how many are not finite and how many exactly zero; its histogram's counts (`HISTOGRAM_EDGES`) follow
where the step asks for one. Each adds over whatever a gradient adds over (the microbatches of a pipeline,
the devices of a data axis), so the statistics follow from the total (`summarized`)."""

_DISCOVERING: contextvars.ContextVar[bool] = contextvars.ContextVar("discovering_probes", default=False)

BY_PATH = "__by_path__"
"""The mark a stacked run's module sows where a glob takes it for one of its layers alone: the log keeps
the entries of the layers the glob names (`kept`)."""

_STACKED = re.compile(r"(?:^|(?<=/))(?:stages/)?layers_\d+(?:_\d+)?(?=/|$)")
"""A traced path's stacked layer component: a scanned run (`layers_3_7`) or a pipeline stage's layer
(`stages/layers_1`), whose module runs layers the log names apart."""

_LAYER = re.compile(r"(?:^|(?<=/))layers_[^/]*(?=/|$)")
_LIVE: contextvars.ContextVar[jax.Array | None] = contextvars.ContextVar("live_rows", default=None)
_SCALE: contextvars.ContextVar[jax.Array | None] = contextvars.ContextVar("loss_scale", default=None)


@contextlib.contextmanager
def unscaled(scale: jax.Array | None) -> Iterator[None]:
    """Within the block, a probe reads its cotangent divided by `scale`, the loss scale the step's backward
    is seeded with (`dew.training.transaction.Transaction.local_reduction`), so its statistics, a
    histogram's decades among them, are the unscaled loss's; None scales nothing."""
    token = _SCALE.set(scale)
    try:
        yield
    finally:
        _SCALE.reset(token)


@contextlib.contextmanager
def counted(live: jax.Array | None) -> Iterator[None]:
    """Within the block, a probe counts its cotangent's entries only where `live`, a boolean the call runs
    under, holds: false for a pipeline stage's bubble iteration, whose output no microbatch keeps, so its
    cotangent, zero, is none of the step's (`dew.nn.backbones.decoder_stack.run_pipeline`)."""
    token = _LIVE.set(live)
    try:
        yield
    finally:
        _LIVE.reset(token)


def probe_width(histogram: bool) -> int:
    """The entries of a probe (`COTANGENT_TERMS`, and a histogram's bins with `histogram`)."""
    return len(COTANGENT_TERMS) + (len(HISTOGRAM_EDGES) + 1 if histogram else 0)


def sufficient(values: jax.Array, where: jax.Array | None, histogram: bool,
               center: jax.Array | float = 0.0) -> jax.Array:
    """`values`' sufficient statistics in float32 over the entries `where` holds (`statistics`' `where`),
    about `center` (`COTANGENT_TERMS`, the first two moments of the entries less it), with `histogram` its
    magnitudes' counts in each decade; each term adds over whatever a sum of them adds over, so the
    summary of several follows from their total (`summarized`)."""
    values = values.astype(jnp.float32)
    keep = _kept(values, where)
    weights = keep.astype(jnp.float32)
    moved = jnp.where(keep, values - center, 0)
    terms = [jnp.sum(weights), jnp.sum(moved), jnp.sum(moved * moved),
             jnp.sum(weights * ~jnp.isfinite(values)), jnp.sum(weights * (values == 0))]
    if histogram:
        exponents = jnp.log10(jnp.maximum(jnp.abs(values), 1e-30)).reshape(-1)
        bins = jnp.searchsorted(jnp.asarray(HISTOGRAM_EDGES, jnp.float32), exponents)
        terms.extend(jnp.zeros(len(HISTOGRAM_EDGES) + 1, jnp.float32).at[bins].add(weights.reshape(-1)))
    return jnp.stack(terms)


def _passed(x: jax.Array, probe: jax.Array, where: jax.Array | None, scale: jax.Array,
            histogram: bool) -> jax.Array:
    """`x` itself, its cotangent passed through unchanged, and `probe`'s the sufficient statistics
    (`sufficient`) over `where`'s entries of that cotangent over `scale`, the loss scale its backward was
    seeded with: a probe's gradient summarizes the loss's cotangent at `x` without a tensor of its size. A
    surrogate (`tapped`, `dew.nn.derivatives.declared_derivative`): the probe's gradient is a derivative of
    Dew's choosing."""
    del probe, where, scale, histogram
    return x


def _tapped_forward(x, probe, where, scale, histogram):
    del probe
    return x, (where, scale)


def _tapped_backward(histogram, saved, cotangent):
    where, scale = saved
    terms = sufficient(cotangent.astype(jnp.float32) / scale, where, histogram)
    return cotangent, terms, zero_from_primal(where), zero_from_primal(scale)


tapped = contracted("test_a_probes_tap_holds_its_orders", "surrogate")(
    declared_derivative(Surrogate(), _passed, _tapped_forward, _tapped_backward, nondiff_argnums=(4,)))


@contextlib.contextmanager
def discovering() -> Iterator[None]:
    """Within the block, a watched module sows a zero probe (`PROBE`) where it would read one, for the
    step to lay its probes out as the objective's model calls nest them (`probes_of`)."""
    token = _DISCOVERING.set(True)
    try:
        yield
    finally:
        _DISCOVERING.reset(token)


def probes_of(inspected: Mapping) -> dict:
    """The zero probes a discovering step sowed (`discovering`), laid out as `PROBES` holds them: each
    under its module's path, stacked where a lifted transform stacked its module, nested as the objective
    nests what its model calls sow. Of a probing step's, the probes its model calls read, each marked where
    its module read it (`watching`)."""
    def kept(tree):
        if not isinstance(tree, Mapping):
            return None
        found = {}
        for key, value in tree.items():
            if key in (PROBE, STAGED_PROBE):
                # A stage's probe less the microbatch axis its pipeline laid it out along.
                found[PROBE] = jnp.zeros(value.shape[1:] if key == STAGED_PROBE else value.shape, value.dtype)
            elif key != SOWN and (held := kept(value)) is not None:
                found[key] = held
        return found or None
    return kept(inspected) or {}


def summarized(terms: jax.Array, *, histogram: bool,
               center: jax.Array | float = 0.0) -> dict[str, jax.Array]:
    """The summary of entries whose sufficient statistics about `center` are `terms` (`sufficient`): mean,
    standard deviation, root mean square, the share not finite and the share exactly zero, and with
    `histogram` the share of their magnitudes in each decade. No entries (a row set all padding, a probe
    no cotangent reached, its call under no gradient) summarize as a zero tensor: every statistic 0 but
    the zero share, 1, its magnitudes all below the lowest decade."""
    reached = terms[0] > 0
    count = jnp.where(reached, terms[0], 1)
    offset = terms[1] / count
    variance = jnp.maximum(terms[2] / count - jnp.square(offset), 0)
    mean = jnp.where(reached, center + offset, 0)
    found = {"mean": mean, "std": jnp.sqrt(variance), "rms": jnp.sqrt(variance + jnp.square(mean)),
             "non_finite": terms[3] / count, "zero_fraction": jnp.where(reached, terms[4] / count, 1)}
    if histogram:
        found.update({f"histogram_{index}": jnp.where(reached, value / count, index == 0)
                      for index, value in enumerate(terms[len(COTANGENT_TERMS):])})
    return found


@contextlib.contextmanager
def watching(modules: tuple[str, ...], *, histogram: bool = False,
             cotangents: bool = False) -> Iterator[None]:
    """Within the block, each module whose class name or path (`layers_3/attn`) a glob of `modules` matches
    (`*Attention`, `MLP`, `layers_3/*`) sows its output's summary (`statistics`) into `INSPECTED` where
    its apply opened it, under `SOWN`, the latest call's where a module runs more than once. A module
    whose output is no single array (a tuple, a cache) sows its first array's, over the positions a
    caller names (`watched_positions`) where its leading axes are theirs, whole otherwise. Nothing is sown
    where the apply opened no such collection, as a module sows nothing into a collection it cannot
    write. With `cotangents`, that array passes through the module's probe, which the apply reads from
    `PROBES` (`tapped`), or, while the step discovers its probes (`discovering`), a zero probe is sown
    in its place."""
    def interceptor(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        module = context.module
        path = "/".join(module.path)
        named = any(fnmatch.fnmatchcase(name, glob)
                    for glob in modules for name in (type(module).__name__, path))
        # A stacked run of layers (`layers_3_7`, a pipeline stage's `stages/layers_j`) is any of the layers
        # the log names its entries by: a glob over one of them takes the run, and the log keeps that layer's
        # entries alone (`flattened`).
        widened = _STACKED.sub("layers_*", path)
        candidate = widened != path and any(
            fnmatch.fnmatchcase(widened, _LAYER.sub("layers_*", glob)) for glob in modules)
        if context.method_name != "__call__" or not (named or candidate):
            return output
        if not module.is_mutable_collection(INSPECTED):
            return output
        if not named:
            module.sow(INSPECTED, BY_PATH, jnp.zeros((), jnp.bool_), init_fn=dict,
                       reduce_fn=lambda previous, latest: latest)
        leaves, tree = jax.tree.flatten(output)
        index = next((index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.Array)), None)
        if index is None:
            return output
        first, positions = leaves[index], _POSITIONS.get()
        where = (positions if positions is not None and first.shape[:positions.ndim] == positions.shape
                 else None)
        module.sow(INSPECTED, SOWN, statistics(first, histogram=histogram, where=where),
                   init_fn=dict, reduce_fn=lambda previous, latest: latest)
        if not cotangents:
            return output
        discovering = _DISCOVERING.get()
        if discovering and in_pipeline_stage() and module.path[-1:] == ("stages",):
            raise ValueError(f"{'/'.join(module.path)} runs a pipeline stage's layers, whose cotangent no "
                             f"layer of the log holds; watch its layers' (`layers_*/*`) for their cotangents")
        if not discovering and not module.has_variable(PROBES, PROBE):
            return output
        # A zero probe where the step discovers its probes, and where it reads one, a mark that it did,
        # nested as the objective nests what the call sows (`probes_of`); a stage's a microbatch apart.
        staged = in_pipeline_stage()
        zero = jnp.zeros(probe_width(histogram), jnp.float32)
        module.sow(INSPECTED, STAGED_PROBE if staged else PROBE, zero[None] if staged else zero,
                   init_fn=dict, reduce_fn=lambda previous, latest: latest)
        if discovering:
            return output
        live = _LIVE.get()
        if live is not None:
            rows = jnp.broadcast_to(live, first.shape[:1])
            where = rows if where is None else where & rows.reshape(rows.shape + (1,) * (where.ndim - 1))
        scale = jnp.asarray(1. if _SCALE.get() is None else _SCALE.get(), jnp.float32)
        leaves[index] = tapped(first, module.get_variable(PROBES, PROBE), where, scale, histogram)
        return jax.tree.unflatten(tree, leaves)

    with nn.intercept_methods(interceptor):
        yield


def kept(inspected: Mapping, modules: tuple[str, ...]) -> set[tuple[str, ...]]:
    """The module paths of a sown `INSPECTED` collection the log keeps: each a glob named (`watching`), a
    stacked run's module only at the layers a glob names by their logged path (`BY_PATH`)."""
    found: set[tuple[str, ...]] = set()

    def walk(tree, path: tuple[str, ...]):
        if not isinstance(tree, Mapping):
            return
        if SOWN in tree and (BY_PATH not in tree or any(
                fnmatch.fnmatchcase("/".join(path), glob) for glob in modules)):
            found.add(path)
        for key, value in tree.items():
            walk(value, (*path, str(key)))

    walk(inspected, ())
    return found


def flattened(inspected: Mapping, modules: tuple[str, ...] = ()) -> dict[str, jax.Array]:
    """A sown `INSPECTED` collection's scalars by name, `activations/<module path>/<statistic>`, of the
    modules the log keeps (`kept`, by the watched globs `modules`). A summary a lifted transform stacked
    (a pipeline's, one a microbatch) names each of its entries by its index,
    `.../<module path>/<i>/<statistic>`, one index a stacking axis."""
    logged = kept(inspected, modules)
    found: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(inspected)[0]:
        keys = [str(entry.key) if isinstance(entry, jax.tree_util.DictKey) else str(entry) for entry in path]
        statistic, module = keys[-1], [key for key in keys[:-1] if key != SOWN]
        if statistic in (PROBE, STAGED_PROBE, BY_PATH) or tuple(module) not in logged:
            continue
        for index in np.ndindex(*leaf.shape):
            found["/".join(["activations", *module, *map(str, index), statistic])] = leaf[index]
    return found


def cotangent_summaries(gradients: Mapping, histogram: bool,
                        logged: set[tuple[str, ...]] | None = None) -> dict[str, jax.Array]:
    """A step's probes' gradients (`tapped`) as the statistics of each watched output's cotangent
    (`summarized`), by name, `cotangents/<module path>/<statistic>`, a stacked probe's entries
    each by its index as `flattened` names an activation's; with `logged`, of those module paths alone
    (`kept`)."""
    found: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(gradients)[0]:
        keys = [str(entry.key) if isinstance(entry, jax.tree_util.DictKey) else str(entry) for entry in path]
        module = [key for key in keys if key != PROBE]
        if logged is not None and tuple(module) not in logged:
            continue
        for index in np.ndindex(*leaf.shape[:-1]):
            for statistic, value in summarized(leaf[index], histogram=histogram).items():
                found["/".join(["cotangents", *module, *map(str, index), statistic])] = value
    return found


@dataclasses.dataclass(frozen=True)
class Watched:
    """The activations a sampled step watches (`dew.training.inspection.Inspection.modules`): the modules
    by a glob of their class name or path (`watching`), and whether each summary holds a histogram."""

    modules: tuple[str, ...]
    histogram: bool = False
    cotangents: bool = False
    """Whether the watched outputs' cotangents are summarized too (`tapped`)."""
