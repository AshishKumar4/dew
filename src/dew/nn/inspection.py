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
import functools
from collections.abc import Iterator, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import PROBES
from dew.nn.sharding import in_pipeline_stage

INSPECTED = "inspected"
"""The collection watched modules sow their outputs' summaries into."""

HISTOGRAM_EDGES = tuple(float(edge) for edge in range(-8, 5))
"""The base-10 exponents bounding an opt-in histogram's bins of magnitudes: a value below 1e-8 in the
first, above 1e4 in the last, one bin a decade between."""


def statistics(values: jax.Array, *, histogram: bool = False,
               where: jax.Array | None = None) -> dict[str, jax.Array]:
    """A tensor's summary in float32 over its global array: mean, standard deviation, root mean square,
    largest magnitude, the share of its entries that are not finite, which makes the others non-finite as
    they are of the values, and the share that is exactly zero; with `histogram`, the share of its
    magnitudes in each decade (`HISTOGRAM_EDGES`), one scalar a bin, `histogram_<bin>`. `where`, given,
    holds the entries summarized, its axes the values' leading ones, each true entry's every trailing
    one in."""
    values = values.astype(jnp.float32)
    keep = None if where is None else jnp.broadcast_to(
        where.reshape(where.shape + (1,) * (values.ndim - where.ndim)), values.shape)
    magnitudes = jnp.abs(values)
    found = {"mean": jnp.mean(values, where=keep), "std": jnp.std(values, where=keep),
             "rms": jnp.sqrt(jnp.mean(jnp.square(values), where=keep)),
             "abs_max": magnitudes.max() if keep is None else jnp.max(magnitudes, where=keep, initial=0.0),
             "non_finite": jnp.mean(~jnp.isfinite(values), where=keep),
             "zero_fraction": jnp.mean(values == 0, where=keep)}
    if histogram:
        exponents = jnp.log10(jnp.maximum(magnitudes, 1e-30)).reshape(-1)
        bins = jnp.searchsorted(jnp.asarray(HISTOGRAM_EDGES, jnp.float32), exponents)
        weights = 1.0 if keep is None else keep.reshape(-1).astype(jnp.float32)
        counted = exponents.size if keep is None else jnp.sum(keep)
        counts = jnp.zeros(len(HISTOGRAM_EDGES) + 1, jnp.float32).at[bins].add(weights) / counted
        found.update({f"histogram_{index}": counts[index] for index in range(counts.shape[0])})
    return found


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

COTANGENT_TERMS = ("count", "sum", "sum_squares", "non_finite", "zeros")
"""A probe's gradient's leading entries: the entries its cotangent counts, their sum and sum of squares,
and how many are not finite and how many exactly zero; its histogram's counts (`HISTOGRAM_EDGES`) follow
where the step asks for one. Each adds over whatever a gradient adds over (the microbatches of a pipeline,
the devices of a data axis), so the statistics follow from the total (`cotangent_statistics`)."""

_DISCOVERING: contextvars.ContextVar[bool] = contextvars.ContextVar("discovering_probes", default=False)


def probe_width(histogram: bool) -> int:
    """The entries of a probe (`COTANGENT_TERMS`, and a histogram's bins with `histogram`)."""
    return len(COTANGENT_TERMS) + (len(HISTOGRAM_EDGES) + 1 if histogram else 0)


def sufficient(values: jax.Array, where: jax.Array | None, histogram: bool) -> jax.Array:
    """`values`' sufficient statistics in float32 (`COTANGENT_TERMS`, a histogram's counts with
    `histogram`), over the entries `where` holds (`statistics`' `where`), every entry where None."""
    values = values.astype(jnp.float32)
    keep = (jnp.ones(values.shape, bool) if where is None else jnp.broadcast_to(
        where.reshape(where.shape + (1,) * (values.ndim - where.ndim)), values.shape))
    weights = keep.astype(jnp.float32)
    terms = [jnp.sum(weights), jnp.sum(jnp.where(keep, values, 0)),
             jnp.sum(jnp.where(keep, values * values, 0)), jnp.sum(weights * ~jnp.isfinite(values)),
             jnp.sum(weights * (values == 0))]
    if histogram:
        exponents = jnp.log10(jnp.maximum(jnp.abs(values), 1e-30)).reshape(-1)
        bins = jnp.searchsorted(jnp.asarray(HISTOGRAM_EDGES, jnp.float32), exponents)
        terms.extend(jnp.zeros(len(HISTOGRAM_EDGES) + 1, jnp.float32).at[bins].add(weights.reshape(-1)))
    return jnp.stack(terms)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def tapped(x: jax.Array, probe: jax.Array, where: jax.Array | None, histogram: bool) -> jax.Array:
    """`x` itself, its cotangent passed through unchanged, and `probe`'s the cotangent's sufficient
    statistics (`sufficient`) over `where`'s entries: a probe's gradient summarizes the loss's cotangent at
    `x` without a tensor of its size."""
    del probe, where
    return x


def _tapped_forward(x, probe, where, histogram):
    del probe
    return x, where


def _tapped_backward(histogram, where, cotangent):
    # The positions are booleans: their cotangent is float0's zero.
    unmoved = None if where is None else np.zeros(where.shape, jax.dtypes.float0)
    return cotangent, sufficient(cotangent, where, histogram), unmoved


tapped.defvjp(_tapped_forward, _tapped_backward)


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
            held = (jnp.zeros(value.shape, value.dtype) if key == PROBE
                    else kept(value) if key != SOWN else None)
            if held is not None:
                found[key] = held
        return found or None
    return kept(inspected) or {}


def cotangent_statistics(terms: jax.Array, factor: jax.Array, histogram: bool) -> dict[str, jax.Array]:
    """The statistics of a cotangent whose sufficient statistics are `terms` (`sufficient`), its loss
    scaled by `factor`: mean, standard deviation, root mean square, the share not finite and the share
    exactly zero, and with `histogram` the share of its magnitudes in each decade."""
    count = terms[0]
    mean = terms[1] / factor / count
    squares = terms[2] / jnp.square(factor) / count
    found = {"mean": mean, "std": jnp.sqrt(jnp.maximum(squares - jnp.square(mean), 0)),
             "rms": jnp.sqrt(squares), "non_finite": terms[3] / count, "zero_fraction": terms[4] / count}
    if histogram:
        found.update({f"histogram_{index}": value / count
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
        names = (type(module).__name__, "/".join(module.path))
        if context.method_name != "__call__" or not any(
                fnmatch.fnmatchcase(name, glob) for glob in modules for name in names):
            return output
        if not module.is_mutable_collection(INSPECTED):
            return output
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
        if discovering and in_pipeline_stage():
            raise ValueError(f"the cotangent of {'/'.join(module.path)}, in a pipelined stack's stage, is "
                             f"not summarized: its stage reads variables by layer, a probe its microbatches "
                             f"would lay out; watch it without cotangents, or train it unpipelined")
        if not discovering and not module.has_variable(PROBES, PROBE):
            return output
        # A zero probe where the step discovers its probes, and where it reads one, a mark that it did,
        # nested as the objective nests what the call sows (`probes_of`).
        module.sow(INSPECTED, PROBE, jnp.zeros(probe_width(histogram), jnp.float32),
                   init_fn=dict, reduce_fn=lambda previous, latest: latest)
        if discovering:
            return output
        leaves[index] = tapped(first, module.get_variable(PROBES, PROBE), where, histogram)
        return jax.tree.unflatten(tree, leaves)

    with nn.intercept_methods(interceptor):
        yield


def flattened(inspected: Mapping) -> dict[str, jax.Array]:
    """A sown `INSPECTED` collection's scalars by name, `activations/<module path>/<statistic>`. A summary
    a lifted transform stacked (a scanned stack's, one a layer; a pipeline's, one a microbatch) names
    each of its entries by its index, `.../<module path>/<i>/<statistic>`, one index a stacking axis."""
    found: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(inspected)[0]:
        keys = [str(entry.key) if isinstance(entry, jax.tree_util.DictKey) else str(entry) for entry in path]
        statistic, module = keys[-1], [key for key in keys[:-1] if key != SOWN]
        if statistic == PROBE:
            continue
        for index in np.ndindex(*leaf.shape):
            found["/".join(["activations", *module, *map(str, index), statistic])] = leaf[index]
    return found


def cotangent_summaries(gradients: Mapping, factor: jax.Array, histogram: bool) -> dict[str, jax.Array]:
    """A step's probes' gradients (`tapped`) as the statistics of each watched output's cotangent
    (`cotangent_statistics`), by name, `cotangents/<module path>/<statistic>`, a stacked probe's entries
    each by its index as `flattened` names an activation's."""
    found: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(gradients)[0]:
        keys = [str(entry.key) if isinstance(entry, jax.tree_util.DictKey) else str(entry) for entry in path]
        module = [key for key in keys if key != PROBE]
        for index in np.ndindex(*leaf.shape[:-1]):
            for statistic, value in cotangent_statistics(leaf[index], factor, histogram).items():
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
