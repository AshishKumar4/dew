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
from collections.abc import Iterator, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

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


@contextlib.contextmanager
def watching(modules: tuple[str, ...], *, histogram: bool = False) -> Iterator[None]:
    """Within the block, each module whose class name or path (`layers_3/attn`) a glob of `modules` matches
    (`*Attention`, `MLP`, `layers_3/*`) sows its output's summary (`statistics`) into `INSPECTED` where
    its apply opened it, under `SOWN`, the latest call's where a module runs more than once. A module
    whose output is no single array (a tuple, a cache) sows its first array's, over the positions a
    caller names (`watched_positions`) where its leading axes are theirs, whole otherwise. Nothing is sown
    where the apply opened no such collection, as a module sows nothing into a collection it cannot
    write."""
    def interceptor(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        module = context.module
        names = (type(module).__name__, "/".join(module.path))
        if context.method_name != "__call__" or not any(
                fnmatch.fnmatchcase(name, glob) for glob in modules for name in names):
            return output
        if not module.is_mutable_collection(INSPECTED):
            return output
        arrays = [leaf for leaf in jax.tree.leaves(output) if isinstance(leaf, jax.Array)]
        if arrays:
            first, positions = arrays[0], _POSITIONS.get()
            where = (positions if positions is not None and first.shape[:positions.ndim] == positions.shape
                     else None)
            module.sow(INSPECTED, SOWN, statistics(first, histogram=histogram, where=where),
                       init_fn=dict, reduce_fn=lambda previous, latest: latest)
        return output

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
        for index in np.ndindex(*leaf.shape):
            found["/".join(["activations", *module, *map(str, index), statistic])] = leaf[index]
    return found


@dataclasses.dataclass(frozen=True)
class Watched:
    """The activations a sampled step watches (`dew.training.inspection.Inspection.modules`): the modules
    by a glob of their class name or path (`watching`), and whether each summary holds a histogram."""

    modules: tuple[str, ...]
    histogram: bool = False
