"""The mean of runs' weights: a model soup of runs trained apart from one start, any objective's,
read as their loaders read them (`averaged`, `averaged_pinned`)."""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath

from dew.batch import Variables

if TYPE_CHECKING:
    from dew.inference.tasks import SavedRun
    from dew.registry import ModelConfig


@dataclasses.dataclass(frozen=True)
class Averaged:
    """The fp32 mean of runs' weights (`averaged`).

    `variables` is the tree their model applies with, its frozen leaves back in `params`; `model` the
    runs' model record with any LoRA adapter merged away, which builds the model the mean applies to;
    and `runs` each run's directory, step and whether its average (EMA) was read."""

    variables: Variables
    model: ModelConfig
    runs: tuple[tuple[str, int, bool], ...]


def averaged(runs: Sequence[str | os.PathLike[str]], *, step: int | str | None = None,
             ema: bool | None = None, mesh=None, layout=None, trust: Sequence[str] = ()) -> Averaged:
    """The fp32 mean of `runs`' effective weights, each run's at `step`, its latest where None
    (`SavedRun.at`, with `trust`), and `ema` as its loaders read them (`averaged_pinned`)."""
    from dew.inference.tasks import SavedRun

    return averaged_pinned([SavedRun.at(str(epath.Path(run)), step, trust) for run in runs], ema=ema,
                           mesh=mesh, layout=layout)


def averaged_pinned(runs: Sequence[SavedRun], *, ema: bool | None = None, mesh=None, layout=None) -> Averaged:
    """The fp32 mean of the pinned `runs`' effective weights (`SavedRun.pinned`), each read with `ema`
    as its loaders read it (its average where None and it keeps one): a model soup of runs trained
    apart from one start, any objective's.

    A run's weights are its model's tree (`SavedRun.variables`, read onto `mesh` as `layout`
    places it, as a run's loaders take them) in fp32 with its frozen leaves back in `params`, a LoRA
    run's factors merged into their kernels first (`Adapter.merge`), as the mean of two runs' factors
    is not the mean of the weights they make: a LoRA run, a partly frozen one and a whole fine-tune of
    one base average. The runs are of one model: their model records, adapters aside, the same, and
    their trees of the same leaves and shapes; any other run is refused (ValueError), naming it. Where
    each run started is the caller's to know: runs drawn from other initial weights average to no
    model of either. The mean is elementwise, so each process takes it over the shards it holds, on
    the host in fp32 (`_taken`), one run's shards beside it at a time, a subnormal kept as float64's
    mean rounds it where a device may flush it, and assembles its leaves as the first run's are
    placed: a pool averages runs no one process holds whole. An integer or boolean leaf must agree
    across the runs."""
    if len(runs) < 2:
        raise ValueError(f"an average is of two or more runs, got {len(runs)}")
    from dew._coordination import agreed
    from dew.distributed.host import place_leaf

    means: list = []
    structure, firsts, model, read = None, [], None, []
    for count, run in enumerate(runs, start=1):
        tree, base, entry = _effective_weights(run, ema, mesh, layout)
        if model is not None and base != model:
            raise ValueError(f"the run in {entry[0]} trains another model than {read[0][0]}'s, so their "
                             "weights are not one model's to average")
        model = base
        leaves, shape = jax.tree.flatten(tree)
        del tree
        if structure is None:
            structure, firsts = shape, [getattr(leaf, "sharding", None) for leaf in leaves]
            means = [_held(leaf, sharding) for leaf, sharding in zip(leaves, firsts, strict=True)]
            del leaves
            read.append(entry)
            continue
        if shape != structure or any(np.shape(leaf) != tuple(held.shape) for held, leaf in zip(
                means, leaves, strict=True)):
            raise ValueError(f"the run in {entry[0]} holds other variables than {read[0][0]}")

        # Every process places the run's leaves as the first run's before any compares them: a placement
        # is a collective, which no process may be left in because a peer refused. Then a process whose
        # shards refuse the run reports it where every process agrees on the run, so none of them reads
        # the next one alone.
        for index, placement in enumerate(firsts):
            leaf = leaves[index]
            if placement is not None and not (isinstance(leaf, jax.Array) and leaf.sharding == placement):
                leaves[index] = jax.device_put(leaf, placement)
        agreed(f"averaging {entry[0]}", functools.partial(_taken_in, means, leaves, firsts, count, entry[0],
                                                          read[0][0]))
        del leaves
        read.append(entry)
    assert structure is not None and model is not None
    placed = [held if sharding is None else place_leaf(held, sharding)
              for held, sharding in zip(means, firsts, strict=True)]
    return Averaged(jax.tree.unflatten(structure, placed), model, tuple(read))


def _taken_in(means: list, leaves: list, firsts: list, count: int, directory: str, first: str) -> None:
    """`means` with the `count`th run's `leaves` taken in (`_mean_of`), each let go in turn, so the run's
    tree leaves as the mean takes it."""
    for index, leaf in enumerate(leaves):
        means[index] = _mean_of(means[index], _held(leaf, firsts[index]), count, directory, first)
        leaves[index] = None


def _held(leaf, sharding):
    """`leaf`, placed as the first run's (`averaged` places it), on the host as the shards this process
    holds of it (`dew.distributed.host.Held`, which holds a replicated block once), a floating one in fp32;
    a host leaf whole. It reads this process's shards alone, no collective."""
    from dew.distributed.host import Held

    if sharding is None:
        return _on_host(leaf)
    held = Held.of(leaf)
    return Held(held.shape, {index: _on_host(block) for index, block in held.shards.items()})


def _mean_of(mean, taken, count: int, directory: str, first: str):
    """The running mean `mean` with the `count`th run's `taken` (`_held`'s), block by block (`_taken`)."""
    if isinstance(mean, np.ndarray):
        return _taken(mean, taken, count, directory, first)
    return type(mean)(mean.shape, {index: _taken(block, taken.shards[index], count, directory, first)
                                   for index, block in mean.shards.items()})


def _effective_weights(run: SavedRun, ema: bool | None, mesh, layout
                       ) -> tuple[Variables, ModelConfig, tuple[str, int, bool]]:
    """The pinned run's model tree as `averaged_pinned` takes it, its model record with any adapter
    merged away, and its directory, step and average read: the adapter's factors merged where the
    recorded model's tree is (`Objective._recorded_part`), beside any part its record leaves out, and
    the frozen leaves back in `params` (`thaw`). What the merge holds goes as this returns."""
    from dew.batch import part, thaw
    from dew.checkpoints import Checkpoints
    from dew.inference.tasks import saved_model
    from dew.lora import Adapter, AdapterRecord, recorded_adapter
    from dew.objectives.base import joined

    directory, pinned, objective = run.directory, run.step, run.objective
    config = saved_model(run.record, None)
    tree = run.variables(ema=ema, mesh=mesh, layout=layout, param_dtype=jnp.float32)
    adapter = recorded_adapter(config.wrappers)
    if adapter is not None:
        name = objective._recorded_part
        own = tree if name is None else part(tree, name)
        merged = Adapter.recorded(config.build(), own, adapter).merge(own)
        if name is None:
            tree = merged
        else:
            others = {key for collection in tree.values() for key in collection} - {name}
            tree = joined({name: merged, **{other: part(tree, other) for other in sorted(others)}})
    with Checkpoints(directory) as checkpoints:
        kept = checkpoints.stored(pinned).get("ema") is not None
    base = dataclasses.replace(config, wrappers=tuple(
        wrapper for wrapper in config.wrappers if not isinstance(wrapper, AdapterRecord)))
    return thaw(tree), base, (directory, pinned, not objective.ema_is_reference and (
        kept if ema is None else bool(ema)))


def _on_host(leaf) -> np.ndarray:
    """`leaf` on the host, a floating one in fp32, bf16 and the float8 types among them, which
    NumPy's own type hierarchy does not count as floating, and a complex one in complex64, both of its
    components fp32."""
    value = np.asarray(leaf)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return value.astype(np.complex64)
    return value.astype(np.float32) if jnp.issubdtype(value.dtype, jnp.inexact) else value


def _taken(held: np.ndarray, leaf: np.ndarray, count: int, directory: str, first: str) -> np.ndarray:
    """`held`, the fp32 mean of a leaf over `count - 1` runs, with the `count`th run's `leaf` taken in,
    in NumPy's fp32, which keeps subnormals: for a second run `(held + leaf) / 2` below 2**126, a sum
    that cannot overflow (and is exact wherever its half could round), and `held / 2 + leaf / 2`
    above, exact halves, one rounding either way, so float64's mean rounded to fp32; for a later one
    `held + (leaf - held) / count` where both share a sign, whose difference cannot overflow, and
    `held * (1 - 1/count) + leaf / count` where they differ, whose terms cannot. A complex leaf's real
    and imaginary parts are each so averaged. An integer or boolean leaf must equal `held`."""
    if jnp.issubdtype(held.dtype, jnp.complexfloating):
        # Each component as a real leaf of the same shape, a scalar's included.
        real, imaginary = (_taken(np.array(part(held)), np.array(part(leaf)), count, directory, first)
                           for part in (np.real, np.imag))
        averaged = real.astype(np.complex64)
        averaged.imag = imaginary
        return averaged
    if not jnp.issubdtype(held.dtype, jnp.inexact):
        if not np.array_equal(held, leaf):
            raise ValueError(f"an integer or boolean variable of the run in {directory} differs from "
                             f"{first}'s, which no mean of their weights reconciles")
        return held
    two = np.float32(2)
    # Each branch is computed whole and the other's lanes discarded: an overflow there is never read.
    with np.errstate(over="ignore"):
        if count == 2:
            small = np.maximum(np.abs(held), np.abs(leaf)) < np.float32(2.0**126)
            midpoint = np.where(small, (held + leaf) / two, held / two + leaf / two)
            return midpoint.astype(np.float32, copy=False)
        same = np.sign(held) == np.sign(leaf)
        moved = held + (leaf - np.where(same, held, np.float32(0))) / np.float32(count)
        weighted = held * np.float32((count - 1) / count) + leaf / np.float32(count)
        return np.where(same, moved, weighted).astype(np.float32, copy=False)
