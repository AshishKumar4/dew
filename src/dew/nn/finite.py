"""Whether values are finite, and a tree zeroed where they are not: the one statement a step's transaction
(`dew.training.transaction`) and a run that updates in its backward (`dew.nn.in_backward`) both read where a
run skips steps that are not finite (`TrainerAdvanced(skip_nonfinite=True)`), and the trees of a state a
step keeps finite and a save refuses non-finite (`checked`, `refuse_non_finite`), which the transaction and
`dew.checkpoints` read alike."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from dew._coordination import read_by_holders
from dew.nn.kernels.options import program
from dew.optim_schedules import injected_edited

if TYPE_CHECKING:
    from dew.batch import Variables
    from dew.state import ReplicaState, TrainState


def _inexact(leaf) -> bool:
    """Whether `leaf` is an array of floating or complex values, the only ones that can be non-finite."""
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)


def leaf_finite(leaf: jax.Array) -> jax.Array:
    """Whether `leaf` is finite throughout."""
    return jnp.isfinite(leaf).all()


def all_finite(tree) -> jax.Array:
    """Whether every floating leaf of `tree` is finite throughout, within a traced program."""
    flags = [leaf_finite(leaf) for leaf in jax.tree.leaves(tree) if _inexact(leaf)]
    return jnp.logical_and.reduce(jnp.stack(flags)) if flags else jnp.asarray(a=True)


def zeroed_unless(ok: jax.Array, tree):
    """`tree` where `ok`, its every floating leaf zero otherwise: an elementwise select against a constant,
    which the computation reading it fuses, so no second copy of the tree is held."""
    def gate(leaf):
        return jnp.where(ok, leaf, jnp.zeros((), leaf.dtype)) if _inexact(leaf) else leaf

    return jax.tree.map(gate, tree)


def checked(state: TrainState | ReplicaState) -> dict[str, Variables | optax.OptState | None]:
    """The trees of `state` (a run's, or DiLoCo's replicas' inner copies) a finite step keeps finite and a
    save refuses non-finite: the parameters, the optimizer state and the averages. The one statement of
    them, which the step's acceptance (`finish_attempt`) and a save's refusal
    (`refuse_non_finite`) read. The optimizer state is read without the
    hyperparameters injected beside it at any depth (`_numerical`): control values a schedule or the user
    sets, not written by a step's gradient, and legally infinite (a clip's bound of `math.inf`)."""
    return {"variables": state.variables, "opt_state": _numerical(state.opt_state), "ema": state.ema}


def _numerical(opt_state: optax.OptState) -> optax.OptState:
    """`opt_state` without the hyperparameters injected beside it at any depth (`injected_edited`): the
    state a step's gradients write."""
    return injected_edited(opt_state, lambda _, node: node._replace(hyperparams={}))



def _finite_each(leaves: list[jax.Array]) -> list[jax.Array]:
    """Whether each of `leaves` is finite throughout, one program over them."""
    return [leaf_finite(leaf) for leaf in leaves]


def finite_flags(leaves: Sequence[jax.Array]) -> list[bool]:
    """Whether each of `leaves`, arrays placed anywhere, is finite throughout (`leaf_finite`), read outside a
    step: one program for each placement the leaves share (their devices and memory kind), as no one
    program spans two meshes (a CPU-owned run's host companion and its accelerator), which hands back one
    flag for them all and, only where it is false, each leaf's; a leaf in host memory (a layout's
    pinned-host field) brought to its devices' memory one leaf at a time. A flag a leaf, read
    each check, took 0.19 s a check of the 4B's 1,493 leaves on a GH200 (UB-6861-FIT2). A placement no device
    of this process's holds (a pipeline stage's mesh another process's devices make, `StageLocal.saved`) is
    read by the processes that hold it, and every process takes their verdicts in one round, so each answers
    alike (row 10780: each rank read each stage's scalar, those outside a stage holding none of it)."""
    groups: dict[tuple, list[int]] = {}
    for index, leaf in enumerate(leaves):
        sharding = leaf.sharding
        groups.setdefault((frozenset(sharding.device_set), sharding.memory_kind), []).append(index)
    # Each placement's flags as this process read them, None where it holds none of its devices (on any
    # backend: a CPU-owned state's companion mesh is no default backend's).
    read: list[list[bool] | None] = []
    for (_, memory), members in groups.items():
        if not leaves[members[0]].sharding.addressable_devices:
            read.append(None)
        elif memory not in (None, "device"):
            read.append([_held_flag(program(leaf_finite)(jax.device_put(
                leaves[index], leaves[index].sharding.with_memory_kind("device")))) for index in members])
        else:
            held = [leaves[index] for index in members]
            read.append([True] * len(members) if _held_flag(program(all_finite)(held)) else
                        [_held_flag(finite) for finite in program(_finite_each)(held)])
    flags: list[bool] = [True] * len(leaves)
    for members, held in zip(groups.values(), read_by_holders(read, [devices for devices, _ in groups]),
                             strict=True):
        for index, finite in zip(members, held, strict=True):
            flags[index] = bool(finite)
    return flags


def _held_flag(finite: jax.Array) -> bool:
    """A verdict replicated over its placement's devices, read from one this process holds."""
    return bool(jax.device_get(finite.addressable_data(0)))


def first_non_finite(state: TrainState) -> str | None:
    """The first leaf of `state`'s checked trees (`checked`: the parameters, every optimizer
    leaf and the averages, and under DiLoCo each replica's inner copy of them) holding a value that is not
    finite, by its path; None where none does. Read wherever the leaves are placed, one compiled check a
    placement (`finite_flags`), the same on every process."""
    trees = {**checked(state), **({} if state.inner is None else
                                  {f"inner.{name}": tree for name, tree in checked(state.inner).items()})}
    leaves = [(f"{name}{jax.tree_util.keystr(path)}", leaf) for name, tree in trees.items()
              for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
              if _inexact(leaf)]
    finite = finite_flags([leaf for _, leaf in leaves])
    return next((path for (path, _), held in zip(leaves, finite, strict=True) if not held), None)


def refuse_non_finite(state: TrainState, step: int, latest: Callable[[], int | None]) -> None:
    """Refuse to save a `state` holding a value that is not finite (`first_non_finite`), on any tier: no
    checkpoint is written and none rotated out, and the run stops, naming the step and the first such leaf,
    so the checkpoint a resume finds is the last finite one (`latest`, read only to say so) (training.md,
    checkpoints and resume). Read at the writers' own boundary (`Checkpoints.save`, `save_local`), so a
    direct save is held to it as a trainer's is (row 9021)."""
    first = first_non_finite(state)
    if first is not None:
        kept = latest()
        said = "no checkpoint" if kept is None else f"the latest checkpoint, step {kept}"
        raise RuntimeError(f"step {step}'s state holds a non-finite value at {first}, so no checkpoint of it "
                           f"is written and {said} is the one a resume finds")
