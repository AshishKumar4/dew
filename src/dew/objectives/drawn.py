"""A draw outside any trace as one program: the variables an objective's init makes (`drawn_once`) and the
start a trainer builds (`dew.training.distributed.start_build`), each lowered as a fresh state's draws are
(`drawing`), the held arrays its init returns as they are kept as those arrays (`passed_through`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import jax
from jax._src import config as jax_config

from dew._cache import holding
from dew.algebra.program import replaying
from dew.nn.kernels.options import abstract_signature, held_context, owned_program


def drawing() -> contextlib.AbstractContextManager[None]:
    """Where a fresh state's program is lowered, wherever it is (`placed_start`, `start_compiled`,
    `start_compiling`): on a GPU, JAX's threefry draws as rolled loops over their rounds
    (`jax_threefry_rolled_lowering`, Dew's JAX build's from post100), in place of each draw's twenty rounds
    unrolled. The draws are the same bits; the program is what an init is, a draw for nearly every leaf, and
    unrolled it is mostly hash arithmetic whose code generation is its compile: a 142M DiT's start compiled
    on a GH200 in 23.8 s, 4.9 s rolled, and placed its state in 26.7 s where it now takes 7.6 s (a 4x1024
    MoE decoder's, 26.5 s and 5.4 s; placed in 41.5 s and 20.2 s; M-MC-13, row 6740).
    Rolled, a draw's rounds no longer fuse into the leaf it makes, which costs one round's state a leaf,
    partitioned as the leaf is (a 1 GB draw over eight GPUs held one shard of temporaries, and no
    collective, where JAX's threefry kernel, unpartitioned, held 6.44 GB). The setting is in JAX's trace
    context and jit key, so every site of the program enters it alike; a step, whose draws fuse into what
    reads them, keeps JAX's default."""
    return jax_config.threefry_rolled_lowering(new_val=True)


def passed_through[Tree, Built, Picked](function: Callable[[Tree, jax.Array], Built],
                                         picked: Callable[[Built], Picked]
                                         ) -> tuple[Callable[[Tree, jax.Array], Built], dict[int, int]]:
    """`function` as a trace records which leaves of `picked` of what it builds are leaves of its first
    argument as they are, by reference (`Objective.held_variables`): each such leaf's index among
    `picked`'s leaves, to the index of the argument's leaf it is, filled as it traces."""
    through: dict[int, int] = {}

    def traced(tree: Tree, key: jax.Array) -> Built:
        built = function(tree, key)
        inputs = {id(leaf): index for index, leaf in enumerate(jax.tree.leaves(tree))}
        through.update({index: inputs[id(leaf)] for index, leaf in enumerate(jax.tree.leaves(picked(built)))
                        if id(leaf) in inputs})
        return built

    return traced, through


@dataclasses.dataclass(frozen=True)
class Parts[Built]:
    """How what a draw builds splits into the rest and the part whose leaves may be held arrays passed
    through as they are (`split`), and joins again (`join`)."""

    split: Callable[[Built], tuple[Any, Any]]
    join: Callable[[Any, Any], Built]


WHOLE: Parts = Parts(lambda built: (None, built), lambda _, picked: picked)
"""A draw whose whole tree may hold arrays passed through: an objective's init."""


@dataclasses.dataclass(frozen=True)
class TracedDraw:
    """A draw's one trace (`traced_draw`): the shapes of what it builds (`abstract`), which leaves of its
    picked part are its held tree's arrays as they are (`through`, each one's index among the picked part's
    leaves to the held leaf's), and its build replayed from that trace (`replay`), which runs no model code;
    the programs an eager draw made of it, by the leaves they build (`programs`)."""

    abstract: Any
    through: dict[int, int]
    replay: Callable
    programs: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)


KEPT_DRAWS = 8
"""The traced draws a process keeps for a later call to read (`traced_draw`'s `kept`), the latest of them."""

_TRACED: collections.OrderedDict[Hashable, TracedDraw] = collections.OrderedDict()


def traced_draw[Tree, Built](function: Callable[[Tree, jax.Array], Built], held: Tree, key: jax.Array,
                             parts: Parts[Built], kept: Hashable | None = None) -> TracedDraw:
    """`function(held, key)` traced once (`TracedDraw`): the one trace a fresh state's start reads
    (`dew.training.distributed.start_build`, `placed_start`) and an eager draw runs (`drawn_once`). A 4B
    decoder's init traced twice took 41 and 36 s on a B200 host before its first step. Where `kept` states
    what it traces from beside its arguments, the trace its equal made under the same context
    (`held_context`) over arguments of the same shapes in this process is read (`dew._cache.holding`), the
    arguments by their leaves alone (the held tree holds its objective's bound init, which a key held past
    its trainer would keep alive): a trainer's refit traces its start once (PF03)."""
    def made() -> TracedDraw:
        traced, through = passed_through(function, lambda built: parts.split(built)[1])
        trace = owned_program(traced).trace(held, key)
        abstract = trace.out_info
        return TracedDraw(abstract, through, replaying(trace.jaxpr, jax.tree.structure(abstract), (), ()))

    if kept is None:
        return made()
    signature = (kept, held_context(), abstract_signature(jax.tree.leaves((held, key))))
    return holding(_TRACED, signature, made, KEPT_DRAWS)


def drawn_rest[Tree, Built](function: Callable[[Tree, jax.Array], Built], parts: Parts[Built],
                            built: Sequence[int]) -> Callable[[Tree, jax.Array], tuple[Any, tuple]]:
    """`function`'s build less the held arrays it passes through: its rest and its picked part's leaves at
    `built`, by their index among that part's leaves, what a draw's program makes (`drawn_once`,
    `dew.training.distributed.start_build`)."""
    def rest(tree: Tree, draw: jax.Array) -> tuple[Any, tuple]:
        other, picked = parts.split(function(tree, draw))
        leaves = jax.tree.leaves(picked)
        return other, tuple(leaves[index] for index in built)

    return rest


def reassembled[Built, Other](parts: Parts[Built], other: Other, made: Sequence[jax.Array],
                              built: Sequence[int], structure: jax.tree_util.PyTreeDef,
                              held_at: Callable[[int], jax.Array]) -> Built:
    """What a draw builds, from its program's rest and the leaves it made (`made`, at `built`), the picked
    part's other leaves the held arrays it passes through (`held_at`, by their index among its leaves)."""
    drawn = dict(zip(built, made, strict=True))
    return parts.join(other, jax.tree.unflatten(structure, [
        drawn[index] if index in drawn else held_at(index) for index in range(structure.num_leaves)]))


def drawn_once[Tree, Built](function: Callable[[Tree, jax.Array], Built], held: Tree, key: jax.Array,
                            parts: Parts[Built], *, kept: Hashable | None = None) -> Built:
    """`function(held, key)` for a caller outside any trace, by one program, its draws rolled (`drawing`),
    where run eagerly each operation of a model's draw compiled alone (tens to hundreds a model, row 8682):
    its trace (`traced_draw`, kept by `kept`) replayed, the held arrays it passes through those arrays
    themselves, no output of the program (`drawn_rest`, `reassembled`), its program kept with its trace so a
    second call compiles nothing (row 8731). The eager draw of an objective's init (`Objective.init`, its
    whole tree `WHOLE`) and a trainer's fresh state (`Trainer.initial_state`)."""
    draw = traced_draw(function, held, key, parts, kept)
    structure = jax.tree.structure(parts.split(draw.abstract)[1])
    built = tuple(index for index in range(structure.num_leaves) if index not in draw.through)
    if built not in draw.programs:
        draw.programs[built] = owned_program(drawn_rest(draw.replay, parts, built))
    with drawing():
        other, made = draw.programs[built](held, key)
    given = jax.tree.leaves(held)
    return reassembled(parts, other, made, built, structure, lambda index: given[draw.through[index]])
