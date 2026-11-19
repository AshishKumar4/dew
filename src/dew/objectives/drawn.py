"""A draw outside any trace as one program: the variables an objective's init makes (`drawn_once`) and the
start a trainer builds (`dew.training.distributed.start_build`), each lowered as a fresh state's draws are
(`drawing`), the held arrays its init returns as they are kept as those arrays (`passed_through`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
from collections.abc import Callable, Hashable, Sequence

import jax
from jax._src import config as jax_config

from dew._cache import holding
from dew.algebra.program import replaying
from dew.nn.kernels.options import OuterProgram, abstract_signature, held_context, owned_program


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
class Parts[Built, Rest, Picked]:
    """How what a draw builds splits into the rest and the part whose leaves may be held arrays passed
    through as they are (`split`), and joins again (`join`)."""

    split: Callable[[Built], tuple[Rest, Picked]]
    join: Callable[[Rest, Picked], Built]


def _whole[Built](built: Built) -> tuple[None, Built]:
    """`built` whole, its rest none."""
    return None, built


def _joined[Built](_: None, picked: Built) -> Built:
    """`picked`, a whole build."""
    return picked


WHOLE: Parts = Parts(_whole, _joined)
"""A draw whose whole tree may hold arrays passed through: an objective's init."""


type Made[Rest] = tuple[Rest, tuple[jax.Array, ...]]
"""What a draw's program returns (`drawn_rest`): its build's rest and the leaves it made."""


@dataclasses.dataclass(frozen=True)
class TracedDraw[Tree, Built, Rest]:
    """A draw's one trace (`traced_draw`): the shapes of what it builds (`abstract`), which leaves of its
    picked part are its held tree's arrays as they are (`through`, each one's index among the picked part's
    leaves to the held leaf's) and which it builds (`built`, the others), its build replayed from that trace
    (`replay`), which runs no model code, and that build less the arrays it passes through (`rest`,
    `drawn_rest`), the eager draw's one program (`program`), made the first time one runs it."""

    abstract: Built
    through: dict[int, int]
    built: tuple[int, ...]
    replay: Callable[[Tree, jax.Array], Built]
    rest: Callable[[Tree, jax.Array], Made[Rest]]

    @functools.cached_property
    def program(self) -> OuterProgram:
        """The program an eager draw runs (`drawn_once`): `rest`'s, one for the trace."""
        return owned_program(self.rest)


KEPT_DRAWS = 8
"""The traced draws a process keeps for a later call to read (`traced_draw`'s `kept`), the latest of them."""

_TRACED: collections.OrderedDict[Hashable, TracedDraw] = collections.OrderedDict()


def held_signature[Tree](tree: Tree) -> Hashable:
    """What a kept trace over `tree` is keyed by: each leaf's path in it, as a string, and the leaves'
    abstract signatures (`abstract_signature`). The paths and not the tree's structure, which holds what
    its nodes bind (an initializer's objective, its bound init), so a trace held past its trainer keeps none
    of it alive; the paths all the same, so two trees of leaves alike under other keys are two keys (row
    8989: flattened first, they shared one trace, the second replaying the first's structure)."""
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    return (tuple(jax.tree_util.keystr(path) for path, _ in flat),
            abstract_signature([leaf for _, leaf in flat]))


def traced_draw[Tree, Built, Rest, Picked](function: Callable[[Tree, jax.Array], Built], held: Tree,
                                           key: jax.Array, parts: Parts[Built, Rest, Picked],
                                           kept: Hashable | None = None) -> TracedDraw[Tree, Built, Rest]:
    """`function(held, key)` traced once (`TracedDraw`): the one trace a fresh state's start reads
    (`dew.training.distributed.start_build`, `placed_start`) and an eager draw runs (`drawn_once`). A 4B
    decoder's init traced twice took 41 and 36 s on a B200 host before its first step. Where `kept` states
    what it traces from beside its arguments, the trace its equal made under the same context
    (`held_context`) over arguments of the same paths and shapes in this process is read
    (`dew._cache.holding`, `held_signature`): a trainer's refit traces its start once (PF03)."""
    def made() -> TracedDraw[Tree, Built, Rest]:
        traced, through = passed_through(function, lambda built: parts.split(built)[1])
        trace = owned_program(traced).trace(held, key)
        abstract = trace.out_info
        leaves = jax.tree.structure(parts.split(abstract)[1]).num_leaves
        built = tuple(index for index in range(leaves) if index not in through)
        replay = replaying(trace.jaxpr, jax.tree.structure(abstract), (), ())
        return TracedDraw(abstract, through, built, replay, drawn_rest(replay, parts, built))

    if kept is None:
        return made()
    signature = (kept, held_context(), held_signature((held, key)))
    return holding(_TRACED, signature, made, KEPT_DRAWS)


def drawn_rest[Tree, Built, Rest, Picked](function: Callable[[Tree, jax.Array], Built],
                                          parts: Parts[Built, Rest, Picked], built: Sequence[int]
                                          ) -> Callable[[Tree, jax.Array], Made[Rest]]:
    """`function`'s build less the held arrays it passes through: its rest and its picked part's leaves at
    `built`, by their index among that part's leaves, what a draw's program makes (`drawn_once`,
    `dew.training.distributed.start_build`)."""
    def rest(tree: Tree, draw: jax.Array) -> Made[Rest]:
        other, picked = parts.split(function(tree, draw))
        leaves = jax.tree.leaves(picked)
        return other, tuple(leaves[index] for index in built)

    return rest


def reassembled[Built, Rest, Picked](parts: Parts[Built, Rest, Picked], other: Rest,
                                     made: Sequence[jax.Array], built: Sequence[int],
                                     unflatten: Callable[[list[jax.Array]], Picked], leaves: int,
                                     held_at: Callable[[int], jax.Array]) -> Built:
    """What a draw builds, from its program's rest and the leaves it made (`made`, at `built`), the picked
    part's other leaves the held arrays it passes through (`held_at`, by their index among its `leaves`),
    the picked part made of them by `unflatten` (its tree structure's)."""
    drawn = dict(zip(built, made, strict=True))
    return parts.join(other, unflatten([drawn[index] if index in drawn else held_at(index)
                                        for index in range(leaves)]))


def drawn_once[Tree, Built, Rest, Picked](function: Callable[[Tree, jax.Array], Built], held: Tree,
                                          key: jax.Array, parts: Parts[Built, Rest, Picked], *,
                                          kept: Hashable | None = None) -> Built:
    """`function(held, key)` for a caller outside any trace, by one program, its draws rolled (`drawing`),
    where run eagerly each operation of a model's draw compiled alone (tens to hundreds a model, row 8682):
    its trace (`traced_draw`, kept by `kept`) replayed, the held arrays it passes through those arrays
    themselves, no output of the program (`drawn_rest`, `reassembled`), its program kept with its trace so a
    second call compiles nothing (row 8731). A draw whose every leaf is a held array passed through, and
    whose rest holds none, runs no program: it builds nothing (row 9069). The eager draw of an objective's
    init (`Objective.init`, its whole tree `WHOLE`) and a trainer's fresh state (`Trainer.initial_state`)."""
    draw = traced_draw(function, held, key, parts, kept)
    rest, picked = parts.split(draw.abstract)
    structure, nothing = jax.tree.structure(picked), not jax.tree.leaves(rest)
    if draw.built or not nothing:
        with drawing():
            other, made = draw.program(held, key)
    else:
        other, made = jax.tree.structure(rest).unflatten([]), ()
    given = jax.tree.leaves(held)
    return reassembled(parts, other, made, draw.built, structure.unflatten, structure.num_leaves,
                       lambda index: given[draw.through[index]])
