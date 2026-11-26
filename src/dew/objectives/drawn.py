"""A draw outside any trace as one program: the variables an objective's init makes (`drawn_once`) and the
start a trainer builds (`dew.training.distributed.start_build`), each lowered as a fresh state's draws are
(`drawing`), the held arrays its init returns as they are kept as those arrays (`passed_through`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import json
from collections.abc import Callable, Hashable, Iterator, Sequence

import jax
from jax._src import config as jax_config

from dew._cache import holding
from dew._facts import Held
from dew._records import JSON
from dew._stated import tree_shape
from dew.algebra.program import replaying
from dew.nn.kernels.options import OuterProgram, abstract_signature, held_context, owned_program


@contextlib.contextmanager
def drawing() -> Iterator[None]:
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
    reads them, keeps JAX's default.
    The algebra pass is off within it (`dew.algebra.passes`): the forward an init traces makes no variable,
    so its bodies, rewritten again under this program's own trace context, rewrote nothing the run reads (an
    every-template step's start missed 562 of its 803 structural lookups on an sm100, row 11803)."""
    from dew.algebra import passes

    with jax_config.threefry_rolled_lowering(new_val=True), passes(on=False):
        yield


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

_TRACED: collections.OrderedDict[Hashable, Held[TracedDraw]] = collections.OrderedDict()


def held_signature[Tree](tree: Tree) -> Hashable:
    """What a kept trace over `tree` is keyed by: its structure node by node (`dew._stated.tree_shape`), an
    empty collection and a None among its nodes, each node's data as plain data (`_plainly`), and its leaves'
    abstract signatures (`abstract_signature`). Two trees of leaves alike under other keys (row 8989), or
    apart only by an empty collection (row 9548: `{**variables, "batch_stats": {}}` replayed without it),
    are two keys; a node's data that is no plain value (`Partial`'s bound function) by its kind alone, so
    a trace held past its trainer keeps none of what its nodes bind alive."""
    shape = tree_shape(jax.tree.structure(tree), "a kept draw's held tree", _plainly)
    return json.dumps(shape, sort_keys=True), abstract_signature(jax.tree.leaves(tree))


def _plainly(value: Hashable | list[Hashable], *, at: str) -> JSON:
    """A node's kind or data as a kept draw's key states it (`held_signature`): a class by its path, an enum
    member by its class's path and its name, a number, a string, a truth value or None as itself, a sequence
    (a mapping's keys) item by item; anything else by its kind's path alone."""
    del at
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, enum.Enum):
        return [_plainly(type(value), at=""), value.name]
    if isinstance(value, list | tuple):
        return [_plainly(part, at="") for part in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return _plainly(type(value), at="")


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
        from dew.nn.kernels import generation
        from dew.nn.kernels.templates import report

        traced, through = passed_through(function, lambda built: parts.split(built)[1])
        # The launches its trace reaches certified aside and settled as it ends (`report.certifying_aside`):
        # certified in turn they took a start's tracing thread before its step's lowering (row 5324); a
        # launch the settling refused leaves the start's lowering to step past it.
        # A draw selects no kernel (`generation.drawing`): its forward's values reach none of the variables.
        from dew.algebra import passes

        # Nor is a body of its forward rewritten (`drawing`'s algebra pass off): none of its values is a
        # variable.
        with report.certifying_aside(), generation.drawing(), passes(on=False):
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
