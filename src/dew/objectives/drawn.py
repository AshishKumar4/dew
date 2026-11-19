"""A draw outside any trace as one program: the variables an objective's init makes (`drawn_once`) and the
start a trainer builds (`dew.training.distributed.start_build`), each lowered as a fresh state's draws are
(`drawing`), the held arrays its init returns as they are kept as those arrays (`passed_through`)."""

from __future__ import annotations

import collections
import contextlib
from collections.abc import Callable, Hashable

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


KEPT_DRAWS = 8
"""The draws a process keeps for a later call to read (`drawn_once`'s `kept`), the latest of them."""

_DRAWS: collections.OrderedDict[Hashable, tuple] = collections.OrderedDict()


def drawn_once[Tree, Built, Picked](function: Callable[[Tree, jax.Array], Built], held: Tree, key: jax.Array,
                                    picked: Callable[[Built], Picked], *, kept: Hashable | None = None
                                    ) -> Built:
    """`function(held, key)` for a caller outside any trace, by one program, its draws rolled (`drawing`),
    where run eagerly each operation of a model's draw compiled alone (tens to hundreds a model, row 8682):
    traced once, the leaves of `picked` of what it builds that are `held`'s as they are (`passed_through`)
    those arrays themselves, no output of the program, which draws the rest from its replay
    (`dew.algebra.program.replaying`); the rest are buffers of their own. Where `kept` states what it traces
    from beside its arguments, its trace and program are kept by it, the trace's context and the arguments'
    shapes (`dew._cache.holding`), so a second call compiles nothing (rows 8731, train 1080's F2). The one
    eager draw: an objective's init (`Objective.init`) and a trainer's fresh state (`Trainer.initial_state`,
    its frozen tree `picked`)."""

    def made() -> tuple:
        traced, through = passed_through(function, picked)
        trace = owned_program(traced).trace(held, key)
        structure = jax.tree.structure(trace.out_info)
        # Each picked leaf's place among the built tree's leaves, read off a tree of its places.
        places = jax.tree.leaves(picked(jax.tree.unflatten(structure, list(range(structure.num_leaves)))))
        given = {places[index]: leaf for index, leaf in through.items()}
        replay = replaying(trace.jaxpr, structure, (), ())
        drawn = [index for index in range(structure.num_leaves) if index not in given]

        def rest(tree: Tree, draw: jax.Array) -> list[jax.Array]:
            leaves = jax.tree.leaves(replay(tree, draw))
            return [leaves[index] for index in drawn]

        return structure, given, drawn, owned_program(rest)

    signature = (held_context(), abstract_signature(jax.tree.leaves((held, key))))
    structure, given, drawn, program = (made() if kept is None
                                        else holding(_DRAWS, (kept, signature), made, KEPT_DRAWS))
    with drawing():
        leaves = dict(zip(drawn, program(held, key), strict=True))
    inputs = jax.tree.leaves(held)
    return jax.tree.unflatten(structure, [leaves[index] if index in leaves else inputs[given[index]]
                                          for index in range(structure.num_leaves)])
