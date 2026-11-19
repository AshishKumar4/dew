"""A draw outside any trace as one program: the variables an objective's init makes (`drawn_once`) and the
start a trainer builds (`dew.training.distributed.start_build`), each lowered as a fresh state's draws are
(`drawing`), the held arrays its init returns as they are kept as those arrays (`passed_through`)."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import jax
from jax._src import config as jax_config

from dew.algebra.program import replaying
from dew.nn.kernels.options import owned_program


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


def passed_through[Tree, Result](function: Callable[[Tree, jax.Array], Result],
                                  picked: Callable[[Result], object] = lambda result: result
                                  ) -> tuple[Callable[[Tree, jax.Array], Result], dict[int, int]]:
    """`function` as a trace records which leaves of `picked` of its result (all of it where unstated)
    are leaves of its first argument as they are, by reference (`Objective.held_variables`): each such
    leaf's index among `picked`'s leaves, to the index of the argument's leaf it is, filled as it traces."""
    through: dict[int, int] = {}

    def traced(tree: Tree, key: jax.Array) -> Result:
        result = function(tree, key)
        inputs = {id(leaf): index for index, leaf in enumerate(jax.tree.leaves(tree))}
        through.update({index: inputs[id(leaf)] for index, leaf in enumerate(jax.tree.leaves(picked(result)))
                        if id(leaf) in inputs})
        return result

    return traced, through


def drawn_once[Tree, Result](function: Callable[[Tree, jax.Array], Result], held: Tree, key: jax.Array
                             ) -> Result:
    """`function(held, key)` for a caller outside any trace, by one program, its draws rolled (`drawing`),
    where run eagerly each operation of a model's draw compiled alone (tens to hundreds a model, row 8682):
    traced once, its leaves that are `held`'s as they are (`passed_through`) those arrays themselves, no
    output of the program, which draws the rest from its replay (`dew.algebra.program.replaying`)."""
    traced, through = passed_through(function)
    trace = owned_program(traced).trace(held, key)
    structure = jax.tree.structure(trace.out_info)
    replay = replaying(trace.jaxpr, structure, (), ())
    made = [index for index in range(structure.num_leaves) if index not in through]

    def rest(tree: Tree, draw: jax.Array) -> list[jax.Array]:
        leaves = jax.tree.leaves(replay(tree, draw))
        return [leaves[index] for index in made]

    with drawing():
        drawn = dict(zip(made, owned_program(rest)(held, key), strict=True))
    given = jax.tree.leaves(held)
    return jax.tree.unflatten(structure, [drawn[index] if index in drawn else given[through[index]]
                                          for index in range(structure.num_leaves)])
