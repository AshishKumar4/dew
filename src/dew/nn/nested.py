"""A nested program: a function an outer program compiles as one sub-program of its own, traced once for its
arguments' types, so a function called many times within one trace (a contraction under Dew's wide
arithmetic, row 3142) is traced once. It is the only bare jit beside Dew's compile point
(`dew.nn.kernels.options.OuterProgram`, SLOP025), and it compiles nothing at the top level: outside any
trace it runs its function as Python, so every program a process compiles is the compile point's."""

import functools
from collections.abc import Callable

import jax
from jax._src import core as jax_core

__all__ = ["nested"]


def nested[**P, R](function: Callable[P, R], **arguments) -> Callable[P, R]:
    """`function` as a nested program (module docstring), under `jax.jit`'s `arguments` (its static
    arguments): within another trace, its jit, which the enclosing program compiles as its own; outside
    any, `function` itself."""
    jitted = jax.jit(function, **arguments)

    @functools.wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        return function(*args, **kwargs) if jax_core.trace_state_clean() else jitted(*args, **kwargs)

    return call
