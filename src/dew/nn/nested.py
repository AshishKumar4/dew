"""A nested program: a function an outer program compiles as one sub-program of its own, traced once for its
arguments' types, so a function called many times within one trace (a contraction under Dew's wide
arithmetic, row 3142) is traced once. It is the only bare jit beside Dew's compile point
(`dew.nn.kernels.options.OuterProgram`, SLOP025), and it compiles nothing at the top level: outside any
staging trace it runs its function as Python, so every program a process compiles is the compile point's."""

import functools
from collections.abc import Callable

import jax
from jax._src import core as jax_core
from jax._src.interpreters import partial_eval
from jax.extend import core

__all__ = ["nested", "staged"]


def staged() -> bool:
    """Whether a staging trace (a jit's, a scan's, `jax.make_jaxpr`'s) encloses the current call, beneath
    any eager transformation: a program called here is a part of the enclosing one, which compiles it.
    Under an eager `jax.vmap` or `jax.grad` alone, none does, and a jit called there compiles itself."""
    stack = jax_core.unsafe_get_trace_stack(jax_core.unsafe_get_current_trace())
    return any(isinstance(trace, partial_eval.DynamicJaxprTrace) for trace in stack)


def nested[**P, R](function: Callable[P, R], *, traced: Callable[[core.ClosedJaxpr], None] | None = None,
                   **arguments) -> Callable[P, R]:
    """`function` as a nested program (module docstring), under `jax.jit`'s `arguments` (its static
    arguments): within a staging trace (`staged`), its jit, which the enclosing program compiles as its
    own; outside any, `function` itself, run as Python or by the eager transformation around it.
    `traced`, where given, takes each staged call's program, the very one the enclosing program's call
    holds (its jit's trace of the call's arguments, which the call then reuses): what identifies Dew's
    own programs to a reader of the enclosing one (`dew.interop.sequence_causality.attested`)."""
    jitted = jax.jit(function, **arguments)

    @functools.wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        if not staged():
            return function(*args, **kwargs)
        if traced is not None:
            traced(jitted.trace(*args, **kwargs).jaxpr)
        return jitted(*args, **kwargs)

    return call
