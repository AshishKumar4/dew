"""A fusion that reads one value both directly and through a view computes from what it reads: its result is
written over none of the value before the value's last read (the fork's post118 XLA, upstream bdf1e5789f,
"Check all operand uses before sharing buffers"). Before it, XLA's buffer assignment let an
elementwise fusion's result share the buffer of an operand the fusion also reads reversed through a bitcast
of it: on post117 half of such a sum's elements came out wrong on a CPU and on an A100, the sum placed in its
operand's own buffer (FS-VIEW-CPU, PK-POST117-VIEW-a100), none at post118. The programs are upstream's
ViewReversedDoesNotCorruptSharedBuffer and TemporaryViewReversedDoesNotCorruptSharedBuffer, in JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SIZE = 1 << 22
"""The value's elements: on a GPU many more blocks than run at once, so a sharing fusion's first writes land
before its last reads."""


def _summed_with_its_reverse(a):
    """`-a` plus its reverse, read once directly and once through a bitcast view, each kept apart from the
    fusion by an optimization barrier as upstream's HLO keeps them."""
    x = jax.lax.optimization_barrier(-a)
    view = jax.lax.optimization_barrier(x.reshape(1, a.size))
    return x + jnp.flip(view, 1).reshape(a.size)


def _as_a_temporary(a):
    """The same sum, a temporary the program reads again."""
    return -jax.lax.optimization_barrier(_summed_with_its_reverse(a))[::2]


PROGRAMS = {"the sum the result": (_summed_with_its_reverse, lambda a: -(a + a[::-1])),
            "the sum a temporary": (_as_a_temporary, lambda a: (a + a[::-1])[::2])}
"""Each program and its exact integer result."""


def a_value_read_through_a_reversed_view_is_summed_exactly(program: str) -> None:
    """`program` compiled once on this process's default device and run three times, each run its exact
    integer result: an arm with no site (`reference_error.ahead_arm`), in a GPU tier part prepared into its
    store and run from it (row 11859: compiled where its run held the GPU), elsewhere the program jitted."""
    from reference_error import ahead_arm

    fn, result = PROGRAMS[program]
    a = np.arange(SIZE, dtype=np.int32)
    run = ahead_arm(jax.jit(fn), f"a reversed view's sum: {program}")
    want = result(a)
    for _ in range(3):
        got = np.asarray(run(a))
        assert np.array_equal(got, want), f"{int(np.sum(got != want))} of {got.size} elements differ"


@pytest.mark.parametrize("program", list(PROGRAMS))
def test_a_value_read_through_a_reversed_view_is_summed_exactly(program):
    a_value_read_through_a_reversed_view_is_summed_exactly(program)
