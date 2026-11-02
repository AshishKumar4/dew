"""JAX's compile events as Dew hears them (`dew.telemetry.compiles`): a block counting the programs it
needs, a thread held to loaded programs alone."""

import threading

import jax
import jax.numpy as jnp
import pytest

from dew.telemetry.compiles import Compiles, ExecutableOnly, executable_only


def test_a_process_held_to_loaded_programs_refuses_each_compile_step_and_runs_what_was_compiled():
    """Within `executable_only`, a function jitted anew, a traced program lowered and a lowered program
    compiled each stop the run, and an executable compiled before runs; another thread is not held, and past
    the block jitting compiles again (ledger 2716: a node measuring programs compiled ahead stops where one is
    missing). One listener hears JAX's events for it and for an open `Compiles` alike, which counts every
    compile done, the one the block stopped as JAX recorded it done, another thread's and those after it."""
    x = jnp.arange(8.0)
    compiled = jax.jit(lambda value: value * 3).lower(x).compile()
    traced = jax.jit(lambda value: value * 5).trace(x)
    lowered = jax.jit(lambda value: value * 7).lower(x)
    elsewhere = []
    with Compiles(None) as compiles:
        with executable_only():
            assert np.asarray(compiled(x)).tolist() == (np.arange(8.0) * 3).tolist()
            for step in (lambda: jax.jit(lambda value: value + 1)(x), traced.lower, lowered.compile):
                with pytest.raises(ExecutableOnly):
                    step()
            thread = threading.Thread(
                target=lambda: elsewhere.append(np.asarray(jax.jit(lambda v: v + 2)(x))))
            thread.start()
            thread.join()
        assert elsewhere[0].tolist() == (np.arange(8.0) + 2).tolist()
        assert np.asarray(jax.jit(lambda value: value - 1)(x)).tolist() == (np.arange(8.0) - 1).tolist()
    assert compiles.count == 3


def test_a_compiles_block_that_closes_waits_for_the_events_already_told_to_it():
    """A `Compiles` block that closes while another thread's event is being told to it closes once that event
    is held: what it counts when it closes is what it counts from then on, never a program late."""
    told, release = threading.Event(), threading.Event()

    class Slow(Compiles):
        """An open block that takes its time over each event of its function it is told."""

        def heard(self, kind, duration_secs, metadata):
            if metadata.get("fun_name") == self.name:
                told.set()
                release.wait(60)
            super().heard(kind, duration_secs, metadata)

    def elevenfold(value):
        return value * 11

    x = jnp.arange(8.0)
    compiles = Slow("elevenfold").__enter__()
    compiler = threading.Thread(target=lambda: jax.jit(elevenfold).trace(x))
    compiler.start()
    assert told.wait(60)
    closed = []
    closer = threading.Thread(target=lambda: (compiles.__exit__(None, None, None),
                                              closed.append(len(compiles.programs))))
    closer.start()
    closer.join(0.5)
    release.set()
    closer.join()
    compiler.join()
    assert closed == [1] and [program.kind for program in compiles.programs] == ["traced"]
