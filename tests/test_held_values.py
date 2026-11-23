"""How a process holds what it derives once (`dew._cache.holding`): a held value read back, None among them,
and one made again where it no longer holds."""

import collections

import jax
import jax.numpy as jnp

from dew import _cache


def test_a_held_none_is_read_back_and_a_value_no_longer_holding_is_made_again():
    """A value of None is held as any value is, made once; a held value `still` refuses (a held trace whose
    facts answer otherwise now, row 6708) is made again and held in its place."""
    cache, made = collections.OrderedDict(), []

    def nothing():
        made.append("nothing")

    assert _cache.holding(cache, "none", nothing, 4) is None
    assert _cache.holding(cache, "none", nothing, 4) is None
    assert made == ["nothing"]

    facts = {"refused": 0}

    def traced():
        made.append("traced")
        return dict(facts)

    def still(held):
        return held == facts

    assert _cache.holding(cache, "trace", traced, 4, still=still) == {"refused": 0}
    assert _cache.holding(cache, "trace", traced, 4, still=still) == {"refused": 0}
    facts["refused"] = 1
    assert _cache.holding(cache, "trace", traced, 4, still=still) == {"refused": 1}
    assert _cache.holding(cache, "trace", traced, 4, still=still) == {"refused": 1}
    assert made == ["nothing", "traced", "traced"]


def test_a_held_trace_made_again_past_its_bound_records_its_facts_afresh():
    """A held trace (`generation.held_trace`) dropped past its cache's bound by others is made again where it
    is read next, and the facts its making reads (`gpu_placed`) are recorded around that read as around its
    first: its holder's report of a reuse is exact across an eviction, never a stale replay (Train 1205's
    review F1, the one owner of a held trace's facts)."""
    from dew.nn.kernels import generation

    cache: collections.OrderedDict = collections.OrderedDict()
    made: list[str] = []

    def making(key: str):
        def make() -> str:
            made.append(key)
            generation.gpu_placed()
            return key
        return make

    with generation.recording() as first:
        assert generation.held_trace(cache, "first", making("first"), 1) == "first"
    generation.held_trace(cache, "second", making("second"), 1)
    with generation.recording() as again:
        assert generation.held_trace(cache, "first", making("first"), 1) == "first"
    with generation.recording() as held:
        generation.held_trace(cache, "first", making("first"), 1)
    assert made == ["first", "second", "first"]
    # Each recording holds the one fact the making read, and its answer.
    assert first and [(fact, answer) for fact, (_, answer) in first.items()] == [
        (fact, answer) for fact, (_, answer) in again.items()] == [
        (fact, answer) for fact, (_, answer) in held.items()]


def test_a_refusal_scopes_lookup_leaves_a_held_traces_key_as_it_was():
    """Within a refusal scope (`report.refusals`), a lookup of a launch's refusals (`report.refusals_of`,
    which fills the scope's index of them) leaves the key a held trace is kept by (`options.held_context`)
    as it was: the scope's registry enters it by its refusals alone (`report.refusal_entries`, read under
    its lock), not by its mutable records, where a first probe made its own key stale and a compile worker
    filling it raced the key's walk of its dicts (row 12134's review)."""
    from dew.nn.kernels import options
    from dew.nn.kernels.templates import report

    with report.refusals():
        before = options.held_context()
        report.refusals_of("a launch no compile refused")
        assert options.held_context() == before


def test_a_program_called_after_a_seeded_refusal_traces_past_it():
    """A program (`options.owned_program`) whose trace reads a launch's refusals (`report.refusals_of`, as a
    launch's derivation reads them), called, then called again over arrays alike after a refusal of that
    launch is seeded (`report.refuse`, a handed plan's), runs a program traced past it: what its executables
    are kept by and its jit holds key on the refusals as its held traces do (row 12165: the executable
    traced before the refusal ran again)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    def chosen(x):
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    run, x = owned_program(chosen), jnp.zeros((), jnp.float32)
    with report.refusals():
        before = float(run(x))
        report.refuse([("a launch", 1, ("its kernel",))])
        after = float(run(x))
    assert (before, after) == (0.0, 1.0)


def test_a_program_whose_trace_a_refusal_follows_runs_a_trace_past_it_after():
    """A program whose trace reads a launch's refusals, a refusal of that launch recorded as it traces
    (before its executable is kept, as another thread's compile records one): a call after it runs a
    program traced past the refusal, never the executable traced before it kept under the refusal's key
    (theme K01-e's review P0: the key read after the compile)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    seeded = []

    def chosen(x):
        seen = report.refusals_of("a launch")
        if not seeded:
            seeded.append(True)
            report.refuse([("a launch", 1, ("its kernel",))])
        return x + (1.0 if seen else 0.0)

    run, x = owned_program(chosen), jnp.zeros((), jnp.float32)
    with report.refusals():
        assert float(run(x)) == float(run(x)) == 1.0
    # Kept by the refusals it traced past alone: a scope of none traces its own (theme K01-e's review P0:
    # the executable kept under the reservation's refusals too).
    with report.refusals():
        assert float(run(x)) == 0.0


def test_a_trace_a_refusal_races_leaves_no_program_for_the_refusals_it_began_under():
    """A program whose first trace a refusal races, recorded before its launch reads the refusals (as
    another thread's would be), runs the program past it, and a scope of no refusal after it runs a program
    of its own: no jit its raced trace cached is held for the refusals it began under (theme K01-e's review
    P0: a jit claimed before its trace held the refusal's program for every later call of none)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    seeded = []

    def chosen(x):
        if not seeded:
            seeded.append(True)
            report.refuse([("a launch", 1, ("its kernel",))])
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    run, x = owned_program(chosen), jnp.zeros((), jnp.float32)
    with report.refusals():
        assert float(run(x)) == 1.0
    with report.refusals():
        assert float(run(x)) == 0.0


def test_a_call_waiting_on_a_compile_kept_past_its_refusals_compiles_its_own():
    """A call joined on another thread's compile of the same key, whose trace that thread's scope refuses a
    launch as it traces, is not given the program its compile kept past the refusal: in a scope of none,
    it runs a program of its own (theme K01-e's review P0: the reservation's waiters given a program of
    refusals their scope holds none of)."""
    import threading
    import time

    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    tracing, released, seeded, found = threading.Event(), threading.Event(), [], {}

    def chosen(x):
        if not seeded:
            seeded.append(True)
            report.refuse([("a launch", 1, ("its kernel",))])
            tracing.set()
            released.wait(timeout=60)
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    run, x = owned_program(chosen), jnp.zeros((), jnp.float32)

    def called(name: str) -> None:
        with report.refusals():
            found[name] = float(run(x))

    first = threading.Thread(target=called, args=("refusing",))
    first.start()
    tracing.wait(timeout=60)
    second = threading.Thread(target=called, args=("of none",))
    second.start()
    # The second call joined on the first's reservation before its compile settles.
    time.sleep(1)
    released.set()
    first.join(timeout=120)
    second.join(timeout=120)
    assert found == {"refusing": 1.0, "of none": 0.0}


def test_a_held_jit_raced_at_a_new_shape_answers_a_scope_of_none_with_its_own_program():
    """A program traced in a scope of no refusal at one shape, then at another while another thread records
    a refusal into that scope, runs the program past the refusal there, and a scope of none after it runs
    its own program at both shapes: a trace reads the refusals it began under and its own alone, and a held
    jit answers its one call (theme K01-e's review P0: a held jit traced at a new shape read the other
    thread's refusal, which its trace cache then answered a scope of none with)."""
    import contextvars
    import threading

    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    scope: list[contextvars.Context] = []

    def chosen(x):
        if x.shape == (2,) and len(scope) == 1:
            # Another thread's refusal into this scope as it traces, recorded before the trace reads it.
            other = threading.Thread(target=scope.pop().run, args=(report.refuse, [("a launch", 1, ("k",))]))
            other.start()
            other.join()
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    run = owned_program(chosen)
    one, two = jnp.zeros((1,), jnp.float32), jnp.zeros((2,), jnp.float32)
    with report.refusals():
        scope.append(contextvars.copy_context())
        assert float(run(one)[0]) == 0.0
        assert float(run(two)[0]) == 1.0
    with report.refusals():
        assert float(run(one)[0]) == 0.0 and float(run(two)[0]) == 0.0


def test_a_held_jits_trace_that_refuses_a_launch_leaves_its_cache_its_own_refusals():
    """A program run in a scope of no refusal, JAX's trace caches then cleared and its executables released,
    run in another such scope whose trace records an inline refusal (a launch certified as it traces) by the
    jit the first left, runs the program past the refusal there, and a third scope of none runs its own:
    a held jit's trace takes no refusal of its own, ended before its cache keeps the refusal's program
    (theme K01-e's review P0: its cache dropped, a held jit's trace read its own refusal and kept that
    program for every call it answered)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    armed: list[bool] = []

    def chosen(x):
        if armed == [True]:
            armed.append(True)
            report.refuse([("a launch", 1, ("k",))])
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    run, x = owned_program(chosen), jnp.zeros((), jnp.float32)
    with report.refusals():
        assert float(run(x)) == 0.0
    jax.clear_caches()
    run.released()
    armed.append(True)
    with report.refusals():
        assert float(run(x)) == 1.0
    run.released()
    with report.refusals():
        assert float(run(x)) == 0.0
