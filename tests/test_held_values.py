"""How a process holds what it derives once (`dew._cache.holding`): a held value read back, None among them,
and one made again where it no longer holds."""

import collections

import jax
import jax.numpy as jnp
import pytest

from dew import _cache, _facts


def test_a_held_none_is_read_back_and_a_value_whose_fact_answers_otherwise_is_made_again():
    """A value of None is held as any value is, made once; a held value whose making read a fact that
    answers otherwise now (`dew._facts.traced_fact`: a launch refused since, row 6708) is made again and held
    in its place, and one whose fact answers alike is read back, its fact read again around it (row 12296:
    every held value held with the facts its making read, `_cache.holding` their one owner)."""
    cache, made, facts = collections.OrderedDict(), [], {"refused": 0}

    def nothing():
        made.append("nothing")

    assert _cache.holding(cache, "none", nothing, 4) is None
    assert _cache.holding(cache, "none", nothing, 4) is None
    assert made == ["nothing"]

    @_facts.traced_fact
    def refused() -> int:
        return facts["refused"]

    def traced():
        made.append("traced")
        return refused()

    assert _cache.holding(cache, "trace", traced, 4) == 0
    with _facts.recording() as again:
        assert _cache.holding(cache, "trace", traced, 4) == 0
    facts["refused"] = 1
    assert _cache.holding(cache, "trace", traced, 4) == 1
    assert _cache.holding(cache, "trace", traced, 4) == 1
    assert made == ["nothing", "traced", "traced"]
    assert [answer for _, answer in again.values()] == [0]


def test_a_held_trace_made_again_past_its_bound_records_its_facts_afresh():
    """A held trace (`_cache.holding`) dropped past its cache's bound by others is made again where it
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

    with _facts.recording() as first:
        assert _cache.holding(cache, "first", making("first"), 1) == "first"
    _cache.holding(cache, "second", making("second"), 1)
    with _facts.recording() as again:
        assert _cache.holding(cache, "first", making("first"), 1) == "first"
    with _facts.recording() as held:
        _cache.holding(cache, "first", making("first"), 1)
    assert made == ["first", "second", "first"]
    # Each recording holds the one fact the making read, and its answer.
    assert first and [(fact, answer) for fact, (_, answer) in first.items()] == [
        (fact, answer) for fact, (_, answer) in again.items()] == [
        (fact, answer) for fact, (_, answer) in held.items()]


def test_a_refusal_scopes_lookup_leaves_a_held_traces_key_as_it_was():
    """Within a refusal scope (`report.refusals`), a lookup of a launch's refusals (`report.refusals_of`,
    which fills the scope's index of them) leaves the key a held trace is kept by (`options.held_context`)
    as it was: the scope's registry enters it by its refusals alone (`report.refusal_snapshot`, read under
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


def test_a_held_jits_trace_refusing_again_what_its_scope_refused_runs_past_it():
    """A program whose every trace refuses one launch (a launch certified inline, refused each time), run
    twice in one scope, JAX's caches cleared and its executables released between: the second's trace, by
    the jit the first left, refuses again what its scope holds already, which ends nothing, and it runs the
    program past the refusal (theme K01-e's review: a held jit's trace ended on every refusal it recorded,
    one it held already among them, retried by that same jit without end)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    def chosen(x):
        report.refuse([("a launch", 1, ("k",))])
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    run, x = owned_program(chosen), jnp.zeros((), jnp.float32)
    with report.refusals():
        assert float(run(x)) == 1.0
        jax.clear_caches()
        run.released()
        assert float(run(x)) == 1.0


def test_a_program_within_a_held_jits_trace_that_refuses_a_launch_leaves_it_to_its_holder():
    """A program run in a scope of no refusal whose trace traces another (`trace`), JAX's caches cleared and
    both released, run in another such scope where the inner one's trace records an inline refusal: the
    outer's held jit traces it, the refusal ends that trace, and the outer traces again past it (1.0), where
    the inner, its reading frozen as the outer's, retried past the outer's snapshot without end; a third
    scope of none then runs its own (theme K01-e's review P0). The refused scope runs in a thread, so a call
    that never ends fails, under its caller's trace-affecting states (`_cache.carried`), so its call keys as
    the caller's (`ALGEBRA`, which the suite sets per test, holds per thread)."""
    import threading

    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    armed: list[bool] = []

    def refusing(x):
        # As a derivation certified inline does: its launch refused wherever its reading holds no refusal.
        if armed and not report.refusals_of("a launch"):
            report.refuse([("a launch", 1, ("k",))])
        return x

    inner = owned_program(refusing)

    def chosen(x):
        # The inner program's own trace (`trace`, through its `_stable_trace`), not a nested call's.
        inner.trace(jax.ShapeDtypeStruct(x.shape, x.dtype))
        return x + (1.0 if report.refusals_of("a launch") else 0.0)

    outer, x = owned_program(chosen), jnp.zeros((), jnp.float32)
    with report.refusals():
        assert float(outer(x)) == 0.0
    jax.clear_caches()
    outer.released()
    inner.released()
    armed.append(True)
    ran: list[float] = []
    entered = _cache.carried()

    def refused_scope() -> None:
        with entered(), report.refusals():
            ran.append(float(outer(x)))

    running = threading.Thread(target=refused_scope, daemon=True)
    running.start()
    running.join(timeout=120)
    assert ran == [1.0], "the refused scope's call did not end"
    armed.clear()
    outer.released()
    inner.released()
    with report.refusals():
        assert float(outer(x)) == 0.0


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("taken", ["called", "traced", "compiled", "nested"])
def test_a_held_value_stands_past_a_refusal_of_a_launch_it_read_none_of(taken, warm):
    """A value held whose making ran a program reading one launch's refusals (`report.refusals_of`), called,
    traced, compiled or called within a plain jit, stands after another launch is refused, and is made again
    once its own is: a trace reads the refusals of the launches it reads alone, the whole registry no fact of
    it (row 12296: any refusal made every held value again, the registry read whole as a program keyed its
    executables). Run, traced, begun compiling (`options.compiling`) or called within another plain jit once
    before, its executable, its held jit, its begun compile or its nested jit answering the making with no
    trace, it holds the facts its trace read as a trace does (row 12296's reviews' P0s: each recorded none,
    and the value stood past its own launch's refusal)."""
    from dew.nn.kernels import options
    from dew.nn.kernels.templates import report

    run = options.owned_program(lambda x: x + (1.0 if report.refusals_of("a launch") else 0.0))
    x = jnp.zeros(())
    cache: collections.OrderedDict = collections.OrderedDict()
    made: list[float] = []
    begun: list[options.Compiling] = []

    def taking() -> float:
        if taken == "called":
            return float(run(x))
        if taken == "traced":
            return float(run.trace(x).lower().compile()(x))
        if taken == "compiled":
            with options.attempting(begun[-1] if begun else None):
                return float(run.compiled(x)(x))
        # Each a plain jit of its own, its trace running the program's nested jit, held across them.
        return float(jax.jit(lambda y: run(y))(x))

    def making() -> float:
        made.append(taking())
        return made[-1]

    with report.refusals():
        if warm and taken == "compiled":
            begun.append(options.compiling(run, x))
        elif warm:
            assert taking() == 0.0
        assert _cache.holding(cache, "held", making, 4) == 0.0
        report.refuse([("another launch", 1, ("k",))])
        assert _cache.holding(cache, "held", making, 4) == 0.0
        assert made == [0.0]
        report.refuse([("a launch", 1, ("k",))])
        assert _cache.holding(cache, "held", making, 4) == 1.0
        assert made == [0.0, 1.0]


def test_a_held_value_made_over_a_program_traced_again_holds_the_facts_that_trace_read():
    """A program traced once, a fact its trace reads (`_facts.traced_fact`) then answering otherwise and JAX's
    caches cleared, traced again by the jit it holds: a value made from that trace holds the fact as it read
    it now, and once the fact answers as before it is made again, from a trace of that answer (row 12296's
    review P0: the facts the held jit's first trace read were replayed over the fresh trace's, and the value
    held on an answer its program never read)."""
    from dew.nn.kernels.options import owned_program

    answer = [0]

    @_facts.traced_fact
    def flag() -> int:
        return answer[0]

    run, x = owned_program(lambda x: x + flag()), jnp.zeros(())
    cache: collections.OrderedDict = collections.OrderedDict()

    def making() -> float:
        return float(run.trace(x).lower().compile()(x))

    assert making() == 0.0
    answer[0] = 1
    jax.clear_caches()
    assert _cache.holding(cache, "held", making, 4) == 1.0
    answer[0] = 0
    assert _cache.holding(cache, "held", making, 4) == 0.0


@pytest.mark.parametrize("enclosed", ["jitted", "mapped"])
def test_a_program_called_within_another_trace_reads_its_facts_at_each_call(enclosed):
    """A program called within a plain jit of its own each time (or an eager `jax.vmap`), at one shape under a
    fact's answer, at another under a second answer, then held made at the first shape: the value is the
    program's under the answer as it stands, held on it (row 12296's review P0s: the program's jit held across
    enclosing traces answered the first shape from its cached trace, made under the first answer, with the
    facts its trace at the other shape read)."""
    from dew.nn.kernels.options import owned_program

    answer = [0]

    @_facts.traced_fact
    def flag() -> int:
        return answer[0]

    run = owned_program(lambda x: x + flag())

    def within(x) -> float:
        if enclosed == "jitted":
            return float(jax.jit(lambda y: run(y))(x)[0])
        return float(jax.vmap(run)(x[None])[0, 0])

    one, two = jnp.zeros((1,)), jnp.zeros((2,))
    assert within(one) == 0.0
    answer[0] = 1
    assert within(two) == 1.0
    cache: collections.OrderedDict = collections.OrderedDict()
    assert _cache.holding(cache, "held", lambda: within(one), 4) == 1.0
    answer[0] = 0
    assert _cache.holding(cache, "held", lambda: within(one), 4) == 0.0


def test_a_trace_reading_a_fact_under_a_gpu_it_assumes_is_taken_as_it_ran():
    """A program whose trace assumes a GPU of its own and reads its memory (`generation.device_memory`, a
    traced fact), traced under another assumed GPU of other memory: its trace is taken as it ran, the fact as
    it read it, where each trace of its own was let go because the fact, asked again outside the GPU it
    assumed, answered otherwise, and traced again without end (row 12296's review P0)."""
    import dataclasses
    import threading

    from jax.extend import core

    from dew.nn.kernels import generation
    from dew.nn.kernels.options import owned_program

    outer = dataclasses.replace(generation.gpu_target("sm90"), memory=80 * 2 ** 30)
    inner = dataclasses.replace(outer, memory=40 * 2 ** 30)

    def read(x):
        with generation.assumed(inner):
            return x + generation.device_memory() / 2 ** 30

    run, x = owned_program(read), jnp.zeros(())
    traced: list = []
    entered = _cache.carried()

    def tracing() -> None:
        with entered(), generation.assumed(outer):
            traced.append(run.trace(x))

    running = threading.Thread(target=tracing, daemon=True)
    running.start()
    running.join(timeout=120)
    assert len(traced) == 1, "the trace was let go and traced again without end"
    assert float(core.jaxpr_as_fun(traced[0].jaxpr)(x)[0]) == 40.0


def test_a_program_called_under_disable_jit_within_a_trace_stages_its_operations_inline():
    """A program called within `jax.make_jaxpr` under `jax.disable_jit`: its function runs as Python in the
    trace around it, which stages its operations as a disabled jit does, binding no program of its own
    (row 12296's review P0: a call within a trace took its held jit's trace and bound that program past the
    disabled jit)."""
    from dew.nn.kernels.options import owned_program

    run = owned_program(lambda x: jnp.sin(x) * 2.0)
    with jax.disable_jit():
        staged = jax.make_jaxpr(run)(jax.ShapeDtypeStruct((), jnp.float32))
    names = [equation.primitive.name for equation in staged.jaxpr.eqns]
    assert "sin" in names and "jit" not in names, names


def test_a_program_of_a_static_argument_counted_from_the_end_binds_its_dynamic_operands_alone():
    """A program whose static argument is named by a negative position (`static_argnums=-1`), called within
    another jit: the program its trace made is bound to its one array, the static argument read as written,
    no operand of it (row 12296's review P0: the negative position kept the static string among the operands
    bound)."""
    from dew.nn.kernels.options import owned_program

    run = owned_program(lambda x, mode: x * 2.0 if mode == "double" else x, static_argnums=-1)
    assert float(jax.jit(lambda y: run(y, "double"))(jnp.ones(()))) == 2.0


def test_a_programs_executable_stands_past_a_refusal_of_a_launch_its_trace_read_none_of():
    """A program run, then a launch its trace reads none of refused: its next call runs the executable it
    holds, compiling none (rows 12311, 12328: every program's key held the whole registry, so any refusal,
    an evaluation's spill among them, compiled the training step again); once a launch it reads is refused,
    it runs one of its own past the refusal."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    run, x = owned_program(lambda x: x + (1.0 if report.refusals_of("a launch") else 0.0)), jnp.zeros(())
    with report.refusals():
        first = run.executable(x)
        assert float(run(x)) == 0.0
        report.refuse([("another launch", 1, ("k",))])
        assert run.executable(x) is first
        assert float(run(x)) == 0.0
        report.refuse([("a launch", 1, ("k",))])
        assert run.executable(x) is not first
        assert float(run(x)) == 1.0


def test_a_program_called_in_two_refusal_scopes_keeps_an_executable_for_each():
    """A program called in turn in a scope that refused a launch its trace reads and in one that refused
    none: each scope's call runs the executable its first call compiled, neither compiled again by the
    other's (rows 12311, 12328: an executable held by the refusals its trace read, one a key)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    run, x = owned_program(lambda x: x + (1.0 if report.refusals_of("a launch") else 0.0)), jnp.zeros(())
    with report.refusals():
        report.refuse([("a launch", 1, ("k",))])
        refused = run.executable(x)
    with report.refusals():
        clear = run.executable(x)
    for _ in range(2):
        with report.refusals():
            report.refuse([("a launch", 1, ("k",))])
            assert run.executable(x) is refused
            assert float(run(x)) == 1.0
        with report.refusals():
            assert run.executable(x) is clear
            assert float(run(x)) == 0.0


def test_a_compile_begun_aside_is_taken_past_a_refusal_of_a_launch_it_read_none_of():
    """A program's compile begun aside (`options.compiling`), then a launch its trace reads none of refused:
    the program compiled after takes the begun compile (`attempting`), lowering nothing again (row 12311:
    the run start's begun compile no longer matched once the step's planning refused a launch, and the
    start compiled twice)."""
    from dew.nn.kernels import options
    from dew.nn.kernels.templates import report

    run = options.owned_program(lambda x: x + (1.0 if report.refusals_of("a launch") else 0.0))
    x = jnp.zeros(())
    with report.refusals():
        begun = options.compiling(run, x)
        report.refuse([("another launch", 1, ("k",))])
        with options.attempting(begun):
            assert float(run.compiled(x)(x)) == 0.0
        assert begun.taken.is_set()


def test_a_programs_executable_past_a_refusal_of_another_launch_holds_its_other_facts_still():
    """A program run under one answer of a fact its trace reads (`_facts.traced_fact`), the fact then
    answering otherwise and a launch its trace reads none of refused: its next call runs a program of the
    fact's answer now, its executable judged by every fact its trace read once the refusals moved (rows
    12311, 12328's review P0: judged by the refusals alone, the program of the first answer ran on)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.kernels.templates import report

    answer = [0]

    @_facts.traced_fact
    def flag() -> int:
        return answer[0]

    run, x = owned_program(lambda x: x + flag()), jnp.zeros(())
    with report.refusals():
        assert float(run(x)) == 0.0
        answer[0] = 1
        report.refuse([("another launch", 1, ("k",))])
        assert float(run(x)) == 1.0
