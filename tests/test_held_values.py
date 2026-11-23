"""How a process holds what it derives once (`dew._cache.holding`): a held value read back, None among them,
and one made again where it no longer holds."""

import collections

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
        run(x)
        assert float(run(x)) == 1.0
