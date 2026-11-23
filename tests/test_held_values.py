"""How a process holds what it derives once (`dew._cache.holding`): a held value read back, None among them,
and one made again where it no longer holds."""

import collections

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
    assert first and again == first and held == first
