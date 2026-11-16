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
