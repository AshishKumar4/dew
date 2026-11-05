"""Scratch probe: where the derivative contracts' time goes."""
import cProfile
import collections
import pstats
import time

import contract_support
import pytest

import test_contracts_recurrence as rec
import test_contracts_objectives as obj
import test_contracts_attention as att

TIMES = collections.defaultdict(float)
_ran = contract_support.ran


def timed(order, label, name):
    start = time.perf_counter()
    try:
        return _ran(order, label, name)
    finally:
        TIMES[(label.split(" ")[1] if " " in label else label, name)] += time.perf_counter() - start


@pytest.mark.parametrize("case", ["chunked", "decode", "short", "derived", "att", "obj"])
def test_probe(case, monkeypatch):
    monkeypatch.setattr(contract_support, "ran", timed)
    calls = {
        "chunked": lambda: [rec.test_the_chunked_op_holds_every_order(edge, False) for edge in contract_support.EDGES],
        "decode": lambda: [rec.test_the_decode_step_holds_every_order(edge) for edge in contract_support.EDGES],
        "short": lambda: [rec.test_the_short_conv_holds_every_order(edge) for edge in contract_support.EDGES],
        "derived": lambda: [rec.test_the_derived_delta_rule_chunks_hold_their_orders(True, edge)
                            for edge in ("normal", "near the least normal")],
        "att": lambda: [f() for name, f in vars(att).items() if name.startswith("test_") and not getattr(f, "pytestmark", None)],
        "obj": lambda: None,
    }
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    calls[case]()
    profile.disable()
    print(f"\nPROBE {case}: {time.perf_counter() - start:.1f} s")
    for key, seconds in sorted(TIMES.items(), key=lambda kv: -kv[1])[:25]:
        print(f"PROBE {case} {key}: {seconds:.2f}")
    pstats.Stats(profile).sort_stats("cumulative").print_stats(30)
    TIMES.clear()
