"""Probe (not for landing): size the offloaded step's check-1 seeds, paired, by `seeds_for`."""

import jax
import numpy as np
from reference_error import seeds_for
from test_host_offload import offloading, stepping, tight
from test_plan import LIMIT, ROWS, SEQ, free


def test_probe(monkeypatch):
    room = tight(monkeypatch)
    initial = offloading().initial_state()

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(0, 64, (ROWS, SEQ + 1)).astype(np.int32),)

    host, device, truth = (stepping(offloading(), initial), stepping(offloading(), initial),
                           stepping(offloading(), initial, exact=True))
    free(monkeypatch, room)
    host(*draw(0))
    free(monkeypatch, LIMIT)
    device(*draw(0))
    with jax.enable_x64(new_val=True):
        truth(*draw(0))
    print("PROBE step seeds", seeds_for(device, draw, "offload", truth_fn=truth, dew_fn=host))
