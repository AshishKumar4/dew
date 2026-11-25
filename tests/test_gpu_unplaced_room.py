"""A fresh start's step compiled, before its state is placed, in the room its admission reads (row 12882):
the arguments in use already, which its admission counts once, are given back to the compile's limit."""

import jax.numpy as jnp
import numpy as np
import pytest
from test_plan import free_until_placed, lm

ROWS, SEQ = 512, 512
TEXT = (np.arange(ROWS * (SEQ + 1)).reshape(ROWS, SEQ + 1) * 7) % 64
"""Rows enough that the batch, an argument of the step, outweighs the bytes XLA rounds a limit by."""


@pytest.mark.gpu(reason="a compile's memory limit applies on a GPU alone (`memory.compile_limit`)")
def test_a_fresh_step_admitted_before_its_state_is_placed_compiles_in_that_room(monkeypatch, tmp_path):
    """A fresh start whose devices read room for its step by a few bytes, as its admission before the state
    is placed counts it (each argument once, its batch among them): its first compile fits that room, the
    limit XLA is given its admission's, where it held back the arguments already in use a second time and
    stopped the compile its admission admitted (row 12882: 25 MiB of a 2xH100 decision batch, 4 GiB a
    device of a LoRA base kept as it is)."""
    from affine_run import Replaying

    from dew.data.dataset import Loader

    rows = Replaying(lambda position: {"text": jnp.asarray(TEXT, jnp.int32)})

    def first_probe(room: int):
        # Each fit's rung its own (`DEW_RUNG_RECORDS`): the second starts where the first did.
        monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / str(room)))
        free_until_placed(monkeypatch, room)
        trainer = lm(split=1, seq=SEQ)
        trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
        return trainer.plan["probes"][0]

    roomy = 1 << 36
    measured = first_probe(roomy)
    assert measured["fits"] is True and isinstance(measured["headroom"], int), measured
    # The room its admission reads as the step's exactly, and 64 KiB beside it.
    probe = first_probe(roomy - measured["headroom"] + (1 << 16))
    assert probe["fits"] is True and 0 <= probe["headroom"] <= 1 << 16, probe
