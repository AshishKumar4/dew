"""A fresh start's step compiled, before its state is placed, in the room its admission reads (row 12882):
the arguments in use already, which its admission counts once, are given back to the compile's limit."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import allocating
from test_frozen_tree import adapted

from dew.training import MeshSpec, Trainer

ROWS = 8
TEXT = np.random.default_rng(0).integers(0, 256, (ROWS, 9))
LIMIT = 1 << 40


@pytest.mark.gpu(reason="a compile's memory limit applies on a GPU alone (`memory.compile_limit`)")
def test_a_lora_start_keeping_its_base_compiles_in_the_room_its_admission_reads(monkeypatch, tmp_path):
    """A LoRA run's fresh start whose base its devices hold already, kept as it is through the start
    (`Start._kept_through`), on devices reading room for its step by a few bytes as its admission before
    the state is placed counts it (each argument once, the base among them): its first compile fits that
    room, the limit XLA is given its admission's, where it held back the base in use a second time and
    stopped the compile its admission admitted (row 12882: 4.1 GiB short on 2xH100 with 4 GiB a device of
    a LoRA base kept)."""
    from affine_run import Replaying

    from dew._cache import uncached
    from dew.data.dataset import Loader

    rows = Replaying(lambda position: {"text": jnp.asarray(TEXT, jnp.int32)})

    def first_probe(room: int):
        # Each fit's rung its own (`DEW_RUNG_RECORDS`): the second starts where the first did.
        monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / str(room)))
        # A BFC pool's reading, the whole limit its pool, `room` of it free before the run places anything.
        allocating(monkeypatch, LIMIT, LIMIT, LIMIT - room)
        # Loaded once the devices read room: the base is in use beside the start, as a loaded run's is.
        trainer = Trainer(adapted(placed=True), optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec(),
                          split=1)
        trainer.fit(Loader(lambda partition: rows, None, None, ROWS), steps=1)
        return trainer.plan["probes"][0]

    roomy = 1 << 36
    measured = first_probe(roomy)
    assert measured["fits"] is True and isinstance(measured["headroom"], int), measured
    # The room its admission reads as the step's exactly, and 64 KiB beside it: less than the base's bytes.
    # Compiled anew, not read from the persistent cache, whose binary no limit stops (`limit_exited`).
    with uncached():
        probe = first_probe(roomy - measured["headroom"] + (1 << 16))
    assert probe["fits"] is True and 0 <= probe["headroom"] <= 1 << 16, probe
