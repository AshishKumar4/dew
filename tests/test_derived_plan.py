"""The fit search's first compile is at the rung the step's derived footprint says fits (row 2114): the
derivation walks the ladder from the base rung before anything compiles, and the compile confirms it or
climbs on as a correction. A run that starts from a plan, recorded or resumed, tries that plan first."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from objective_cases import jepa
from test_plan import LIMIT, free, needs, one_device

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.objectives.lm import LMObjective
from dew.telemetry.instrumentation import Compiles
from dew.training import MeshSpec, Trainer


def deep(remat=None):
    """A decoder whose residuals are most of what its step holds, on one device, reading its batch
    whole: eight layers over 256 tokens, the derivation's case."""
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=8, num_heads=2, mlp_features=64,
                              max_seq_len=256, remat=remat)
    return one_device(Trainer(LMObjective(model, seq_len=256), optax.sgd(1e-2), key=jax.random.key(0),
                              split=1))


DEEP = {"text": jnp.asarray((np.arange(8 * 257).reshape(8, 257) * 7) % 64, jnp.int32)}


def compiled(trainer, batch=DEEP):
    """`trainer`'s step compiled for `batch`, its plan made."""
    state, _, _ = trainer.place()
    return trainer.compile(state, batch)


def room_for_full_recompute_only(monkeypatch, records) -> int:
    """Bytes free with room for the deep step under full recomputation, as compiled, and not under
    none, as compiled or as derived, measured by runs that record their plans under `records`, apart
    from the run the test makes."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))
    recomputed, base = deep(REMAT_POLICIES["full"]), deep()
    full = needs(monkeypatch, recomputed, lambda: compiled(recomputed))
    whole = needs(monkeypatch, base, lambda: compiled(base))
    return (full + min(whole, LIMIT - base.plan["derived"][0]["headroom"])) // 2


def test_the_first_compile_is_at_the_least_recompute_rung_the_derivation_says_fits(monkeypatch, tmp_path):
    """With room for the step under full recomputation and not under none, the search's first compile
    is at the least-recompute rung whose derived footprint fits, which the compile confirms: the step
    compiles once, where the climb from the base rung compiled every rung below. The plan records each
    rung the derivation read, refused below and fitting at its rung."""
    room = room_for_full_recompute_only(monkeypatch, tmp_path / "sizing")
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "run"))
    free(monkeypatch, room)
    trainer = deep()
    with Compiles("step") as compiles:
        compiled(trainer)
    labels = [entry["label"] for entry in trainer.plan["derived"]]
    assert labels[0] == "refuses" and labels[-1] == "fits" and set(labels[1:-1]) <= {"refuses"}
    assert trainer.plan["remat"] is not None and trainer.plan["derived"][-1]["remat"] == trainer.plan["remat"]
    assert compiles.count == 1 and trainer.plan["probes"][0]["fits"]


def test_a_derivation_that_fits_the_base_rung_or_cannot_judge_starts_the_search_there(monkeypatch, tmp_path):
    """With room for every rung, the derivation fits the base rung, which the search compiles first,
    as it did before any derivation: no rung above it is read. A device that states no memory leaves
    the derivation unable to judge, and the search starts at the base rung too."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "fits"))
    free(monkeypatch, LIMIT)
    trainer = deep()
    compiled(trainer)
    assert [entry["label"] for entry in trainer.plan["derived"]] == ["fits"]
    assert trainer.plan["remat"] is None and trainer.plan["derived"][0]["remat"] is None
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "unknown"))
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: None)
    trainer = deep()
    compiled(trainer)
    assert [entry["label"] for entry in trainer.plan["derived"]] == ["unknown"]
    assert trainer.plan["probes"][0]["remat"] is None


def test_a_run_that_starts_from_a_recorded_plan_compiles_it_first(monkeypatch, tmp_path):
    """A restart whose plan an earlier run recorded (beside the compilation cache) compiles that plan
    first, with no derivation read, where its devices now hold room for full recomputation alone: the
    compile, not the derivation, decides whether it fits, and a miss climbs from there."""
    room = room_for_full_recompute_only(monkeypatch, tmp_path / "sizing")
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "run"))
    free(monkeypatch, LIMIT)
    compiled(deep())
    free(monkeypatch, room)
    again = deep()
    compiled(again)
    assert again.plan["derived"] == [] and again.plan["probes"][0]["remat"] is None


def test_an_objective_that_trains_two_modules_unset_reads_the_derivation(monkeypatch, tmp_path):
    """An objective whose two trained modules, an encoder and a predictor, set no remat reads the
    derivation before it compiles, as a one-module objective does."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    free(monkeypatch, LIMIT)
    objective, batch = jepa()
    trainer = one_device(Trainer(objective, optax.sgd(1e-2), key=jax.random.key(0), split=1))
    compiled(trainer, batch)
    assert trainer.plan["derived"]


def test_a_run_that_sets_its_remat_is_told_the_lighter_rung_its_derivation_fits(monkeypatch, tmp_path):
    """A run that sets its remat compiles there alone, once, and before it compiles reads the derivation
    from no remat up to its own (row 2536): with room for every rung, a run set to full recomputation
    reads the base rung fit, which its plan records beside the rung it compiled; a run set to the least
    rung that fits reads every rung below it refused, and the derivation stops there."""
    room = room_for_full_recompute_only(monkeypatch, tmp_path / "sizing")
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "unset"))
    free(monkeypatch, room)
    unset = deep()
    compiled(unset)
    for free_bytes, remat, lighter in ((LIMIT, "full", True), (room, unset.plan["remat"], False)):
        monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / f"run-{remat}"))
        free(monkeypatch, free_bytes)
        trainer = deep(REMAT_POLICIES[remat])
        with Compiles("step") as compiles:
            compiled(trainer)
        derived = trainer.plan["derived"]
        assert compiles.count == 1 and trainer.plan["probes"][0]["remat"] == trainer.plan["remat"] == remat
        assert derived and derived[0]["remat"] is None and all(entry["remat"] != remat for entry in derived)
        assert [entry["label"] for entry in derived][-1] == ("fits" if lighter else "refuses")
        assert {entry["label"] for entry in derived[:-1]} <= {"refuses"}
    # A policy off the ladder, which no climb from no remat reaches, has no rung known below it.
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "off"))
    free(monkeypatch, LIMIT)
    off = deep(REMAT_POLICIES["save_qkv_proj"])
    compiled(off)
    assert off.plan["derived"] == [] and off.plan["remat"] == "save_qkv_proj"


@pytest.mark.mesh(devices=2)
def test_a_run_over_two_devices_that_sets_its_remat_is_told_the_rung_its_derivation_fits(
        monkeypatch, tmp_path, caplog):
    """A run sharded over two devices (fsdp 2, as the two-GPU run that raised row 2536) that sets
    full recomputation reads its derivation per device, each its share of the batch and the state
    (`candidate_footprint`), and with room for every rung is told the base rung fits, where its search
    compiles at its own alone; the derivation's climb compiles nothing, so it says it compiles nothing
    (train 333's review: rows 2615, 2616)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "two"))
    free(monkeypatch, LIMIT)
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=8, num_heads=2, mlp_features=64,
                              max_seq_len=256, remat=REMAT_POLICIES["full"])
    trainer = Trainer(LMObjective(model, seq_len=256), optax.sgd(1e-2), key=jax.random.key(0), split=1,
                      mesh=MeshSpec(fsdp=2).build(jax.devices()[:2]))
    with caplog.at_level(logging.WARNING, logger="dew"), Compiles("step") as compiles:
        compiled(trainer)
    derived = trainer.plan["derived"]
    assert compiles.count == 1 and trainer.plan["remat"] == "full"
    assert derived and derived[0]["remat"] is None and derived[-1]["label"] == "fits"
    assert not [record for record in caplog.records if "compiling it again" in record.getMessage()]
    assert [record for record in caplog.records if "recomputes less" in record.getMessage()]
