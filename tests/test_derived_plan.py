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
from dew.training.planning import PlanRefused


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


@pytest.mark.mesh(devices=2)
def test_a_run_over_a_tensor_axis_that_sets_its_remat_hears_its_derivation_cannot_judge_it(
        monkeypatch, tmp_path, caplog):
    """A run over a tensor axis of two that sets full recomputation, whose footprint the derivation reads
    per device only over data and fsdp axes, hears once that its derivation cannot judge its layout,
    rather than nothing, and compiles at its own rung alone (row 2721; 2615 stays open for such layouts)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "tensor"))
    free(monkeypatch, LIMIT)
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=8, num_heads=2, mlp_features=64,
                              max_seq_len=256, remat=REMAT_POLICIES["full"])
    trainer = Trainer(LMObjective(model, seq_len=256), optax.sgd(1e-2), key=jax.random.key(0), split=1,
                      mesh=MeshSpec(tensor=2).build(jax.devices()[:2]))
    with caplog.at_level(logging.WARNING, logger="dew"), Compiles("step") as compiles:
        compiled(trainer)
    assert compiles.count == 1 and trainer.plan["remat"] == "full"
    assert trainer.plan["derived"][-1]["label"] == "unknown"
    assert len([record for record in caplog.records if record.name == "dew.training.planning"]) == 1


def test_a_compile_the_derivation_judged_otherwise_is_told_as_a_model_miss(monkeypatch, tmp_path, caplog):
    """Every compile's plan record holds the headroom the derivation judged beside the one the compile
    measured, and each compile whose fit the derivation judged otherwise is told as a model miss, the
    compile deciding: with room between what the derivation and the compile say a step at full
    recomputation needs, the run hears a miss for its first compile and one for each it judged so."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "sizing"))
    free(monkeypatch, LIMIT)
    sizing = deep(REMAT_POLICIES["full"])
    compiled(sizing)
    (measured,) = sizing.plan["probes"]
    derived, compiled_need = LIMIT - measured["derived_headroom"], LIMIT - measured["headroom"]
    assert derived != compiled_need
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "between"))
    free(monkeypatch, (derived + compiled_need) // 2)
    trainer = deep(REMAT_POLICIES["full"])
    with caplog.at_level(logging.WARNING, logger="dew"), Compiles("step") as compiles:
        if compiled_need > derived:
            with pytest.raises(PlanRefused):
                compiled(trainer)
        else:
            compiled(trainer)
    missed = [record for record in caplog.records if "model miss" in record.getMessage()]
    probes = (trainer.plan or {}).get("probes") or []
    judged = [probe for probe in probes if (probe["derived_headroom"] >= 0) != probe["fits"]]
    assert compiles.count >= 1 and missed and (not probes or len(missed) == len(judged))


GATE_LAYERS = {
    "decoder": {},
    "hybrid": {"layer_types": ("mamba", "attention") * 2, "kinds": {"mamba": {"mixer": {
        "class": "mamba2", "fields": {"num_heads": 4, "head_dim": 16, "state_size": 8, "n_groups": 1,
                                      "chunk_size": 32}}}, "attention": {}}},
    "moe": {"mixture": {"experts": 4, "top_k": 2, "layers": (1, 3)}},
}
"""The gate's decoders (row 2113): dense, hybrid of Mamba-2 and attention layers, and mixture of experts."""


def gated(layers: str, generation: str, remat: str, capacity: int, records) -> list[dict]:
    """A gate case's step compiled ahead for one compile-only GPU of `generation` stating `capacity`
    bytes, its rungs recorded under `records`: every compile its plan made, refused or not."""
    from dew.registry import models
    from dew.training.ahead import compile_ahead, compile_only_gpus

    model = models.build("causal_transformer", vocab_size=128, emb_features=64, num_layers=4, num_heads=4,
                         num_kv_heads=2, mlp_features=128, max_seq_len=128,
                         remat=None if remat == "none" else REMAT_POLICIES[remat], **GATE_LAYERS[layers])
    trainer = Trainer(LMObjective(model, seq_len=128), optax.adamw(1e-3), key=jax.random.key(0), split=1,
                      mesh=MeshSpec().build(compile_only_gpus(generation, 1)))
    tokens = ((np.arange(4 * 129).reshape(4, 129) * 7) % 128).astype(np.int32)
    # At JAX's default matmul precision, a run's fast forms', not the suite's highest: an expert
    # decoder's step at highest on an Ampere holds twice what its derivation finds (row 2750).
    with pytest.MonkeyPatch.context() as patch, jax.default_matmul_precision("default"):
        patch.setenv("DEW_RUNG_RECORDS", str(records))
        try:
            compile_ahead(trainer, {"text": tokens}, capacity=capacity)
        except PlanRefused as refused:
            return [probe.record() for probe in refused.probes]
    return list(trainer.plan["probes"])


@pytest.mark.parametrize("remat", ["none", "full"])
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
@pytest.mark.parametrize("layers", list(GATE_LAYERS))
def test_the_derivation_decides_a_steps_fit_as_its_compile_does(layers, generation, remat, tmp_path):
    """The derivation judges a step's fit as its compile does at 0.6 and 1.6 times what the compile needs,
    over decoders dense, hybrid and of experts, Ampere and Hopper, no recomputation and full (row 2113's
    gate, in the kernel gate's evidence shape, `compile_evidence`). The need is the program's as XLA lays
    it out on a roomy GPU, and the derivation's, read off that compile, admits it at 1.6 times and refuses
    it at 0.6. On a GPU stating 1.6 times, the compile fits and the derivation judged so. On one stating
    0.6, a step set to full recomputation compiles there alone: where XLA refuses it the derivation did,
    and where XLA fits it, it fits by XLA's own rematerialization against the stated memory, which the
    derivation does not model and the compile decides (its temporaries below the roomy compile's)."""
    from compile_evidence import Evidence, Facts, Measurement, Prediction

    (roomy,) = gated(layers, generation, remat, 1 << 34, tmp_path / "roomy")
    need, derived = (1 << 34) - roomy["headroom"], (1 << 34) - roomy["derived_headroom"]
    for margin, admits in ((1.6, True), (0.6, False)):
        budget = int(need * margin)
        evidence = Evidence(
            f"{layers} {generation} remat {remat} at {margin} of its need", "",
            Facts((layers, remat), generation),
            Prediction({"peak": roomy["derived"]["peak"]}, derived, budget, derived <= budget, "bytes"),
            Measurement({"temp": roomy["temp"]}, need <= budget, "xla", jax.__version__))
        assert evidence.compiled.admits == admits and evidence.agrees, evidence.describe()
    (fitting, *_) = gated(layers, generation, remat, int(need * 1.6), tmp_path / "1.6")
    assert fitting["fits"] and fitting["derived_headroom"] >= 0, fitting
    if remat == "full":
        tight = gated(layers, generation, remat, int(need * 0.6), tmp_path / "0.6")[0]
        assert tight["derived_headroom"] < 0, tight
        assert not tight["fits"] or tight["temp"] < roomy["temp"], tight
