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
from test_plan import LIMIT, asked, free, needs, one_device

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.objectives.lm import LMObjective
from dew.telemetry.compiles import Compiles
from dew.training import MeshSpec, Trainer
from dew.training.planning import PlanRefused


def deep(remat=None, split: int | None = 1):
    """A decoder whose residuals are most of what its step holds, on one device, reading its batch
    whole, or in `split` microbatches, or in the split its plan finds (None): two layers over 256 tokens,
    the derivation's case (each layer's residuals there are some forty times its parameters)."""
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2, mlp_features=64,
                              max_seq_len=256, remat=remat)
    return one_device(Trainer(LMObjective(model, seq_len=256), optax.sgd(1e-2), key=jax.random.key(0),
                              split=split))


DEEP = {"text": jnp.asarray((np.arange(8 * 257).reshape(8, 257) * 7) % 64, jnp.int32)}


def compiled(trainer, batch=DEEP):
    """`trainer`'s step compiled for `batch`, its plan made."""
    state, _, _ = trainer.place()
    return trainer.compile(state, batch)


@pytest.fixture(scope="module")
def room(tmp_path_factory) -> int:
    """Bytes free with room for the deep step under full recomputation, as compiled, and not under
    none, as compiled or as derived, measured once for the module by runs that record their plans
    apart from the runs its tests make."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path_factory.mktemp("sizing")))
        recomputed, base = deep(REMAT_POLICIES["full"]), deep()
        full = needs(monkeypatch, recomputed, lambda: compiled(recomputed))
        whole = needs(monkeypatch, base, lambda: compiled(base))
        return (full + min(whole, LIMIT - base.plan["derived"][0]["headroom"])) // 2


def test_the_first_compile_is_at_the_least_recompute_rung_the_derivation_says_fits(
        monkeypatch, tmp_path, room):
    """With room for the step under full recomputation and not under none, the search's first compile
    is at the least-recompute rung whose derived footprint fits, which the compile confirms: the step
    compiles once, where the climb from the base rung compiled every rung below. The plan records each
    rung the derivation read, refused below and fitting at its rung."""
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
    the derivation nothing to judge against ("unstated"), and the search starts at the base rung too."""
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
    assert [entry["label"] for entry in trainer.plan["derived"]] == ["unstated"]
    assert trainer.plan["probes"][0]["remat"] is None


def test_a_run_that_starts_from_a_recorded_plan_compiles_it_first(monkeypatch, tmp_path, room):
    """A restart whose plan an earlier run recorded (beside the compilation cache) compiles that plan
    first, with no derivation read, where its devices now hold room for full recomputation alone: the
    compile, not the derivation, decides whether it fits, and a miss climbs from there."""
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


def test_the_first_compile_is_at_the_fewest_microbatches_the_derivation_says_fit(monkeypatch, tmp_path):
    """With room for the step at four microbatches and not at two, as both its compile and its derivation
    say, the search's first compile is at four, the fewest the derivation says fit, which the compile
    confirms: the step compiles once, where the search compiled the whole batch first and halved the
    range by compiles (row 3341). The plan records the splits the derivation read."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "sizing"))
    need = {}
    for split in (2, 4):
        sizing = deep(split=split)
        compiled_need = needs(monkeypatch, sizing, lambda sizing=sizing: compiled(sizing))
        (judged,) = [entry for entry in sizing.plan["derived"] if entry["split"] == split]
        need[split] = (compiled_need, LIMIT - judged["headroom"])
    room = max(need[4])
    assert room < min(need[2]), need
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "run"))
    free(monkeypatch, room)
    trainer = deep(split=None)
    with Compiles("step") as compiles:
        compiled(trainer)
    assert compiles.count == 1 and trainer.plan["split"] == 4 and trainer.plan["remat"] is None
    assert {entry["split"]: entry["label"] for entry in trainer.plan["derived"]}[2] == "refuses"


def test_a_split_the_derivation_leaves_no_byte_to_spare_is_one_it_fits(monkeypatch, tmp_path):
    """A split whose derived headroom is exactly nothing fits as the derivation judges a fit (a headroom of
    at least nothing), so with room for four microbatches to the byte the search's first compile is at
    four, the fewest the derivation says fit, not the eight past it (row 3341's review: a headroom of
    nothing read as a refusal)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "sizing"))
    sizing = deep(split=4)
    needs(monkeypatch, sizing, lambda: compiled(sizing))
    (judged,) = [entry for entry in sizing.plan["derived"] if entry["split"] == 4]
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "run"))
    free(monkeypatch, LIMIT - judged["headroom"])
    trainer = deep(split=None)
    compiled(trainer)
    headroom = {entry["split"]: entry["headroom"] for entry in trainer.plan["derived"]}
    assert headroom[4] == 0 and headroom[2] < 0 and asked(trainer)[0][0] == 4


def test_a_run_that_sets_its_remat_derives_its_own_rung_alone_and_compiles_there_once(monkeypatch, tmp_path):
    """A run that sets its remat compiles at it alone (row 2111), once, its split derived there; no rung
    below its own is traced for it, which cost a run every lighter rung's trace before its first step to
    say only whether one fits (row 3341)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    free(monkeypatch, LIMIT)
    for remat in ("full", "save_qkv_proj"):
        trainer = deep(REMAT_POLICIES[remat])
        with Compiles("step") as compiles:
            compiled(trainer)
        assert compiles.count == 1 and trainer.plan["remat"] == remat
        assert trainer.plan["derived"] and {entry["remat"] for entry in trainer.plan["derived"]} == {remat}


@pytest.mark.mesh(devices=2)
def test_a_run_over_two_devices_that_sets_its_remat_derives_its_share_and_compiles_once(
        monkeypatch, tmp_path):
    """A run sharded over two devices (fsdp 2, as the two-GPU run that raised row 2536) that sets full
    recomputation derives its footprint per device, each its share of the batch and the state
    (`candidate_footprint`), at its own rung, which fits with room for every rung, and compiles there
    once (rows 2615, 3341)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "two"))
    free(monkeypatch, LIMIT)
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=8, num_heads=2, mlp_features=64,
                              max_seq_len=256, remat=REMAT_POLICIES["full"])
    trainer = Trainer(LMObjective(model, seq_len=256), optax.sgd(1e-2), key=jax.random.key(0), split=1,
                      mesh=MeshSpec(fsdp=2).build(jax.devices()[:2]))
    with Compiles("step") as compiles:
        compiled(trainer)
    derived = trainer.plan["derived"]
    assert compiles.count == 1 and trainer.plan["remat"] == "full"
    assert [(entry["remat"], entry["label"]) for entry in derived] == [("full", "fits")]


@pytest.mark.mesh(devices=2)
def test_a_run_over_a_tensor_axis_whose_footprint_the_derivation_cannot_judge_compiles_its_own_rung(
        monkeypatch, tmp_path):
    """A run over a tensor axis of two that sets full recomputation, whose footprint the derivation reads
    per device only over data and fsdp axes ("unknown", row 2615), compiles at its own rung once, as it
    would with no derivation (row 2721)."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "tensor"))
    free(monkeypatch, LIMIT)
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=8, num_heads=2, mlp_features=64,
                              max_seq_len=256, remat=REMAT_POLICIES["full"])
    trainer = Trainer(LMObjective(model, seq_len=256), optax.sgd(1e-2), key=jax.random.key(0), split=1,
                      mesh=MeshSpec(tensor=2).build(jax.devices()[:2]))
    with Compiles("step") as compiles:
        compiled(trainer)
    assert compiles.count == 1 and trainer.plan["remat"] == "full"
    assert trainer.plan["derived"][-1]["label"] == "unknown"


def test_a_compile_the_derivation_judged_otherwise_is_told_as_a_model_miss(monkeypatch, tmp_path, caplog):
    """Every compile's plan record holds the headroom the derivation judged beside the one the compile
    measured, and each compile whose fit the derivation judged otherwise is told as a model miss, the
    compile deciding: with room between what the derivation and the compile say a step at full
    recomputation needs, the run hears a miss for its first compile and one for each it judged so. Whether
    a head policy past that compile then fits the room, a plan, or none does, a refusal, is the
    compiler's schedule's: under XLA:CPU's memory-optimized schedule the head tiled, its gradient taken
    in the backward, fits it (7263045940), under its default none did; the misses are told either way."""
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
    refused: PlanRefused | None = None
    with caplog.at_level(logging.WARNING, logger="dew"), Compiles("step") as compiles:
        try:
            compiled(trainer)
        except PlanRefused as error:
            refused = error
    missed = [record for record in caplog.records if "model miss" in record.getMessage()]
    probes = [probe.record() for probe in refused.probes] if refused else trainer.plan["probes"]
    judged = [probe for probe in probes
              if probe["derived_headroom"] is not None and (probe["derived_headroom"] >= 0) != probe["fits"]]
    assert compiles.count >= 1 and probes[0]["fits"] == (compiled_need < derived)
    assert missed and len(missed) == len(judged)


GATE_LAYERS = {
    "decoder": {},
    "hybrid": {"layer_types": ("mamba", "attention"), "kinds": {"mamba": {"mixer": {
        "class": "mamba2", "fields": {"num_heads": 4, "head_dim": 16, "state_size": 8, "n_groups": 1,
                                      "chunk_size": 32}}}, "attention": {}}},
    "moe": {"mixture": {"experts": 4, "top_k": 2, "layers": (1,)}},
}
"""The gate's decoders (row 2113), each the least of its family: dense, a Mamba-2 layer beside an attention
layer, and an attention layer beside one of experts, two layers each."""


def gated(layers: str, generation: str, remat: str, capacity: int, records, precision: str) -> list[dict]:
    """A gate case's step compiled ahead for one compile-only GPU of `generation` stating `capacity`
    bytes at JAX's matmul `precision`, its rungs recorded under `records`: every compile its plan made,
    refused or not."""
    from dew.registry import models
    from dew.training.ahead import compile_ahead, compile_only_gpus

    model = models.build("causal_transformer", vocab_size=128, emb_features=64, num_layers=2, num_heads=4,
                         num_kv_heads=2, mlp_features=128, max_seq_len=128,
                         remat=None if remat == "none" else REMAT_POLICIES[remat], **GATE_LAYERS[layers])
    trainer = Trainer(LMObjective(model, seq_len=128), optax.adamw(1e-3), key=jax.random.key(0), split=1,
                      mesh=MeshSpec().build(compile_only_gpus(generation, 1)))
    tokens = ((np.arange(4 * 129).reshape(4, 129) * 7) % 128).astype(np.int32)
    with pytest.MonkeyPatch.context() as patch, jax.default_matmul_precision(precision):
        patch.setenv("DEW_RUNG_RECORDS", str(records))
        try:
            compile_ahead(trainer, {"text": tokens}, capacity=capacity)
        except PlanRefused as refused:
            return [probe.record() for probe in refused.probes]
    return list(trainer.plan["probes"])


@pytest.mark.parametrize("precision", ["default", "highest"])
@pytest.mark.parametrize("remat", ["none", "full"])
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
@pytest.mark.parametrize("layers", list(GATE_LAYERS))
def test_the_derivation_decides_a_steps_fit_as_its_compile_does(layers, generation, remat, precision,
                                                                   tmp_path):
    """The derivation judges a step's fit as its compile does at 0.6 and 1.6 times what the compile needs,
    over decoders dense, hybrid and of experts, Ampere and Hopper, no recomputation and full, JAX's
    default and highest matmul precision (row 2113's gate, in the kernel gate's evidence shape,
    `compile_evidence`). The need is the program's as XLA lays it out on a roomy GPU, and the
    derivation's, read off that compile, admits it at 1.6 times and refuses it at 0.6. A gate decoder's
    peak is the program's own (`Footprint.exact`), an expert decoder's at JAX's default precision too: its
    ragged products run on CUDA as Pallas grouped kernels, custom calls that hold their outputs alone
    (`dew.nn.programs.opaque`). At highest precision they are XLA's ragged products, the GEMM template's
    portable route (`grouped_matmul.xla_ragged_dot`), which XLA lays out as its target, dtype and flags
    choose: that peak alone is a bound from below, never above the compile's need, which never refuses a
    step the compile fits and may admit one it refuses, a miss its compile corrects. On a GPU stating 1.6
    times the compile is the roomy one, whose need it holds, XLA's rematerialization changing nothing in a
    module already within its limit (`RematerializeComputation`,
    xla/hlo/transforms/simplifiers/hlo_rematerialization.cc), so its fit is read off that compile, at 0.6
    times as at 1.6."""
    from compile_evidence import Evidence, Facts, Measurement, Prediction

    (roomy,) = gated(layers, generation, remat, 1 << 34, tmp_path / "roomy", precision)
    need, derived = (1 << 34) - roomy["headroom"], (1 << 34) - roomy["derived_headroom"]
    exact = roomy["derived"]["peak_exact"]
    assert exact == (layers != "moe" or precision != "highest"), roomy["derived"]
    assert exact or derived <= need, (derived, need)
    for margin, admits in ((1.6, True), (0.6, False)):
        budget = int(need * margin)
        evidence = Evidence(
            f"{layers} {generation} remat {remat} {precision} at {margin} of its need", "",
            Facts((layers, remat), generation),
            Prediction({"peak": roomy["derived"]["peak"]}, derived, budget, derived <= budget, "bytes"),
            Measurement({"temp": roomy["temp"]}, need <= budget, "xla", jax.__version__))
        assert evidence.compiled.admits == admits, evidence.describe()
        assert evidence.agrees or (not exact and derived <= budget), evidence.describe()
