"""The scripts under tools/ that no other test imports and runs.

Each is loaded from its file, the way tests/test_benchmark_data.py loads
benchmark_data.py, and run on a case small enough for CPU in seconds. A
reference generator writes its tiny fixture into a temporary directory and
the result is compared with what is committed: data/config/weight identity,
and named arithmetic outputs under the existing family parity contracts.
This catches generator drift without requiring bit-identical CPU math. A
benchmark runs its pure pieces, and its real entry point where the step
compiles on CPU in seconds.
"""

import dataclasses
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# The existing family parity contracts, not bounds fitted to a CI CPU:
# test_moe.py (routers and expert sums), test_text_encoders.py (tiny towers),
# test_t5_encoders.py (tiny hidden states), test_vae_16ch.py (raw encode/decode).
MOE_OUTPUTS = {
    "mixtral.npz": {"router_weights": 1e-6, "block_output": 2e-5},
    "deepseek.npz": {"router_weights": 1e-6, "block_output": 2e-5},
    "deepseek_v2.npz": {"router_weights": 1e-6},
    "deepseek_v4.npz": {"router_weights": 1e-6, "experts_output": 2e-5},
}
CLIP_OUTPUTS = dict.fromkeys(
    ("last_hidden_state", "pooler_output", "text_embeds", "image_embeds"), 1e-4)
T5_OUTPUTS = {"last_hidden_state": 1e-4}
VAE_OUTPUTS = {"latent": 1e-5, "decoded": 1e-5}


def load(name: str):
    """tools/ holds scripts, not a package, so a tool is loaded from its file."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_fixture_arrays(written: Path, committed: Path, numerical: dict[str, float]) -> None:
    """Only named arithmetic outputs use the family's fp32 parity bound.

    Seeds do not promise cross-platform bit identity for arithmetic:
    https://docs.pytorch.org/docs/2.14/notes/randomness.html
    https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html
    Inputs (including floats), weights and discrete choices remain exact.
    These bounds do not guarantee every CPU/release; a failure requires
    investigation, never automatic fixture regeneration.
    """
    with np.load(written) as ours, np.load(committed) as theirs:
        assert set(ours.files) == set(theirs.files)
        assert numerical.keys() <= set(theirs.files)
        for name in theirs.files:
            actual, expected = ours[name], theirs[name]
            assert actual.shape == expected.shape, name
            assert actual.dtype == expected.dtype, name
            if name in numerical:
                assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
                difference = float(np.max(np.abs(actual - expected)))
                assert difference < numerical[name], f"{name}: max error {difference:.3e}"
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=name)


def assert_fixture_json(written: Path, committed: Path) -> None:
    assert {p.name for p in written.iterdir()} == {p.name for p in committed.iterdir()}
    for path in committed.glob("*.json"):
        assert json.loads((written / path.name).read_text()) == json.loads(path.read_text()), path.name


def assert_same_tensors(written: Path, committed: Path) -> None:
    from safetensors.numpy import load_file

    ours, theirs = load_file(str(written)), load_file(str(committed))
    assert sorted(ours) == sorted(theirs)
    for name in theirs:
        assert ours[name].dtype == theirs[name].dtype, name
        assert np.array_equal(ours[name], theirs[name]), name


# ---------------------------------------------------------------------------
# tools/moe_reference.py
# ---------------------------------------------------------------------------

def test_moe_fixtures_are_what_the_generator_writes(tmp_path):
    """Exact recipe, parameters and router choices; fp32 router/block parity."""
    load("moe_reference").main(["--out", str(tmp_path)])

    committed = FIXTURES / "moe"
    assert_fixture_json(tmp_path, committed)
    for name, outputs in MOE_OUTPUTS.items():
        assert_fixture_arrays(tmp_path / name, committed / name, outputs)


def test_moe_expert_tensors_undo_the_gate_up_merge():
    """transformers fuses gate_proj and up_proj into one tensor with the gate
    rows first; put back together in that order, the per-expert tensors the
    generator writes are the fused ones."""
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    tool = load("moe_reference")
    experts = MixtralExperts(MixtralConfig(**tool.MIXTRAL))
    tool.scatter_weights(experts, seed=3)
    written = tool.expert_tensors(experts)

    gate_up = experts.get_parameter("gate_up_proj").detach().numpy()
    down = experts.get_parameter("down_proj").detach().numpy()
    for index in range(tool.MIXTRAL["num_local_experts"]):
        fused = np.concatenate([written[f"mlp.experts.{index}.gate_proj.weight"],
                                written[f"mlp.experts.{index}.up_proj.weight"]])
        assert np.array_equal(fused, gate_up[index])
        assert np.array_equal(written[f"mlp.experts.{index}.down_proj.weight"], down[index])
    assert not np.array_equal(gate_up[0, :tool.EXPERT_HIDDEN], gate_up[0, tool.EXPERT_HIDDEN:]), (
        "gate and up rows are identical here, so a swap would pass")


# ---------------------------------------------------------------------------
# tools/clip_reference.py
# ---------------------------------------------------------------------------

def test_clip_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, tokenizer and pixels; both towers meet fp32 parity."""
    load("clip_reference").write_tiny(tmp_path)

    committed = FIXTURES / "clip" / "tiny"
    assert_fixture_json(tmp_path, committed)
    assert_same_tensors(tmp_path / "model.safetensors", committed / "model.safetensors")
    assert_fixture_arrays(tmp_path / "reference.npz", committed / "reference.npz", CLIP_OUTPUTS)


# ---------------------------------------------------------------------------
# tools/t5_reference.py
# ---------------------------------------------------------------------------

def test_t5_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, config, tokenizer and tokens; fp32 encoder parity."""
    load("t5_reference").main(["--out", str(tmp_path)])

    written, committed = tmp_path / "tiny", FIXTURES / "t5" / "tiny"
    assert_fixture_json(written, committed)
    assert_same_tensors(written / "model.safetensors", committed / "model.safetensors")
    assert_fixture_arrays(written / "reference.npz", committed / "reference.npz", T5_OUTPUTS)


# ---------------------------------------------------------------------------
# tools/vae_reference.py
# ---------------------------------------------------------------------------

def test_vae_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, config and sample; fp32 encode/decode parity."""
    load("vae_reference").main(["--out", str(tmp_path)])

    written, committed = tmp_path / "sd3-tiny", FIXTURES / "vae" / "sd3-tiny"
    assert_fixture_json(written, committed)
    assert_same_tensors(written / "diffusion_pytorch_model.safetensors",
                        committed / "diffusion_pytorch_model.safetensors")
    assert_fixture_arrays(written / "reference.npz", committed / "reference.npz", VAE_OUTPUTS)


# ---------------------------------------------------------------------------
# tools/optimizer_curve.py
# ---------------------------------------------------------------------------

def token_directory(tmp_path: Path) -> Path:
    """A byte-tokenized corpus, written by the tool the curve reads from."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"line {i}: the quick brown fox jumps over the lazy dog\n"
                              for i in range(60)))
    tokenize = load("tokenize_text")
    out = tmp_path / "tokens"
    tokenize.main(tokenize.TokenizeArgs(input=str(corpus), out=str(out),
                                        tokenizer="byte", val_fraction=0.1))
    return out


def test_optimizer_curve_arms_share_the_model_and_the_batches(tmp_path):
    """The step-0 loss is computed before any update, so two arms at one seed
    agree on it exactly, and only the solver separates them afterwards. An
    arm that seeded its own init, or shuffled its own order, would differ at
    step 0; two arms running the same solver would never separate."""
    tool = load("optimizer_curve")
    tokens = token_directory(tmp_path)
    curves = {}
    for solver in ("adamw", "muon-unsplit"):
        out = tmp_path / f"{solver}.json"
        tool.main(tool.Comparison(dataset=str(tokens), out=str(out), optimizer=solver,
                                  steps=3, batch_size=8, sequence_length=8,
                                  emb_features=16, num_layers=1, num_heads=2, seed=1))
        curves[solver] = json.loads(out.read_text())

    adamw, muon = curves["adamw"]["losses"], curves["muon-unsplit"]["losses"]
    assert len(adamw) == len(muon) == 3
    assert all(np.isfinite(adamw)) and all(np.isfinite(muon))
    assert adamw[0] == muon[0]
    assert adamw[1:] != muon[1:]
    assert curves["adamw"]["tokens"] == 3 * 8 * 8
    assert curves["adamw"]["corpus_tokens"] == json.loads(
        (tokens / "meta.json").read_text())["train_tokens"]


# ---------------------------------------------------------------------------
# tools/lm_step_parity.py
# ---------------------------------------------------------------------------

def test_lm_step_parity_records_a_repeatable_fixed_batch_run():
    """Two runs are comparable only if one run is repeatable: the same seed
    and the same batch give the same losses to the last bit. On one fixed
    batch the loss also has to fall, which a loop feeding fresh random
    tokens each step would not show."""
    tool = load("lm_step_parity")
    config = dict(vocab_size=64, emb_features=16, num_layers=1, num_heads=2,
                  mlp_features=32, max_seq_len=8)

    first = tool.run(config, batch=8, seq=8, steps=4)
    second = tool.run(config, batch=8, seq=8, steps=4)

    assert len(first.losses) == len(first.token_accuracy) == 4
    assert first == second
    assert all(np.isfinite(first.losses))
    assert first.losses[-1] < first.losses[0]
    assert all(0.0 <= accuracy <= 1.0 for accuracy in first.token_accuracy)


# ---------------------------------------------------------------------------
# tools/benchmark_lm_head.py
# ---------------------------------------------------------------------------

def test_lm_head_variant_names_parse_as_documented():
    """A name is the head, an optional chunk count and optional suffixes:
    the rows docs/research/lm-head.md ran, plus both suffixes at once."""
    tool = load("benchmark_lm_head")
    parsed = {text: tool.parse_variant(text) for text in
              ("baseline", "stored4", "stored8", "remat4", "stored4-noacc", "remat8-noacc-fp32")}

    assert [(v.head, v.chunks, v.accuracy, v.states_dtype.name) for v in parsed.values()] == [
        ("baseline", 4, True, "bfloat16"),
        ("stored", 4, True, "bfloat16"),
        ("stored", 8, True, "bfloat16"),
        ("remat", 4, True, "bfloat16"),
        ("stored", 4, False, "bfloat16"),
        ("remat", 8, False, "float32"),
    ]
    for bad in ("chunked4", "stored4-fast", "stored-8"):
        with pytest.raises(ValueError):
            tool.parse_variant(bad)


def test_lm_head_variants_compute_the_same_loss_accuracy_and_gradients():
    """stored and remat are the baseline head rearranged into vocabulary
    tiles, so on one small case all three agree on the loss, the top-1
    accuracy and both gradients. The vocabulary of 12 does not split into
    4 equal tiles, so the short last tile is on the path."""
    tool = load("benchmark_lm_head")
    key = jax.random.PRNGKey(0)
    states = jax.random.normal(key, (2, 3, 8), jnp.float32)
    table = jax.random.normal(jax.random.fold_in(key, 1), (12, 8), jnp.float32)
    targets = jnp.array([[0, 5, 11], [3, 11, 4]], jnp.int32)
    variant = tool.parse_variant("stored4-fp32")

    outputs = {}
    for name in ("baseline", "stored", "remat"):
        head = tool.HEADS[name]
        (loss, accuracy), (d_states, d_table) = jax.value_and_grad(
            lambda s, t: head(s, t, targets, variant), argnums=(0, 1), has_aux=True)(states, table)
        outputs[name] = (float(loss), float(accuracy), np.asarray(d_states), np.asarray(d_table))

    reference = outputs["baseline"]
    for name in ("stored", "remat"):
        loss, accuracy, d_states, d_table = outputs[name]
        assert loss == pytest.approx(reference[0], abs=1e-5), name
        assert accuracy == reference[1], name
        np.testing.assert_allclose(d_states, reference[2], atol=1e-5, err_msg=name)
        np.testing.assert_allclose(d_table, reference[3], atol=1e-5, err_msg=name)


# ---------------------------------------------------------------------------
# tools/benchmark_step.py
# ---------------------------------------------------------------------------

TINY_LM = {"vocab_size": 64, "emb_features": 16, "num_layers": 1, "num_heads": 2,
           "mlp_features": 32, "max_seq_len": 8}


def composite_case(tool, architecture: str, **changes):
    """The cpu-smoke preset's case for one composite, resized by `changes`."""
    (case,) = [c for c in tool.cpu_smoke_cases() if c.architecture == architecture]
    return dataclasses.replace(case, **changes)


def parameter_movement(tool, case, steps: int = 2):
    """How far every parameter moved over `steps` of the case's real compiled
    step, by tree path, with the loss and the step's measured FLOPs.

    The trainer, the batches and the compiled step are the tool's own: a
    parameter that does not move here is one the measured step never trains.
    """
    def named(tree):
        return {jax.tree_util.keystr(path): leaf
                for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}

    trainer = tool.build_trainer(case, "reference")
    source = tool.batches(case)
    state = jax.jit(trainer.initial_state)()
    before = named(state.params)
    compiled = trainer.compile(state, next(source))
    for _ in range(steps):
        state, loss, _, finite, _ = compiled(state, next(source))
    after = named(state.params)
    assert bool(finite) and np.isfinite(float(loss))
    moved = {name: float(jnp.max(jnp.abs(value - before[name])))
             for name, value in after.items()}
    return moved, trainer.flops_per_step


def test_step_benchmark_keeps_the_cases_it_measured_when_a_later_one_fails(tmp_path):
    """A sweep writes --json-out after every case, so the rows measured before
    a case that cannot be built are kept: the file holds the finished row,
    and only that row."""
    tool = load("benchmark_step")
    out = tmp_path / "rows.json"
    good = tool.Case("causal_transformer", dict(TINY_LM), batch_size=8, seq_len=8,
                     fsdp_min_param_size=256)
    bad = tool.Case("no_such_architecture")
    config = tool.BenchmarkConfig(cases=[good, bad], warmup=1, steps=2, json_out=str(out))

    with pytest.raises(KeyError, match="no model named 'no_such_architecture'"):
        tool.run(config)

    rows = json.loads(out.read_text())
    assert [row["architecture"] for row in rows] == ["causal_transformer"]
    (row,) = rows
    assert row["measured_steps"] == 2 and row["finite"] and np.isfinite(row["loss"])
    assert row["samples_per_sec"] == pytest.approx(8 / (row["ms_per_step"] / 1e3), rel=1e-2)


def test_step_benchmark_overrides_reach_only_the_cases_they_apply_to():
    """--frames resizes the video cases and leaves an image model's rank
    alone; --packed-documents packs the plain token windows and nobody else,
    so a canvas row keeps the geometry its objective reads and a media row
    keeps the one its processor emitted; --batch-size reaches every case."""
    tool = load("benchmark_step")
    cases = tool.build_cases(tool.BenchmarkConfig(
        preset="small", frames=4, packed_documents=2, batch_size=2))

    by_name = {}
    for case in cases:
        by_name.setdefault(case.architecture, []).append(case)
    assert {c.frames for c in by_name["video_dit"] + by_name["unet_3d"]} == {4}
    assert {c.frames for c in by_name["simple_dit"] + by_name["unet"]} == {0}
    assert {c.packed_documents for c in by_name["causal_transformer"]} == {2}
    assert all(c.packed_documents == 0 for c in cases if not c.is_lm)
    composites = by_name["multimodal_transformer"] + by_name["diffusion_gemma"]
    assert [c.packed_documents for c in composites] == [0, 0]
    assert {c.batch_size for c in cases} == {2}


def test_step_benchmark_refuses_a_case_it_cannot_name():
    """An architecture outside the preset and a JSON field outside Case are
    both errors before anything compiles, and so is a composite whose row
    does not hold what its wrapper reads."""
    tool = load("benchmark_step")
    with pytest.raises(ValueError, match="no_such"):
        tool.build_cases(tool.BenchmarkConfig(preset="cpu-smoke", architectures=["no_such"]))
    with pytest.raises(ValueError, match="bogus"):
        tool.cases_from_json('[{"architecture": "unet", "bogus": 1}]')
    with pytest.raises(ValueError, match="JSON list"):
        tool.cases_from_json('{"architecture": "unet"}')
    ragged = composite_case(tool, "diffusion_gemma", seq_len=14)
    with pytest.raises(ValueError, match="whole canvases"):
        tool.build_trainer(ragged, "reference")
    crowded = composite_case(tool, "multimodal_transformer", seq_len=2)
    with pytest.raises(ValueError, match="no room"):
        next(tool.batches(crowded))


def test_step_benchmark_packed_rows_restart_positions_at_every_document():
    """A row of 17 tokens packed as 4 documents is three of 5 and one of 2:
    the segment ids count the documents from 1 and the positions count from
    0 inside each, the form the packed loader's mask and RoPE read."""
    tool = load("benchmark_step")
    case = tool.Case("causal_transformer", dict(TINY_LM), batch_size=2, seq_len=16,
                     packed_documents=4)

    batch = next(tool.batches(case))

    assert batch["text"].shape == (2, 17) and batch["text"].max() < 64
    for row in range(2):
        assert batch["text_segment_ids"][row].tolist() == [1] * 5 + [2] * 5 + [3] * 5 + [4] * 2
        assert batch["text_positions"][row].tolist() == [0, 1, 2, 3, 4] * 3 + [0, 1]


def test_step_benchmark_small_preset_exempts_only_the_jepa_predictor():
    """Every registry architecture has a case of its own except
    jepa_predictor, which has no step of its own: the JEPA cases build it
    through the registry inside their objective, so their rows are its rows.
    An architecture named as covered without a case measuring it would leave
    the difference here nonempty."""
    from dew import models

    tool = load("benchmark_step")
    cases = tool.small_cases("bfloat16")

    assert set(models) - {case.architecture for case in cases} == {"jepa_predictor"}
    (jepa,) = [case for case in cases if case.architecture == "jepa_encoder"]
    predictor = tool.build_trainer(jepa, "reference").objective.predictor
    assert models.name_of(type(predictor)) == "jepa_predictor"


def test_step_benchmark_media_rows_mark_one_slot_per_projected_feature():
    """A media row is what a processor hands the model: the image tokens fill
    exactly the slots the projector has features for, each slot naming its
    own feature and every other slot naming none. Fewer slots would pay for
    features the decoder never reads."""
    tool = load("benchmark_step")
    case = composite_case(tool, "multimodal_transformer", batch_size=2)
    case = dataclasses.replace(case, media={**case.media, "images": 2})

    inputs = next(tool.batches(case))["text"]

    slots = 2 * tool.image_tokens(case)
    assert slots == 8  # two images, four soft tokens each from a 2x2 patch grid
    indices = np.asarray(inputs.token_fields["image_indices"])
    assert inputs.tokens.shape == (2, 16)
    assert np.array_equal(np.asarray(inputs.tokens)[:, :slots], np.full((2, slots), 255))
    for row in range(2):
        assert indices[row].tolist() == list(range(slots)) + [-1] * (16 - slots)
    assert np.asarray(inputs.conditioning["pixel_values"]).shape == (2, 2, 3, 16, 16)


def test_step_benchmark_canvas_rows_are_whole_unpadded_canvases():
    """A canvas row is a clean prompt followed by whole canvases, and carries
    no pad id: the block loss reads its target masks off that id, so a drawn
    zero would move the measured target support with the batch seed."""
    tool = load("benchmark_step")
    case = composite_case(tool, "diffusion_gemma", batch_size=2)

    assert tool.canvas_split(case) == (8, 4, 2)
    tokens = next(tool.batches(case))["text"]
    assert tokens.shape == (2, 16) and tokens.min() >= 1 and tokens.max() < 256


def test_step_benchmark_media_step_trains_the_image_tower():
    """The measured step is the whole conditioned model's: the tower and the
    projector move under it, and a second image a row raises the FLOPs the
    row reports. A step that fed no pixels would leave both untouched."""
    tool = load("benchmark_step")
    case = composite_case(tool, "multimodal_transformer")

    moved, flops = parameter_movement(tool, case)

    media = {name: value for name, value in moved.items()
             if "tower" in name or "projector" in name}
    assert media and min(media.values()) > 0
    assert min(value for name, value in moved.items() if "language_model" in name) > 0
    _, wider = parameter_movement(
        tool, dataclasses.replace(case, media={**case.media, "images": 2}), steps=1)
    assert wider > flops


def test_step_benchmark_canvas_step_trains_the_self_conditioning_decoder():
    """The measured step is the official fine-tuning step: the
    self-conditioning MLP the second decoder pass feeds moves under it, and a
    wider response raises the FLOPs the row reports. A plain next-token step
    on the same trunk has no such parameter to move."""
    tool = load("benchmark_step")
    narrow = composite_case(tool, "diffusion_gemma",
                            canvas={"prompt_length": 12, "canvas_size": 4})

    moved, one_canvas = parameter_movement(tool, narrow)

    conditioning = {name: value for name, value in moved.items()
                    if "self_conditioning" in name}
    assert conditioning and min(conditioning.values()) > 0
    assert min(value for name, value in moved.items() if "text" in name) > 0
    _, two_canvases = parameter_movement(
        tool, composite_case(tool, "diffusion_gemma"), steps=1)
    assert two_canvases > one_canvas


def test_step_benchmark_table_shows_each_column_in_its_unit():
    """FLOPs print as GFLOP, utilisation as a percentage, bytes as GiB, a
    parameter count with separators, and a value the backend did not report
    as n/a."""
    tool = load("benchmark_step")
    row = {"architecture": "simple_dit", "batch_size": 8, "fsdp_size": 1, "expert_size": 1,
           "params": 1234567, "ms_per_step": 5.55, "p10_ms": 3.9, "p50_ms": 4.2,
           "p90_ms": 4.4, "samples_per_sec": 1435.7, "flops_per_step": 2.5e9,
           "utilization": 0.4321, "peak_device_bytes": 3 * 2 ** 30}

    table = tool.format_table([row, {**row, "utilization": None, "peak_device_bytes": None}])

    lines = table.splitlines()
    assert lines[2].split() == ["simple_dit", "8", "1", "1", "1,234,567", "5.5", "3.9", "4.2",
                                "4.4", "1435.7", "2.5", "43.2", "3.00"]
    assert lines[3].split()[-2:] == ["n/a", "n/a"]
