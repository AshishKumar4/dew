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
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tool_support import FIXTURES, REPO_ROOT, load

CLAIMS = json.loads((FIXTURES / "provenance.json").read_text())["references"]

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


def assert_fixture_files(written: Path, committed: Path, tool: str) -> None:
    """The generator's listing against a directory provenance.json gives it
    whole, which test_fixture_provenance keeps every other writer out of."""
    assert CLAIMS.get(committed.relative_to(FIXTURES).as_posix(), {}).get("tool") == f"tools/{tool}.py"
    assert {p.name for p in written.iterdir() if p.is_file()} == {
        p.name for p in committed.iterdir() if p.is_file()}


def assert_fixture_json(written: Path, committed: Path, tool: str) -> None:
    assert_fixture_files(written, committed, tool)
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
    assert_fixture_json(tmp_path, committed, "moe_reference")
    for name, outputs in MOE_OUTPUTS.items():
        assert_fixture_arrays(tmp_path / name, committed / name, outputs)

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
        np.testing.assert_array_equal(fused, gate_up[index])
        np.testing.assert_array_equal(written[f"mlp.experts.{index}.down_proj.weight"], down[index])
    assert not np.array_equal(gate_up[0, :tool.EXPERT_HIDDEN], gate_up[0, tool.EXPERT_HIDDEN:])


# ---------------------------------------------------------------------------
# tools/clip_reference.py
# ---------------------------------------------------------------------------

def test_clip_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, tokenizer and pixels; both towers meet fp32 parity."""
    load("clip_reference").write_tiny(tmp_path)

    committed = FIXTURES / "clip" / "tiny"
    assert_fixture_json(tmp_path, committed, "clip_reference")
    assert_same_tensors(tmp_path / "model.safetensors", committed / "model.safetensors")
    assert_fixture_arrays(tmp_path / "reference.npz", committed / "reference.npz", CLIP_OUTPUTS)


# ---------------------------------------------------------------------------
# tools/t5_reference.py
# ---------------------------------------------------------------------------

def test_t5_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, config, tokenizer and tokens; fp32 encoder parity."""
    load("t5_reference").main(["--out", str(tmp_path)])

    written, committed = tmp_path / "tiny", FIXTURES / "t5" / "tiny"
    assert_fixture_json(written, committed, "t5_reference")
    assert_same_tensors(written / "model.safetensors", committed / "model.safetensors")
    assert_fixture_arrays(written / "reference.npz", committed / "reference.npz", T5_OUTPUTS)


# ---------------------------------------------------------------------------
# tools/vae_reference.py
# ---------------------------------------------------------------------------

def test_vae_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, config and sample; fp32 encode/decode parity."""
    load("vae_reference").main(["--out", str(tmp_path)])

    written, committed = tmp_path / "sd3-tiny", FIXTURES / "vae" / "sd3-tiny"
    assert_fixture_json(written, committed, "vae_reference")
    assert_same_tensors(written / "diffusion_pytorch_model.safetensors",
                        committed / "diffusion_pytorch_model.safetensors")
    assert_fixture_arrays(written / "reference.npz", committed / "reference.npz", VAE_OUTPUTS)


# ---------------------------------------------------------------------------
# tools/flaxdiff_reference.py
# ---------------------------------------------------------------------------

# The modules both pinned models import, including models/__init__.py's UNet.
FLAXDIFF_COMMON_SOURCES = ("flaxdiff/__init__.py", "flaxdiff/models/__init__.py",
                          "flaxdiff/models/attention.py", "flaxdiff/models/common.py",
                          "flaxdiff/models/hilbert.py", "flaxdiff/models/simple_dit.py",
                          "flaxdiff/models/simple_unet.py", "flaxdiff/models/vit_common.py")
FLAXDIFF_SOURCES = {
    "simple_udit": (*FLAXDIFF_COMMON_SOURCES, "flaxdiff/models/simple_vit.py"),
    "hybrid_dit": (*FLAXDIFF_COMMON_SOURCES, "flaxdiff/models/ssm_dit.py"),
}


@pytest.mark.network
@pytest.mark.parametrize("architecture, committed", [
    ("simple_udit", FIXTURES / "flaxdiff"),
    ("hybrid_dit", FIXTURES / "flaxdiff" / "hybrid_dit"),
], ids=["simple_udit", "hybrid_dit"])
def test_flaxdiff_fixture_is_what_the_generator_writes(tmp_path, architecture, committed):
    """FlaxDiff's own code at the pinned commit writes the committed fixture:
    exact config, weights and inputs, and the output at test_flaxdiff.py's bound."""
    pytest.importorskip("matplotlib")  # FlaxDiff's hilbert module imports it
    import urllib.request

    tool = load("flaxdiff_reference")
    commit = {"simple_udit": tool.COMMIT, "hybrid_dit": tool.HYBRID_COMMIT}[architecture]
    source = tmp_path / "flaxdiff-src"
    for name in FLAXDIFF_SOURCES[architecture]:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/AshishKumar4/FlaxDiff/{commit}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    written = tmp_path / "out"
    # A fresh interpreter keeps one pin's imported modules out of the other pin.
    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "flaxdiff_reference.py"),
                    "--flaxdiff-path", str(source), "--architecture", architecture,
                    "--out", str(written)], check=True)

    assert_fixture_files(written, committed, "flaxdiff_reference")
    actual = json.loads((written / "config.json").read_text())
    expected = json.loads((committed / "config.json").read_text())
    # jax_version records the generating environment, not the fixture's recipe.
    assert actual.pop("jax_version") == jax.__version__
    expected.pop("jax_version", None)
    assert actual == expected
    assert_fixture_arrays(written / "reference.npz", committed / "reference.npz", {"output": 1e-6})


# ---------------------------------------------------------------------------
# tools/optimizer_curve.py
# ---------------------------------------------------------------------------

def token_directory(tmp_path: Path) -> Path:
    """A byte-tokenized corpus, in the directory the curve reads from."""
    from token_support import encoded_token_fixture

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"line {i}: the quick brown fox jumps over the lazy dog\n"
                              for i in range(60)))
    out = tmp_path / "tokens"
    encoded_token_fixture(out, [corpus.read_text()], tokenizer="byte", val_fraction=0.1)
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
    config = {"vocab_size": 64, "emb_features": 16, "num_layers": 1, "num_heads": 2,
              "mlp_features": 32, "max_seq_len": 8}

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
            lambda s, t, head=head: head(s, t, targets, variant), argnums=(0, 1), has_aux=True)(states, table)
        outputs[name] = (float(loss), float(accuracy), np.asarray(d_states), np.asarray(d_table))

    reference = outputs["baseline"]
    for name in ("stored", "remat"):
        loss, accuracy, d_states, d_table = outputs[name]
        assert loss == pytest.approx(reference[0], abs=1e-5), name
        assert accuracy == reference[1], name
        np.testing.assert_allclose(d_states, reference[2], atol=1e-5, err_msg=name)
        np.testing.assert_allclose(d_table, reference[3], atol=1e-5, err_msg=name)


# ---------------------------------------------------------------------------
# tools/layout_parity.py
# ---------------------------------------------------------------------------

def test_layout_parity_orders_preserve_the_tiny_nemotron_function():
    from residual_orders import permuted

    from dew.interop import Pretrained

    loaded = Pretrained.load(FIXTURES / 'hf/nemotron-h-moe-tiny', dtype='float32',
                             attention_impl='reference')
    order = np.array([3, 0, 12, 5, 10, 1, 8, 7, 2, 14, 4, 13, 6, 15, 9, 11])
    ids = jnp.array([[2, 7, 11, 3]], jnp.int32)
    tool = load('layout_parity')
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: leaf.astype(jnp.float64), loaded.variables)
        model = loaded.model.clone(dtype=jnp.float64)
        truth = np.asarray(model.apply(wide, ids))
        reordered = np.asarray(model.apply(permuted(wide, order), ids))
    tool.order_invariance({'logits': reordered}, {'logits': truth},
                          float(reordered.sum()), float(truth.sum()))


@pytest.mark.parametrize("model", ["dit", "dense_mdlm"])
def test_layout_parity_anchors_an_objective_that_draws_per_row_on_its_own_draws(model):
    """The fp64 anchor is the reference's step with every sum in fp64, the
    noise and masks the trainer's first step draws included, so each leaf is a
    rounding of the reference's: an fp32 distance, not a different step. A
    diffusion objective and a masked one draw by row, so their floor rests on
    the anchor alone; the anchor built only decoders, and computed no step for
    these (765a6f97)."""
    tool = load("layout_parity")
    import benchmark_models

    case = dataclasses.replace(tool.zoo()[model], dtype="float32")
    # As the tool runs: the reference and the anchor both under x64.
    with jax.enable_x64(new_val=True):
        batch = benchmark_models.global_batch(case)
        reference = tool.computed_reference(case, batch, 1)
        loss, gradient = tool.anchor_step(case, batch)

    assert loss == pytest.approx(reference.losses[0], rel=1e-5)
    errors = tool.leaf_errors(gradient, reference.gradient, "float32")
    assert max(errors.values()) <= tool.rounding_limit("float32"), max(
        errors.items(), key=lambda item: item[1])


def test_layout_parity_reaches_every_leaf_of_a_model_that_initializes_its_output_to_zeros():
    """A DiT zero-initializes its output projection and its modulations, so
    its first step's gradient reached 2 of the zoo DiT's 70 leaves, and every
    DiT, UNet and MMDiT layout was judged on its output layer alone. The
    tool draws every all-zero leaf, so each one carries a gradient."""
    tool = load("layout_parity")
    import benchmark_models

    case = dataclasses.replace(tool.zoo()["dit"], dtype="float32")
    _, gradient, _ = tool.trained(case, {}, benchmark_models.global_batch(case), steps=1, one_device=True)

    silent = [leaf for leaf, values in gradient.items() if not np.any(values)]
    assert not silent, f"{len(silent)} of {len(gradient)} leaves get no gradient: {silent[:4]}"


# ---------------------------------------------------------------------------
# tools/benchmark_step.py
# ---------------------------------------------------------------------------

def composite_case(tool, architecture: str, **changes):
    """The cpu-smoke preset's case for one composite, in fp32 as the preset
    runs it, resized by `changes`."""
    (case,) = [c for c in tool.cpu_smoke_cases() if c.architecture == architecture]
    return dataclasses.replace(case, **{"dtype": "float32", **changes})


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
    source = tool.batches(case, trainer.device_mesh)
    state = jax.jit(trainer.initial_state)()
    before = jax.tree.map(np.asarray, named(state.variables))  # the step consumes the state
    compiled = trainer.compile(state, next(source))
    for _ in range(steps):
        state, loss, _, finite, _, _ = compiled(state, next(source))
    after = named(state.variables)
    assert bool(finite) and np.isfinite(float(loss))
    moved = {name: float(jnp.max(jnp.abs(value - before[name])))
             for name, value in after.items()}
    return moved, trainer.step_flops and trainer.step_flops.hardware


@pytest.mark.parametrize("architecture", ["sd3_transformer", "flux_transformer"])
def test_step_benchmark_native_diffusion_trains_text_and_pooled_conditioning(architecture):
    tool = load("benchmark_step")
    case = composite_case(tool, architecture)
    trainer = tool.build_trainer(case, "reference")
    encoder = trainer.objective.inputs.conditions["conditioning"].encoder
    encoded = encoder.encode(encoder.params, tool.global_batch(case)["text"])
    assert encoded.context.shape == (case.batch_size, tool.TEXT_TOKENS,
                                     trainer.objective.model.joint_attention_dim)
    assert encoded.pooled.shape == (case.batch_size, trainer.objective.model.pooled_projection_dim)
    if architecture == "flux_transformer":
        np.testing.assert_array_equal(encoded.guidance, np.full(case.batch_size, 3.5))
    moved, flops = parameter_movement(tool, case)
    assert flops > 0
    for projection in ("context_embedder", "text_embedder_linear_1"):
        changes = [value for path, value in moved.items() if projection in path]
        assert changes and min(changes) > 0, projection
    if architecture == "flux_transformer":
        changes = [value for path, value in moved.items() if "guidance_embedder_linear_1" in path]
        assert changes and min(changes) > 0


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


# ---------------------------------------------------------------------------
# tools/trace_window.py
# ---------------------------------------------------------------------------

def test_a_traced_window_splits_into_compute_exposed_collectives_and_idle():
    """One device's kernels, in microseconds: an all-reduce half hidden
    behind compute, a gap a batch's host-to-device copy ends, and a gap
    compute ends. benchmark_step and the reference runs both report these
    figures, and the window is their sum."""
    tool = load("trace_window")
    events = [("loop_add_fusion", 0, 10), ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 5, 20),
              ("MemcpyH2D", 25, 26), ("gemm_fusion_dot", 26, 30), ("loop_multiply_fusion", 40, 50)]
    split = tool.window_split([(name, start * 1000, end * 1000) for name, start, end in events])

    assert {key: value / 1000 for key, value in split.items()
            if key in ("window", "busy", "compute", "communication", "exposed_communication",
                       "idle_input", "idle_host", "AllReduce", "AllGather")} == {
        "window": 50, "busy": 35, "compute": 25, "communication": 15,
        "exposed_communication": 10, "idle_input": 5, "idle_host": 10, "AllReduce": 15,
        "AllGather": 0}
    assert split["window"] == (split["compute"] + split["exposed_communication"]
                               + split["idle_input"] + split["idle_host"])


# ---------------------------------------------------------------------------
# tools/reference_runs/scoreboard.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tools/benchmark_quantized_serving.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tools/check_distribution.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tools/torch_optim_reference.py
# ---------------------------------------------------------------------------

def test_torch_optim_fixture_is_what_the_generator_writes(tmp_path):
    """The stored gradients and initial parameters exactly; torch's schedule
    values and Adam steps within float32 rounding of a parameter of order 1."""
    pytest.importorskip("torch")
    load("torch_optim_reference").main(["--out", str(tmp_path)])
    committed = FIXTURES / "torch_optim"
    assert_fixture_files(tmp_path, committed, "torch_optim_reference")
    with np.load(committed / "reference.npz") as stored:
        computed = [name for name in stored.files if not name.startswith(("initial/", "grads/"))]
    assert_fixture_arrays(tmp_path / "reference.npz", committed / "reference.npz",
                          dict.fromkeys(computed, 1e-6))


# ---------------------------------------------------------------------------
# tools/lint_slop.py
# ---------------------------------------------------------------------------

def test_a_plugin_package_checks_its_own_source_with_dews_rules(tmp_path):
    """#46: `--root` and `--package` point the gate at another checkout, whose
    `src/<package>` gets the contract rules and whose own modules SLOP008
    will not see patched."""
    lint = load("lint_slop")
    (tmp_path / "src" / "sparx").mkdir(parents=True)
    (tmp_path / "src" / "sparx" / "graph.py").write_text(
        "def edges(table: dict[str, object]):\n    return table\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_graph.py").write_text(
        "def test_edges(monkeypatch):\n    monkeypatch.setattr('sparx.graph.edges', len)\n")
    modules = list(lint.collect(["src/sparx", "tests"], tmp_path, "sparx"))
    found = sorted((module.relative, finding.code) for module in modules for finding in lint.check(module))
    assert found == [("src/sparx/graph.py", "SLOP002"), ("tests/test_graph.py", "SLOP008")]
    assert lint.main(["--root", str(tmp_path), "--package", "sparx"]) == 1


def _linted(tmp_path, files: dict[str, str]) -> list[tuple[str, int, str]]:
    """Write `files` into a sparx checkout and return each finding as (path, line, code)."""
    lint = load("lint_slop")
    for relative, text in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    modules = list(lint.collect(["src/sparx", "tests", "tools"], tmp_path, "sparx"))
    return sorted((finding.path, finding.line, finding.code)
                  for module in modules for finding in lint.check(module))


def test_a_test_that_checks_wording_or_source_text_is_reported_and_one_that_checks_behaviour_is_not(tmp_path):
    """SLOP011: `match=` on pytest's raise, a read of a function's source and a
    test that names a source file each prove text, not behaviour, however the
    test imports them. A raise without `match`, another object's `raises`, a
    local `getsource`, and a library that names its own file, are not reported."""
    found = _linted(tmp_path, {
        "src/sparx/graph.py": 'HOME = "src/sparx/graph.py"\n',
        "tests/test_graph.py": (
            "import inspect\n"
            "import pytest\n"
            "from inspect import getsource as source\n"
            "from pytest import raises as refused\n"
            "from sparx import graph\n"
            "def test_wording():\n"
            "    with pytest.raises(ValueError, match='empty'):\n"
            "        graph.edges({})\n"
            "    with refused(ValueError, match='empty'):\n"
            "        graph.edges({})\n"
            "def test_source():\n"
            "    assert 'return' in inspect.getsource(graph) + source(graph)\n"
            "def test_file():\n"
            "    assert open('src/sparx/graph.py').read()\n"
            "def test_refusal(table):\n"
            "    def source():\n"
            "        return ''\n"
            "    with pytest.raises(ValueError), table.raises(ValueError, match='empty'):\n"
            "        graph.edges({source(): 1, table().inspect.getsource(graph): 2})\n"
            "def test_imported_here():\n"
            "    from pytest import raises as source\n"
            "    with source(ValueError, match='empty'):\n"
            "        graph.edges({})\n"
            "class TestReading:\n"
            "    from inspect import getsource as read\n"
            "    def test_default(self, text=read(graph)):\n"
            "        assert text\n"
            "def test_comprehension(readers):\n"
            "    import inspect\n"
            "    assert [inspect.getsource() for inspect in readers]\n"
            "def test_lambda_default(text=(lambda inspect: inspect.getsource())(graph)):\n"
            "    assert text\n"),
    })
    assert found == [("tests/test_graph.py", line, "SLOP011") for line in (7, 9, 12, 12, 14, 22, 26)]


def test_a_module_rebound_for_every_caller_is_reported_outside_the_suite(tmp_path):
    """SLOP012: a script that replaces or deletes the package's own functions,
    by any assignment or `setattr`, and a library that does so to anything it
    imported, a sibling module by a relative import included, change them for
    every caller. A script's rebinding of another library, an interpreter
    hook however it is spelled, and an attribute of an object a scope holds
    as its own name or reaches from a call are not reported; each scope
    reads its own imports, and a comprehension's names stay in it."""
    found = _linted(tmp_path, {
        "src/sparx/conv.py": "",
        "src/sparx/graph.py": (
            "import sys\n"
            "import sys as system\n"
            "import json\n"
            "from . import conv\n"
            "class Edges:\n"
            "    pass\n"
            "def install():\n"
            "    json.dumps = repr\n"
            "    conv.convolve = print\n"
            "    sys.excepthook = print\n"
            "    system.displayhook = print\n"
            "    setattr(sys, 'breakpointhook', print)\n"
            "def fill(json):\n"
            "    conv = Edges()\n"
            "    conv.count = 0\n"
            "    json.dumps = repr\n"
            "    Edges().json.dumps = repr\n"
            "def after_comprehension():\n"
            "    [conv for conv in (Edges(),)]\n"
            "    conv.convolve = repr\n"
            "def rebuilt():\n"
            "    import json\n"
            "    json = Edges()\n"
            "    json.count = 0\n"
            "def install_again(json):\n"
            "    import json\n"
            "    json.dumps = repr\n"),
        "tools/bench.py": (
            "import json\n"
            "from sparx import graph\n"
            "import sparx.graph\n"
            "import sparx.graph as sys\n"
            "graph.install = print\n"
            "setattr(sparx.graph, 'install', print)\n"
            "graph.install: object = print\n"
            "graph.install, graph.edges = print, repr\n"
            "del graph.edges\n"
            "sys.excepthook = print\n"
            "json.dumps = repr\n"
            "def fill(graph):\n"
            "    graph.count = 0\n"
            "def first_party():\n"
            "    from sparx import graph as engine\n"
            "    engine.run = repr\n"
            "def third_party():\n"
            "    import json as engine\n"
            "    engine.run = repr\n"
            "def defined():\n"
            "    class graph:\n"
            "        pass\n"
            "    graph.run = repr\n"),
    })
    assert found == [*[("src/sparx/graph.py", line, "SLOP012") for line in (8, 9, 20, 27)],
                     *[("tools/bench.py", line, "SLOP012") for line in (5, 6, 7, 8, 8, 9, 10, 16)]]


# ---------------------------------------------------------------------------
# tools/hf_coverage.py
# ---------------------------------------------------------------------------

def test_a_tutorials_outputs_are_stale_after_a_dependency_only_commit(tmp_path, monkeypatch):
    """The outputs come from the library and its pinned dependencies, so a
    commit that changes only pyproject.toml makes them stale, as one to a
    module the notebook imported does."""
    checker = load("check_tutorial_outputs")

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    (tmp_path / "src" / "dew").mkdir(parents=True)
    (tmp_path / "src" / "dew" / "probe.py").write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text('dependencies = ["jax==0.11.2"]\n')
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "outputs")
    recorded = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    notebook = tmp_path / "probe.ipynb"
    notebook.write_text(json.dumps({"metadata": {"dew": {"outputs": {"commit": recorded}}}, "cells": []}))
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    imports = {"probe.ipynb": ["dew.probe"]}
    assert checker.verdict(notebook, imports)[0] == "current"
    (tmp_path / "pyproject.toml").write_text('dependencies = ["jax==0.11.3"]\n')
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "dependency")
    label, lines = checker.verdict(notebook, imports)
    assert label == "STALE" and "pyproject.toml" in lines[0]


@pytest.mark.parametrize("stop", ["CellTimeoutError", "DeadKernelError"])
def test_a_tutorial_run_records_a_stalled_notebook_and_runs_the_rest(tmp_path, monkeypatch, stop):
    """A cell that times out or a kernel that dies fails its notebook, and the
    run goes on to the next one; the kernel spec a reused --workdir holds is
    written again for this interpreter."""
    import nbformat

    runner = load("run_tutorials")
    (tmp_path / "tutorials").mkdir()
    for name in ("01-stalls.ipynb", "02-runs.ipynb"):
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")]),
                       tmp_path / "tutorials" / name)
    workdir = tmp_path / "work"
    stale = workdir / ".jupyter" / "kernels" / "dew-tutorials"
    stale.mkdir(parents=True)
    (stale / "kernel.json").write_text(json.dumps({"argv": ["/gone/python", "-m", "ipykernel_launcher"],
                                                   "display_name": "old", "language": "python"}))
    ran = []

    class Client:
        def __init__(self, notebook, **options):
            self.notebook = notebook

        def execute(self):
            ran.append(len(ran))
            if len(ran) == 1:
                raise getattr(runner, stop)("stalled")
            probe = {"modules": ["dew"], "device": "cpu", "jax": "0"}
            self.notebook.cells[-1].outputs = [nbformat.v4.new_output(
                "stream", name="stdout", text=runner.PROBE_MARK + json.dumps(probe) + "\n")]

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "NotebookClient", Client)
    monkeypatch.setenv("JUPYTER_PATH", "")
    monkeypatch.setattr(sys, "argv", ["run_tutorials.py", "--workdir", str(workdir)])
    assert runner.main() == 1
    assert ran == [0, 1]
    assert json.loads((stale / "kernel.json").read_text())["argv"][0] == sys.executable
