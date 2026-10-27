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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, chain_roundings
from test_chunked_cross_entropy import EXERCISED, eighths
from test_moe import OUTPUT_TOLERANCE, ROUTER_TOLERANCE
from test_text_encoders import transformers_clip
from test_vae_16ch import TOLERANCE as VAE_TOLERANCE
from tool_support import FIXTURES, REPO_ROOT, load

from dew.telemetry.trace import window_split

CLAIMS = json.loads((FIXTURES / "provenance.json").read_text())["references"]

# The family parity contracts, read from the tests that hold Dew to these fixtures, not bounds fitted
# to a CI CPU: test_moe.py (routers and expert sums), test_vae_16ch.py (raw encode/decode). The CLIP
# towers' outputs are held as their own test holds them, to the reference's float64 run
# (`transformers_clip`).
MOE_OUTPUTS = {
    "mixtral.npz": {"router_weights": ROUTER_TOLERANCE, "block_output": OUTPUT_TOLERANCE},
    "deepseek.npz": {"router_weights": ROUTER_TOLERANCE, "block_output": OUTPUT_TOLERANCE},
    "deepseek_v2.npz": {"router_weights": ROUTER_TOLERANCE},
    "deepseek_v4.npz": {"router_weights": ROUTER_TOLERANCE, "experts_output": OUTPUT_TOLERANCE},
}
CLIP_OUTPUTS = ("last_hidden_state", "pooler_output", "text_embeds", "image_embeds")
VAE_OUTPUTS = {"latent": VAE_TOLERANCE, "decoded": VAE_TOLERANCE}


def assert_fixture_arrays(written: Path, committed: Path, numerical: dict[str, float],
                          truth: dict[str, np.ndarray] | None = None) -> None:
    """Only named arithmetic outputs use the family's fp32 parity bound, or, where `truth` gives
    their float64 values, are held as exact as the committed float32 output from them
    (tests/reference_error.py).

    Seeds do not promise cross-platform bit identity for arithmetic:
    https://docs.pytorch.org/docs/2.14/notes/randomness.html
    https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html
    Inputs (including floats), weights and discrete choices remain exact.
    These bounds do not guarantee every CPU/release; a failure requires
    investigation, never automatic fixture regeneration.
    """
    with np.load(written) as ours, np.load(committed) as theirs:
        assert set(ours.files) == set(theirs.files)
        truth = {} if truth is None else truth
        assert numerical.keys() <= set(theirs.files) and truth.keys() <= set(theirs.files)
        for name in theirs.files:
            actual, expected = ours[name], theirs[name]
            assert actual.shape == expected.shape, name
            assert actual.dtype == expected.dtype, name
            if name in truth:
                assert_as_exact_as_the_reference(actual.astype(np.float64), expected.astype(np.float64),
                                                 truth[name], name)
            elif name in numerical:
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
    """Exact checkpoint, tokenizer and pixels; both towers' outputs as exact as the committed ones,
    both from the reference's float64 run."""
    import torch

    load("clip_reference").write_tiny(tmp_path)

    committed = FIXTURES / "clip" / "tiny"
    assert_fixture_json(tmp_path, committed, "clip_reference")
    assert_same_tensors(tmp_path / "model.safetensors", committed / "model.safetensors")
    with np.load(committed / "reference.npz") as inputs:
        truth = transformers_clip(tmp_path, dict(inputs), torch.float64, CLIP_OUTPUTS)
    assert_fixture_arrays(tmp_path / "reference.npz", committed / "reference.npz", {}, truth)


# ---------------------------------------------------------------------------
# tools/t5_reference.py
# ---------------------------------------------------------------------------

def assert_generated_fixture(tool: str, written: Path, committed: Path, weights: str, outputs) -> None:
    """`tool`'s fixture written to `written` (its `main(["--out", ...])`) is the committed one: its
    JSON files, its `weights` checkpoint bit for bit, and its reference arrays at their tolerances."""
    load(tool).main(["--out", str(written.parent)])
    assert_fixture_json(written, committed, tool)
    assert_same_tensors(written / weights, committed / weights)
    assert_fixture_arrays(written / "reference.npz", committed / "reference.npz", outputs)


def test_t5_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, config, tokenizer and tokens; the encoder's parity is its recording's
    (tests/test_t5_encoders.py)."""
    assert_generated_fixture("t5_reference", tmp_path / "tiny", FIXTURES / "t5" / "tiny", "model.safetensors",
                             {})


# ---------------------------------------------------------------------------
# tools/vae_reference.py
# ---------------------------------------------------------------------------

def test_vae_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """Exact checkpoint, config and sample; fp32 encode/decode parity."""
    assert_generated_fixture("vae_reference", tmp_path / "sd3-tiny", FIXTURES / "vae" / "sd3-tiny",
                             "diffusion_pytorch_model.safetensors", VAE_OUTPUTS)


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

LINES = [f"line {i}: the quick brown fox jumps over the lazy dog" for i in range(60)]


def token_directory(tmp_path: Path) -> Path:
    """A byte-tokenized corpus, as the curve's documented producer writes it: `dew data export` over
    a text file's lines, each a document closed by EOS."""
    from dew.cli.main import DataExport

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"{line}\n" for line in LINES))
    out = tmp_path / "tokens"
    assert DataExport(str(corpus), str(out), tokenizer="byte", eos=True, read="memory").run_command() == 0
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
    # Every line's bytes, its newline and its EOS.
    assert curves["adamw"]["corpus_tokens"] == sum(len(line.encode()) + 2 for line in LINES)


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

LM_HEAD_SEEDS = 16
"""Check 1's declared seeds for each lm head variant against the baseline, sized paired (`seeds_for`
with the variant as `dew_fn`)."""


@pytest.mark.parametrize("states_dtype", ["float32", "bfloat16"])
def test_lm_head_variants_compute_the_same_loss_accuracy_and_gradients(states_dtype):
    """stored and remat are the baseline head rearranged into vocabulary
    tiles, so each takes the baseline's loss and both gradients on par over
    drawn states and tables (check 1), the baseline in float64 the truth, in
    the states' own precision: bf16 states take the bf16 head product in every
    variant. The vocabulary of 12 does not split into 4 equal tiles, so the
    short last tile is on the path. Over eighths (`eighths`), whose products
    sum exactly in any order, a tile's bf16 logits are the whole product's
    whatever order a device's GEMM of the tile's shape sums in, so every
    variant takes the baseline's top-1 accuracy exactly."""
    tool = load("benchmark_lm_head")
    dtype = jnp.float32 if states_dtype == "float32" else jnp.bfloat16
    targets = jnp.array([[0, 5, 11], [3, 11, 4]], jnp.int32)
    variant = tool.parse_variant("stored4-fp32" if states_dtype == "float32" else "stored4")

    def draw(seed):
        key = jax.random.PRNGKey(seed)
        return (jax.random.normal(key, (2, 3, 8), jnp.float32).astype(dtype),
                jax.random.normal(jax.random.fold_in(key, 1), (12, 8), jnp.float32))

    def taken(name):
        """The variant's loss, top-1 accuracy and gradients of the states and the table."""
        def run(states, table):
            with jax.default_matmul_precision(EXERCISED[dtype]):
                (loss, accuracy), (d_states, d_table) = jax.value_and_grad(
                    lambda s, t: tool.HEADS[name](s, t, targets, variant), argnums=(0, 1),
                    has_aux=True)(states, table)
            return {"loss": loss, "accuracy": accuracy, "states": d_states, "table": d_table}

        return jax.jit(run)

    # Each arm compiled once, so every draw runs its one program.
    arms = {name: taken(name) for name in ("baseline", "stored", "remat")}

    def compared(name):
        def run(states, table):
            return {key: value for key, value in arms[name](states, table).items() if key != "accuracy"}

        return run

    exact = eighths(*draw(0))
    for name in ("stored", "remat"):
        assert_as_exact(compared(name), compared("baseline"), draw, f"{states_dtype} {name} lm head",
                        seeds=LM_HEAD_SEEDS)
        assert arms[name](*exact)["accuracy"] == arms["baseline"](*exact)["accuracy"], name


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
        roundings = chain_roundings(jax.make_jaxpr(lambda variables: model.apply(variables, ids))(wide))
    tool.order_invariance({'logits': reordered}, {'logits': truth},
                          float(reordered.sum()), float(truth.sum()), roundings=roundings)


def test_residual_orders_preserve_a_decoder_whose_heads_are_the_residual_width():
    """A residual order permutes only the parameters that hold the residual stream: with 2 heads of
    16 in a 32-wide decoder, Kimi Delta Attention's dt_bias and short-convolution weights are the
    residual's width too, and permuting them, as a size-matching guess would, changes the function."""
    from model_support import TINY_DECODER
    from residual_orders import orders, permuted, residual_width

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import mixers

    mixer = mixers.build("kimi_delta_attention", linear_num_heads=2, linear_head_dim=16)
    model = CausalTransformer(**{**TINY_DECODER, "vocab_size": 16, "max_seq_len": 128}, mixer=mixer)
    ids = jnp.array([[2, 7, 11, 3, 5, 1, 9, 4]], jnp.int32)
    variables = model.init(jax.random.key(4), ids)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), variables)
        model = model.clone(dtype=jnp.float64)
        truth = np.asarray(model.apply(wide, ids))
        order = orders(residual_width(variables), 2, seed=6)[1]
        reordered = np.asarray(model.apply(permuted(wide, order), ids))
        roundings = chain_roundings(jax.make_jaxpr(lambda variables: model.apply(variables, ids))(wide))
    load('layout_parity').order_invariance({'logits': reordered}, {'logits': truth},
                                           float(reordered.sum()), float(truth.sum()), roundings=roundings)


def test_a_nonfinite_order_fails_the_invariance_check():
    """A residual order whose function or loss is not finite fails the check, as a NaN compares
    with nothing."""
    tool = load('layout_parity')
    for logits, loss in ((np.array([np.nan, 1.0]), 1.0), (np.array([1.0, 2.0]), np.nan)):
        with pytest.raises(ValueError):
            tool.order_invariance({'logits': logits}, {'logits': np.array([1.0, 2.0])}, loss, 1.0,
                                  roundings=1)


def test_residual_orders_preserve_a_looped_ngpt():
    """A residual order is an exact symmetry of Dory's looped nGPT decoder: its per-feature step rates
    (`attn_alpha`, `mlp_alpha`) live on the residual stream and are reordered with it, as its norms'
    scales are; left in place, the reordered model computes another function (the 4B release's order 1
    sat at an RMS of 10 from vLLM's log probabilities)."""
    from residual_orders import orders, permuted, residual_width

    from dew.interop import Pretrained

    loaded = Pretrained.load(FIXTURES / 'hf/dory-tiny', dtype='float32', attention_impl='reference')
    ids = jnp.array([[2, 7, 11, 3, 5, 1, 9, 4]], jnp.int32)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), loaded.variables)
        model = loaded.model.clone(dtype=jnp.float64)
        truth = np.asarray(model.apply(wide, ids))
        order = orders(residual_width(loaded.variables), 2, seed=6)[1]
        reordered = np.asarray(model.apply(permuted(wide, order), ids))
        roundings = chain_roundings(jax.make_jaxpr(lambda variables: model.apply(variables, ids))(wide))
    load('layout_parity').order_invariance({'logits': reordered}, {'logits': truth},
                                           float(reordered.sum()), float(truth.sum()), roundings=roundings)


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

def owners(monkeypatch):
    """tools/benchmark_cases.py and tools/benchmark_models.py, the cases and the trainer, batches and
    compiled step the step benchmark measures (benchmark_models imports benchmark_cases by name)."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    import benchmark_cases
    import benchmark_models
    return benchmark_cases, benchmark_models


def composite_case(cases, architecture: str, **changes):
    """The cpu-smoke preset's case for one composite, in fp32 as the preset
    runs it, resized by `changes`."""
    (case,) = [c for c in cases.cpu_smoke_cases() if c.architecture == architecture]
    return dataclasses.replace(case, **{"dtype": "float32", **changes})


def parameter_movement(models, case, steps: int = 2):
    """How far every parameter moved over `steps` of the case's real compiled
    step, by tree path, with the loss and the step's measured FLOPs.

    The trainer, the batches and the compiled step are the benchmark's own: a
    parameter that does not move here is one the measured step never trains.
    """
    def named(tree):
        return {jax.tree_util.keystr(path): leaf
                for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}

    trainer = models.build_trainer(case, "reference")
    with models.compiled_step(case, trainer) as compiled:
        before = jax.tree.map(np.asarray, named(compiled.state.variables))  # the step consumes the state
        state, loss, finite, _ = models.timed_steps(compiled.step, compiled.state, steps)
    after = named(state.variables)
    assert bool(finite) and np.isfinite(float(loss))
    moved = {name: float(jnp.max(jnp.abs(value - before[name])))
             for name, value in after.items()}
    return moved, trainer.step_flops and trainer.step_flops.hardware


@pytest.mark.parametrize("architecture", ["sd3_transformer", "flux_transformer"])
def test_step_benchmark_native_diffusion_trains_text_and_pooled_conditioning(architecture, monkeypatch):
    cases, models = owners(monkeypatch)
    case = composite_case(cases, architecture)
    trainer = models.build_trainer(case, "reference")
    encoder = trainer.objective.inputs.conditions["conditioning"].encoder
    encoded = encoder.encode(encoder.params, models.global_batch(case)["text"])
    assert encoded.context.shape == (case.batch_size, cases.TEXT_TOKENS,
                                     trainer.objective.model.joint_attention_dim)
    assert encoded.pooled.shape == (case.batch_size, trainer.objective.model.pooled_projection_dim)
    if architecture == "flux_transformer":
        np.testing.assert_array_equal(encoded.guidance, np.full(case.batch_size, 3.5))
    moved, flops = parameter_movement(models, case)
    assert flops > 0
    for projection in ("context_embedder", "text_embedder_linear_1"):
        changes = [value for path, value in moved.items() if projection in path]
        assert changes and min(changes) > 0, projection
    if architecture == "flux_transformer":
        changes = [value for path, value in moved.items() if "guidance_embedder_linear_1" in path]
        assert changes and min(changes) > 0


def test_step_benchmark_media_step_trains_the_image_tower(monkeypatch):
    """The measured step is the whole conditioned model's: the tower and the
    projector move under it, and a second image a row raises the FLOPs the
    row reports. A step that fed no pixels would leave both untouched."""
    cases, models = owners(monkeypatch)
    case = composite_case(cases, "multimodal_transformer")

    moved, flops = parameter_movement(models, case)

    media = {name: value for name, value in moved.items()
             if "tower" in name or "projector" in name}
    assert media and min(media.values()) > 0
    assert min(value for name, value in moved.items() if "language_model" in name) > 0
    _, wider = parameter_movement(
        models, dataclasses.replace(case, media={**case.media, "images": 2}), steps=1)
    assert wider > flops


def test_step_benchmark_canvas_step_trains_the_self_conditioning_decoder(monkeypatch):
    """The measured step is the official fine-tuning step: the
    self-conditioning MLP the second decoder pass feeds moves under it, and a
    wider response raises the FLOPs the row reports. A plain next-token step
    on the same trunk has no such parameter to move."""
    cases, models = owners(monkeypatch)
    narrow = composite_case(cases, "diffusion_gemma",
                            canvas={"prompt_length": 12, "canvas_size": 4})

    moved, one_canvas = parameter_movement(models, narrow)

    conditioning = {name: value for name, value in moved.items()
                    if "self_conditioning" in name}
    assert conditioning and min(conditioning.values()) > 0
    assert min(value for name, value in moved.items() if "text" in name) > 0
    _, two_canvases = parameter_movement(
        models, composite_case(cases, "diffusion_gemma"), steps=1)
    assert two_canvases > one_canvas



def test_a_canvas_case_trains_with_the_head_chunks_and_objective_keywords_it_names(monkeypatch):
    """A canvas case's head chunking and objective keywords reach the block objective its measured
    step trains, as a decoder case's reach its objective: with the canvas loss weighed at nothing,
    the self-conditioning MLP only the canvas pass feeds stays where it was while the trunk the
    encoder loss trains moves, over the head in the one chunk the case names (chunking changes no
    number, so that the step trains with it is what is asked). A case that names another decoder
    objective is refused: the canvas case would benchmark the block objective under its name."""
    cases, models = owners(monkeypatch)
    case = dataclasses.replace(
        composite_case(cases, "diffusion_gemma", canvas={"prompt_length": 12, "canvas_size": 4}),
        head_chunks=1, objective={"decoder_loss_weight": 0.0})

    moved, _ = parameter_movement(models, case, steps=1)

    conditioning = [value for name, value in moved.items() if "self_conditioning" in name]
    assert conditioning and max(conditioning) == 0
    assert max(value for name, value in moved.items() if "text" in name) > 0
    with pytest.raises(ValueError):
        models.build_objective(dataclasses.replace(case, decoder_objective="dpo"))


# ---------------------------------------------------------------------------
# tools/trace_window.py
# ---------------------------------------------------------------------------

def test_trace_reports_keep_global_busy_time_separate_from_per_device_averages():
    tool = load("benchmark_step")
    common = load("reference_runs/common")
    rows = [
        (0, "gemm_fusion_dot", 0, 10, None),
        (0, "ncclDevKernel_SendRecv", 5, 20, "all_to_all.124.1"),
        (0, "MemcpyH2D", 25, 26, None),
        (0, "gemm_fusion_dot", 26, 30, None),
        (0, "ncclDevKernel_SendRecv", 40, 50, "collective-permute.3"),
        (1, "loop_add_fusion", 100, 110, None),
        (1, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 105, 120, "all-reduce.3"),
        (1, "loop_add_fusion", 125, 130, None),
    ]
    planes = {f"/device:GPU:{device}": [
        SimpleNamespace(name=name, start_ns=start * 1_000_000, end_ns=end * 1_000_000,
                        stats=[] if op is None else [("hlo_op", op)])
        for owner, name, start, end, op in reversed(rows) if owner == device] for device in (0, 1, 2)}
    kernels = [(name, start * 1_000_000, end * 1_000_000, device)
               for device, name, start, end, _ in rows]

    timeline = tool.device_timeline("unused", 2, planes=planes)
    reference = common.kernel_summary(kernels, 2)
    communication = tool.communication("unused", 2, planes=planes)

    assert timeline["device_busy_ms_per_step"] == pytest.approx(30)
    assert timeline["device_window_ms_per_step"] == pytest.approx(65)
    assert timeline["device_busy_percent"] == pytest.approx(100 * 60 / 130)
    assert timeline["kernels_per_step"] == 4
    assert timeline["kernel_ms_by_category"] == pytest.approx(
        {"collective": 20, "elementwise": 7.5, "gemm": 7, "other": 0.5})
    assert dict(timeline["top_kernels"])["gemm_fusion_dot"] == pytest.approx(7)
    assert reference["device_busy_ms_per_step"] == pytest.approx(15)
    assert reference["device_window_ms_per_step"] == pytest.approx(20)
    assert reference["device_busy_percent"] == pytest.approx(75)
    assert reference["kernels_per_step"] == 2
    assert reference["kernel_ms_by_category"] == pytest.approx(
        {"collective": 10, "elementwise": 3.75, "gemm": 3.5, "other": 0.25})
    gemm = next(row for row in reference["top_kernels"] if row["name"] == "gemm_fusion_dot")
    assert (gemm["ms_per_step"], gemm["calls_per_step"]) == pytest.approx((3.5, 0.5))
    assert communication["traced_devices"] == reference["devices"] == 2
    assert communication["window_ms_per_step"] == pytest.approx(20)
    for key, expected in {"compute": 7.5, "communication": 10, "exposed_communication": 7.5,
                          "idle_input": 1.25, "idle_host": 3.75}.items():
        assert (communication[f"{key}_ms_per_step"]
                == reference[f"{key}_ms_per_step"] == pytest.approx(expected))
    assert communication["all_to_all_ms_per_step"] == pytest.approx(3.75)
    assert communication["all_to_all_exposed_ms_per_step"] == pytest.approx(2.5)
    assert communication["collective_permute_ms_per_step"] == pytest.approx(2.5)
    assert communication["collective_permute_exposed_ms_per_step"] == pytest.approx(2.5)
    assert communication["all_reduce_exposed_ms_per_step"] == pytest.approx(2.5)
    assert communication["exposed_communication_percent"] == pytest.approx(37.5)


@pytest.mark.parametrize("scale", [1_000, 0.125])
def test_kernel_sums_preserve_decode_categories_counts_and_overlapping_streams(scale):
    tool = load("trace_window")
    rows = [("cudnn_conv", 0, 30), ("flash_gemm", 5, 15), ("cutlass_scatter", 30, 40),
            ("argmax", 40, 50), ("philox", 60, 70), ("unclassified", 70, 80),
            ("unclassified", 85, 90), ("ncclDevKernel_SendRecv", 90, 95)]
    events = [(name, start * scale, end * scale) for name, start, end in reversed(rows)]
    durations, counts = tool.kernel_totals(events)
    classes, class_counts = tool.kernel_totals(events, tool.decode_category)
    training, _ = tool.kernel_totals(events, tool.kernel_category)

    assert {name: duration / scale for name, duration in durations.items()} == pytest.approx({
        "cudnn_conv": 30, "flash_gemm": 10, "cutlass_scatter": 10, "argmax": 10,
        "philox": 10, "unclassified": 15, "ncclDevKernel_SendRecv": 5})
    assert counts == {name: 2 if name == "unclassified" else 1 for name in durations}
    assert {kind: duration / scale for kind, duration in classes.items()} == pytest.approx(
        {"attention": 40, "gemm": 10, "reduction": 10, "random": 10, "other": 20})
    assert class_counts == {"attention": 2, "gemm": 1, "reduction": 1, "random": 1, "other": 3}
    assert {kind: duration / scale for kind, duration in training.items()} == pytest.approx(
        {"conv": 30, "attention": 10, "gemm": 10, "other": 35, "collective": 5})
    split = window_split(events)
    assert split["busy"] / scale == pytest.approx(80)
    assert split["window"] / scale == pytest.approx(95)
    assert sum(durations.values()) == sum(classes.values()) == sum(training.values())
    assert sum(durations.values()) > split["busy"]


# ---------------------------------------------------------------------------
# tools/reference_runs/scoreboard.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tools/benchmark_host_offload.py
# ---------------------------------------------------------------------------

def test_the_host_offload_benchmark_counts_each_parameter_in_the_memory_space_it_was_compiled_for():
    """A program compiled with one argument in pinned host memory and one on the device counts one in each,
    as its compiled input shardings name them."""
    tool = load("benchmark_host_offload")
    device = jax.devices()[0]
    host = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
    resident = jax.sharding.SingleDeviceSharding(device, memory_kind="device")
    compiled = jax.jit(lambda banked, x: x + 1.0, in_shardings=(host, resident), keep_unused=True).lower(
        jax.ShapeDtypeStruct((4,), jnp.float32), jax.ShapeDtypeStruct((4,), jnp.float32)).compile()
    assert tool.entry_spaces(compiled) == {"pinned_host": 1, "device": 1}


# ---------------------------------------------------------------------------
# tools/benchmark_quantized_serving.py
# ---------------------------------------------------------------------------

def test_the_quantized_serving_benchmark_samples_through_the_task_and_reads_its_real_rows():
    """The benchmark's samples are the task's own walk and decode, read back as their real rows: a
    pixel-space model's too, which has no autoencoder, and its twelve prompts once on eight devices,
    which the walk pads to sixteen rows, so quality() scores twelve images against twelve prompts."""
    from flax import linen as nn

    from dew.diffusion import DirectPredictionTransform, FlowMatchingScheduler, Process
    from dew.inference import TextToImage
    from dew.inputs import Field, InputSpec
    from dew.sampling import Euler

    class Zero(nn.Module):
        def __call__(self, x, time, train=False):
            return jnp.zeros_like(x)

    tool = load("benchmark_quantized_serving")
    pipe = TextToImage(Zero(), Process(FlowMatchingScheduler(), DirectPredictionTransform()),
                       InputSpec(Field("image", (2, 2, 1))), {}, steps=1, guidance=None, solver=Euler())
    pixels, found = tool.sample(pipe, 0)
    assert pixels.shape == (len(tool.PROMPTS), 2, 2, 1) and pixels.dtype == np.uint8
    assert found == {"latents": 0, "pixels": 0}


# ---------------------------------------------------------------------------
# tools/check_distribution.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tools/torch_optim_reference.py
# ---------------------------------------------------------------------------

def test_torch_optim_fixture_is_what_the_generator_writes(tmp_path):
    """The stored gradients and initial parameters exactly; torch's schedule
    values and Adam steps within float32 rounding of a parameter of order 1. PyTorch, the generator's
    reference, is the suite's own dependency: a missing one fails here."""
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


def test_the_script_reads_its_files_in_processes_and_reports_as_one_process_does(tmp_path, capsys):
    """tools/lint_slop.py run as a script reads its files in a pool of processes, one a processor; its
    findings, their order and its counts are those main reports reading them in its caller's process."""
    lint = load("lint_slop")
    (tmp_path / "src" / "sparx").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    for index in range(40):
        (tmp_path / "src" / "sparx" / f"graph_{index}.py").write_text(
            f"def edges_{index}(table: dict[str, object]):\n    return table\n")
        patched = f"sparx.graph_{index}.edges_{index}"
        (tmp_path / "tests" / f"test_graph_{index}.py").write_text(
            f"def test_edges(monkeypatch):\n    monkeypatch.setattr('{patched}', len)\n")
    arguments = ["--root", str(tmp_path), "--package", "sparx"]
    assert lint.main(arguments) == 1
    serial = capsys.readouterr()
    script = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "lint_slop.py"), *arguments],
                            capture_output=True, text=True, timeout=300)
    assert script.returncode == 1
    assert (script.stdout, script.stderr) == (serial.out, serial.err)
    # One finding a file, eighty files.
    assert len(serial.out.splitlines()) == 80


def _linted(tmp_path, files: dict[str, str]) -> list[tuple[str, int, str]]:
    """Write `files` into a sparx checkout and return each finding as (path, line, code)."""
    lint = load("lint_slop")
    for relative, text in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    modules = list(lint.collect(["src/sparx", "tests", "tools"], tmp_path, "sparx"))
    return sorted((finding.path, finding.line, finding.code)
                  for module in modules for finding in lint.check(module))


def assert_reported(tmp_path, path: str, source: str, code: str, lines: tuple[int, ...]) -> None:
    """`source`, written at `path` in a sparx checkout, is reported as `code` at `lines` and nowhere else."""
    assert _linted(tmp_path, {path: source}) == [(path, line, code) for line in lines]


def test_the_gate_reads_the_site_the_docs_generators_and_the_kernel_packages_but_not_installed_npm(tmp_path):
    """With no roots named, the gate reads the site's scripts, the docs' figure generators and the
    kernel packages beside the package's source and the scripts, and none of the Python an npm
    install leaves under the site."""
    lint = load("lint_slop")
    roots = lint.default_roots("dew")
    found = {module.relative for module in lint.collect(roots, REPO_ROOT, "dew")}
    assert {"site/scripts/gen_api.py", "docs/assets/figures.py",
            "kernels/flash_attn/dew_flash_attn/__init__.py"} <= found
    for relative in ("site/scripts/build.py", "site/node_modules/node-gyp/gyp/pylib/gyp/common.py"):
        (tmp_path / relative).parent.mkdir(parents=True)
        (tmp_path / relative).write_text("WIDTH = 1  # noqa: E501\n")
    assert [module.relative for module in lint.collect(roots, tmp_path, "dew")] == ["site/scripts/build.py"]


def test_the_gate_skips_the_hero_script_while_its_bytes_are_the_recorded_ones(tmp_path):
    """The landing hero's script is recorded data: the hero capture pins its bytes (script_sha256),
    so the gate leaves it alone while they are the recorded ones and reads it once they are not, as
    an edit the hero is then recorded again for."""
    lint = load("lint_slop")
    hero, capture = "site/src/data/hero.py", "site/src/data/capture.json"
    assert {hero: capture} == lint.RECORDED
    recorded = (REPO_ROOT / hero).read_bytes()
    (tmp_path / hero).parent.mkdir(parents=True)
    (tmp_path / capture).write_bytes((REPO_ROOT / capture).read_bytes())
    (tmp_path / hero).write_bytes(recorded)
    assert [module.relative for module in lint.collect(["site"], tmp_path, "dew")] == []
    (tmp_path / hero).write_bytes(recorded + b"\n")
    assert [module.relative for module in lint.collect(["site"], tmp_path, "dew")] == [hero]


def test_a_suppression_is_reported_in_every_file_and_a_string_or_function_that_spells_one_is_not(tmp_path):
    """SLOP003 runs in the suite and the tools as in the source: an ignore comment,
    `typing.cast` however it was imported, and a warning filter whose action is
    "ignore" are each reported. A string that spells a comment, a function of the
    file's own named `cast`, a filter that makes warnings errors, and an import kept
    for the registry entry it makes are not."""
    found = _linted(tmp_path, {
        "src/sparx/graph.py": "EDGES = 1\n",
        "tests/test_graph.py": (
            "import warnings\n"
            "from typing import cast as as_type\n"
            "import pytest\n"
            "import sparx.plugins  # noqa: F401  (registers the kind)\n"
            "WIDTH = 1  # noqa: E501\n"
            "SPELLED = 'x = 1  # noqa: E501'\n"
            "def cast(value):\n"
            "    return value\n"
            "@pytest.mark.filterwarnings('ignore::DeprecationWarning')\n"
            "def test_quiet():\n"
            "    warnings.simplefilter('ignore')\n"
            "    with warnings.catch_warnings(action='ignore'):\n"
            "        assert as_type(int, cast(1))\n"
            "@pytest.mark.filterwarnings('error')\n"
            "def test_loud():\n"
            "    warnings.filterwarnings(action='error')\n"
            "@pytest.mark.filterwarnings('error', 'ignore::DeprecationWarning')\n"
            "def test_quiet_after_all():\n"
            "    pass\n"),
        "tools/report.py": ("from typing_extensions import cast\n"
                            "COUNT: int = '1'  # type: ignore[assignment]\n"
                            "WIDTH = cast(int, COUNT)\n"),
    })
    assert found == [*(("tests/test_graph.py", line, "SLOP003") for line in (5, 9, 11, 12, 13, 17)),
                     ("tools/report.py", 2, "SLOP003"), ("tools/report.py", 3, "SLOP003")]


def test_a_first_party_patch_is_reported_however_its_owner_was_imported_and_a_boundary_is_not(tmp_path):
    """SLOP008: a first-party module's member or table replaced in a test is reported through an
    alias, a from-import or a local import, by setattr, delattr, setitem or delitem, and a member
    named like a seam (`narrowed_paths`) is none. The clock, a Hub snapshot and a subprocess are
    boundaries a test may replace."""
    found = _linted(tmp_path, {
        "src/sparx/graph.py": "EDGES = {}\n",
        "tests/test_graph.py": (
            "import time\n"
            "import sparx.graph as g\n"
            "from sparx import graph\n"
            "def test_alias(monkeypatch):\n"
            "    monkeypatch.setattr(g, 'narrowed_paths', dict)\n"
            "def test_table(monkeypatch):\n"
            "    monkeypatch.setitem(graph.EDGES, 'edge', 1)\n"
            "    monkeypatch.delitem(g.EDGES, 'edge')\n"
            "def test_local(monkeypatch):\n"
            "    from sparx.graph import EDGES as edges\n"
            "    monkeypatch.setitem(edges, 'edge', 1)\n"
            "    monkeypatch.delattr(graph, 'narrowed_paths')\n"
            "def test_boundaries(monkeypatch):\n"
            "    monkeypatch.setattr(time, 'sleep', lambda seconds: None)\n"
            "    monkeypatch.setattr(graph, 'snapshot', lambda: None)\n"
            "    monkeypatch.setattr('sparx.graph.subprocess.run', print)\n"),
    })
    assert found == [("tests/test_graph.py", line, "SLOP008") for line in (5, 7, 8, 11, 12)]


ROW_1878_LINTS = {
    "SLOP008": ({
        "src/sparx/store.py": "class Store:\n    def _open(self):\n        pass\ndef open(path):\n    pass\n",
        "tests/test_store.py": (
            "from sparx.store import Store\n"
            "import sparx.store as store\n"
            "def test_private(monkeypatch):\n"
            "    monkeypatch.setattr(Store, '_open', lambda self: None)\n"
            "def test_public(monkeypatch):\n"
            "    monkeypatch.setattr(store, 'open', lambda path: None)\n"
            "def test_a_seams_own_private(monkeypatch):\n"
            "    monkeypatch.setattr('sparx.store.os._exit', lambda code: None)\n"
            "def test_a_first_party_private_under_a_seam_word(monkeypatch):\n"
            "    monkeypatch.setattr('sparx.sources.hub.Hub._selected', lambda self: None)\n"),
    }, "tests/test_store.py", (4, 10)),
    "SLOP011": ({
        "src/sparx/graph.py": "EDGES = 1\n",
        "tests/test_graph.py": (
            "def test_table(table):\n"
            "    assert report.table().splitlines() == ['   a  b', 'x  1  2']\n"
            "    assert table.plain == 'data 4,824 rows  ·  0 rejected'\n"
            "    assert '4,824' in table.split()\n"
            "    assert table.title == 'data rows'\n"
            "    assert table.index('b') + 1 == len('   a  b')\n"
            "    assert json.loads(payload) == ['a  b']\n"
            "    assert parsed.name == 'two  spaces'\n"),
    }, "tests/test_graph.py", (2, 3)),
}
"""Row 1878's two lint cases, by code: the files linted, the file the findings are in, their lines.
SLOP008: a private attribute of a first-party object is the code under test, so replacing it is
reported even where its name holds a seam's (`Checkpoints._open`); a public seam of the same name and a
standard-library module's own private attribute are not. SLOP011: a rendering (its text read through
`plain`, `table()`, captured output or `str`) compared whole with a string, or a list of strings, with a
run of two or more spaces between words is a spacing snapshot; its words and numbers looked for, a
one-space string, a column end measured and parsed data holding two spaces are not."""


@pytest.mark.parametrize("code", sorted(ROW_1878_LINTS))
def test_row_1878s_lints_report_their_cases_and_spare_the_rest(code, tmp_path):
    files, where, lines = ROW_1878_LINTS[code]
    assert _linted(tmp_path, files) == [(where, line, code) for line in lines]


def test_a_test_that_skips_on_its_own_packages_import_is_reported(tmp_path):
    """SLOP015: a test that reaches its own package through `pytest.importorskip`,
    however it imports the skip, turns an import the package breaks into a skip; an
    optional dependency's skip, or a package that only shares the prefix, is not reported."""
    found = _linted(tmp_path, {
        "src/sparx/graph.py": "EDGES = 1\n",
        "tests/test_graph.py": ("import pytest\n"
                                "from pytest import importorskip\n"
                                "graph = pytest.importorskip('sparx.graph')\n"
                                "sparx = importorskip('sparx')\n"
                                "torch = pytest.importorskip('torch')\n"
                                "near = pytest.importorskip('sparxish')\n"
                                "named = pytest.importorskip(modname='sparx.graph')\n"
                                "optional = pytest.importorskip(modname='torch')\n"),
    })
    assert found == [("tests/test_graph.py", line, "SLOP015") for line in (3, 4, 7)]


def test_a_choice_keyed_on_the_device_generation_is_reported_outside_the_target_table(tmp_path):
    """SLOP016: the device's generation, `device_generation()` or a `gpu_target()`'s `.generation`,
    however imported, and through a name bound to it in the function, its module or a class body,
    compared with or prefix-matched against one the source spells, matched against a spelled case,
    cut up, or used as a table's key or attribute name, is reported outside nn/kernels/generation.py.
    A value is read before its name takes it, and a comprehension's first iterable before its targets.
    Comparing it with a recorded generation, reading a fact off the target, passing it to a function
    (a `getattr` of the caller's among them), a name rebound to something else or shadowed by a
    parameter or a comprehension (a lambda's in it too), and a class attribute, which no method reads
    as a name, are not."""
    found = _linted(tmp_path, {
        "src/sparx/nn/kernels/generation.py": (
            "def device_generation():\n"
            "    return 'sm80'\n"
            "def gpu_target(name=None):\n"
            "    return name\n"
            "def wide():\n"
            "    return device_generation().startswith('sm')\n"),
        "src/sparx/tiles.py": (
            "from sparx.nn.kernels import generation as hardware\n"
            "from sparx.nn.kernels.generation import device_generation, gpu_target\n"
            "TILES = {'sm80': 4096}\n"
            "def tile():\n"
            "    return TILES.get(device_generation(), 1024)\n"
            "def held():\n"
            "    target = hardware.gpu_target()\n"
            "    return 0 if target.generation == 'sm90' else 1\n"
            "def narrow():\n"
            "    named = device_generation()\n"
            "    return named.startswith('sm8') or int(named[2:]) > 89 or named in ('sm86', 'sm89')\n"
            "def measured(on):\n"
            "    named = device_generation()\n"
            "    return named != on and gpu_target().shared_limit and gpu_target(named) and print(named)\n"
            "GENERATION = device_generation()\n"
            "def later():\n"
            "    return lambda: GENERATION == 'sm90' or getattr(TILES, GENERATION)\n"
            "class Launch:\n"
            "    PICK = TILES[device_generation()]\n"
            "def matched():\n"
            "    named: str = device_generation()\n"
            "    match named:\n"
            "        case 'sm90' | 'sm100':\n"
            "            return 0\n"
            "    return 1\n"
            "def relabeled(record):\n"
            "    named = device_generation()\n"
            "    named = record.label\n"
            "    return named == 'label' or [named == 'sm80' for named in record.names]\n"
            "def shadowed(GENERATION):\n"
            "    return GENERATION == 'sm80'\n"
            "def listed(record):\n"
            "    named = device_generation()\n"
            "    return [named == 'sm80' for named in record.names] and named == 'sm90'\n"
            "def looked_up(TILES):\n"
            "    named = device_generation()\n"
            "    named = TILES[named]\n"
            "    return named\n"
            "def iterated(TILES):\n"
            "    named = device_generation()\n"
            "    return [named for named in TILES[named]]\n"
            "class Scoped:\n"
            "    LABEL = device_generation()\n"
            "    def method(self):\n"
            "        return LABEL == 'label'\n"
            "def deferred(records):\n"
            "    named = device_generation()\n"
            "    return [lambda: named == 'label' for named in records]\n"
            "def passed(getattr, TILES):\n"
            "    return getattr(TILES, device_generation())\n"
            "def copied(TILES):\n"
            "    named = device_generation()\n"
            "    named: str = named\n"
            "    return TILES[named]\n"
            "def relabeled_inside():\n"
            "    named = device_generation()\n"
            "    [(named := label) for label in ('label',)]\n"
            "    held = [(found := device_generation()) for _ in ()]\n"
            "    return named == 'label' or held and found == 'sm80'\n"
            "def later_or_inner(TILES):\n"
            "    named = device_generation()\n"
            "    deferred = ((named := label) for label in ('label',))\n"
            "    inner = [lambda: (named := 'label') for _ in ('x',)]\n"
            "    return TILES[named], deferred, inner\n"),
    })
    assert found == [("src/sparx/tiles.py", line, "SLOP016")
                     for line in (5, 8, 11, 11, 11, 17, 17, 19, 22, 34, 37, 41, 54, 59, 64)]


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
            "    assert text\n"
            "class TestClassComprehension:\n"
            "    from inspect import getsource as read\n"
            "    texts = [read(graph) for _ in range(1)]\n"),
    })
    assert found == [("tests/test_graph.py", line, "SLOP011") for line in (7, 9, 12, 12, 14, 22, 26)]


def test_a_sentence_looked_for_in_captured_output_is_reported_and_a_parsed_report_is_not(tmp_path):
    """SLOP011: a sentence the test spells, looked for in the output pytest captured, however the test
    holds it (`capsys.readouterr()`'s `.out`, a name for the capture or its `.out`, the pair unpacked,
    added to), proves wording. A name bound again to a parse of the output, the same name in a test
    that captured nothing or only in a function it defines, and a mapping an item of the output is
    put in, hold data, and are not reported."""
    assert_reported(tmp_path, "tests/test_report.py", ("import json\n"
        "def test_direct(capsys):\n"
        "    assert 'saved' in capsys.readouterr().out\n"
        "def test_held(capsys):\n"
        "    captured = capsys.readouterr()\n"
        "    assert 'saved' in captured.out\n"
        "def test_unpacked(capsys):\n"
        "    out, err = capsys.readouterr()\n"
        "    assert 'saved' in out\n"
        "def test_added(capsys):\n"
        "    text = capsys.readouterr().out\n"
        "    text += capsys.readouterr().err\n"
        "    assert text.count('saved')\n"
        "def test_parsed(capsys):\n"
        "    report = capsys.readouterr().out\n"
        "    report = json.loads(report)\n"
        "    assert 'tokens' in report\n"
        "def test_unrelated():\n"
        "    out = {'tokens': 8}\n"
        "    assert 'tokens' in out\n"
        "def test_nested(capsys):\n"
        "    report = {'tokens': 8}\n"
        "    def later():\n"
        "        report = capsys.readouterr().out\n"
        "    assert 'tokens' in report\n"
        "def test_item(capsys):\n"
        "    report = {'tokens': 8}\n"
        "    report['err'] = capsys.readouterr().err\n"
        "    assert 'tokens' in report\n"), "SLOP011", (3, 6, 9, 13))


def test_an_errors_or_a_processs_wording_is_reported_and_its_parsed_output_is_not(tmp_path):
    """SLOP011: a sentence the test spells, looked for in a raised error's text (`str` or `repr` of
    an `except` name or of `pytest.raises`'s value, imported under another name too) or in a
    process's `stdout` or `stderr`, by `in`, `==`, `startswith`, `endswith` or `count`, proves
    wording. A refusal's type, a process's exit code, a value the process prints and the test
    parses, and an error's own attribute hold behaviour, as does a process's whole output compared
    with the value it computes, and are not reported."""
    assert_reported(tmp_path, "tests/test_refusal.py", ("import json\n"
        "import subprocess\n"
        "import pytest\n"
        "def test_caught():\n"
        "    try:\n"
        "        int('x')\n"
        "    except ValueError as error:\n"
        "        assert 'invalid literal' in str(error)\n"
        "def test_raised():\n"
        "    with pytest.raises(ValueError) as raised:\n"
        "        int('x')\n"
        "    assert str(raised.value).startswith('invalid')\n"
        "    assert repr(raised.value) == 'ValueError()'\n"
        "def test_aliased():\n"
        "    from pytest import raises as refused\n"
        "    with refused(ValueError) as caught:\n"
        "        int('x')\n"
        "    assert str(caught.value).startswith('invalid')\n"
        "def test_process():\n"
        "    done = subprocess.run(['false'], capture_output=True, text=True)\n"
        "    assert 'refused' in done.stderr\n"
        "    assert done.stdout.count('saved')\n"
        "def test_behaviour():\n"
        "    with pytest.raises(ValueError) as raised:\n"
        "        int('x')\n"
        "    assert raised.type is ValueError\n"
        "    done = subprocess.run(['true'], capture_output=True, text=True)\n"
        "    assert done.returncode == 0\n"
        "    assert json.loads(done.stdout) == {'refused': True}\n"
        "    assert done.stdout.split()[-1] == 'Scalar'\n"
        "    assert done.stdout == '42\\n'\n"
        "    try:\n"
        "        int('x')\n"
        "    except ValueError as error:\n"
        "        assert error.args\n"), "SLOP011", (8, 12, 13, 18, 21, 22))


def test_a_sleep_a_poll_or_a_retry_is_reported_where_nothing_leaves_the_process(tmp_path):
    """SLOP018: in the package's source, a sleep, a timed wait round a loop (a poll) and a retry of
    one piece of work are reported where the function deals with no subprocess, network, service or
    other process, itself or through the module's functions and methods it calls. A timed wait made
    once is a deadline, a future's wait is the deadline of work handed elsewhere, a loop over items
    that skips the failed ones retries nothing, and a retry or a sleep beside a subprocess is at its
    boundary. A method or a function named like another, a definition only defined, a parameter
    or a local import called, a helper of an enclosing function, a class body run where it is
    defined, and a nested loop's or a `match`'s exits are each read as what they are; a failure
    that falls through to the loop's exit retries nothing. A notebook's display reaches its frontend,
    another process, so its frames redrawn on a clock are at that boundary, while a representation
    built round a loop publishes nothing."""
    assert_reported(tmp_path, "src/sparx/waits.py", ("import concurrent.futures\n"
        "import subprocess\n"
        "import time\n"
        "def poll(done):\n"
        "    while not done.wait(0.1):\n"
        "        pass\n"
        "def nap():\n"
        "    time.sleep(1)\n"
        "def deadline(done):\n"
        "    return done.wait(5)\n"
        "def retry(work):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            return work()\n"
        "        except OSError:\n"
        "            continue\n"
        "    return None\n"
        "def each(steps):\n"
        "    for step in steps:\n"
        "        try:\n"
        "            step()\n"
        "        except OSError:\n"
        "            continue\n"
        "def run(command):\n"
        "    return subprocess.run(command, check=True)\n"
        "def external(command):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            return run(command)\n"
        "        except subprocess.CalledProcessError:\n"
        "            time.sleep(1)\n"
        "    return None\n"
        "def futures(pending):\n"
        "    while pending:\n"
        "        done, pending = concurrent.futures.wait(pending)\n"
        "class Waiting:\n"
        "    def run(self):\n"
        "        time.sleep(1)\n"
        "class Other:\n"
        "    def run(self):\n"
        "        return 1\n"
        "def given(run):\n"
        "    run()\n"
        "    time.sleep(1)\n"
        "def defined():\n"
        "    def helper():\n"
        "        subprocess.run(['true'])\n"
        "    time.sleep(1)\n"
        "def after(work):\n"
        "    while True:\n"
        "        try:\n"
        "            work()\n"
        "        except ValueError:\n"
        "            continue\n"
        "        break\n"
        "def inner(rows):\n"
        "    for index in range(len(rows)):\n"
        "        try:\n"
        "            for row in rows[index]:\n"
        "                break\n"
        "        except ValueError:\n"
        "            continue\n"
        "def counted():\n"
        "    from math import factorial as run\n"
        "    run(5)\n"
        "    time.sleep(1)\n"
        "def outer():\n"
        "    def helper():\n"
        "        subprocess.run(['true'])\n"
        "    def enclosed():\n"
        "        helper()\n"
        "        time.sleep(1)\n"
        "    enclosed()\n"
        "def classy():\n"
        "    class Probe:\n"
        "        subprocess.run(['true'])\n"
        "    time.sleep(1)\n"
        "def once(work):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            work()\n"
        "        except ValueError:\n"
        "            failed = True\n"
        "        break\n"
        "def matched(work):\n"
        "    while True:\n"
        "        try:\n"
        "            match work():\n"
        "                case None:\n"
        "                    break\n"
        "        except ValueError:\n"
        "            continue\n"
        "def frames(done, frame):\n"
        "    from IPython.display import update_display\n"
        "    while not done.wait(0.125):\n"
        "        update_display(frame(), display_id='panel', raw=True)\n"
        "def built(done):\n"
        "    from IPython.display import HTML\n"
        "    while not done.wait(0.125):\n"
        "        HTML('<b>waiting</b>')\n"), "SLOP018", (5, 8, 12, 38, 44, 48, 50, 66, 86, 99))


def test_a_retry_is_told_from_a_scan_by_its_work_and_from_a_failure_by_its_paths(tmp_path):
    """SLOP018 reads a loop over a range as attempts at one piece of work only where its body reads
    the number it counts nowhere but in a handler: a scan by index, like one over the items,
    retries nothing, and so, by the rule's stated boundary, does a retry that reads its attempt
    number outside the handler; a name the work only binds is no read of it. A failure that
    raises on its last attempt and goes round on the others retries; one that raises on every
    attempt does not. A failure that runs on into the loop's tail retries where some path of the
    tail goes round, and a `try` inside the loop's own `with`, branch or `match` case is read as
    one in its body."""
    assert_reported(tmp_path, "src/sparx/retries.py", ("def retry(work):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            return work()\n"
        "        except ValueError:\n"
        "            continue\n"
        "def last_raises(work):\n"
        "    for attempt in range(3):\n"
        "        try:\n"
        "            return work()\n"
        "        except ValueError:\n"
        "            if attempt == 2:\n"
        "                raise\n"
        "def raised(work):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            return work()\n"
        "        except ValueError:\n"
        "            raise\n"
        "def direct(rows, parse):\n"
        "    for row in rows:\n"
        "        try:\n"
        "            return parse(row)\n"
        "        except ValueError:\n"
        "            continue\n"
        "def indexed(rows, parse):\n"
        "    for index in range(len(rows)):\n"
        "        try:\n"
        "            return parse(rows[index])\n"
        "        except ValueError:\n"
        "            continue\n"
        "def tail_returns(work):\n"
        "    while True:\n"
        "        try:\n"
        "            parsed = work()\n"
        "        except ValueError:\n"
        "            parsed = None\n"
        "        if parsed is not None:\n"
        "            return parsed\n"
        "def tail_once(work):\n"
        "    while True:\n"
        "        try:\n"
        "            parsed = work()\n"
        "        except ValueError:\n"
        "            parsed = None\n"
        "        return parsed\n"
        "def mentioned(work):\n"
        "    for attempt in range(3):\n"
        "        try:\n"
        "            return work(), attempt\n"
        "        except ValueError:\n"
        "            continue\n"
        "def fed(work):\n"
        "    for attempt in range(3):\n"
        "        try:\n"
        "            return work(attempt)\n"
        "        except ValueError:\n"
        "            continue\n"
        "def locked(work, lock):\n"
        "    for _ in range(3):\n"
        "        with lock:\n"
        "            try:\n"
        "                return work()\n"
        "            except ValueError:\n"
        "                continue\n"
        "def conditional(work, ready):\n"
        "    while True:\n"
        "        if ready():\n"
        "            try:\n"
        "                return work()\n"
        "            except ValueError:\n"
        "                failed = True\n"
        "def cased(work, mode):\n"
        "    while True:\n"
        "        match mode:\n"
        "            case 'fast':\n"
        "                try:\n"
        "                    return work()\n"
        "                except ValueError:\n"
        "                    continue\n"
        "            case _:\n"
        "                return None\n"
        "def preread(rows, parse):\n"
        "    for index in range(len(rows)):\n"
        "        row = rows[index]\n"
        "        try:\n"
        "            return parse(row)\n"
        "        except ValueError:\n"
        "            continue\n"
        "def unpacked(work):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            parsed, _ = work()\n"
        "            return parsed\n"
        "        except ValueError:\n"
        "            continue\n"), "SLOP018", (2, 8, 33, 60, 67, 74, 91))


def test_a_sleeps_boundary_is_found_through_calls_as_python_resolves_them(tmp_path):
    """SLOP018 follows a call to the definition Python's names reach: `self.f` to the first class
    that binds `f` in its method resolution order, from a method or a closure in one; a name to its
    binding, which a comprehension's own variable does not make outside it but does inside, and a
    pattern's capture, an exception's name or a class body's assignment does, and a receiver a
    comprehension rebinds is another object's. A generator's element runs only as it is consumed,
    at once where a loop iterates it, and a list's again and again, as a loop's body does. A sleep
    beside a call that reaches a subprocess, a list built of them included, is at its boundary."""
    assert_reported(tmp_path, "src/sparx/waits.py", ("import subprocess\n"
        "import time\n"
        "def run(command):\n"
        "    return subprocess.run(command, check=True)\n"
        "class Base:\n"
        "    def work(self):\n"
        "        subprocess.run(['true'])\n"
        "class Running(Base):\n"
        "    def spin(self):\n"
        "        self.work()\n"
        "        time.sleep(1)\n"
        "    def launch(self):\n"
        "        def poll():\n"
        "            self.work()\n"
        "            time.sleep(1)\n"
        "        poll()\n"
        "def comprehended():\n"
        "    doubled = [run * 2 for run in range(3)]\n"
        "    run(doubled)\n"
        "    time.sleep(1)\n"
        "def matching(value, callback):\n"
        "    match callback:\n"
        "        case run:\n"
        "            run(value)\n"
        "    time.sleep(1)\n"
        "def namespaced():\n"
        "    class Local:\n"
        "        run = abs\n"
        "        run(-1)\n"
        "    time.sleep(1)\n"
        "class Pure(Base):\n"
        "    def work(self):\n"
        "        return 1\n"
        "class Left(Base):\n"
        "    pass\n"
        "class Diamond(Left, Pure):\n"
        "    def spin(self):\n"
        "        self.work()\n"
        "        time.sleep(1)\n"
        "class Plain(Base):\n"
        "    work = staticmethod(abs)\n"
        "    def spin(self):\n"
        "        self.work()\n"
        "        time.sleep(1)\n"
        "def inline():\n"
        "    [run(-1) for run in (abs,)]\n"
        "    time.sleep(1)\n"
        "def deferred():\n"
        "    pending = (subprocess.run(['true']) for _ in range(3))\n"
        "    time.sleep(1)\n"
        "    return pending\n"
        "def handled():\n"
        "    try:\n"
        "        int('x')\n"
        "    except ValueError as subprocess:\n"
        "        subprocess.add_note('local')\n"
        "    time.sleep(1)\n"
        "def eager():\n"
        "    [subprocess.run(['true']) for _ in range(3)]\n"
        "    time.sleep(1)\n"
        "class Peers(Base):\n"
        "    def spin(self, peers):\n"
        "        [self.work() for self in peers]\n"
        "        time.sleep(1)\n"
        "def polled(done):\n"
        "    return [done.wait(0.1) for _ in range(3)]\n"
        "def consumed(commands):\n"
        "    for finished in (subprocess.run(command) for command in commands):\n"
        "        time.sleep(1)\n"), "SLOP018", (25, 30, 39, 44, 47, 50, 57, 64, 66))


def test_a_feature_detected_by_its_exception_is_reported(tmp_path):
    """SLOP019: a handler of the ImportError, ModuleNotFoundError or AttributeError a missing feature
    raises, by any name it is imported as, that carries on instead of raising, in the package's
    source. One that raises, naming what to install, requires the feature and is not reported, as
    is one whose class body raises where it is defined; a raise in a function the handler only
    defines is no raise of its own, and one that raises on some paths carries on along the rest. A
    `finally` that completes resumes the raise it interrupted, and a `match` whose last case
    catches all, through `as` or `|`, leaves by its cases alone, unless a guard lets it through."""
    assert_reported(tmp_path, "src/sparx/features.py", ("from builtins import ImportError as Missing\n"
        "def optional():\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError:\n"
        "        yaml = None\n"
        "    return yaml\n"
        "def required():\n"
        "    try:\n"
        "        import yaml\n"
        "    except ModuleNotFoundError as error:\n"
        "        raise RuntimeError('install sparx[yaml]') from error\n"
        "    return yaml\n"
        "def attribute(module):\n"
        "    try:\n"
        "        return module.fast\n"
        "    except AttributeError:\n"
        "        return module.slow\n"
        "def aliased():\n"
        "    try:\n"
        "        import yaml\n"
        "    except (ValueError, Missing):\n"
        "        def later():\n"
        "            raise RuntimeError('never')\n"
        "        yaml = None\n"
        "    return yaml\n"
        "def strict():\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError:\n"
        "        class Missing:\n"
        "            raise RuntimeError('install sparx[yaml]')\n"
        "    return yaml\n"
        "def unrelated():\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError as error:\n"
        "        if error.name != 'yaml':\n"
        "            raise\n"
        "        yaml = None\n"
        "    return yaml\n"
        "def released(lock):\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError as error:\n"
        "        try:\n"
        "            raise RuntimeError('install sparx[yaml]') from error\n"
        "        finally:\n"
        "            lock.release()\n"
        "    return yaml\n"
        "def released_fallback(lock):\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError:\n"
        "        try:\n"
        "            yaml = None\n"
        "        finally:\n"
        "            lock.release()\n"
        "    return yaml\n"
        "def aliased_catch_all(name):\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError as error:\n"
        "        match name:\n"
        "            case 'yaml':\n"
        "                raise RuntimeError('install sparx[yaml]') from error\n"
        "            case _ as other:\n"
        "                raise RuntimeError(f'install {other}') from error\n"
        "    return yaml\n"
        "def or_catch_all(name):\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError as error:\n"
        "        match name:\n"
        "            case 'yaml' | _:\n"
        "                raise RuntimeError('install sparx[yaml]') from error\n"
        "    return yaml\n"
        "def guarded_catch_all(name):\n"
        "    try:\n"
        "        import yaml\n"
        "    except ImportError as error:\n"
        "        match name:\n"
        "            case _ if name:\n"
        "                raise RuntimeError('install sparx[yaml]') from error\n"
        "        yaml = None\n"
        "    return yaml\n"), "SLOP019", (5, 17, 22, 37, 54, 81))


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
# tools/benchmark_data.py
# ---------------------------------------------------------------------------

def test_the_data_benchmark_times_a_chain_under_its_own_loading_unless_its_flags_set_one():
    """Without a `--loading.` flag the benchmark compiles the chain with no loading of its own, so the
    chain's `loading(...)` is what it times, as a run reads it; a flag sets the tool's, which the
    compile takes over the chain's."""
    from dew.data.dataset import Loading

    tool = load("benchmark_data")
    assert tool.benchmark(["--steps", "3"]).loading is None
    assert tool.benchmark(["--loading.workers", "2", "--steps", "3"]).loading == Loading(workers=2)


@pytest.mark.parametrize("timeline", [None, 0.01, 0.2])
def test_the_data_benchmark_checks_the_device_timelines_idle_and_reports_the_hosts_wait_beside_it(timeline):
    """The loop's wait in `next(reader)` overlaps the updates still running on the device, so a host
    that waits half of each step leaves the check to the device's own traced idle: it passes at 1%,
    fails at 20%, and a run with no accelerator to trace has no check."""
    tool = load("benchmark_data")
    device = None if timeline is None else {"device_idle_fraction": timeline,
                                            "device_input_idle_fraction": 0.0}

    def arm(engine):
        return {"workload": "text", "engine": engine, "loader": {"tokens_per_second": 100.0},
                "parent_rss_peak_bytes": 1, "worker_rss_peak_bytes": 1,
                "first_batch_from_open_seconds": 1.0,
                "training": {"data_wait_ms": 2.0, "step_ms": 2.0, "device_timeline": device}}

    [verdict] = tool.comparison([arm("dew"), arm("grain")])
    assert verdict["host_data_wait_fraction"] == 0.5
    checked = verdict["checks"].get("device_idle_under_5_percent")
    assert checked == (None if timeline is None else timeline < 0.05)


def test_the_data_benchmark_counts_a_grain_packing_and_dews_alike_as_the_objective_counts_them():
    """The data benchmark counts a batch's tokens and work as the run's objective counts them
    (`real_lengths`, `training_flops`), whichever packer laid the batch out: Grain's first-fit
    windows, laid out as Dew's read lays its own (`Coordinated`), hold the same real tokens as Dew's
    windows of the same documents. An image's rows count, and its objective declares no work."""
    import grain.experimental as gx
    import grain.python as grain

    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import Pipeline
    from dew.data.sources.memory import Memory
    from dew.data.steps.generic import Coordinated

    tool = load("benchmark_data")
    worker = tool.Worker("dew", "text", Path("in"), Path("out"), tool.Measurement(seq_len=8))
    objective = tool.objective_for(worker)
    documents = [{"text": np.arange(1, size + 1, dtype=np.int32)} for size in (3, 5, 2, 4, 6)]
    rows = grain.MapDataset.source(documents).to_iter_dataset()
    native = list(gx.FirstFitPackIterDataset(rows, length_struct={"text": 9}, num_packing_bins=2,
                                             shuffle_bins=False))
    stacked = {name: np.stack([window[name] for window in native]) for name in native[0]}
    laid = Coordinated(("text",), None).map(stacked)
    loader = compile(Pipeline(Memory(documents)).pack(9, bins=2).batch(len(native)), read="memory",
                     loading=Loading(workers=0, threads=1, read_buffer=0))
    reader = loader.train(DataPartition())
    try:
        ours = next(reader)
    finally:
        reader.close()
    expected = int(np.sum(objective.real_lengths(ours)))
    counted = [tool.batch_counts(batch, objective)["tokens"] for batch in (laid, ours)]
    assert counted == [expected, expected]
    work = tool.Counted(iter([laid]), objective)
    next(work)
    assert work.counts.get()["model_flops"] == objective.training_flops(None, ours) > 0
    image = tool.objective_for(tool.Worker("dew", "image", Path("in"), Path("out"), tool.Measurement()))
    pictures = tool.Counted(iter([{"image": np.zeros((2, 8, 8, 3), np.uint8)}]), image)
    next(pictures)
    assert {key: value for key, value in pictures.counts.get().items() if key != "padded_tokens"} == {
        "rows": 2, "tokens": 2, "model_flops": None}


def test_every_remote_code_release_carries_the_fields_its_loader_reads(monkeypatch):
    """Each remote-code release, the loader's own and those the LLaDA-MoE tool merges into it, is the
    four fields `released_model` reads: the Hub repo, its commit, the files its code is, and the auto
    class its model registers as. A record of another shape fails at its first read, a model never
    built."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    for tool in ("remote_code_reference", "llada_moe_reference"):
        for name, (repo, revision, files, auto) in load(tool).RELEASES.items():
            assert (repo.count("/"), len(revision), auto.startswith("Auto")) == (1, 40, True), name
            assert files and all(file.endswith(".py") for file in files), name


def test_a_float64_twin_rotates_by_its_whole_complex_table():
    """A float64 twin keeps a complex table its constructor computes complex (Llama 4's vision
    `freqs_ci`): it rotates by the table's cosines and sines both, so its output is the float64
    product of its input and the table, which a table cast to a real dtype is not."""
    import torch

    from tools.torch_precision import float64, float64_twin

    class Rotation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            angles = torch.arange(1, 5).float()
            self.table = torch.nn.Buffer(torch.polar(torch.ones_like(angles), angles), persistent=False)
            self.gain = torch.nn.Parameter(torch.ones(4))

        def forward(self, pairs):
            return torch.view_as_real(torch.view_as_complex(pairs.float()) * self.table * self.gain)

    pairs = torch.from_numpy(np.random.default_rng(0).standard_normal((4, 2)))
    with float64():
        rotated = float64_twin(Rotation(), Rotation)(pairs)
    angles = torch.arange(1, 5, dtype=torch.float64)
    table = torch.polar(torch.ones_like(angles), angles)
    assert torch.equal(rotated, torch.view_as_real(torch.view_as_complex(pairs) * table))
