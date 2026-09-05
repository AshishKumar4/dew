"""The scripts under tools/ that no other test imports and runs.

Each is loaded from its file, the way tests/test_benchmark_data.py loads
benchmark_data.py, and run on a case small enough for CPU in seconds. A
reference generator writes its tiny fixture into a temporary directory and
the result is compared with what is committed, so a generator that drifts
from its fixture, or a library upgrade that changes the reference, shows up
here rather than in a parity test measuring Dew against stale evidence. A
benchmark runs its pure pieces, and its real entry point where the step
compiles on CPU in seconds.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def load(name: str):
    """tools/ holds scripts, not a package, so a tool is loaded from its file."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_same_arrays(written: Path, committed: Path) -> None:
    ours, theirs = np.load(written), np.load(committed)
    assert sorted(ours.files) == sorted(theirs.files)
    for name in theirs.files:
        assert np.array_equal(ours[name], theirs[name]), name


def assert_same_tensors(written: Path, committed: Path) -> None:
    from safetensors.numpy import load_file

    ours, theirs = load_file(str(written)), load_file(str(committed))
    assert sorted(ours) == sorted(theirs)
    for name in theirs:
        assert np.array_equal(ours[name], theirs[name]), name


# ---------------------------------------------------------------------------
# tools/moe_reference.py
# ---------------------------------------------------------------------------

def test_moe_fixtures_are_what_the_generator_writes(tmp_path):
    """The committed router fixtures regenerate byte for byte, so the config
    the test reads and the arrays it compares against come from one run of
    this generator and not from an edit that forgot to rerun it."""
    load("moe_reference").main(["--out", str(tmp_path)])

    committed = FIXTURES / "moe"
    assert json.loads((tmp_path / "config.json").read_text()) == json.loads(
        (committed / "config.json").read_text())
    for name in ("mixtral.npz", "deepseek.npz"):
        assert_same_arrays(tmp_path / name, committed / name)


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
    """tiny/ regenerates byte for byte: the prompts and image recipe, the
    random-weight checkpoint, and the reference outputs the parity test reads.
    The real tower is left to the network-marked test; nothing downloads."""
    load("clip_reference").write_tiny(tmp_path)

    committed = FIXTURES / "clip" / "tiny"
    assert json.loads((tmp_path / "prompts.json").read_text()) == json.loads(
        (committed / "prompts.json").read_text())
    assert_same_tensors(tmp_path / "model.safetensors", committed / "model.safetensors")
    assert_same_arrays(tmp_path / "reference.npz", committed / "reference.npz")


# ---------------------------------------------------------------------------
# tools/t5_reference.py
# ---------------------------------------------------------------------------

def test_t5_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """tiny/ regenerates byte for byte: the prompts, the config the loader
    builds the encoder from (decoder_start_token_id included, as every
    published T5 config carries it), the full state dict and the reference
    hidden states."""
    load("t5_reference").main(["--out", str(tmp_path)])

    written, committed = tmp_path / "tiny", FIXTURES / "t5" / "tiny"
    for name in ("prompts.json", "config.json"):
        assert json.loads((written / name).read_text()) == json.loads(
            (committed / name).read_text()), name
    assert_same_tensors(written / "model.safetensors", committed / "model.safetensors")
    assert_same_arrays(written / "reference.npz", committed / "reference.npz")


# ---------------------------------------------------------------------------
# tools/vae_reference.py
# ---------------------------------------------------------------------------

def test_vae_tiny_fixture_is_what_the_generator_writes(tmp_path):
    """sd3-tiny/ regenerates byte for byte: the image recipe, the config the
    loader builds the autoencoder from, its weights, and the encode and
    decode of the one committed image."""
    load("vae_reference").main(["--out", str(tmp_path)])

    written, committed = tmp_path / "sd3-tiny", FIXTURES / "vae" / "sd3-tiny"
    for name in ("inputs.json", "config.json"):
        assert json.loads((written / name).read_text()) == json.loads(
            (committed / name).read_text()), name
    assert_same_tensors(written / "diffusion_pytorch_model.safetensors",
                        committed / "diffusion_pytorch_model.safetensors")
    assert_same_arrays(written / "reference.npz", committed / "reference.npz")


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
