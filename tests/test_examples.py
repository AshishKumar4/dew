"""Every example runs end to end on stub data: train, evaluate, export.

The older three are called in process, with a stub dataset built here. The
end-to-end four ship their own `--smoke` mode over the repo's fixtures, and
run the way a reader runs them: their own process, their own command line,
one artifact each to show they got to the end.
"""

import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from test_diffusion_objective import RES, TOKENS, StubText

from dew.data import Dataset
from dew.inputs import Condition, Field, InputSpec
from dew.interop import load_params

REPO_ROOT = Path(__file__).resolve().parents[1]


def smoke(name, out, *arguments):
    """One example's `--smoke` run, in its own process, on one CPU device.

    The environment is the one the docstrings tell a reader to use, minus
    the suite's eight simulated devices: a smoke run is a single-device run,
    and `HF_HUB_OFFLINE` keeps a fixture path from becoming a download.
    """
    environment = {**os.environ,
                   "PYTHONPATH": str(REPO_ROOT / "src"),
                   "JAX_PLATFORMS": "cpu",
                   "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                   "HF_HUB_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
    finished = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / f"{name}.py"), "--smoke",
         "--out", str(out), *arguments],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=900)
    assert finished.returncode == 0, (
        f"{name} --smoke exited {finished.returncode}\n"
        f"--- stdout ---\n{finished.stdout}\n--- stderr ---\n{finished.stderr}")
    return finished


def load_example(name):
    path = REPO_ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _batches(batch, classes=None, size=RES):
    """Endless captioned batches of noise images; labels when `classes` is set."""
    def stream():
        rng = np.random.RandomState(0)
        while True:
            record = {"image": rng.randint(0, 256, (batch, size, size, 3), np.uint8),
                      "text": {"input_ids": np.ones((batch, TOKENS), np.int32),
                               "attention_mask": np.ones((batch, TOKENS), np.int32)}}
            if classes is not None:
                record["label"] = rng.randint(0, classes, (batch,), np.int32)
            yield record
    return stream


def fake_dataset(batch, classes=None, size=RES):
    return Dataset(train=_batches(batch, classes, size),
                   val=lambda: itertools.islice(_batches(batch, classes, size)(), 1),
                   records=4 * batch, batch=batch)


@pytest.mark.mesh
def test_train_diffusion_example_trains_samples_and_exports(tmp_path):
    example = load_example("train_diffusion")
    config = example.Config(image_size=RES, batch_size=8, steps=3, prompts=("a", "b"),
                            model=dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2),
                            out=tmp_path)
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained("stub"))})

    state = example.main(config, data=fake_dataset(8), inputs=inputs)

    assert int(state.step) == 3
    grid = np.asarray(__import__("PIL.Image").Image.open(tmp_path / "samples.png"))
    assert grid.shape == (RES, 2 * RES, 3) and grid.dtype == np.uint8
    assert (tmp_path / "export" / "model.safetensors").exists()
    assert (tmp_path / "export" / "config.json").exists()
    assert (tmp_path / "checkpoints" / "3").is_dir(), "the last step was not checkpointed"


@pytest.mark.mesh
def test_train_jepa_example_trains_probes_and_saves_the_encoder(tmp_path):
    example = load_example("train_jepa")
    # An 8x8 patch grid, the smallest the default mask geometry fits on.
    config = example.Config(classes=5, image_size=32, patch_size=4, batch_size=8, steps=3,
                            model=dict(emb_features=32, num_layers=2, num_heads=2), out=tmp_path)

    state = example.main(config, data=fake_dataset(8, classes=5, size=32))

    assert int(state.step) == 3
    saved = load_params(tmp_path / "encoder.safetensors")
    averaged = state.averaged["params"]["context_encoder"]
    assert jax.tree.structure(saved) == jax.tree.structure(averaged), "not the encoder's tree"
    assert all(np.array_equal(np.asarray(a), b) for a, b in zip(
        jax.tree.leaves(averaged), jax.tree.leaves(saved), strict=True))


@pytest.mark.mesh
def test_train_lm_example_trains_and_generates(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    text = ("ab" * 2000).encode()
    (tokens / "train.bin").write_bytes(text[:3600])
    (tokens / "val.bin").write_bytes(text[3600:])
    (tokens / "meta.json").write_text(json.dumps(
        {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8"}))
    example = load_example("train_lm")
    config = example.Config(tokens=tokens, sequence_length=32, batch_size=8, steps=3,
                            model=dict(emb_features=16, num_layers=1, num_heads=2),
                            prompt="ab", sample_tokens=8, out=tmp_path / "run")

    state = example.main(config)

    assert int(state.step) == 3
    sample = (tmp_path / "run" / "sample.txt").read_text()
    assert sample.startswith("ab") and len(sample) > 2


# ---------------------------------------------------------------------------------
# The end-to-end scripts, run the way their docstrings say to run them
# ---------------------------------------------------------------------------------

def test_train_flowers_tpu_smoke_samples_a_grid_and_scores_it(tmp_path):
    """The diffusion run's whole arc: synthetic ArrayRecords in, a run
    directory with its record and checkpoint, a samples grid out of
    `dew.pipeline`, and the CLIPScore of that grid."""
    smoke("train_flowers_tpu", tmp_path)

    grid = np.asarray(__import__("PIL.Image").Image.open(tmp_path / "samples.png"))
    assert grid.shape == (16, 4 * 16, 3) and grid.dtype == np.uint8
    assert "clip_score" in json.loads((tmp_path / "eval.json").read_text())
    assert (tmp_path / "checkpoints" / "smoke" / "run.json").is_file()


def test_sft_diffusion_gemma_smoke_writes_an_adapter_and_generates_from_it(tmp_path):
    """LoRA over a host-streamed base: the PEFT directory the run publishes,
    and the canvas `dew.pipeline` decodes once that directory is read back
    onto the base weights."""
    smoke("sft_diffusion_gemma", tmp_path)

    adapter = tmp_path / "adapter"
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["target_modules"]
    assert len((tmp_path / "samples.txt").read_text().splitlines()) == 2
