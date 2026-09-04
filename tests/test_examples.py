"""The three examples run end to end on stub data: train, evaluate, export."""

import dataclasses
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.data import Dataset, TokenWindows
from dew.inputs import Condition, Field, InputSpec
from test_diffusion_objective import RES, TOKENS, StubText

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert any((tmp_path / "checkpoints").iterdir())


def test_train_jepa_example_trains_probes_and_saves_the_encoder(tmp_path):
    example = load_example("train_jepa")
    # An 8x8 patch grid, the smallest the default mask geometry fits on.
    config = example.Config(classes=5, image_size=32, patch_size=4, batch_size=8, steps=3,
                            model=dict(emb_features=32, num_layers=2, num_heads=2), out=tmp_path)

    state = example.main(config, data=fake_dataset(8, classes=5, size=32))

    assert int(state.step) == 3
    assert (tmp_path / "encoder.safetensors").stat().st_size > 0


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
