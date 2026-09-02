"""The example scripts run end to end with a tiny model and no downloads."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dew.inputs import ConditionalInputConfig, DiffusionInputConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_config_cli import RES, TOKENS, StubTextEncoder  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def fake_captioned_dataset(batch):
    """Uniform-noise images with empty captions, batch sized for the 8 simulated devices."""
    def batches():
        rs = np.random.RandomState(0)
        while True:
            yield {"image": rs.uniform(0, 255, (batch, RES, RES, 3)).astype(np.float32),
                   "text": {"input_ids": np.zeros((batch, TOKENS), np.int32),
                            "attention_mask": np.ones((batch, TOKENS), np.int32)}}
    return {"train": batches, "val": batches, "train_len": 4 * batch,
            "local_batch_size": batch, "global_batch_size": batch}


def load_example(name):
    spec = importlib.util.spec_from_file_location(f"examples.{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_diffusion_example_trains_samples_and_exports(tmp_path):
    example = load_example("train_diffusion")
    inputs = DiffusionInputConfig(
        sample_data_key="image", sample_data_shape=(RES, RES, 3),
        conditions=[ConditionalInputConfig(
            encoder=StubTextEncoder(model="stub", tokenizer=None), conditioning_data_key="text",
            pretokenized=True, unconditional_input="", model_key_override="textcontext")])
    config = example.Config(
        image_size=RES, batch_size=8, epochs=1, steps_per_epoch=3,
        model=dict(patch_size=8, emb_features=32, num_layers=1, num_heads=2), prompts=("a", "b"),
        out=tmp_path / "run")
    config.out.mkdir()

    state = example.main(config, data=fake_captioned_dataset(8), inputs=inputs)

    assert int(state.step) == 3
    grid = np.asarray(Image.open(config.out / "samples.png"))
    assert grid.shape == (RES, 2 * RES, 3) and grid.dtype == np.uint8
    assert (config.out / "export" / "model.safetensors").exists()
    assert (config.out / "export" / "config.json").exists()
    assert any((config.out / "checkpoints").rglob("*"))


def fake_labelled_dataset(batch, classes):
    def batches():
        rs = np.random.RandomState(0)
        while True:
            yield {"image": rs.uniform(0, 255, (batch, RES, RES, 3)).astype(np.float32),
                   "label": rs.randint(0, classes, batch).astype(np.int32)}
    return {"train": batches, "val": batches, "train_len": 4 * batch,
            "local_batch_size": batch, "global_batch_size": batch}


def test_train_jepa_example_trains_probes_and_saves_the_encoder(tmp_path):
    example = load_example("train_jepa")
    config = example.Config(
        classes=5, image_size=RES, patch_size=4, batch_size=8, epochs=1, steps_per_epoch=3,
        emb_features=32, num_layers=2, num_heads=2, out=tmp_path / "run")
    config.out.mkdir()

    state = example.main(config, data=fake_labelled_dataset(8, 5))

    assert int(state.step) == 3
    assert (config.out / "encoder.safetensors").stat().st_size > 0
