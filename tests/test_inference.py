"""Inference: a run directory through to a sample.

A run writes its manifest next to its checkpoints; `TextToImage.from_run`
rebuilds the model, the process, the inputs and the weights from those two
things alone, and `from_pretrained` is the same on a pulled hub snapshot.
Everything here drives the real pipeline from a checkpoint the trainer has
just written.
"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import dew.nn.backbones  # registers the models
from dew.data import Dataset
from dew.diffusion import FlowMatchPredictionTransform
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.interop.manifest import Manifest
from dew.nn.dit import TextContext
from dew.objectives.base import merge
from dew.objectives.diffusion import DiffusionObjective
from dew.registry import encoders, models, presets
from dew.sampling import CFG, Heun, TextToImage
from dew.sampling import pipelines
from dew.training import Checkpoints, Trainer

RES = 8
TOKENS = 5
FEATURES = 6
VOCAB = 11
MODEL = dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)


@encoders("stub_text")
@dataclass(frozen=True, eq=False)
class StubText(ConditionEncoder):
    """A table lookup standing in for CLIP, rebuilt from its manifest fields
    the way the real encoder is."""

    checkpoint: str
    params: dict

    @classmethod
    def from_pretrained(cls, checkpoint: str, **fields):
        seed = int(checkpoint.rsplit("-", 1)[-1])
        return cls(checkpoint=checkpoint, params={"table": jnp.asarray(
            np.random.RandomState(seed).normal(size=(VOCAB, FEATURES)).astype(np.float32))})

    def tokenize(self, texts):
        ids = np.zeros((len(texts), TOKENS), np.int32)
        mask = np.zeros((len(texts), TOKENS), np.int32)
        for row, text in enumerate(texts):
            codes = [1] + [2 + (ord(char) % (VOCAB - 2)) for char in text[:TOKENS - 1]]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, params, tokens):
        return TextContext(hidden=params["table"][jnp.asarray(tokens["input_ids"])],
                           mask=jnp.asarray(tokens["attention_mask"]))

    def to_json(self):
        return {"checkpoint": self.checkpoint}


def make_run(directory, preset=presets.EDM(), seed=3):
    """One training step of the tiny conditional DiT, its checkpoint and its
    manifest in `directory`, as a recipe leaves them."""
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained(f"stub-{seed}"))})
    objective = DiffusionObjective(models.SimpleDiT(**MODEL), preset(), inputs)
    encoder = inputs.conditions["textcontext"].encoder
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (8, 1, RES, 3)).astype(np.uint8)
    batch = {"image": images, "text": encoder.tokenize(["a", "b", "c", "d", "e", "f", "g", "h"])}

    class Stream:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.position += 1
            return batch

        def get_state(self):
            return str(self.position).encode()

        def set_state(self, state):
            self.position = int(state.decode())

    checkpoints = Checkpoints(str(directory), keep=1)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0),
                      checkpoints=checkpoints)
    state = trainer.fit(Dataset(train=Stream, val=None, records=None, batch=8),
                        steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    Manifest(
        config={"run": "test"},
        model={"name": "simple_dit", "fields": MODEL},
        inputs=inputs.to_json(),
        preset={"name": presets.name_of(type(preset)), "fields": dataclasses.asdict(preset)},
        autoencoder=None,
    ).write(str(directory))
    return objective, state


def test_pipeline_generates_from_a_run_directory(tmp_path):
    """The whole offline path: the manifest and checkpoint a run wrote, the
    model, process, inputs and weights rebuilt from them, and a sample out."""
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    assert type(pipe.model).__name__ == "SimpleDiT" and pipe.model.emb_features == 16
    assert pipe.inputs.sample == Field("image", (RES, RES, 3))
    assert pipe.inputs.conditions["textcontext"].encoder.checkpoint == "stub-3"

    images = pipe(["a water lily", "a sunflower"], steps=3, guidance=2.0,
                  key=jax.random.PRNGKey(0))
    assert images.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(images))
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_from_run_restores_the_averaged_weights_by_default(tmp_path):
    """The EMA copy is what a run publishes; `ema=False` reads the live ones."""
    objective, state = make_run(tmp_path)
    averaged = merge(state.params, state.ema)

    pipe = TextToImage.from_run(str(tmp_path))
    for expected, loaded in zip(jax.tree.leaves(averaged["params"]),
                                jax.tree.leaves(pipe.params["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))
    live = TextToImage.from_run(str(tmp_path), ema=False)
    for expected, loaded in zip(jax.tree.leaves(state.params["params"]),
                                jax.tree.leaves(live.params["params"]), strict=True):
        np.testing.assert_allclose(np.asarray(loaded), np.asarray(expected))
    assert not all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(pipe.params["params"]), jax.tree.leaves(live.params["params"])))
    # the frozen encoder's table is the manifest's, not something re-drawn
    np.testing.assert_array_equal(
        np.asarray(pipe.params["encoders"]["textcontext"]["table"]),
        np.asarray(objective.inputs.conditions["textcontext"].encoder.params["table"]))


def test_from_run_rebuilds_the_training_process_exactly(tmp_path):
    """The manifest holds the preset's fields, so inference samples with the
    shift the run trained with and not the preset default."""
    make_run(tmp_path, preset=presets.Flow(shift=3.0, logit_mean=0.5))
    pipe = TextToImage.from_run(str(tmp_path))
    assert isinstance(pipe.process.schedule, FlowMatchingScheduler)
    assert pipe.process.schedule.shift == 3.0 and pipe.process.schedule.logit_mean == 0.5
    assert type(pipe.process.prediction) is FlowMatchPredictionTransform
    assert pipe.process.sampling is None


def test_from_pretrained_is_from_run_on_the_pulled_snapshot(tmp_path, monkeypatch):
    make_run(tmp_path)
    import dew.interop.hub as hub
    monkeypatch.setattr(hub, "pull_from_hub", lambda repo_id, revision=None: tmp_path)
    pipe = TextToImage.from_pretrained("user/flowers-dit")
    assert pipe.inputs.conditions["textcontext"].encoder.checkpoint == "stub-3"


def test_sampler_and_guidance_are_call_arguments(tmp_path):
    """Two steps of training leave the zero-initialised head near zero, so
    the weights are nudged off it for the conditional and unconditional
    branches to differ; then guidance is visible in the sample."""
    make_run(tmp_path)
    loaded = TextToImage.from_run(str(tmp_path))
    pipe = dataclasses.replace(loaded, params=jax.tree.map(lambda leaf: leaf + 0.05, loaded.params))
    key = jax.random.PRNGKey(1)
    plain = pipe(["x"], steps=8, guidance=None, sampler=Heun(), key=key)
    guided = pipe(["x"], steps=8, guidance=CFG(4.0, interval=(0.2, 0.8)), sampler=Heun(), key=key)
    assert plain.shape == guided.shape == (1, RES, RES, 3)
    assert not np.allclose(plain, guided)
    assert np.array_equal(pipe(["x"], steps=8, guidance=None, sampler=Heun(), key=key), plain)
    assert np.array_equal(pipe(["x"], steps=8, guidance=4.0, sampler=Heun(), key=key),
                          pipe(["x"], steps=8, guidance=CFG(4.0), sampler=Heun(), key=key))


def test_an_autoencoder_without_a_loader_is_refused():
    with pytest.raises(ValueError, match="simple_autoencoder"):
        pipelines.load_autoencoder({"name": "simple_autoencoder", "fields": {}})
    assert pipelines.load_autoencoder(None) is None
