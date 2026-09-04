"""An SD3-shaped run end to end: MMDiT over 16-channel latents, conditioned
on T5, through the real Trainer, and a sample out of it.

Everything here is the shipped path: the run's own `DiffusionRunConfig` names
the T5 encoder and the 16-channel autoencoder from the committed fixtures,
`build()` returns the objective the recipe trains, `Trainer.fit` takes the
steps, and `evaluate` samples through `dew.sampling.sample`. The pieces are
tiny (two T5 layers, a two-stage VAE, one MMDiT block) so it runs on CPU in
seconds; what it proves is that the shapes and the seams line up, not that
the model learns anything.

What the released SD3.5 and Flux checkpoints would need beyond this, named
rather than approximated:

- `pooled_projection_dim`: SD3 conditions adaLN on a pooled projection of its
  two CLIP text encoders, a second condition beside the T5 context that
  enters the joint stream. `SimpleMMDiT` pools the one context it is given
  (`ConditioningEmbed`, mask-weighted), so a released checkpoint's
  `time_text_embed.text_embedder` has nowhere to load.
- `pos_embed_max_size`: SD3 adds learned patch position embeddings and crops
  that table to the sampled resolution. `SimpleMMDiT` uses RoPE, so there is
  no `pos_embed.pos_embed` table in its tree.
- `dual_attention_layers`: SD3.5-large runs an extra image-stream
  self-attention in some layers. `MMDiTBlock` has one attention per stream.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import dew.nn.backbones  # registers the models
from dew.artifacts import ImageGrid
from dew.config import ModelConfig, TrainerConfig
from dew.data import Dataset, OxfordFlowers
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionRunConfig, StableDiffusionAutoencoder, TextCondition
from dew.objectives.diffusion.objective import VALIDATION_SAMPLES
from dew.registry import presets, samplers
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
T5_TINY = FIXTURES / "t5" / "tiny"
VAE_TINY = FIXTURES / "vae" / "sd3-tiny"

RES = 16
BATCH = 8
PROMPTS = ["a red bird", "two cats", "a harbour at dawn", "x",
           "rain on the roof", "bread and jam", "birds at dawn", "a short note"]


def run_config(directory):
    """The SD3-shaped run: MMDiT, flow matching, T5 text, the 16-channel VAE."""
    return DiffusionRunConfig(
        model=ModelConfig("simple_mmdit", dict(patch_size=2, emb_features=32, num_layers=1,
                                               num_heads=2, mlp_ratio=1),
                          dtype="float32", attention_impl="reference"),
        data=OxfordFlowers(image_size=RES),
        trainer=TrainerConfig(checkpoint_dir=str(directory), batch_size=BATCH, steps=2),
        preset=presets.Flow(), sampler=samplers.Euler(), sampling_steps=3, guidance=0.0,
        text=TextCondition(encoder="t5", checkpoint=str(T5_TINY), max_length=8),
        autoencoder=StableDiffusionAutoencoder(modelname=str(VAE_TINY), dtype="float32"),
        val_metrics=[])


def batches(objective):
    """Endless uint8 images with their tokenized prompts, the shape the data
    workers write."""
    encoder = objective.inputs.conditions["textcontext"].encoder
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (BATCH, 1, RES, 3)).astype(np.uint8)
    batch = {"image": images, "text": encoder.tokenize(PROMPTS)}

    def stream():
        while True:
            yield batch

    return stream


def test_the_run_builds_an_mmdit_over_sixteen_channel_latents(tmp_path):
    """The autoencoder decides what the model denoises: 16 channels at the
    latent resolution, not 3 at the image resolution."""
    config = run_config(tmp_path)
    objective = config.build()

    assert objective.autoencoder.latent_channels == 16
    assert objective.latent_shape == (RES // 2, RES // 2, 16)
    assert objective.model.output_channels == 16
    assert config.model_fields(objective.autoencoder)["output_channels"] == 16


def test_one_trainer_step_and_a_sample(tmp_path):
    """The whole path: the objective's tree holds the model and the frozen T5,
    the trainer takes two steps on it, and evaluate samples through the same
    solver inference uses, decoded back to pixels."""
    config = run_config(tmp_path)
    objective = config.build()

    variables = objective.init(jax.random.PRNGKey(0))
    assert set(variables) == {"params", "encoders"}
    assert set(variables["encoders"]) == {"textcontext"}

    data = Dataset(train=batches(objective), val=None, records=None, batch=BATCH)
    state = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0)).fit(
        data, steps=2, log_every=100)
    assert int(state.step) == 2
    leaves = jax.tree.leaves(state.params["params"])
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)

    encoder = objective.inputs.conditions["textcontext"].encoder
    batch = {"image": np.zeros((BATCH, RES, RES, 3), np.uint8),
             "text": encoder.tokenize(PROMPTS)}
    artifact = objective.evaluate(
        state.params, batch, Step(step=state.step, key=jax.random.PRNGKey(1), ema=None))

    assert isinstance(artifact, ImageGrid)
    assert artifact.images.shape == (VALIDATION_SAMPLES, RES, RES, 3)
    assert np.all(np.isfinite(np.asarray(artifact.images)))
    assert float(jnp.min(artifact.images)) >= -1.0
    assert float(jnp.max(artifact.images)) <= 1.0
    # max_length is 8 here, so a long prompt is truncated and the caption is
    # what the model was actually conditioned on.
    assert PROMPTS[0].startswith(artifact.captions[0]) and artifact.captions[0]


def test_the_frozen_text_tower_is_not_optimized(tmp_path):
    """T5's weights ride in the tree as state, so the optimizer never sees
    them and two steps leave them exactly as loaded."""
    config = run_config(tmp_path)
    objective = config.build()
    loaded = objective.inputs.conditions["textcontext"].encoder.params["params"]

    data = Dataset(train=batches(objective), val=None, records=None, batch=BATCH)
    state = Trainer(objective, optax.adam(1e-1), key=jax.random.PRNGKey(0)).fit(
        data, steps=2, log_every=100)

    trained = state.params["encoders"]["textcontext"]["params"]
    for before, after in zip(jax.tree.leaves(loaded), jax.tree.leaves(trained), strict=True):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    assert objective.ema.select(("params",)) and not objective.ema.select(("encoders",))
