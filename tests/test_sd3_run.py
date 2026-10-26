"""An SD3-shaped run end to end: MMDiT over 16-channel latents, conditioned
on T5, through the real Trainer, and a sample out of it.

Everything here is the shipped path: the run's own `DiffusionRunConfig` names
the T5 encoder and the 16-channel autoencoder from the committed fixtures,
`build()` returns the objective the recipe trains, `Trainer.fit` takes the
steps, and `evaluate` samples through `dew.sampling.sample`. The pieces are
tiny (two T5 layers, a two-stage VAE, one MMDiT block) so it runs on CPU in
seconds; what it proves is that the shapes and the seams line up, not that
the model learns anything.

What the released SD3.5 checkpoints would need beyond this, named rather
than approximated. The values are `transformer/config.json` of
stable-diffusion-3.5-large (read from an ungated mirror, stabilityai's own
repo being gated), against `SD3Transformer2DModel.__init__` in diffusers
0.34:

- `pooled_projection_dim` (2048): SD3 conditions adaLN on a pooled
  projection of its two CLIP text encoders, a second condition beside the
  4096-wide T5 context that enters the joint stream. `SimpleMMDiT` pools the
  one context it is given (`ConditioningEmbed`, mask-weighted), so a released
  checkpoint's `time_text_embed.text_embedder` has nowhere to load.
- `pos_embed_max_size` (192): SD3 adds a learned patch position embedding
  table and crops it to the sampled resolution. `SimpleMMDiT` uses RoPE, so
  its tree holds no `pos_embed.pos_embed`.
- `dual_attention_layers`: an extra image-stream self-attention in the
  layers it names, null in 3.5-large and set in 3.5-medium. `SimpleMMDiT`
  builds its blocks without it.

What does line up: `in_channels` 16 and `patch_size` 2 are what this test
runs, `caption_projection_dim` is `emb_features` through `txt_embed`,
`joint_attention_dim` is the T5 width the tower emits, and `qk_norm`
("rms_norm") is `SimpleMMDiT(qk_norm=True)`.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffusion_support import images_of

from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.data.dataset import Loader
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.nn.autoencoders.pretrained import PretrainedAutoencoder
from dew.objectives.base import Step
from dew.objectives.diffusion.objective import VALIDATION_SAMPLES
from dew.registry import ModelConfig
from dew.sampling import Euler
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
        model=ModelConfig("simple_mmdit", {"patch_size": 2, "emb_features": 32, "num_layers": 1,
                                               "num_heads": 2, "mlp_ratio": 1,
                                               "dtype": "float32", "attention_impl": "reference"}),
        data=images_of("tfds://oxford_flowers102", RES, batch=BATCH),
        trainer=TrainerConfig(checkpoint_dir=str(directory), steps=2),
        preset=Flow(), text=TextCondition(encoder="t5", checkpoint=str(T5_TINY), max_length=8),
        autoencoder=PretrainedAutoencoder(modelname=str(VAE_TINY), dtype="float32"),
        val_metrics=(),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 3, "guidance": None}))


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


def test_one_trainer_step_and_a_sample(tmp_path):
    """The whole path: the objective's tree holds the model and the frozen T5,
    the trainer takes two steps on it, and preview samples through the same
    solver inference uses, decoded back to pixels."""
    config = run_config(tmp_path)
    objective = config.build()
    loaded = objective.inputs.conditions["textcontext"].encoder.params["params"]


    data = Loader(train=lambda partition: batches(objective)(), val=None, records=None, batch=BATCH)
    state = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0)).fit(
        data, steps=2, log_every=100)
    assert int(state.step) == 2
    trained = state.variables["encoders"]["textcontext"]["params"]
    for before, after in zip(jax.tree.leaves(loaded), jax.tree.leaves(trained), strict=True):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    leaves = jax.tree.leaves(state.variables["params"])
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)

    encoder = objective.inputs.conditions["textcontext"].encoder
    batch = {"image": np.zeros((BATCH, RES, RES, 3), np.uint8),
             "text": encoder.tokenize(PROMPTS)}
    artifact = objective.preview(
        state.variables, batch, Step(step=state.step, key=jax.random.PRNGKey(1), ema=None))

    assert artifact.images.shape == (VALIDATION_SAMPLES, RES, RES, 3)
    assert np.all(np.isfinite(np.asarray(artifact.images)))
    assert float(jnp.min(artifact.images)) >= -1.0
    assert float(jnp.max(artifact.images)) <= 1.0
    # max_length is 8 here, so a long prompt is truncated and the caption is
    # what the model was actually conditioned on.
    assert PROMPTS[0].startswith(artifact.captions[0]) and artifact.captions[0]
