"""Shortcut models against kvfrans/shortcut-models' `get_targets`
(`tools/shortcut_reference.py`), and the objective and run config around
them."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.diffusion import presets
from dew.objectives.diffusion.few_step import ShortcutObjective, shortcut_levels, shortcut_target

CASE = np.load(Path(__file__).resolve().parent / "fixtures" / "shortcut" / "targets.npz")
SETTINGS = json.loads(str(CASE["settings"]))


def reference(x, t, level, labels):
    """`tools/shortcut_reference.py`'s velocity, data minus noise, in the
    reference's time and step level."""
    def column(value):
        return jnp.asarray(value, jnp.float32).reshape(-1, 1, 1, 1)

    return (jnp.tanh(x) * 0.5 + jnp.sin(2 * column(t)) * x * 0.3
            + 0.1 * column(level) * jnp.cos(x) + 0.05 * column(labels))


def test_the_self_consistency_targets_are_the_references():
    """Dew's sigma is 1 - t and its velocity noise minus data, the
    reference's negated; a step of 2^-level is the reference's level. On the
    reference's draws the levels and every bootstrapped target agree."""
    rows = SETTINGS["batch_size"] // SETTINGS["bootstrap_every"]
    levels = shortcut_levels(rows, SETTINGS["denoise_timesteps"])
    np.testing.assert_array_equal(np.asarray(levels), CASE["level"][:rows])
    labels = jnp.asarray(CASE["classes"][:rows])

    def velocity(x, sigma, following):
        return -reference(x, 1 - sigma, jnp.log2(1 / (sigma - following)), labels)

    target = shortcut_target(velocity, jnp.asarray(CASE["x_t"][:rows]), 1 - jnp.asarray(CASE["t"][:rows]),
                             2.0 ** -levels)
    # Dyadic times and steps map exactly; the rest is the same float32
    # arithmetic with the sign flipped.
    np.testing.assert_allclose(np.asarray(target), -CASE["v_t"][:rows], rtol=1e-6, atol=1e-6)


def test_a_run_config_trains_a_shortcut_model_on_its_own_targets():
    from test_diffusion_run_sources import batch_for

    from dew.config import ModelConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, ShortcutTraining, TextCondition
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2},
                          dtype="float32", attention_impl="xla"),
        data=TFDSImages(image_size=4), preset=presets.Shortcut(), solver=Euler(), guidance=None,
        sampling_steps=3, val_metrics=(), text=TextCondition(encoder="char_table", checkpoint="char_table"),
        shortcut=ShortcutTraining(sections=4, bootstrap_every=2))
    task = config.build()
    assert isinstance(task, ShortcutObjective) and task.model.interval
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(task, 4)
    step = trainer.compile(state, batch)
    for _ in range(3):
        state, *_ = step(state, batch)
    images = task.pipeline(state)(["a red bird"], key=9).host().images
    assert np.all(np.isfinite(images))
