"""Shortcut models against kvfrans/shortcut-models' `get_targets` and the
loss its `update` takes (`tools/shortcut_reference.py`), and the run config
around them."""


import jax
import numpy as np
import optax

from dew.diffusion import presets
from dew.objectives.diffusion.few_step import ShortcutObjective


def test_a_run_config_trains_a_shortcut_model_on_its_own_targets():
    from diffusion_stubs import batch_for
    from diffusion_support import images_of

    from dew.config import ObjectiveConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.inputs import TextCondition
    from dew.registry import ModelConfig
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
             "dtype": "float32", "attention_impl": "xla"},
        ),
        data=images_of("tfds://oxford_flowers102", 4),
        preset=presets.Shortcut(),
        val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("shortcut",
                                  {"sections": 4, "bootstrap_every": 2, "solver": Euler(), "steps": 3}),
    )
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
