"""The masked-diffusion sampler on a loaded checkpoint, tokens in and tokens out."""

import json

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

import dew.nn.backbones.causal_transformer  # noqa: F401, registers the backbone
from dew.diffusion.discrete import MDLM
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def test_unmask_sampler_runs_on_a_loaded_model_end_to_end():
    """Evaluation on llada-tiny unmasks the fully masked rows into vocabulary
    ids: [4, 12] int32 with no mask id left. A sampler returning its input
    would fail on the mask check."""
    from safetensors.numpy import load_file

    directory = FIXTURES / "llada-tiny"
    config = translate_config(json.loads((directory / "config.json").read_text()))
    assert config["mask_token_id"] == 120
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))
    variables = translate_weights(load_file(str(directory / "model.safetensors")), config)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                        steps=8, samples=4)
    out = objective.preview(
        variables, {}, Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(0), ema=None))
    tokens = np.asarray(out.tokens)
    assert tokens.shape == (4, 12) and tokens.dtype == np.int32
    assert bool(((tokens != 120) & (tokens >= 0) & (tokens < 128)).all())
    scored = objective.evaluate(
        variables, {"text": np.zeros((7, 12), np.int32)},
        Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=None))
    assert scored.tokens.shape == (7, 12) and scored.texts == ()
    assert bool((np.asarray(scored.tokens) != 120).all())
