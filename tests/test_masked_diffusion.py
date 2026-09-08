"""The masked-diffusion sampler and the held-weights seam on a loaded checkpoint.

The sampler half runs tokens in and tokens out. The seam half is what lets a
run continue from LLaDA's or Dream's released weights: the objective reports
the loaded tree through `held_variables`, the trainer binds it as the
initializer's argument and builds its state from it. The trained
source-format export of both families lives in test_masked_diffusion_export.py.
"""

import json

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import dew.nn.backbones.causal_transformer  # noqa: F401, registers the backbone
from dew.diffusion.discrete import MDLM
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.registry import models, with_precision
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def loaded(name: str):
    """A committed tiny masked-diffusion checkpoint as its model and variables."""
    from safetensors.numpy import load_file

    directory = FIXTURES / name
    config = translate_config(json.loads((directory / "config.json").read_text()))
    assert config["mask_token_id"] == 120
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))
    variables = translate_weights(load_file(str(directory / "model.safetensors")), config)
    return model, variables


def flat(tree):
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def test_unmask_sampler_runs_on_a_loaded_model_end_to_end():
    """Evaluation on llada-tiny unmasks the fully masked rows into vocabulary
    ids: [4, 12] int32 with no mask id left. A sampler returning its input
    would fail on the mask check."""
    model, variables = loaded("llada-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                        steps=8, samples=4)
    out = objective.preview(
        variables, {}, Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(0), ema=None))
    assert out is not None, "preview returns the artifact on process zero"
    tokens = np.asarray(out.tokens)
    assert tokens.shape == (4, 12) and tokens.dtype == np.int32
    assert bool(((tokens != 120) & (tokens >= 0) & (tokens < 128)).all())
    scored = objective.evaluate(
        variables, {"text": np.zeros((7, 12), np.int32)},
        Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=None))
    assert scored.tokens.shape == (7, 12) and scored.texts == ()
    assert bool((np.asarray(scored.tokens) != 120).all())


def test_the_objective_reports_the_loaded_tree_and_returns_it_from_init():
    """The held-weights contract. `held_variables` is what the trainer's
    boundary binds as an argument, and `init` returns that tree whether the
    caller passes it or leaves the objective to read its own. An objective
    holding nothing binds nothing and draws the model's tree from the key."""
    model, variables = loaded("llada-tiny")
    holding = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                       ema_decay=None, pretrained=variables)
    fresh = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12, ema_decay=None)
    tree_bytes = sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(variables))

    assert holding.held_variables() is variables and fresh.held_variables() is None
    assert sum(int(np.asarray(leaf).nbytes)
               for leaf in jax.tree.leaves(holding.initializer)) == tree_bytes
    assert jax.tree.leaves(fresh.initializer) == []

    key = jax.random.key(0)
    held = flat(variables)
    for tree in (holding.init(key), holding.initializer(key), fresh.init(key, variables)):
        assert flat(tree).keys() == held.keys()
        for name, leaf in flat(tree).items():
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]))
    drawn = flat(fresh.init(key))
    assert drawn.keys() == held.keys()
    assert any(not np.array_equal(np.asarray(leaf), np.asarray(held[name]))
               for name, leaf in drawn.items()), "the fresh init returned the held tree"


def test_a_tree_that_is_not_the_variables_dict_is_refused():
    """`pretrained` is the whole variables dict, so a params collection
    handed over on its own is named rather than initialising a tree whose
    leaves sit one level too high."""
    model, variables = loaded("llada-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                         ema_decay=None, pretrained=variables["params"])

    with pytest.raises(ValueError, match="params"):
        objective.init(jax.random.key(0))


def test_the_trainer_builds_its_state_from_the_held_checkpoint():
    """What the seam exists for: a real `Trainer` accepts the objective and
    its initial state is the loaded checkpoint, so the run continues from
    Dream's weights instead of a fresh draw."""
    model, variables = loaded("dream-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                         ema_decay=None, pretrained=variables)

    state = Trainer(objective, optax.sgd(1e-2), key=jax.random.key(0)).initial_state()

    built, held = flat(state.params), flat(variables)
    assert built.keys() == held.keys()
    for name, leaf in built.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]))
