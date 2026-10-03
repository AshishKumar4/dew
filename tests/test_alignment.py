"""REPA and iREPA against their official code (`tools/repa_reference.py`),
and the alignment inside the diffusion objective."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion import Alignment, DiffusionObjective
from dew.objectives.diffusion.alignment import ALIGNMENT, REPRESENTATION, spatial_zscore
from dew.sampling import Euler, TextToImage

CASES = np.load(Path(__file__).resolve().parent / "fixtures" / "repa" / "losses.npz")
SETTINGS = json.loads(str(CASES["settings"]))


def projector(kind: str) -> dict:
    prefix = f"{kind}."
    tree: dict = {}
    for key in CASES.files:
        if key.startswith(prefix):
            module, leaf = key[len(prefix):].split(".")
            tree.setdefault(module, {})[leaf] = jnp.asarray(CASES[key], jnp.float32)
    return {"params": tree}


# A float32 mean of 32 cosines, each a few roundings of an O(1) value: its
# error is a small multiple of 2^-24, so 1e-6 separates rounding from any
# difference in what is computed (a wrong normalization or projection moves
# the loss by more than 1e-3 here).
LOSS_ATOL = 1e-6


def test_the_repa_loss_is_the_official_one():
    """REPA's MLP projector and its -cos loop over tokens and examples."""
    alignment = Alignment(nn.Module(), {}, "block", width=SETTINGS["width"])
    loss = alignment.loss(projector("mlp"), jnp.asarray(CASES["hidden"]), jnp.asarray(CASES["features"]))
    np.testing.assert_allclose(float(loss), float(CASES["repa"]), rtol=0, atol=LOSS_ATOL)


def test_the_irepa_loss_is_the_official_one():
    """iREPA's 3x3 convolution projector against spatially z-scored targets."""
    alignment = Alignment(nn.Module(), {}, "block", projector="conv", spatial_norm=SETTINGS["gamma"])
    targets = spatial_zscore(jnp.asarray(CASES["features"]), SETTINGS["gamma"])
    loss = alignment.loss(projector("conv"), jnp.asarray(CASES["hidden"]), targets)
    np.testing.assert_allclose(float(loss), float(CASES["irepa"]), rtol=0, atol=LOSS_ATOL)


class Patches(nn.Module):
    """A frozen encoder of 4-pixel patches, one token per DiT patch."""

    @nn.compact
    def __call__(self, pixels):
        return nn.Conv(5, (4, 4), strides=(4, 4))(pixels)


def aligned(kind: str = "mlp"):
    model = SimpleDiT(patch_size=4, emb_features=16, num_layers=2, num_heads=2, mlp_ratio=1)
    encoder = Patches()
    variables = encoder.init(jax.random.PRNGKey(9), jnp.zeros((1, 8, 8, 3)))
    alignment = Alignment(encoder, variables, "dit_block_0", weight=0.5, projector=kind, width=8,
                          spatial_norm=0.6 if kind == "conv" else None)
    return DiffusionObjective(model, presets.Flow()(), InputSpec(Field("image", (8, 8, 3))),
                              guidance=None, solver=Euler(), steps=2, alignment=alignment)


@pytest.mark.parametrize("kind", ["mlp", "conv"])
def test_the_objective_adds_the_weighted_alignment_to_the_denoising_mean(kind):
    """The loss is the denoising mean plus `weight` times the alignment, whose
    gradient reaches the projector and the layers up to the aligned one; the
    encoder is held beside the model, and a published task drops it and the
    projector."""
    objective = aligned(kind)
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (4, 8, 8, 3), 0, 256), np.uint8)}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    loss, aux = objective.scalar_loss(params, batch, step)

    plain = DiffusionObjective(objective.model, objective.process, objective.inputs, guidance=None,
                               solver=Euler(), steps=2)
    denoising, _ = plain.scalar_loss({**objective.model_variables(params), "encoders": params["encoders"]},
                               batch, step)
    # REPA's total is mse + proj_coeff * alignment; Dew's L2 halves the
    # first, so the second is halved with it: 0.5 / 2.
    assert float(loss) == pytest.approx(float(denoising) + 0.25 * float(aux.metrics["alignment"]), rel=1e-6)
    assert -1.0 <= float(aux.metrics["alignment"]) <= 1.0

    def alignment_only(tree):
        return objective.scalar_loss({**params, "params": tree}, batch, step)[1].metrics["alignment"]

    grads = jax.grad(alignment_only)(params["params"])
    assert float(jnp.abs(jax.tree.leaves(grads[ALIGNMENT])[0]).sum()) > 0
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads["dit_block_0"]))) > 0
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads["dit_block_1"]))) == 0
    published = TextToImage.from_objective(objective, params).params
    assert REPRESENTATION not in published and ALIGNMENT not in published["params"]


def test_a_layer_the_model_lacks_is_refused():
    objective = aligned()
    alignment = Alignment(objective.alignment.encoder, objective.alignment.variables, "dit_block_9")
    with pytest.raises(ValueError, match="no submodule 'dit_block_9'"):
        DiffusionObjective(objective.model, objective.process, objective.inputs, guidance=None,
                           solver=Euler(), steps=2, alignment=alignment).init(jax.random.PRNGKey(0))


def test_from_run_publishes_the_model_without_the_alignment_head(tmp_path):
    """An inference record restores the denoiser, not the frozen encoder or
    projector that only its training loss reads, as the objective's own
    pipeline does."""
    import optax

    from dew.checkpoints import Checkpoints
    from dew.training import Trainer

    objective = aligned()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))
    state = trainer.initial_state()
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None, artifact=objective.inference_record())
    checkpoints.wait()

    restored = TextToImage.from_run(str(tmp_path), ema=False)
    published = objective.pipeline(state, ema=False)
    assert REPRESENTATION not in restored.variables
    assert ALIGNMENT not in restored.variables["params"]
    assert jax.tree.structure(restored.variables) == jax.tree.structure(published.variables)
    for actual, expected in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(published.variables),
                                strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(restored([{}, {}], key=9).host().images,
                                  published([{}, {}], key=9).host().images)
