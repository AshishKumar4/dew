"""Optional moving averages and method-required policy references."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.checkpoints import Checkpoints
from dew.diffusion.discrete import MDLM
from dew.diffusion.presets import EDM
from dew.inputs import Field, InputSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives import Step
from dew.objectives.base import select
from dew.objectives.diffusion import DiffusionObjective, MaskedDiffusionObjective
from dew.objectives.lm import LMObjective, Samples
from dew.objectives.rl import GRPOObjective
from dew.sampling import Sampling
from dew.training import Trainer
from dew.training.state import TrainState


class Denoiser(nn.Module):
    @nn.compact
    def __call__(self, x, t, train=False):
        return nn.Conv(3, (1, 1))(x)


def decoder(causal=True):
    return CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                             num_heads=2, mlp_features=16, max_seq_len=8, causal=causal)


def test_metadata_inspection_and_restore_share_the_committed_snapshot(tmp_path, monkeypatch):
    """A shape inspection opens arrays once; restore still reads every value."""
    import orbax.checkpoint as ocp

    def state(width, step):
        return TrainState(
            step=jnp.asarray(step),
            microstep=jnp.asarray(step),
            updates=jnp.asarray(step),
            variables={"params": {"weight": jnp.arange(width, dtype=jnp.float32)}},
            opt_state=(),
            ema=None,
            key=jax.random.key(0),
            scale=None,
            window_size=jnp.asarray(1),
        )

    checkpoints = Checkpoints(str(tmp_path))
    first = state(3, 3)
    checkpoints.save(3, first, None)
    checkpoints.wait()
    stored = checkpoints.stored()
    placement = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    template = {"variables": jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=placement), stored["variables"])}

    def repeated_metadata(self, infos):
        raise AssertionError("the inspected immutable checkpoint metadata was opened again")

    with monkeypatch.context() as context:
        context.setattr(ocp.type_handlers.ArrayHandler, "metadata", repeated_metadata)
        restored, _ = checkpoints.restore(template)
        np.testing.assert_array_equal(restored["variables"]["params"]["weight"],
                                      first.variables["params"]["weight"])

    second = state(5, 5)
    checkpoints.save(5, second, None)
    checkpoints.wait()
    assert checkpoints.stored()["variables"]["params"]["weight"].shape == (5,)
    restored, _ = checkpoints.restore()
    np.testing.assert_array_equal(restored["variables"]["params"]["weight"],
                                  second.variables["params"]["weight"])
    third = state(7, 7)
    checkpoints.save(7, third, None)
    checkpoints.wait()
    assert checkpoints.stored()["variables"]["params"]["weight"].shape == (7,)


def make_case(kind, decay):
    rows = jax.device_count()
    if kind == "lm":
        objective = LMObjective(decoder(), 4, ema_decay=decay, head_chunks=1,
                                samples=Samples([1, 2], 2, sampling=Sampling(temperature=0.)))
        batch = {"text": jnp.tile(jnp.array([[1, 2, 3, 4, 5]], jnp.int32), (rows, 1))}
    elif kind == "masked":
        objective = MaskedDiffusionObjective(decoder(causal=False), MDLM(mask_id=7)(), 4,
                                             ema_decay=decay, head_chunks=1, steps=2)
        batch = {"text": jnp.tile(jnp.array([[1, 2, 3, 4]], jnp.int32), (rows, 1))}
    else:
        objective = DiffusionObjective(Denoiser(), EDM(regime="pixel"),
                                       InputSpec(Field("image", (2, 2, 3))),
                                       ema_decay=decay, guidance=None, steps=2)
        batch = {"image": jnp.arange(rows * 12, dtype=jnp.uint8).reshape(rows, 2, 2, 3)}
    return objective, batch


@pytest.mark.parametrize("kind", ["lm", "masked", "diffusion"])
def test_disabled_ema_trains_previews_and_resumes_without_a_copy(tmp_path, kind):
    objective, batch = make_case(kind, None)
    frozen_objective, _ = make_case(kind, 1.)
    optimizer = optax.sgd(.01)
    checkpoints = Checkpoints(str(tmp_path / kind))
    trainer = Trainer(objective, optimizer, key=jax.random.PRNGKey(1), checkpoints=checkpoints)
    frozen_trainer = Trainer(frozen_objective, optimizer, key=jax.random.PRNGKey(1))
    initial = trainer.initial_state()
    assert initial.ema is None
    with pytest.raises(ValueError, match="keeps no EMA"):
        _ = initial.averaged
    # Each step consumes its state, so the two states share no buffer: the
    # frozen one is a copy with the average as its own copy of the parameters,
    # and the average's starting values are kept on the host to compare with.
    reference = jax.tree.map(np.asarray, select(initial.variables, frozen_objective.ema.select))
    frozen_state = jax.tree.map(jnp.copy, dataclasses.replace(
        initial, ema=select(initial.variables, frozen_objective.ema.select)))
    state = initial
    step = trainer.compile(initial, batch)
    frozen_step = frozen_trainer.compile(frozen_state, batch)
    for _ in range(2):
        state, loss, _, _, accepted = step(state, batch)
        frozen_state, frozen_loss, *_ = frozen_step(frozen_state, batch)
        assert bool(accepted) and state.ema is None
        np.testing.assert_allclose(loss, frozen_loss, rtol=1e-6)
        for got, want in zip(
            jax.tree.leaves(state.variables), jax.tree.leaves(frozen_state.variables), strict=True
        ):
            np.testing.assert_array_equal(got, want)
    for got, want in zip(jax.tree.leaves(frozen_state.ema), jax.tree.leaves(reference), strict=True):
        np.testing.assert_array_equal(got, want)
    info = Step(state.microstep, jax.random.PRNGKey(13), None)
    preview = objective.preview(state.variables, batch, info)
    expected_preview = frozen_objective.preview(state.variables, batch, info)
    for got, want in zip(jax.tree.leaves(preview), jax.tree.leaves(expected_preview), strict=True):
        np.testing.assert_array_equal(got, want)
    checkpoints.save(2, state, None, {"loss": float(loss)})
    checkpoints.wait()
    restored, _, _ = trainer.place()
    assert restored.ema is None
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        from affine_run import raw_leaf
        np.testing.assert_array_equal(raw_leaf(got), raw_leaf(want))
    with pytest.raises(ValueError, match="EMA configuration"):
        Trainer(frozen_objective, optimizer, key=jax.random.PRNGKey(1), checkpoints=checkpoints).place()


def test_zero_beta_grpo_has_no_reference_and_keeps_live_updates_and_preview():
    objective = GRPOObjective(decoder(), 5, beta=0., head_chunks=1,
                              samples=Samples([1, 2], 2, sampling=Sampling(temperature=0.)))
    optimizer = optax.sgd(.01)
    trainer = Trainer(objective, optimizer, key=jax.random.PRNGKey(5))
    initial = trainer.initial_state()
    assert initial.ema is None
    ids = jnp.tile(jnp.array([[1, 2, 3, 4, 5, 6]], jnp.int32), (jax.device_count(), 1))
    mask = jnp.tile(jnp.array([[0, 0, 0, 1, 1, 1]], jnp.float32), (ids.shape[0], 1))
    batch = {"input_ids": ids, "text_segment_ids": jnp.ones_like(ids),
             "text_positions": jnp.tile(jnp.arange(6, dtype=jnp.int32), (ids.shape[0], 1)),
             "response_mask": mask, "advantages": mask}
    old = objective.packed_log_probs(initial.variables, batch)
    batch.update(old_log_probs=old, behavior_log_probs=old)
    def unregularized(params):
        current = objective.packed_log_probs({"params": params}, batch)
        return -jnp.sum(jnp.exp(current - old) * mask) / jnp.sum(mask)
    expected_gradient = jax.grad(unregularized)(initial.variables["params"])
    updates, _ = optimizer.update(expected_gradient, initial.opt_state, initial.variables["params"])
    expected = optax.apply_updates(initial.variables["params"], updates)
    state, loss, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted) and state.ema is None and int(state.updates) == 1
    assert float(loss) == pytest.approx(-1.)
    for got, want in zip(jax.tree.leaves(state.variables["params"]), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    info = Step(state.microstep, jax.random.PRNGKey(7), None)
    live = objective.preview(state.variables, batch, info)
    zeroed = jax.tree.map(jnp.zeros_like, state.variables)
    moved = objective.preview(zeroed, batch, info)
    assert np.any(np.asarray(live.tokens) != np.asarray(moved.tokens))
    np.testing.assert_array_equal(moved.tokens[:, 2:], 0)
    referenced = GRPOObjective(decoder(), 5, beta=.1, head_chunks=1)
    assert Trainer(referenced, optimizer, key=jax.random.PRNGKey(5)).initial_state().ema is not None
