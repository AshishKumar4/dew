"""Optional moving averages and method-required policy references."""
import dataclasses

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.diffusion.discrete import MDLM
from dew.inputs import Field, InputSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives import Step
from dew.objectives.base import select
from dew.objectives.diffusion import DiffusionObjective, MaskedDiffusionObjective
from dew.objectives.lm import LMObjective, Samples
from dew.objectives.rl import GRPOObjective
from dew.registry import presets
from dew.sampling import Sampling
from dew.training import Trainer


class Denoiser(nn.Module):
    @nn.compact
    def __call__(self, x, t, train=False):
        return nn.Conv(3, (1, 1))(x)


def decoder(causal=True):
    return CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                             num_heads=2, mlp_features=16, max_seq_len=8, causal=causal)


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
        objective = DiffusionObjective(Denoiser(), presets.EDM()(),
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
    reference = select(initial.params, frozen_objective.ema.select)
    frozen_state = dataclasses.replace(initial, ema=reference)
    state = initial
    step = trainer.compile(initial, batch)
    frozen_step = frozen_trainer.compile(frozen_state, batch)
    for _ in range(2):
        state, loss, _, _, accepted = step(state, batch)
        frozen_state, frozen_loss, *_ = frozen_step(frozen_state, batch)
        assert bool(accepted) and state.ema is None
        np.testing.assert_allclose(loss, frozen_loss, rtol=1e-6)
        for got, want in zip(jax.tree.leaves(state.params), jax.tree.leaves(frozen_state.params), strict=True):
            np.testing.assert_array_equal(got, want)
    for got, want in zip(jax.tree.leaves(frozen_state.ema), jax.tree.leaves(reference), strict=True):
        np.testing.assert_array_equal(got, want)
    info = Step(state.microstep, jax.random.PRNGKey(13), None)
    preview = objective.preview(state.params, batch, info)
    expected_preview = frozen_objective.preview(state.params, batch, info)
    for got, want in zip(jax.tree.leaves(preview), jax.tree.leaves(expected_preview), strict=True):
        np.testing.assert_array_equal(got, want)
    checkpoints.save(2, state, None, {"loss": float(loss)})
    checkpoints.wait()
    restored, _, _ = trainer.place()
    assert restored.ema is None
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(got, want)
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
    old = objective.per_token_log_probs(initial.params, ids)[:, 2:]
    batch = {"input_ids": ids, "old_log_probs": old, "advantages": jnp.ones_like(old),
             "response_mask": jnp.ones_like(old), "prompt_length": jnp.full((ids.shape[0],), 3, jnp.int32)}
    def unregularized(params):
        current = objective.per_token_log_probs({"params": params}, ids)[:, 2:]
        return -jnp.mean(jnp.exp(current - old))
    expected_gradient = jax.grad(unregularized)(initial.params["params"])
    updates, _ = optimizer.update(expected_gradient, initial.opt_state, initial.params["params"])
    expected = optax.apply_updates(initial.params["params"], updates)
    state, loss, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted) and state.ema is None and int(state.updates) == 1
    assert float(loss) == pytest.approx(-1.)
    for got, want in zip(jax.tree.leaves(state.params["params"]), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    info = Step(state.microstep, jax.random.PRNGKey(7), None)
    live = objective.preview(state.params, batch, info)
    zeroed = jax.tree.map(jnp.zeros_like, state.params)
    moved = objective.preview(zeroed, batch, info)
    assert np.any(np.asarray(live.tokens) != np.asarray(moved.tokens))
    np.testing.assert_array_equal(moved.tokens[:, 2:], 0)
    referenced = GRPOObjective(decoder(), 5, beta=.1, head_chunks=1)
    assert Trainer(referenced, optimizer, key=jax.random.PRNGKey(5)).initial_state().ema is not None
