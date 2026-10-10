"""GRPO: the clipped-surrogate-plus-KL composition against verl.

The loss must match verl 0.9's PPO path with the GRPO settings
(`compute_policy_loss_vanilla` with `clip_ratio` 0.2 both sides,
`clip_ratio_c` 3.0 and `token-mean`, plus `beta` times the `token-mean` k3
from `kl_penalty_forward`). The reference is tests/fixtures/rl/grpo.npz,
written by tools/parity_grpo.py from torch over one fixed rollout: old,
current and reference log-probabilities, both-signed advantages, and a mask
with short tails, with entries past every clip point. `GRPOObjective` reads
the same terms out of the rolled-out batch.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from rl_support import TinyHead, clipped_surrogate, token_mean
from steady_state import steady_state

from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY
from dew.objectives.base import Step
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.rollout import SampledRollout
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
)
from dew.rl import k3_kl, token_log_ratio
from dew.sampling import Sampling

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "grpo.npz"
VOCAB = 8
PROMPT_WIDTH = 5
RESPONSE_WIDTH = 3
ROWS = 2


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE, allow_pickle=True))


def terms(fixture):
    """The fixture rollout as float32 arrays with the KL strength."""
    def get(key):
        return np.asarray(fixture[key], np.float32)
    return (get("old_log_probs"), get("current_log_probs"), get("ref_log_probs"),
            get("advantages"), get("response_mask"), float(fixture["beta"]))


def test_grpo_loss_and_gradient_match_verl(reference):
    """Dew's composition against verl 0.9.0's own `compute_policy_loss_vanilla`
    and `kl_penalty` (tools/parity_grpo.py) on the fixed rollout, held to the
    float64 rule over the loss and its gradient in the current
    log-probabilities: Dew's RMS error from verl's float64 values at most
    twice verl's own fp32 error."""
    old, current, ref, advantages, mask, beta = terms(reference)

    def loss(current):
        ratio = token_log_ratio(current, jnp.asarray(old))
        pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
        kl = token_mean(k3_kl(current, jnp.asarray(ref)), jnp.asarray(mask))
        return pg + beta * kl

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(current))
    assert_as_exact_as_the_reference(
        np.append(value, gradient), np.append(reference["verl_loss"], reference["verl_current_grad"]),
        np.append(reference["verl_loss_f64"], reference["verl_current_grad_f64"]), "GRPO loss and gradient")


# --- the objective -------------------------------------------------------------

def rollout_batch(seed=0):
    """Two packed rows, one chain each, the last RESPONSE_WIDTH ids sampled,
    with a short second response."""
    rng = np.random.RandomState(seed)
    width = PROMPT_WIDTH + RESPONSE_WIDTH
    ids = rng.randint(0, VOCAB, (ROWS, width)).astype(np.int32)
    mask = np.zeros((ROWS, width), np.float32)
    mask[:, PROMPT_WIDTH:] = 1
    mask[1, -1] = 0
    terms = np.where(mask != 0, rng.normal(-0.5, 0.5, (ROWS, width)), 0).astype(np.float32)
    advantages = np.where(mask != 0, rng.normal(0, 1, (ROWS, width)), 0).astype(np.float32)
    return {
        IDS_KEY: jnp.asarray(ids),
        SEGMENT_IDS_KEY: jnp.ones((ROWS, width), jnp.int32),
        POSITIONS_KEY: jnp.tile(jnp.arange(width, dtype=jnp.int32), (ROWS, 1)),
        OLD_LOG_PROBS_KEY: jnp.asarray(terms),
        BEHAVIOR_LOG_PROBS_KEY: jnp.asarray(terms),
        ADVANTAGES_KEY: jnp.asarray(advantages),
        RESPONSE_MASK_KEY: jnp.asarray(mask),
    }


def test_the_loss_reads_the_rolled_out_batch():
    """The objective's loss is the composition over the batch's own terms:
    current log-probabilities scored along each chain, ratio against the
    stored old ones, surrogate and KL masked alike."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    batch = rollout_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)

    loss, aux = objective.scalar_loss(params, batch, step)

    ids = np.asarray(batch[IDS_KEY])
    start = PROMPT_WIDTH - 1
    policy = np.asarray(objective.per_token_log_probs(params, ids))[:, start:]
    ref = np.asarray(objective.per_token_log_probs(frozen, ids))[:, start:]
    old = np.asarray(batch[OLD_LOG_PROBS_KEY])[:, PROMPT_WIDTH:]
    advantages = np.asarray(batch[ADVANTAGES_KEY])[:, PROMPT_WIDTH:]
    mask = np.asarray(batch[RESPONSE_MASK_KEY])[:, PROMPT_WIDTH:]
    ratio = token_log_ratio(jnp.asarray(policy), jnp.asarray(old))
    pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
    kl = token_mean(k3_kl(jnp.asarray(policy), jnp.asarray(ref)), jnp.asarray(mask))
    assert float(loss) == pytest.approx(float(pg + 0.01 * kl), rel=1e-5)
    assert {"pg", "actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower", "kl"} <= set(aux.metrics)


def test_zero_beta_leaves_the_reference_unread():
    """With the KL term off, no frozen tree is needed and none is reported."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.0)
    params = objective.init(jax.random.key(0))
    batch = rollout_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)

    loss, aux = objective.scalar_loss(params, batch, step)

    assert np.isfinite(float(loss))
    assert "kl" not in aux.metrics


def test_a_positive_beta_needs_the_frozen_tree():
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    with pytest.raises(ValueError, match=r"step.ema"):
        objective.scalar_loss(params, rollout_batch(), step)


def test_a_misbuilt_objective_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        GRPOObjective(TinyHead(vocab_size=VOCAB), 7, beta=-0.1)
    with pytest.raises(ValueError, match="unit decay"):
        GRPOObjective(TinyHead(vocab_size=VOCAB), 7, ema_decay=0.999)
    with pytest.raises(ValueError, match="response mask"):
        GRPOObjective(TinyHead(vocab_size=VOCAB), 7, loss_role="x")


def test_a_misshapen_batch_is_refused():
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.0)
    params = objective.init(jax.random.key(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    batch = rollout_batch()

    narrow = {key: value[:, :5] for key, value in batch.items()}
    with pytest.raises(ValueError, match="8 ids per row"):
        objective.scalar_loss(params, narrow, step)

    ragged = dict(batch, **{ADVANTAGES_KEY: jnp.zeros((ROWS, 2), jnp.float32)})
    with pytest.raises(ValueError, match="shape"):
        objective.scalar_loss(params, ragged, step)


def test_evaluation_scores_prompt_perplexity():
    """Validation never samples: each prompt's shifted cross entropy, the
    real suffix weighted off `prompt_length`, pads counting nothing."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB), 4, beta=0.0)
    params = objective.init(jax.random.key(0))
    prompts = np.array([[0, 0, 1, 2, 3], [4, 5, 6, 7, 1]], np.int32)
    batch = {PROMPT_KEY: jnp.asarray(prompts),
             LENGTH_KEY: jnp.asarray([5, 3], np.int32)}
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)

    scores = objective.evaluate(params, batch, step)

    expected = np.asarray(objective.per_token_log_probs(params, prompts))
    weights = np.asarray(scores.weights)
    np.testing.assert_allclose(np.asarray(scores.losses) * weights, -expected * weights, rtol=1e-5)
    np.testing.assert_array_equal(weights, [[1, 1, 1, 1], [0, 0, 1, 1]])

    with pytest.raises(ValueError, match="two tokens"):
        objective.evaluate(params, {PROMPT_KEY: jnp.ones((1, 1), jnp.int32),
                                    LENGTH_KEY: jnp.ones(1, jnp.int32)}, step)


@pytest.mark.parametrize("kind", ["grpo", "ppo"])
def test_checkpointed_policy_fit_validates_source_prompts(tmp_path, kind):
    import itertools

    import optax
    from affine_run import Data
    from flax import linen as nn
    from recording import RecordingTracker

    from dew.objectives.base import VALID_ROWS
    from dew.objectives.rl import PPOObjective
    from dew.training import Checkpoints, Trainer

    class Critic(nn.Module):
        @nn.compact
        def __call__(self, tokens, **packing):
            return nn.Embed(VOCAB, 1)(tokens)[..., 0]

    model = TinyHead(vocab_size=VOCAB)
    width = PROMPT_WIDTH + RESPONSE_WIDTH - 1
    objective = (GRPOObjective(model, width, beta=.01) if kind == "grpo"
                 else PPOObjective(model, width, critic=Critic(), beta=.01))
    rows = 2 * jax.device_count()
    source = {PROMPT_KEY: np.tile(np.arange(1, 6, dtype=np.int32), (rows, 1)),
              LENGTH_KEY: np.tile(np.array([5, 3], np.int32), rows // 2)}
    validation = {**source, VALID_ROWS: np.arange(rows) < rows - 1}
    packed = jax.tree.map(lambda value: jnp.tile(value, (rows // ROWS, 1)), rollout_batch())
    if kind == "ppo":
        packed.update(old_values=jnp.zeros_like(packed[RESPONSE_MASK_KEY]),
                      returns=jnp.ones_like(packed[RESPONSE_MASK_KEY]))
    tracker = RecordingTracker()
    trainer = Trainer(objective, optax.sgd(0.), key=jax.random.key(0), tracker=tracker,
                      checkpoints=Checkpoints(str(tmp_path / kind)),
                      rollout=lambda state, batch, key: packed)
    data = Data(train=lambda: itertools.repeat(source), val=lambda: iter((validation,)), batch=rows)
    final = trainer.fit(data, steps=1, eval_every=1)
    reported = [scalars["val/loss"] for _, scalars in tracker.scalars if "val/loss" in scalars]
    assert reported and np.isfinite(reported).all()
    scores = objective.evaluate(final.variables, validation, Step(jnp.array(1), jax.random.key(1), None))
    weights = np.asarray(scores.weights) * validation[VALID_ROWS][:, None]
    expected = np.sum(np.asarray(scores.losses) * weights) / weights.sum()
    assert reported[-1] == pytest.approx(float(expected), rel=1e-5)


def test_the_rollout_batch_feeds_the_objective():
    """The `SampledRollout` pack and the `GRPOObjective` loss agree on every
    key: a rollout straight into the loss, shapes fixed, loss finite."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    width = PROMPT_WIDTH
    prompts = np.tile(np.arange(1, width + 1, dtype=np.int32), (2, 1))
    info = max(len("rule"), len("other"))
    def pad(text):
        return np.pad(
            np.frombuffer(text.encode(), np.uint8).astype(np.int32), (0, info - len(text)))
    batch = {
        PROMPT_KEY: prompts,
        LENGTH_KEY: np.full(2, width, np.int32),
        SOURCE_KEY: np.stack([pad("rule"), pad("other")]),
        TRUTH_KEY: np.stack([pad("1"), pad("2")]),
        INFO_KEY: np.stack([pad(""), pad("")]),
    }
    state = SimpleNamespace(variables=params, updates=0)
    rollout = SampledRollout(objective, lambda *args: 1.0, groups=2,
                             max_new_tokens=RESPONSE_WIDTH, sampling=Sampling(temperature=0.0))
    rolled = rollout(state, batch, jax.random.key(1))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)

    loss, aux = objective.scalar_loss(params, rolled, step)

    assert np.isfinite(float(loss))
    assert rolled[IDS_KEY].shape == (4, width + RESPONSE_WIDTH)
    assert set(aux.metrics) >= {"pg", "kl"}


def test_a_rollout_after_the_first_reuses_its_programs_and_reads_only_what_it_scores():
    """Sampling a group of the same prompt width again runs the programs the
    first rollout compiled, and the host reads the drawn rows and the group
    advantages by name and nothing else (`steady_state`). The prompts reach
    the device as each batch arrives, so only reads are held."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB), PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    state = SimpleNamespace(variables=objective.init(jax.random.key(0)), updates=0)
    rollout = SampledRollout(objective, lambda *args: float(len(args[0]) % 3), groups=2,
                             max_new_tokens=RESPONSE_WIDTH, sampling=Sampling(temperature=1.0))
    info = len("other")
    def pad(text):
        return np.pad(np.frombuffer(text.encode(), np.uint8).astype(np.int32), (0, info - len(text)))

    def batch(offset):
        rows = np.arange(PROMPT_WIDTH, dtype=np.int32)[None] + np.arange(2)[:, None] + offset
        prompts = rows % (VOCAB - 1) + 1
        return {PROMPT_KEY: prompts, LENGTH_KEY: np.full(2, PROMPT_WIDTH, np.int32),
                SOURCE_KEY: np.stack([pad("rule"), pad("other")]), TRUTH_KEY: np.stack([pad("1"), pad("2")]),
                INFO_KEY: np.stack([pad(""), pad("")])}

    keys = [jax.random.key(index) for index in range(3)]
    rollout(state, batch(0), keys[0])
    with steady_state(allow=("host_to_device",)):
        rolled = [rollout(state, batch(offset), key) for offset, key in zip((1, 2), keys[1:], strict=True)]
    assert all(rows[IDS_KEY].shape == (4, PROMPT_WIDTH + RESPONSE_WIDTH) for rows in rolled)
