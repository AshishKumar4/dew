"""GRPO: the clipped-surrogate-plus-KL composition against verl.

The loss must match verl 0.9's PPO path with the GRPO settings
(`compute_policy_loss_vanilla` with `clip_ratio` 0.2 both sides,
`clip_ratio_c` 3.0 and `token-mean`, plus `beta` times the `token-mean` k3
from `kl_penalty_forward`). The reference is tests/fixtures/rl/grpo.npz,
written by tools/parity_grpo.py from torch over one fixed rollout: old,
current and reference log-probabilities, both-signed advantages, and a mask
with short tails, with entries past every clip point. `GRPOObjective` reads
the same terms out of the rolled-out batch. Each mutation below removes one
term and must move the loss, proving the term binds.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY
from dew.objectives.base import Step
from dew.sampling import Sampling
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.rollout import (ADVANTAGES_KEY, IDS_KEY, OLD_LOG_PROBS_KEY,
                                       RESPONSE_MASK_KEY, SampledRollout)
from dew.rl import clipped_surrogate, k3_kl, token_log_ratio, token_mean

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
    get = lambda key: np.asarray(fixture[key], np.float32)
    return (get("old_log_probs"), get("current_log_probs"), get("ref_log_probs"),
            get("advantages"), get("response_mask"), float(fixture["beta"]))


def test_grpo_loss_matches_verl(reference):
    """Dew's composition against verl 0.9's PPO path on the fixed rollout.
    Largest observed difference: 7.45e-08."""
    old, current, ref, advantages, mask, beta = terms(reference)

    ratio = token_log_ratio(jnp.asarray(current), jnp.asarray(old))
    pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
    kl = token_mean(k3_kl(jnp.asarray(current), jnp.asarray(ref)), jnp.asarray(mask))
    loss = pg + beta * kl

    difference = abs(float(loss) - float(reference["verl_loss"]))
    assert difference < 1e-6, f"largest difference against verl: {difference}"


def test_grpo_gradients_match_verl(reference):
    """Autograd on both sides over the current log-probabilities. Largest
    observed difference: exact, 0.0 across the 16 entries."""
    old, current, ref, advantages, mask, beta = terms(reference)

    def loss(current):
        ratio = token_log_ratio(current, jnp.asarray(old))
        pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
        kl = token_mean(k3_kl(current, jnp.asarray(ref)), jnp.asarray(mask))
        return pg + beta * kl

    grads = jax.grad(loss)(jnp.asarray(current))
    expected = np.asarray(reference["verl_current_grad"], np.float32)
    difference = float(np.abs(np.asarray(grads) - expected).max())
    assert difference < 1e-6, f"largest difference against verl: {difference}"


def test_the_fixture_names_its_reference(reference):
    assert str(reference["verl_version"]) == "0.9.0"
    assert str(reference["torch_version"]).startswith("2.14")
    assert float(reference["beta"]) == 0.01
    assert np.asarray(reference["response_mask"]).shape == (4, 4)
    assert np.asarray(reference["response_mask"]).sum() == 13


def test_an_unclipped_ratio_moves_the_loss(reference):
    """Without the 1 +- 0.2 clip the high-ratio entries dominate: observed
    move 0.154."""
    old, current, ref, advantages, mask, beta = terms(reference)

    raw = token_mean(-jnp.asarray(advantages) * jnp.exp(
        jnp.asarray(current) - jnp.asarray(old)), jnp.asarray(mask))
    kl = token_mean(k3_kl(jnp.asarray(current), jnp.asarray(ref)), jnp.asarray(mask))

    assert abs(float(raw + beta * kl) - float(reference["verl_loss"])) > 1e-4


def test_a_k1_penalty_moves_the_loss(reference):
    """The k1 estimator prices drift linearly instead of exponentially:
    observed move 0.0051 on the KL term, before the beta dilution."""
    _, current, ref, _, mask, _ = terms(reference)

    cubic = token_mean(k3_kl(jnp.asarray(current), jnp.asarray(ref)), jnp.asarray(mask))
    linear = token_mean(jnp.asarray(current) - jnp.asarray(ref), jnp.asarray(mask))

    assert abs(float(linear) - float(cubic)) > 1e-4


def test_a_flat_mean_moves_the_loss(reference):
    """Averaging over the whole rectangle instead of the masked tokens
    dilutes the short tails: observed move 0.082."""
    old, current, ref, advantages, mask, beta = terms(reference)
    ratio = token_log_ratio(jnp.asarray(current), jnp.asarray(old))
    pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
    kl = token_mean(k3_kl(jnp.asarray(current), jnp.asarray(ref)), jnp.asarray(mask))
    flat = jnp.mean(jnp.asarray(mask) * -jnp.asarray(advantages) * jnp.exp(ratio))

    assert abs(float(flat + beta * kl) - float(reference["verl_loss"])) > 1e-4


# --- the objective -------------------------------------------------------------

class TinyHead(nn.Module):
    """A position-wise map with the backbone's scoring contract: int32 ids
    in, float32 logits out, the head split off behind `hidden_states` and
    `head_weight`."""

    vocab_size: int
    final_logit_softcap = None
    precision = None

    def setup(self):
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    @nn.compact
    def hidden_states(self, tokens, train: bool = False):
        x = nn.Embed(self.vocab_size, 8)(tokens)
        h = nn.LayerNorm()(x)
        return nn.LayerNorm()(x + nn.Dense(8)(nn.gelu(nn.Dense(16)(h))))

    @nn.compact
    def init_cache(self, batch_size):
        """A placeholder cache: the trunk mixes nothing across positions, so
        incremental decoding keeps no state, but `generate` threads one."""
        self.variable("cache", "index", lambda: jnp.zeros((batch_size,), jnp.int32))

    def __call__(self, tokens, train: bool = False, decode: bool = False):
        return self.lm_head(
            self.hidden_states(tokens, train=train)).astype(jnp.float32)

    def head_weight(self, params):
        return params["lm_head"]["kernel"].astype(jnp.float32)


def rollout_batch(seed=0):
    """Two full concatenations with response-width terms, as the
    `SampledRollout` packs them."""
    rng = np.random.RandomState(seed)
    width = PROMPT_WIDTH + RESPONSE_WIDTH
    ids = rng.randint(0, VOCAB, (ROWS, width)).astype(np.int32)
    terms = rng.normal(-0.5, 0.5, (ROWS, RESPONSE_WIDTH)).astype(np.float32)
    advantages = rng.normal(0, 1, (ROWS, RESPONSE_WIDTH)).astype(np.float32)
    mask = np.ones((ROWS, RESPONSE_WIDTH), np.float32)
    mask[1, 2:] = 0
    return {
        IDS_KEY: jnp.asarray(ids),
        OLD_LOG_PROBS_KEY: jnp.asarray(terms),
        ADVANTAGES_KEY: jnp.asarray(advantages),
        RESPONSE_MASK_KEY: jnp.asarray(mask),
    }


def test_the_loss_reads_the_rolled_out_batch():
    """The objective's loss is the composition over the batch's own terms:
    current log-probabilities sliced out of the concatenation, ratio against
    the stored old ones, surrogate and KL masked alike."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    batch = rollout_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)

    loss, aux = objective.loss(params, batch, step)

    ids = np.asarray(batch[IDS_KEY])
    start = PROMPT_WIDTH - 1
    policy = np.asarray(objective.per_token_log_probs(params, ids))[:, start:start + 3]
    ref = np.asarray(objective.per_token_log_probs(frozen, ids))[:, start:start + 3]
    old = np.asarray(batch[OLD_LOG_PROBS_KEY])
    advantages = np.asarray(batch[ADVANTAGES_KEY])
    mask = np.asarray(batch[RESPONSE_MASK_KEY])
    ratio = token_log_ratio(jnp.asarray(policy), jnp.asarray(old))
    pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
    kl = token_mean(k3_kl(jnp.asarray(policy), jnp.asarray(ref)), jnp.asarray(mask))
    assert float(loss) == pytest.approx(float(pg + 0.01 * kl), rel=1e-5)
    assert set(aux.metrics) == {"pg", "actor/pg_clipfrac", "actor/ppo_kl",
                                "actor/pg_clipfrac_lower", "kl"}


def test_zero_beta_leaves_the_reference_unread():
    """With the KL term off, no frozen tree is needed and none is reported."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.0)
    params = objective.init(jax.random.key(0))
    batch = rollout_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)

    loss, aux = objective.loss(params, batch, step)

    assert np.isfinite(float(loss))
    assert "kl" not in aux.metrics


def test_a_positive_beta_needs_the_frozen_tree():
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    with pytest.raises(ValueError, match="step.ema"):
        objective.loss(params, rollout_batch(), step)


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

    bare = {IDS_KEY: batch[IDS_KEY]}
    with pytest.raises(ValueError, match="old_log_probs"):
        objective.loss(params, bare, step)

    narrow = dict(batch, **{IDS_KEY: batch[IDS_KEY][:, :5]})
    with pytest.raises(ValueError, match="8 ids per row"):
        objective.loss(params, narrow, step)

    wide = dict(batch, **{OLD_LOG_PROBS_KEY: jnp.zeros((ROWS, 9), jnp.float32),
                          ADVANTAGES_KEY: jnp.zeros((ROWS, 9), jnp.float32),
                          RESPONSE_MASK_KEY: jnp.zeros((ROWS, 9), jnp.float32)})
    with pytest.raises(ValueError, match="concatenation"):
        objective.loss(params, wide, step)

    ragged = dict(batch, **{ADVANTAGES_KEY: jnp.zeros((ROWS, 2), jnp.float32)})
    with pytest.raises(ValueError, match="one term per response token"):
        objective.loss(params, ragged, step)


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

    with pytest.raises(ValueError, match="prompt"):
        objective.evaluate(params, {}, step)
    with pytest.raises(ValueError, match="two tokens"):
        objective.evaluate(params, {PROMPT_KEY: jnp.ones((1, 1), jnp.int32),
                                    LENGTH_KEY: jnp.ones(1, jnp.int32)}, step)


def test_the_rollout_batch_feeds_the_objective():
    """The `SampledRollout` pack and the `GRPOObjective` loss agree on every
    key: a rollout straight into the loss, shapes fixed, loss finite."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    width = PROMPT_WIDTH
    prompts = np.tile(np.arange(1, width + 1, dtype=np.int32), (2, 1))
    info = max(len("rule"), len("other"))
    pad = lambda text: np.pad(
        np.frombuffer(text.encode(), np.uint8).astype(np.int32), (0, info - len(text)))
    batch = {
        PROMPT_KEY: prompts,
        LENGTH_KEY: np.full(2, width, np.int32),
        SOURCE_KEY: np.stack([pad("rule"), pad("other")]),
        TRUTH_KEY: np.stack([pad("1"), pad("2")]),
        INFO_KEY: np.stack([pad(""), pad("")]),
    }
    state = SimpleNamespace(params=params)
    rollout = SampledRollout(objective, lambda *args: 1.0, groups=2,
                             max_new_tokens=RESPONSE_WIDTH, sampling=Sampling(temperature=0.0))
    rolled = rollout(state, batch, jax.random.key(1))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)

    loss, aux = objective.loss(params, rolled, step)

    assert np.isfinite(float(loss))
    assert rolled[IDS_KEY].shape == (4, width + RESPONSE_WIDTH)
    assert set(aux.metrics) >= {"pg", "kl"}
