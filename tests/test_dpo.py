"""DPO: the preference loss against TRL, and the objective around it.

`preference_logsigmoid` must match TRL 1.12's DPO path with the defaults
(`sigmoid` loss, `reverse_kl`): the same per-token terms summed under the
shifted completion mask, the `[chosen, rejected]` chunking, and
`mean(-logsigmoid(beta * delta))`. The reference is
tests/fixtures/rl/dpo.npz, written by tools/parity_dpo.py from torch
autograd over fixed tensors. `DPOObjective` composes that term with the
chunked head's per-token log-probabilities, reading the reference from the
frozen `step.ema`, and `PreferencePairs` stacks the batches it trains on.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.data import Loading, PreferencePairs
from dew.data.preferences import IDS_KEY, MASK_KEY, PreferenceSource
from dew.objectives.base import Step
from dew.objectives.rl import DPOObjective
from dew.rl import preference_logsigmoid

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "dpo.npz"
VOCAB = 8
PAIRS = 2
WIDTH = 6


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE, allow_pickle=True))


def halves(fixture):
    """The fixture's TRL-layout stack as chosen/rejected halves with the mask
    shifted the way both implementations read it."""
    policy = np.asarray(fixture["policy_logps"], np.float32)
    ref = np.asarray(fixture["ref_logps"], np.float32)
    mask = np.asarray(fixture["completion_mask"], np.float32)[:, 1:]
    half = policy.shape[0] // 2
    beta = float(fixture["beta"])
    return (policy[:half], policy[half:], ref[:half], ref[half:],
            mask[:half], mask[half:], beta)


def test_dpo_loss_matches_trl(reference):
    """Dew's term against TRL 1.12's `dpo_loss` on the same fixed tensors.
    Largest observed difference: 5.96e-08."""
    policy_c, policy_r, ref_c, ref_r, mask_c, mask_r, beta = halves(reference)

    loss = preference_logsigmoid(
        jnp.asarray(policy_c), jnp.asarray(policy_r), jnp.asarray(ref_c),
        jnp.asarray(ref_r), jnp.asarray(mask_c), jnp.asarray(mask_r), beta)

    difference = abs(float(loss) - float(reference["trl_loss"]))
    assert difference < 1e-6, f"largest difference against TRL: {difference}"


def test_dpo_gradients_match_trl(reference):
    """Autograd on both sides over the four per-token tensors. Largest
    observed difference: exact, 0.0 across the 120 entries."""
    policy_c, policy_r, ref_c, ref_r, mask_c, mask_r, beta = halves(reference)
    args = [jnp.asarray(a) for a in (policy_c, policy_r, ref_c, ref_r)]
    masks = [jnp.asarray(a) for a in (mask_c, mask_r)]

    def loss(policy_c, policy_r, ref_c, ref_r):
        return preference_logsigmoid(policy_c, policy_r, ref_c, ref_r,
                                     masks[0], masks[1], beta)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*args)
    trl_policy = np.asarray(reference["trl_policy_grad"], np.float32)
    trl_ref = np.asarray(reference["trl_ref_grad"], np.float32)
    half = trl_policy.shape[0] // 2
    expected = [trl_policy[:half], trl_policy[half:], trl_ref[:half], trl_ref[half:]]
    difference = max(float(np.abs(np.asarray(g) - e).max())
                     for g, e in zip(grads, expected, strict=True))
    assert difference < 1e-6, f"largest difference against TRL: {difference}"


def test_the_fixture_names_its_reference(reference):
    assert str(reference["trl_version"]) == "1.12.0"
    assert str(reference["torch_version"]).startswith("2.14")
    assert float(reference["beta"]) == 0.1
    assert np.asarray(reference["completion_mask"]).shape == (6, 6)


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

    def __call__(self, tokens, train: bool = False):
        return self.lm_head(
            self.hidden_states(tokens, train=train)).astype(jnp.float32)

    def head_weight(self, params):
        return params["lm_head"]["kernel"].astype(jnp.float32)


def pair_batch(seed=0):
    """Two pairs of full-length rows, `[PAIRS, 2, WIDTH]`, with prompt
    prefixes under the mask."""
    rng = np.random.RandomState(seed)
    ids = rng.randint(0, VOCAB, (PAIRS, 2, WIDTH)).astype(np.int32)
    mask = np.ones((PAIRS, 2, WIDTH), np.int32)
    mask[:, :, :2] = 0
    mask[0, 1, 4:] = 0
    return {IDS_KEY: jnp.asarray(ids), MASK_KEY: jnp.asarray(mask)}


def flat(batch):
    """A `[B, 2, S]` pair batch as TRL's `[2B, S]` stack with the shifted
    mask, indexed pair by pair, the layout both implementations share below
    the objective."""
    ids = np.asarray(batch[IDS_KEY])
    mask = np.asarray(batch[MASK_KEY])[:, :, 1:]
    return (ids[:, 0], ids[:, 1], mask[:, 0], mask[:, 1])


def test_the_loss_composes_the_term_over_head_log_probs():
    """The objective's loss is the preference term over the chunked head's
    per-token log-probabilities, policy from the live params and reference
    from the frozen tree."""
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.5)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    batch = pair_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)

    loss, aux = objective.loss(params, batch, step)

    chosen_ids, rejected_ids, chosen_mask, rejected_mask = flat(batch)
    stack = np.concatenate([chosen_ids, rejected_ids])
    policy = np.asarray(objective.per_token_log_probs(params, stack))
    ref = np.asarray(objective.per_token_log_probs(frozen, stack))
    half = PAIRS
    expected = preference_logsigmoid(
        jnp.asarray(policy[:half]), jnp.asarray(policy[half:]),
        jnp.asarray(ref[:half]), jnp.asarray(ref[half:]),
        jnp.asarray(chosen_mask), jnp.asarray(rejected_mask), 0.5)
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)
    assert set(aux.metrics) == {"rewards/chosen", "rewards/rejected", "accuracy"}
    assert 0.0 <= float(aux.metrics["accuracy"]) <= 1.0


def test_the_reference_comes_from_the_frozen_tree():
    """Perturbing the live params moves the policy half of the loss; moving
    the frozen tree moves the reference half."""
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.5)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    moved = jax.tree.map(lambda leaf: leaf + 1.0, frozen)
    batch = pair_batch()

    base, _ = objective.loss(
        params, batch, Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen))
    live_moved, _ = objective.loss(
        moved, batch, Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen))
    ref_moved, _ = objective.loss(
        params, batch, Step(step=jnp.asarray(0), key=jax.random.key(1), ema=moved))

    assert abs(float(live_moved) - float(base)) > 1e-3
    assert abs(float(ref_moved) - float(base)) > 1e-3


def test_swapped_halves_change_the_loss():
    """Chosen over rejected is a direction, not a bag: exchanging the halves
    must move the loss, guarding the pair order. The policy starts one step
    off the reference so the deltas are not all zero."""
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.5)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    moved = jax.tree.map(lambda leaf: leaf + 1.0, frozen)
    batch = pair_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)

    loss, _ = objective.loss(moved, batch, step)
    swapped = {IDS_KEY: batch[IDS_KEY][:, ::-1, :],
               MASK_KEY: batch[MASK_KEY][:, ::-1, :]}
    flipped, _ = objective.loss(moved, swapped, step)

    assert abs(float(flipped) - float(loss)) > 1e-4


def test_evaluation_scores_chosen_perplexity():
    """Validation reads the chosen responses' cross entropies under the
    shifted completion mask."""
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.5)
    params = objective.init(jax.random.key(0))
    batch = pair_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)

    scores = objective.evaluate(params, batch, step)

    chosen_ids, _, chosen_mask, _ = flat(batch)
    expected = np.asarray(objective.per_token_log_probs(params, chosen_ids))
    np.testing.assert_allclose(np.asarray(scores.losses), -expected, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(scores.weights), chosen_mask)


def test_a_misbuilt_objective_is_refused():
    with pytest.raises(ValueError, match="positive"):
        DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.0)
    with pytest.raises(ValueError, match="unit decay"):
        DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, ema_decay=0.999)
    with pytest.raises(ValueError, match="completion mask"):
        DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, loss_role="x")


def test_a_misshapen_batch_is_refused():
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1)
    params = objective.init(jax.random.key(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)
    batch = pair_batch()

    flat_batch = {IDS_KEY: batch[IDS_KEY].reshape(-1, WIDTH),
                  MASK_KEY: batch[MASK_KEY].reshape(-1, WIDTH)}
    with pytest.raises(ValueError, match="holds pairs"):
        objective.loss(params, flat_batch, step)

    short = {IDS_KEY: batch[IDS_KEY][:, :, :4], MASK_KEY: batch[MASK_KEY][:, :, :4]}
    with pytest.raises(ValueError, match="6 ids per row"):
        objective.loss(params, short, step)

    wide = {IDS_KEY: batch[IDS_KEY], MASK_KEY: batch[MASK_KEY][:, :, :4]}
    with pytest.raises(ValueError, match="one mark per token"):
        objective.loss(params, wide, step)

    bare = {IDS_KEY: batch[IDS_KEY]}
    with pytest.raises(ValueError, match="completion_mask"):
        objective.loss(params, bare, step)

    no_ref = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    with pytest.raises(ValueError, match="step.ema"):
        objective.loss(params, batch, no_ref)


# --- the source ------------------------------------------------------------------

def records(*rows):
    return tuple(json.dumps(row) for row in rows)


PAIR = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5],
        "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]}


def test_pairs_stack_chosen_over_rejected():
    source = PreferenceSource.from_records(records(PAIR), 0, 4)

    batch = source[0]

    np.testing.assert_array_equal(batch[IDS_KEY], [[1, 2, 3, 4], [1, 2, 5, 0]])
    np.testing.assert_array_equal(batch[MASK_KEY], [[0, 0, 1, 1], [0, 0, 1, 0]])


def test_missing_masks_default_to_all_completion():
    row = {"chosen": [1, 2], "rejected": [3]}
    source = PreferenceSource.from_records(records(row), 0, 4)

    batch = source[0]

    np.testing.assert_array_equal(batch[MASK_KEY], [[1, 1, 0, 0], [1, 0, 0, 0]])

def test_a_ragged_pair_is_refused():
    row = dict(PAIR, chosen_mask=[0, 1, 1])
    with pytest.raises(ValueError, match="as long as its ids"):
        PreferenceSource.from_records(records(row), 0, 4)[0]
    row = dict(PAIR, chosen=[1, "x"])
    with pytest.raises(ValueError, match="token ids"):
        PreferenceSource.from_records(records(row), 0, 4)[0]
    row = dict(PAIR, rejected_mask=[0, 0, 2])
    with pytest.raises(ValueError, match="0/1"):
        PreferenceSource.from_records(records(row), 0, 4)[0]


def test_an_incomplete_row_is_refused():
    with pytest.raises(ValueError, match="both chosen and rejected"):
        PreferenceSource.from_records(records({"chosen": [1]}), 0, 4)
    with pytest.raises(ValueError, match="unknown fields"):
        PreferenceSource.from_records(records(dict(PAIR, reward=1.0)), 0, 4)
    with pytest.raises(ValueError, match="an object"):
        PreferenceSource.from_records(records([1, 2]), 0, 4)[0]
    with pytest.raises(ValueError, match="no pairs"):
        PreferenceSource.from_records((), 0, 4)


def test_parquet_pairs_batch_in_pairs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({
        "chosen": [[1, 2, 3, 4]], "rejected": [[1, 2, 5]],
        "chosen_mask": [[0, 0, 1, 1]], "rejected_mask": [[0, 0, 1]],
    }), path)

    data = PreferencePairs(path=str(path), seq_len=4,
                           loading=Loading(workers=0)).load(batch=1)

    assert data.records == 1
    batch = next(data.train())
    np.testing.assert_array_equal(
        np.asarray(batch[IDS_KEY]), [[[1, 2, 3, 4], [1, 2, 5, 0]]])
    np.testing.assert_array_equal(
        np.asarray(batch[MASK_KEY]), [[[0, 0, 1, 1], [0, 0, 1, 0]]])

def test_varied_rows_pad_to_one_window():
    """Two pairs of different lengths batch without ragged edges: the
    grain batcher stacks static shapes, so every row is already `[2,
    seq_len]`."""
    rows = records(
        {"chosen": [1, 2, 3, 4], "rejected": [5]},
        {"chosen": [6], "rejected": [7, 8]})
    data = PreferencePairs(records=rows, seq_len=4,
                           loading=Loading(workers=0)).load(batch=2)

    batch = next(data.train())

    ids = np.asarray(batch[IDS_KEY])
    assert ids.shape == (2, 2, 4)
    order = np.argsort(-ids[:, 0, :].sum(-1))
    np.testing.assert_array_equal(ids[order][:, 0, :], [[1, 2, 3, 4], [6, 0, 0, 0]])
    np.testing.assert_array_equal(np.asarray(batch[MASK_KEY])[order][:, 1, :],
                                  [[1, 0, 0, 0], [1, 1, 0, 0]])

def test_an_overlong_row_is_refused():
    row = {"chosen": [1, 2, 3, 4, 5], "rejected": [1]}
    with pytest.raises(ValueError, match="shorten the row"):
        PreferenceSource.from_records(records(row), 0, 4)
    with pytest.raises(ValueError, match="seq_len"):
        PreferenceSource.from_records(records(PAIR), 0, 0)


def test_parquet_without_sides_is_refused(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "nopairs.parquet"
    pq.write_table(pa.table({"chosen": [[1]]}), path)
    with pytest.raises(ValueError, match="rejected column is required"):
        PreferencePairs(path=str(path)).load(batch=1)


def test_the_dataset_reads_one_source():
    with pytest.raises(ValueError, match="one source"):
        PreferencePairs().load(batch=2)
