"""DPO: the preference loss against TRL, and the objective around it.

`preference_logsigmoid_terms`, averaged over pairs as `DPOObjective` does,
must match TRL 1.12's DPO path with the defaults (`sigmoid` loss,
`reverse_kl`): the reference is tests/fixtures/rl/dpo.npz, which
tools/parity_dpo.py writes by running TRL's own `DPOTrainer._compute_loss`
on fixed logits. `DPOObjective` composes that term with the
chunked head's per-token log-probabilities, reading the reference from the
frozen `step.ema`, and `Preferences` stacks the batches it trains on.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording
from rl_support import TinyHead
from steady_state import steady_state
from tool_support import load

from dew.data import Files, Loading, Memory, presets
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition
from dew.data.preferences import IDS_KEY, MASK_KEY
from dew.objectives.base import Step
from dew.objectives.rl import DPOObjective


def test_preference_rewards_and_gradients_ignore_nonfinite_completion_padding():
    """Policy and reference likelihoods outside each completion carry no preference evidence,
    even when those positions are nonfinite."""
    from dew.rl.surrogate import preference_logsigmoid_terms

    values = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3) / -8
    chosen = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rejected = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    masks = jnp.stack([chosen, rejected, chosen, rejected])

    def loss(log_probs):
        terms, rewards = preference_logsigmoid_terms(*log_probs, chosen, rejected, beta=0.1)
        return jnp.sum(terms), rewards

    evaluate = jax.value_and_grad(loss, has_aux=True)
    clean = evaluate(values)
    poisoned = evaluate(jnp.where(masks != 0, values, jnp.nan))
    for observed, expected in zip(jax.tree.leaves(poisoned), jax.tree.leaves(clean), strict=True):
        np.testing.assert_array_equal(observed, expected)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "dpo.npz"
VOCAB = 8
PAIRS = 2
WIDTH = 6


@pytest.fixture(scope="module")
def reference():
    return recording(FIXTURE)


def test_dpo_loss_rewards_and_gradient_match_trl(reference):
    """Dew against TRL 1.12.0's own `DPOTrainer._compute_loss` with its
    default settings (tools/parity_dpo.py), on par with TRL's fp32 over the
    recorded policy and reference logits (check 1): the loss, its gradient in
    the policy logits and the chosen and rejected rewards, TRL's float64 run
    the truth."""
    seeds = declared_seeds(FIXTURE)["dpo"]
    draw, recorded, truth = indexed_recording(reference, "dpo", seeds)
    # Dew's composition over the TRL-layout batch, the one the recorder planned its seeds by: token
    # log-probabilities off the shifted logits and `preference_logsigmoid_terms` over the halves.
    composed = load("parity_dpo").proxy

    @jax.jit
    def dew(index, arrays):
        return composed(arrays)

    assert_as_exact(dew, recorded, draw, "DPO loss, gradient and rewards", seeds=seeds, truth_fn=truth)


# --- the objective -------------------------------------------------------------

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


def test_rewards_measure_reference_relative_improvement_on_unequal_pairs():
    model = TinyHead(vocab_size=VOCAB)
    objective = DPOObjective(model, WIDTH - 1, beta=.5)
    reference = objective.init(jax.random.key(0))
    params = jax.tree.map(lambda x: x + .1, reference)
    batch = pair_batch()
    loss, aux = objective.scalar_loss(params, batch, Step(jnp.array(0), jax.random.key(1), reference))
    chosen, rejected, chosen_mask, rejected_mask = flat(batch)
    def score(variables, ids, mask):
        logits = model.apply(variables, jnp.asarray(ids[:, :-1]))
        log_probs = jax.nn.log_softmax(logits)
        selected = np.take_along_axis(np.asarray(log_probs), ids[:, 1:, None], axis=-1)[..., 0]
        return (selected * mask).sum(axis=-1)
    rewards_chosen = .5 * (score(params, chosen, chosen_mask) - score(reference, chosen, chosen_mask))
    rewards_rejected = 0.5 * (
        score(params, rejected, rejected_mask) - score(reference, rejected, rejected_mask)
    )
    np.testing.assert_allclose(aux.metrics["rewards/chosen"], rewards_chosen.mean(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(aux.metrics["rewards/rejected"], rewards_rejected.mean(), rtol=1e-5, atol=1e-6)
    assert float(aux.metrics["accuracy"]) == float((rewards_chosen > rewards_rejected).mean())
    np.testing.assert_allclose(loss, np.logaddexp(0, rewards_rejected - rewards_chosen).mean(), rtol=1e-6)


def test_the_reference_comes_from_the_frozen_tree():
    """Perturbing the live params moves the policy half of the loss; moving
    the frozen tree moves the reference half."""
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.5)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    moved = jax.tree.map(lambda leaf: leaf + 1.0, frozen)
    batch = pair_batch()

    base, _ = objective.scalar_loss(
        params, batch, Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)
    )
    live_moved, _ = objective.scalar_loss(
        moved, batch, Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)
    )
    ref_moved, _ = objective.scalar_loss(
        params, batch, Step(step=jnp.asarray(0), key=jax.random.key(1), ema=moved)
    )

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

    loss, _ = objective.scalar_loss(moved, batch, step)
    swapped = {IDS_KEY: batch[IDS_KEY][:, ::-1, :],
               MASK_KEY: batch[MASK_KEY][:, ::-1, :]}
    flipped, _ = objective.scalar_loss(moved, swapped, step)

    assert abs(float(flipped) - float(loss)) > 1e-4


def test_evaluation_scores_chosen_perplexity():
    """Validation reads the chosen responses' cross entropies under the
    shifted completion mask, compiled once for the objective's modules."""
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.5)
    params = objective.init(jax.random.key(0))
    batch = pair_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)

    scores = objective.evaluate(params, batch, step)
    with steady_state(allow=("host_to_device",)):
        objective.evaluate(params, batch, step)

    chosen_ids, _, chosen_mask, _ = flat(batch)
    expected = np.asarray(objective.per_token_log_probs(params, chosen_ids))
    np.testing.assert_allclose(np.asarray(scores.losses), -expected, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(scores.weights), chosen_mask)
    logits = objective.model.apply(params, chosen_ids[:, :-1], train=False)
    np.testing.assert_array_equal(np.asarray(scores.correct),
                                  np.asarray(jnp.argmax(logits, axis=-1) == chosen_ids[:, 1:]))


def test_a_misbuilt_objective_is_refused():
    with pytest.raises(ValueError):
        DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.0)


def test_a_misshapen_batch_is_refused():
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1)
    params = objective.init(jax.random.key(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)
    batch = pair_batch()

    flat_batch = {IDS_KEY: batch[IDS_KEY].reshape(-1, WIDTH),
                  MASK_KEY: batch[MASK_KEY].reshape(-1, WIDTH)}
    with pytest.raises(ValueError):
        objective.scalar_loss(params, flat_batch, step)

    short = {IDS_KEY: batch[IDS_KEY][:, :, :4], MASK_KEY: batch[MASK_KEY][:, :, :4]}
    with pytest.raises(ValueError):
        objective.scalar_loss(params, short, step)

    wide = {IDS_KEY: batch[IDS_KEY], MASK_KEY: batch[MASK_KEY][:, :, :4]}
    with pytest.raises(ValueError):
        objective.scalar_loss(params, wide, step)

    no_ref = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    with pytest.raises(ValueError):
        objective.scalar_loss(params, batch, no_ref)


# --- the source ------------------------------------------------------------------

def records(*rows):
    return tuple(json.dumps(row) for row in rows)


def preference_rows(rows, pad_id, width):
    pairs = Memory([json.loads(row) for row in rows]).pipe(presets.preferences, seq_len=width, batch=None,
                                                           pad_id=pad_id)
    return list(iterate(pairs, read="memory", loading=Loading(workers=0)))


PAIR = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5],
        "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]}


def test_a_preference_dataset_trains_as_the_preference_config_does():
    """`Trainer.fit` compiles a Dataset with the objective's own stages.
    DPO adds none, so one step over `preferences.PreferencePair` rows moves the policy
    exactly as one step over the same pairs through the preferences preset."""
    import optax
    from bank_support import identical

    from dew.data import preferences
    from dew.training import Layout, Trainer

    pairs = [PAIR] * 8

    def trained(data):
        trainer = Trainer(DPOObjective(TinyHead(vocab_size=8), 3), optax.sgd(0.1), key=jax.random.key(0),
                          layout=Layout(min_shard=1, tolerance=1.0))
        return trainer.fit(data, steps=1).variables

    identical(trained(Memory(pairs).map(preferences.PreferencePair(4)).batch(8)),
              trained(compile(Memory(pairs).pipe(presets.preferences, seq_len=4, batch=8),
                              read="memory", loading=Loading(workers=0))))


def test_a_ragged_pair_is_refused():
    row = dict(PAIR, chosen_mask=[0, 1, 1])
    with pytest.raises(ValueError):
        preference_rows(records(row), 0, 4)[0]
    row = dict(PAIR, chosen=[1, "x"])
    with pytest.raises(ValueError):
        preference_rows(records(row), 0, 4)[0]
    row = dict(PAIR, rejected_mask=[0, 0, 2])
    with pytest.raises(ValueError):
        preference_rows(records(row), 0, 4)[0]


def test_an_incomplete_row_is_refused():
    with pytest.raises(ValueError):
        preference_rows(records({"chosen": [1]}), 0, 4)
    with pytest.raises(ValueError):
        preference_rows(records([1, 2]), 0, 4)[0]
    with pytest.raises(ValueError):
        compile(Memory([]).pipe(presets.preferences, seq_len=4, batch=1), read="memory",
                loading=Loading(workers=0))


def test_unread_preference_fields_pass_through():
    row = preference_rows(records(dict(PAIR, reward=1.0)), 0, 4)[0]
    assert row["reward"] == 1.0
    np.testing.assert_array_equal(row[IDS_KEY], [[1, 2, 3, 4], [1, 2, 5, 0]])
    np.testing.assert_array_equal(row[MASK_KEY], [[0, 0, 1, 1], [0, 0, 1, 0]])


def test_parquet_pairs_batch_in_pairs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({
        "chosen": [[1, 2, 3, 4]], "rejected": [[1, 2, 5]],
        "chosen_mask": [[0, 0, 1, 1]], "rejected_mask": [[0, 0, 1]],
    }), path)

    data = compile(Files(str(path), format="parquet").pipe(presets.preferences, seq_len=4, batch=1),
                   read="memory", loading=Loading(workers=0))

    assert data.records == 1
    batch = next(data.train(DataPartition()))
    np.testing.assert_array_equal(
        np.asarray(batch[IDS_KEY]), [[[1, 2, 3, 4], [1, 2, 5, 0]]])
    np.testing.assert_array_equal(
        np.asarray(batch[MASK_KEY]), [[[0, 0, 1, 1], [0, 0, 1, 0]]])

@pytest.mark.parametrize("mask_case", ["varied", "explicit", "default"])
def test_varied_rows_pad_to_one_window(mask_case):
    """Two pairs of different lengths batch without ragged edges: the
    grain batcher stacks static shapes, so every row is already `[2,
    seq_len]`."""
    rows = records(
        {"chosen": [1, 2, 3, 4], "rejected": [5]},
        {"chosen": [6], "rejected": [7, 8]})
    if mask_case == "explicit":
        rows = records(PAIR)
    elif mask_case == "default":
        rows = records({"chosen": [1, 2], "rejected": [3]})
    batch_size = 2 if mask_case == "varied" else 1
    data = compile(Memory([json.loads(row) for row in rows]).pipe(presets.preferences, seq_len=4,
                                                                  batch=batch_size),
                   read="memory", loading=Loading(workers=0))

    batch = next(data.train(DataPartition()))

    ids = np.asarray(batch[IDS_KEY])
    if mask_case == "explicit":
        np.testing.assert_array_equal(ids[0], [[1, 2, 3, 4], [1, 2, 5, 0]])
        np.testing.assert_array_equal(batch[MASK_KEY][0], [[0, 0, 1, 1], [0, 0, 1, 0]])
        return
    if mask_case == "default":
        np.testing.assert_array_equal(batch[MASK_KEY][0], [[1, 1, 0, 0], [1, 0, 0, 0]])
        return
    assert ids.shape == (2, 2, 4)
    order = np.argsort(-ids[:, 0, :].sum(-1))
    np.testing.assert_array_equal(ids[order][:, 0, :], [[1, 2, 3, 4], [6, 0, 0, 0]])
    np.testing.assert_array_equal(np.asarray(batch[MASK_KEY])[order][:, 1, :],
                                  [[1, 0, 0, 0], [1, 1, 0, 0]])

def test_an_overlong_row_is_refused():
    row = {"chosen": [1, 2, 3, 4, 5], "rejected": [1]}
    with pytest.raises(ValueError):
        preference_rows(records(row), 0, 4)
    with pytest.raises(ValueError):
        preference_rows(records(PAIR), 0, 0)


def test_parquet_without_sides_is_refused(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "nopairs.parquet"
    pq.write_table(pa.table({"chosen": [[1]]}), path)
    with pytest.raises(ValueError):
        next(compile(Files(str(path), format="parquet").pipe(presets.preferences, seq_len=4, batch=1),
                     read="memory").train(DataPartition()))
