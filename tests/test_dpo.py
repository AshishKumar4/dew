"""DPO: the preference loss against TRL, and the objective around it.

`preference_logsigmoid_terms`, averaged over pairs as `DPOObjective` does,
must match TRL 1.12's DPO path with the defaults (`sigmoid` loss,
`reverse_kl`): the reference is tests/fixtures/rl/dpo.npz, which
tools/parity_dpo.py writes by running TRL's own `DPOTrainer._compute_loss`
on fixed logits. `DPOObjective` composes that term with the
chunked head's per-token log-probabilities over windows of packed pairs, reading the reference from the
frozen `step.ema`, and `presets.preferences` packs the windows it trains on, a pair's shared prompt once.
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
from dew.data.preferences import HALVES_KEY, IDS_KEY, MASK_KEY, REAL_KEY, pair_windows, window_shape, windows
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

def pairs(seed=0):
    """Two pairs of full-length rows, `[PAIRS, 2, WIDTH]`, ids and completion masks, a shared prompt of two
    ids under the mask."""
    rng = np.random.RandomState(seed)
    ids = rng.randint(0, VOCAB, (PAIRS, 2, WIDTH)).astype(np.int32)
    ids[:, 1, :2] = ids[:, 0, :2]
    mask = np.ones((PAIRS, 2, WIDTH), np.int32)
    mask[:, :, :2] = 0
    mask[0, 1, 4:] = 0
    return ids, mask


def pair_batch(seed=0):
    """`pairs(seed)` as the windows a DPO run trains on."""
    return pair_windows(*pairs(seed))


def flat(held):
    """`[B, 2, S]` pairs as TRL's `[2B, S]` stack with the shifted mask, indexed pair by pair, the layout
    both implementations share below the objective."""
    ids, mask = held
    mask = mask[:, :, 1:]
    return (ids[:, 0], ids[:, 1], mask[:, 0], mask[:, 1])


def test_rewards_measure_reference_relative_improvement_on_unequal_pairs():
    model = TinyHead(vocab_size=VOCAB)
    objective = DPOObjective(model, WIDTH - 1, beta=.5)
    reference = objective.init(jax.random.key(0))
    params = jax.tree.map(lambda x: x + .1, reference)
    batch = pair_batch()
    loss, aux = objective.scalar_loss(params, batch, Step(jnp.array(0), jax.random.key(1), reference))
    chosen, rejected, chosen_mask, rejected_mask = flat(pairs())
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
    ids, mask = pairs()
    flipped, _ = objective.scalar_loss(moved, pair_windows(ids[:, ::-1], mask[:, ::-1]), step)

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

    # Each chosen completion token's score, pair by pair in the windows' order, as its own row scores it.
    chosen_ids, _, chosen_mask, _ = flat(pairs())
    expected = np.asarray(objective.per_token_log_probs(params, chosen_ids))
    counted = np.asarray(scores.weights) > 0
    assert counted.sum() == chosen_mask.sum()
    np.testing.assert_allclose(np.asarray(scores.losses)[counted], -expected[chosen_mask > 0], rtol=1e-5)
    logits = objective.model.apply(params, chosen_ids[:, :-1], train=False)
    right = np.asarray(jnp.argmax(logits, axis=-1) == chosen_ids[:, 1:])
    np.testing.assert_array_equal(np.asarray(scores.correct)[counted], right[chosen_mask > 0])


def test_a_misbuilt_objective_is_refused():
    with pytest.raises(ValueError):
        DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1, beta=0.0)


def test_a_misshapen_batch_is_refused():
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), WIDTH - 1)
    params = objective.init(jax.random.key(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=params)
    batch = pair_batch()

    ids, mask = pairs()
    with pytest.raises(ValueError):
        objective.scalar_loss(params, {IDS_KEY: ids, MASK_KEY: mask}, step)

    wide = {**batch, MASK_KEY: batch[MASK_KEY][:, :4]}
    with pytest.raises(ValueError):
        objective.scalar_loss(params, wide, step)

    no_ref = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    with pytest.raises(ValueError):
        objective.scalar_loss(params, batch, no_ref)


# --- the source ------------------------------------------------------------------

def records(*rows):
    return tuple(json.dumps(row) for row in rows)


def preference_rows(rows, width):
    laid = Memory([json.loads(row) for row in rows]).pipe(presets.preferences, seq_len=width, batch=None)
    return list(iterate(laid, read="memory", loading=Loading(workers=0)))


PAIR = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5],
        "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]}


def test_a_preference_dataset_trains_as_the_preference_config_does():
    """`Trainer.fit` compiles a Dataset with the objective's own stages.
    DPO adds none, so one step over the windows `preferences.windows` packs of the pairs moves the policy
    exactly as one step over the same pairs through the preferences preset."""
    import optax
    from bank_support import identical

    from dew.training import Layout, Trainer

    held = [PAIR] * 8

    def trained(data):
        trainer = Trainer(DPOObjective(TinyHead(vocab_size=8), 3), optax.sgd(0.1), key=jax.random.key(0),
                          layout=Layout(min_shard=1, tolerance=1.0))
        return trainer.fit(data, steps=1).variables

    # The preset's windows, as wide as its `seq_len` lays them (`window_shape`).
    laid = windows(held, 4, inputs=window_shape(4)[0])
    rows = [{key: value[at] for key, value in laid.items()} for at in range(len(laid[IDS_KEY]))]
    identical(trained(Memory(rows).batch(len(rows))),
              trained(compile(Memory(held).pipe(presets.preferences, seq_len=4, batch=len(rows)),
                              read="memory", loading=Loading(workers=0))))


def test_a_ragged_pair_is_refused():
    row = dict(PAIR, chosen_mask=[0, 1, 1])
    with pytest.raises(ValueError):
        preference_rows(records(row), 4)[0]
    row = dict(PAIR, chosen=[1, "x"])
    with pytest.raises(ValueError):
        preference_rows(records(row), 4)[0]
    row = dict(PAIR, rejected_mask=[0, 0, 2])
    with pytest.raises(ValueError):
        preference_rows(records(row), 4)[0]


def test_an_incomplete_row_is_refused():
    with pytest.raises(ValueError):
        preference_rows(records({"chosen": [1]}), 4)
    with pytest.raises(ValueError):
        preference_rows(records([1, 2]), 4)[0]
    with pytest.raises(ValueError):
        compile(Memory([]).pipe(presets.preferences, seq_len=4, batch=1), read="memory",
                loading=Loading(workers=0))


def test_unread_preference_fields_pass_through_the_pair_step():
    """The pair step keeps a record's fields it does not read beside its pair: a pair a row, its halves two
    rows padded with `pad_id`, or the pair's document its windows pack, its shared prompt once, then each
    half's rest."""
    from dew.data.preferences import PreferencePair

    row = PreferencePair(4, pad_id=9).map(dict(PAIR, reward=1.0))
    assert row["reward"] == 1.0
    np.testing.assert_array_equal(row[IDS_KEY], [[1, 2, 3, 4], [1, 2, 5, 9]])
    np.testing.assert_array_equal(row[REAL_KEY], [[1, 1, 1, 1], [1, 1, 1, 0]])
    document = PreferencePair(4, packed=True).map(dict(PAIR, reward=1.0))
    assert document["reward"] == 1.0
    np.testing.assert_array_equal(document["text"], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(document[MASK_KEY], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(document[HALVES_KEY], [0, 0, 1, 1, 2])


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
    # The prompt once and the chosen half's rest, its final and the rejected's in the tail.
    np.testing.assert_array_equal(np.asarray(batch[IDS_KEY]), [[1, 2, 3, 0, 0, 0, 4, 5]])
    np.testing.assert_array_equal(np.asarray(batch[MASK_KEY]), [[0, 0, 1, 0, 0, 0, 1, 1]])

@pytest.mark.parametrize("share", [True, False], ids=["shared", "whole"])
def test_pairs_pack_into_windows_each_half_at_its_own_places(share):
    """Pairs of different lengths pack first-fit into windows of two halves' tokens, several a window: each
    pair's halves read back, token for token and place for place, as their own rows lay them, its prompt
    once where the halves share it and twice where they are laid whole; a default mask marks every token a
    completion's, so nothing is shared."""
    from dew.data.preferences import PAIRS_KEY, POSITIONS_KEY

    rows = [{"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5], "chosen_mask": [0, 0, 1, 1],
             "rejected_mask": [0, 0, 1]},
            {"chosen": [6], "rejected": [7, 8]}, {"chosen": [1, 2], "rejected": [3]}]
    laid = list(iterate(Memory(rows).pipe(presets.preferences, seq_len=4, batch=None, share=share),
                        read="memory", loading=Loading(workers=0)))
    found = {}
    for batch in laid:
        ids, pair, half = (np.asarray(batch[key]) for key in (IDS_KEY, PAIRS_KEY, HALVES_KEY))
        places = np.asarray(batch[POSITIONS_KEY])
        for number in range(1, int(pair.max()) + 1):
            held = pair == number
            for side in (1, 2):
                own = held & ((half == side) | (half == 0))
                found.setdefault(tuple(ids[held]), []).append(
                    (list(ids[own][np.argsort(places[own], kind="stable")]), list(np.sort(places[own]))))
    halves = sorted((tuple(row["chosen"]), tuple(row["rejected"])) for row in rows)
    read = sorted(tuple(tuple(tokens) for tokens, _ in found[key]) for key in found)
    assert read == halves
    assert all(places == list(range(len(tokens))) for each in found.values() for tokens, places in each)
    shared = sum(int(np.sum(np.asarray(batch[REAL_KEY]))) for batch in laid)
    assert shared == sum(len(row["chosen"]) + len(row["rejected"]) for row in rows) - (2 if share else 0)


def test_an_overlong_row_is_refused():
    row = {"chosen": [1, 2, 3, 4, 5], "rejected": [1]}
    with pytest.raises(ValueError):
        preference_rows(records(row), 4)
    with pytest.raises(ValueError):
        preference_rows(records(PAIR), 0)


def test_parquet_without_sides_is_refused(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "nopairs.parquet"
    pq.write_table(pa.table({"chosen": [[1]]}), path)
    with pytest.raises(ValueError):
        next(compile(Files(str(path), format="parquet").pipe(presets.preferences, seq_len=4, batch=1),
                     read="memory").train(DataPartition()))


def test_a_preference_pair_marks_each_halfs_real_tokens():
    """A window marks its pairs' ids as real (`REAL_KEY`) and its padding as not, whatever their completion
    marks: a chosen half ending in an unmarked token, a rejected half of no completion, and a window of
    padding alone that a batch of more windows than its pairs need holds."""
    laid = windows([{"chosen": [1, 2, 3, 4], "rejected": [1, 2], "chosen_mask": [0, 0, 1, 0],
                     "rejected_mask": [0, 0]}, {"chosen": [7], "rejected": [8]}], 5, count=3)
    assert laid[REAL_KEY].tolist() == [[1, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]]


@pytest.mark.parametrize("share", [True, False], ids=["shared", "whole"])
@pytest.mark.parametrize("kind", ["attention", "gated delta net hybrid", "one document a row"])
def test_packed_pairs_train_as_each_pair_a_row(kind, share):
    """Pairs packed into windows (`preferences.windows`: several pairs a window, a pair's shared prompt once
    where `share`; a user's model that reads one document a row one half a window, `segments=1`) give the
    DPO loss, its rewards and the gradient in every parameter that each pair laid out a row, its halves
    scored apart, gives (check 1, over drawn weights and references, float64 truth): each half's completion
    scored from its own predecessors, a half's first token from the shared prompt's last, each pair's sums
    reduced by its number across the batch (`pair_numbers`)."""
    from model_support import TINY_DECODER, float64_twin
    from optim_support import in_float64

    from dew.batch import coordinates
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm.objective import LMObjective
    from dew.rl.surrogate import preference_logsigmoid_terms

    half = 7
    segments = 1 if kind == "one document a row" else None
    if kind == "one document a row":
        from capability_modules import TokenModel as UserTokens

        model = UserTokens(vocab_size=VOCAB)
    elif kind == "attention":
        model = CausalTransformer(vocab_size=VOCAB, **{**TINY_DECODER, "max_seq_len": 2 * half},
                                  dtype="float32")
    else:
        model = CausalTransformer(
            vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8, mlp_features=32,
            max_seq_len=2 * half, dtype="float32", layer_types=("linear_attention", "full_attention"),
            kinds={"linear_attention": {"mixer": {"class": "gated_delta_net", "fields": {
                "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 8,
                "linear_value_head_dim": 8}}}})
    rng = np.random.default_rng(5)
    rows = []
    for index in range(5):
        prompt = rng.integers(1, VOCAB, 2 + index % 3).tolist()
        chosen = prompt + rng.integers(1, VOCAB, 1 + index % 2).tolist()
        rejected = prompt + rng.integers(1, VOCAB, 2).tolist()
        rows.append({"chosen": chosen, "rejected": rejected,
                     "chosen_mask": [0] * len(prompt) + [1] * (len(chosen) - len(prompt)),
                     "rejected_mask": [0] * len(prompt) + [1] * (len(rejected) - len(prompt))})
    # A pair of default masks, every token a completion's: no prompt shared, each half's first token
    # scored from nothing and so counted in neither arm.
    rows.append({"chosen": rng.integers(1, VOCAB, 4).tolist(),
                 "rejected": rng.integers(1, VOCAB, 3).tolist()})
    rows[-1].update({"chosen_mask": [1] * 4, "rejected_mask": [1] * 3})
    pairs = 1 if segments == 1 else 3
    laid = windows(rows, half, share=share, segments=segments, pairs_a_window=pairs)
    assert laid[IDS_KEY].shape[0] < len(rows) or segments == 1
    objective = DPOObjective(model, half - 1, beta=0.5, segments=segments, pairs=pairs)
    ids = np.zeros((2 * len(rows), half), np.int32)
    marks = np.zeros((2 * len(rows), half), np.float32)
    for index, row in enumerate(rows):
        for side, name in enumerate(("chosen", "rejected")):
            ids[side * len(rows) + index, :len(row[name])] = row[name]
            marks[side * len(rows) + index, :len(row[name])] = row[f"{name}_mask"]
    held = coordinates(valid=jnp.asarray(ids > 0), right_padded=True)

    def packed(objective):
        def run(params, reference):
            step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=reference)
            (loss, aux), grads = jax.value_and_grad(objective.scalar_loss, has_aux=True)(params, laid, step)
            return {"loss": loss, "chosen": aux.metrics["rewards/chosen"], "grads": grads}
        return jax.jit(run)

    def each_row(model):
        scorer = LMObjective(model, half - 1)

        def run(params, reference):
            def loss(params):
                policy, frozen = (-scorer.token_scores(tree, ids, coordinates=held).losses
                                  for tree in (params, reference))
                shifted = jnp.asarray(marks[:, 1:])
                halves = [jnp.split(values, 2) for values in (policy, frozen)]
                terms, (chosen, _) = preference_logsigmoid_terms(
                    halves[0][0], halves[0][1], halves[1][0], halves[1][1], *jnp.split(shifted, 2), 0.5)
                return jnp.mean(terms), jnp.mean(chosen)
            (value, chosen), grads = jax.value_and_grad(loss, has_aux=True)(params)
            return {"loss": value, "chosen": chosen, "grads": grads}
        return jax.jit(run)

    # A user's module states no dtype of its own: its truth is the reference arm over the float64 draws,
    # its scores and rewards in a float64 run's output dtype (`in_float64`) where the run's are float32.
    truth_fn = in_float64(each_row(model)) if segments == 1 else each_row(float64_twin(model))

    initialized = jax.jit(objective.init)

    def draw(seed):
        # The reference another draw of the same model's weights.
        return initialized(jax.random.key(seed)), initialized(jax.random.fold_in(jax.random.key(seed), 1))

    assert_as_exact(packed(objective), each_row(model), draw, f"DPO over packed pairs, {kind}",
                    seeds=48, truth_fn=truth_fn)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("segments", [None, 1], ids=["packed", "a half a window"])
def test_a_step_split_over_parts_and_shards_trains_as_each_pair_a_row(segments):
    """A trainer's step over windows of pairs, read in two microbatches over two data shards (`split=2`
    on a two-device data mesh), moves the policy as one SGD step on the loss of each pair laid out a row
    (check 1, over drawn weights, float64 truth): packed windows, or a pair a row, its halves a window each
    (`segments=1`), which whatever deals the rows keeps whole, and a pair's halves reduce by its number
    however the windows hold them."""
    import optax
    from flax import linen as nn
    from model_support import TINY_DECODER, float64_twin

    from dew.batch import coordinates
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm.objective import LMObjective
    from dew.rl.surrogate import preference_logsigmoid_terms
    from dew.training import MeshSpec, Trainer

    half, rate = 8, 0.5
    model = CausalTransformer(vocab_size=VOCAB, **{**TINY_DECODER, "max_seq_len": 2 * half}, dtype="float32")
    rng = np.random.default_rng(9)
    rows = []
    # Eight rows either way: two pairs of at most seven tokens a window of sixteen, or a pair a row.
    for index in range(8 if segments == 1 else 16):
        prompt = rng.integers(1, VOCAB, 2 + index % 2).tolist()
        chosen, rejected = (prompt + rng.integers(1, VOCAB, 2).tolist() for _ in range(2))
        rows.append({"chosen": chosen, "rejected": rejected,
                     "chosen_mask": [0] * len(prompt) + [1, 1], "rejected_mask": [0] * len(prompt) + [1, 1]})
    pairs = 1 if segments == 1 else 2
    laid = windows(rows, half, segments=segments, pairs_a_window=pairs)
    assert laid[IDS_KEY].shape[0] == 8
    ids = np.zeros((2 * len(rows), half), np.int32)
    marks = np.zeros((2 * len(rows), half), np.float32)
    for index, row in enumerate(rows):
        for side, name in enumerate(("chosen", "rejected")):
            ids[side * len(rows) + index, :len(row[name])] = row[name]
            marks[side * len(rows) + index, :len(row[name])] = row[f"{name}_mask"]
    held = coordinates(valid=jnp.asarray(ids > 0), right_padded=True)

    def trainer_for(seed, split=2):
        objective = DPOObjective(model, half - 1, beta=0.5, segments=segments, pairs=pairs)
        trainer = Trainer(objective, optax.sgd(rate),
                          key=jax.random.key(seed), split=split, checkpoints=None, tracker=None)
        # Two devices, the data axis taking both: two data shards.
        trainer.device_mesh = MeshSpec().build(jax.devices()[:2])
        return trainer

    trainer = trainer_for(0)
    start = trainer.initial_state()
    step = trainer.compile(start, laid)

    def stepped(params):
        # The trainer's step from the drawn weights, its reference the same weights frozen (`step.ema`), run
        # on the host: the gate compiles its arms, and a step is a program of its own.
        # A state of its own buffers each call: the step donates the state it reads.
        state = jax.tree.map(jnp.copy, start.replace(variables={**start.variables, "params": params},
                                                      ema={**start.ema, "params": params}))
        state, *_ = step(state, laid)
        return state.variables["params"]

    def each_row(model):
        scorer = LMObjective(model, half - 1)

        def run(params):
            def loss(params):
                policy, frozen = (-scorer.token_scores({"params": tree}, ids, coordinates=held).losses
                                  for tree in (params, jax.lax.stop_gradient(params)))
                halves = [jnp.split(values, 2) for values in (policy, frozen)]
                terms, _ = preference_logsigmoid_terms(halves[0][0], halves[0][1], halves[1][0], halves[1][1],
                                                       *jnp.split(jnp.asarray(marks[:, 1:]), 2), 0.5)
                return jnp.mean(terms)
            return jax.tree.map(lambda leaf, grad: leaf - rate * grad, params, jax.grad(loss)(params))
        return jax.jit(run)

    initialized = jax.jit(DPOObjective(model, half - 1).init)

    def draw(seed):
        params = nn.unbox(initialized(jax.random.key(seed)))["params"]
        return params, stepped(params)

    plain, wide = each_row(model), each_row(float64_twin(model))
    assert_as_exact(lambda _, moved: moved, lambda params, _: plain(params), draw,
                    f"a split step over pairs, segments={segments}", seeds=8,
                    truth_fn=lambda params, _: wide(params))


def test_pairs_are_numbered_across_the_windows_a_step_reads_whatever_numbers_they_carry():
    """The DPO loss numbers each pair across the windows a step reads (`pair_numbers`): a pair a row's two
    windows (`segments=1`, `[B, 2, W]`, read in turn) one pair, and rows stacked from two batches, whose
    windows carry the same numbers, pairs of their own, packed or a pair a row."""
    from dew.data.preferences import PAIRS_KEY
    from dew.objectives.rl.preference import pair_numbers

    rows = [{"chosen": [1, 2, 3], "rejected": [1, 2, 4], "chosen_mask": [0, 1, 1], "rejected_mask": [0, 1, 1]}
            for _ in range(4)]
    halved = windows(rows, 3, segments=1)
    assert halved[IDS_KEY].shape == (4, 2, 3)
    reading = DPOObjective(TinyHead(vocab_size=VOCAB), 2, segments=1)

    def numbered(batch, halved=True):
        laid = reading.windows(batch) if halved else batch
        return np.asarray(pair_numbers(laid, halved=halved)[0]).max(axis=1)

    assert numbered(halved).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    twice = {key: np.concatenate([value, value]) for key, value in halved.items()}
    assert numbered(twice).tolist() == [*np.repeat(np.arange(1, 9), 2)]
    shared = windows(rows, 3)
    assert numbered({PAIRS_KEY: np.concatenate([shared[PAIRS_KEY]] * 2)}, halved=False).tolist() == list(
        range(1, 2 * len(shared[PAIRS_KEY]) + 1))


def test_a_padded_pair_before_real_ones_leaves_each_real_pair_its_own_loss():
    """Batches of a pair a row (`segments=1`) stacked, each filled with a padded pair after its real one, or
    the padded pairs first, reduce each real pair's halves to the sums and weight the two real pairs alone
    give (`_halves` over the same per-target values): a padded pair numbers no pair, weighs nothing, and
    leaves the real pair after it a pair of its own."""
    from dew.objectives.rl.preference import _halves

    first = {"chosen": [1, 2, 3], "rejected": [1, 2, 4], "chosen_mask": [0, 1, 1], "rejected_mask": [0, 1, 1]}
    second = {"chosen": [5, 6, 7], "rejected": [5, 3], "chosen_mask": [0, 1, 1], "rejected_mask": [0, 1]}
    objective = DPOObjective(TinyHead(vocab_size=VOCAB), 2, segments=1)
    alone = windows([first, second], 3, segments=1)
    padded = {key: np.zeros_like(value) for key, value in alone.items()}
    rng = np.random.default_rng(5)
    # Each real pair's per-target values, the same wherever its rows stand; a padded pair's, noise.
    held = rng.standard_normal((2, 2, 2)).astype(np.float32)

    def reduced(pairs):
        batch = {key: np.stack([alone[key][at] if at is not None else padded[key][0] for at in pairs])
                 for key in alone}
        values = np.stack([held[at] if at is not None else rng.standard_normal((2, 2)).astype(np.float32)
                           for at in pairs]).reshape(-1, 2)
        laid = objective.windows(batch)
        return [np.asarray(part) for part in _halves(jnp.asarray(values), laid,
                                                     jnp.asarray(laid[REAL_KEY])[:, 1:], halved=True)]

    expected = reduced([0, 1])
    for pairs in ([0, None, 1, None], [None, None, 0, 1]):
        found = reduced(pairs)
        real = [number for number, at in enumerate(pairs) if at is not None]
        for part, wanted in zip(found, expected, strict=True):
            np.testing.assert_array_equal(part[real], wanted[:2])
        assert np.all(found[2][[number for number, at in enumerate(pairs) if at is None]] == 0)


def test_two_window_geometries_trace_apart():
    """Two DPO objectives of one model and `seq_len` whose windows hold different pairs (`pairs`) or lay a
    pair a row (`segments=1`) forward different inputs and finals, so their trace keys differ (`traced`),
    and a fit of one never steps on the other's lowered program."""
    model = TinyHead(vocab_size=VOCAB)
    keys = [DPOObjective(model, 5, **options).trace_key() for options in ({}, {"pairs": 2}, {"segments": 1})]
    assert None not in keys and len(set(keys)) == 3


@pytest.mark.parametrize("segments", [None, 1], ids=["packed", "a pair a row"])
def test_a_windows_padding_holds_the_pad_id(segments):
    """Every window's padding is `pad_id` (main's `presets.preferences(..., pad_id=)`), the id a model reads
    past a pair, packed or a pair a row, its real ids as they were."""
    pair = {"chosen": [1, 2, 3], "rejected": [1, 2, 4], "chosen_mask": [0, 1, 1], "rejected_mask": [0, 1, 1]}
    plain, padded = (windows([pair], 5, segments=segments, count=2, pad_id=pad_id) for pad_id in (0, 7))
    real = plain[REAL_KEY] > 0
    assert (~real).any()
    np.testing.assert_array_equal(padded[IDS_KEY], np.where(real, plain[IDS_KEY], 7))
    chain = Memory([pair]).pipe(presets.preferences, seq_len=5, batch=1, segments=segments, pad_id=7)
    (batch,) = list(chain)
    held = np.asarray(batch[REAL_KEY]) > 0
    assert np.all(np.asarray(batch[IDS_KEY])[~held] == 7)
