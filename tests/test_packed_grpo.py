"""GRPO over packed rows against GRPO over one unmerged row per model call.

A multi-call rollout whose history stays append-only merges into one chain,
and several chains share a row. Each sampled id must still be scored with
exactly the prefix its call saw, so the packed loss, its gradient and its
proximal log-probabilities equal the ones computed call by call.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import widened
from test_tools import load

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    CALL_INDEX_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
    SESSION_INDEX_KEY,
    Call,
    Session,
    Status,
    advantages,
    pack,
)
from dew.rl import k3_kl

VOCAB = 16
WIDTH = 24


def _model(dtype="float32", *, attention_impl="xla"):
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2,
                             mlp_features=32, max_seq_len=64, dtype=dtype, attention_impl=attention_impl)


def _rollouts():
    rng = np.random.default_rng(0)

    def ids(count):
        return tuple(int(value) for value in rng.integers(1, VOCAB, count))

    def logps(count):
        return tuple(float(value) for value in -rng.random(count))

    rollouts = []
    for sample in range(4):
        first = Call(ids(3), ids(2), logps(2), "tool_calls", 0)
        # Append-only: the second prompt holds the first call's prompt and sampled ids.
        second = Call(first.prompt_ids + first.sampled_ids + ids(2), ids(3), logps(3), "tool_calls", 0)
        # A rewrite: the third call's history drops the second call's sampled ids.
        third = Call(second.prompt_ids + ids(1), ids(2), logps(2), "stop", 0)
        rollouts.append(Session("t", str(sample // 2), sample, 0, (first, second, third),
                                Status.COMPLETED, float(sample % 2)))
    return rollouts


def _per_call(rollouts):
    """One chain per row, one row per call: the unmerged layout, each call's
    prompt then its sampled ids, carrying its rollout's advantage."""
    values = advantages(rollouts)
    calls = [(call, values[index]) for index, rollout in enumerate(rollouts) for call in rollout.calls]
    shape = (len(calls), WIDTH)
    ids, segments, positions = np.zeros(shape, np.int32), np.zeros(shape, np.int32), np.zeros(shape, np.int32)
    mask, behavior, advantage = (
        np.zeros(shape, np.float32),
        np.zeros(shape, np.float32),
        np.zeros(shape, np.float32),
    )
    for row, (call, value) in enumerate(calls):
        size, count = len(call.prompt_ids), len(call.sampled_ids)
        ids[row, :size + count] = (*call.prompt_ids, *call.sampled_ids)
        segments[row, :size + count] = 1
        positions[row, :size + count] = np.arange(size + count)
        mask[row, size:size + count] = 1
        behavior[row, size:size + count] = call.behavior_log_probs
        advantage[row, :size + count] = value
    return {IDS_KEY: ids, SEGMENT_IDS_KEY: segments, POSITIONS_KEY: positions, RESPONSE_MASK_KEY: mask,
            OLD_LOG_PROBS_KEY: behavior, BEHAVIOR_LOG_PROBS_KEY: behavior, ADVANTAGES_KEY: advantage}


def _loss(objective, params, batch, reference):
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=reference)
    return objective.loss(params, batch, step)[0].mean()[0]


FLOOR_FACTOR = load("layout_parity").FLOOR_FACTOR
"""layout_parity's rule: how far a run may sit from its reference, in
multiples of the rounding measured for it."""


def _step(objective, params, batch, reference):
    """The loss and its gradient."""
    value, gradients = jax.value_and_grad(lambda p: _loss(objective, p, batch, reference))(params)
    return {"loss": value, "gradients": gradients}


def _token_contributions(objective, params, batch, reference, trainable):
    """The N weighted trainable-token contributions to this fixture's loss.

    The fixture has token-mean aggregation and no behavior corrections.
    Keeping contributions separate lets jacrev measure each gradient leaf's
    sum of absolute token contributions, including leaves whose sum cancels.
    """
    terms = objective._terms(params, batch)
    policy, _ = objective._policy_terms(terms, terms.mask)
    kl = k3_kl(terms.policy, objective.packed_log_probs(reference, batch))
    weighted = jnp.where(terms.mask != 0, policy + objective.beta * kl, 0) * terms.mask
    return weighted.reshape(-1)[trainable] / jnp.sum(terms.mask)


def _fp64_reference(objective, params, batch, reference):
    """Reference, legitimate row-reorder spread and the final contraction's magnitudes."""
    exact = _step(objective, params, batch, reference)
    rows = np.arange(batch[IDS_KEY].shape[0])
    orders = (rows[::-1], np.concatenate((rows[::2], rows[1::2])))
    reordered = [_step(objective, params, {key: np.asarray(value)[order] for key, value in batch.items()},
                       reference) for order in orders]
    trainable = np.flatnonzero(np.asarray(batch[RESPONSE_MASK_KEY]).reshape(-1))

    def contributions(parameters):
        return _token_contributions(objective, parameters, batch, reference, trainable)

    tokens = contributions(params)
    jacobian = jax.jacrev(contributions)(params)
    magnitudes = {"loss": jnp.sum(jnp.abs(tokens)),
                  "gradients": jax.tree.map(lambda leaf: jnp.sum(jnp.abs(leaf), axis=0), jacobian)}
    return exact, reordered, magnitudes, trainable.size


def _assert_fp64_layout_envelope(candidate, reference, reordered, magnitudes, terms):
    """An empirical parity convention, with a derived final-contraction floor.

    Both layouts and the reorderings run the same fp64 reference-attention
    graph. FLOOR_FACTOR=4 is layout_parity's existing convention: up to four
    times the largest leaf deviation measured from legitimate reference
    reorderings (reverse and even/odd rows). This is an empirical envelope,
    not a worst-case bound on all upstream arithmetic or all possible ISAs.

    Its floor is arithmetic-derived. Each loss/gradient entry is a sum of N
    trainable-token contributions; here N=28. With fp64 unit roundoff u=2^-53,
    a contraction of N products has error <= gamma_N sum(abs(products)),
    gamma_N=N*u/(1-N*u), for ANY reduction order (one product and at most
    N-1 adds per scalar path). Two contractions differ by at most twice it.
    jacrev supplies those signed token gradients before their final sum,
    so the leaf's magnitude is max_entry sum(abs(token gradients)), not
    max_entry abs(net gradient), matching layout_parity's per-leaf max norm.
    The positive magnitude reduction is inflated by 1/(1-gamma_(N-1)) for
    its own rounding. No fp32 error or eps64/eps32 scaling enters the rule:
    such scaling is invalid under cancellation, and the fp32 XLA attention
    and fp64 reference attention do not even have the same operation graph.
    """
    u = np.finfo(np.float64).eps / 2
    gamma = terms * u / (1 - terms * u)
    magnitude_gamma = (terms - 1) * u / (1 - (terms - 1) * u)
    reordered_leaves = [jax.tree.leaves(value) for value in reordered]
    for index, ((path, expected), actual, magnitude) in enumerate(zip(
        jax.tree.leaves_with_path(reference), jax.tree.leaves(candidate),
        jax.tree.leaves(magnitudes), strict=True,
    )):
        expected, actual, magnitude = (np.asarray(x, np.float64) for x in (expected, actual, magnitude))
        spread = max(float(np.max(np.abs(np.asarray(leaves[index]) - expected)))
                     for leaves in reordered_leaves)
        floor = 2 * gamma * float(np.max(magnitude)) / (1 - magnitude_gamma)
        bound = max(FLOOR_FACTOR * spread, floor)
        difference = float(np.max(np.abs(actual - expected)))
        assert difference <= bound, (
            f"{jax.tree_util.keystr(path)}: max error {difference:.3e}, "
            f"max envelope {bound:.3e} (fp64 reorder spread {spread:.3e}, N={terms})")


@pytest.mark.parametrize("policy_loss", ["ppo", "cispo"])
def test_packed_grpo_equals_per_call_grpo_on_the_unmerged_chains(policy_loss):
    """Packing preserves the per-call loss and gradients within the fp64
    layout envelope. See _assert_fp64_layout_envelope for its measured spread
    and arithmetic-derived floor. Everything runs on CPU, since a TPU has
    no float64."""
    rollouts = _rollouts()
    packed = pack(rollouts, WIDTH)
    assert packed[IDS_KEY].shape[0] < sum(len(rollout.calls) for rollout in rollouts)
    assert packed[SEGMENT_IDS_KEY].max() >= 2, "rows share chains"
    packed[OLD_LOG_PROBS_KEY] = packed[BEHAVIOR_LOG_PROBS_KEY]
    unmerged = _per_call(rollouts)

    with jax.default_device(jax.devices("cpu")[0]):
        objective = GRPOObjective(_model(), WIDTH - 1, beta=0.1, policy_loss=policy_loss)
        params = objective.init(jax.random.key(1))
        reference = jax.tree.map(lambda leaf: leaf * 0.9, params)
        with jax.enable_x64():
            twin = GRPOObjective(_model(jnp.float64, attention_impl="reference"), WIDTH - 1,
                                 beta=0.1, policy_loss=policy_loss)
            exact, reordered, magnitudes, terms = _fp64_reference(
                twin, widened(params), widened(unmerged), widened(reference))
            packed_exact = _step(twin, widened(params), widened(packed), widened(reference))
    _assert_fp64_layout_envelope(packed_exact, exact, reordered, magnitudes, terms)


def test_the_fp64_layout_envelope_rejects_a_dropped_packed_token():
    """A real packing defect remains far outside the empirical envelope."""
    rollouts = _rollouts()
    unmerged, packed = _per_call(rollouts), pack(rollouts, WIDTH)
    packed[OLD_LOG_PROBS_KEY] = packed[BEHAVIOR_LOG_PROBS_KEY]
    row, column = np.argwhere(packed[RESPONSE_MASK_KEY] != 0)[0]
    packed[RESPONSE_MASK_KEY][row, column] = 0
    with jax.default_device(jax.devices("cpu")[0]):
        objective = GRPOObjective(_model(), WIDTH - 1, beta=0.1, policy_loss="cispo")
        params = objective.init(jax.random.key(1))
        reference = jax.tree.map(lambda leaf: leaf * 0.9, params)
        with jax.enable_x64():
            twin = GRPOObjective(_model(jnp.float64, attention_impl="reference"), WIDTH - 1,
                                 beta=0.1, policy_loss="cispo")
            exact, reordered, magnitudes, terms = _fp64_reference(
                twin, widened(params), widened(unmerged), widened(reference))
            changed = _step(twin, widened(params), widened(packed), widened(reference))
    with pytest.raises(AssertionError, match="max error"):
        _assert_fp64_layout_envelope(changed, exact, reordered, magnitudes, terms)


def test_packed_log_probs_score_each_id_with_its_own_calls_prefix():
    rollouts = _rollouts()
    packed = pack(rollouts, WIDTH)
    unmerged = _per_call(rollouts)
    objective = GRPOObjective(_model(), WIDTH - 1)
    params = objective.init(jax.random.key(2))
    scored = np.asarray(objective.packed_log_probs(params, packed))
    alone = np.asarray(objective.packed_log_probs(params, unmerged))
    expected = alone[unmerged[RESPONSE_MASK_KEY] != 0]
    order = [
        (index, number) for index, rollout in enumerate(rollouts) for number in range(len(rollout.calls))
    ]
    placed = np.concatenate([scored[(packed[SESSION_INDEX_KEY] == index) & (packed[CALL_INDEX_KEY] == number)]
                             for index, number in order])
    np.testing.assert_allclose(placed, expected, atol=1e-5)
    assert (scored[packed[RESPONSE_MASK_KEY] == 0] == 0).all()


def test_corrections_without_proximal_log_probs_read_the_current_policy():
    """verl's bypass mode: with behavior standing in for the old policy, the
    band and the sequence masks compare the detached current policy with
    behavior, rather than behavior with itself."""
    packed = pack(_rollouts(), WIDTH)
    objective = GRPOObjective(_model(), WIDTH - 1, behavior_importance=(0.5, 5.0))
    params = objective.init(jax.random.key(3))
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    policy = np.asarray(objective.packed_log_probs(params, packed))
    mask = packed[RESPONSE_MASK_KEY] != 0
    ratio = np.exp(policy - packed[BEHAVIOR_LOG_PROBS_KEY])[mask]
    outside = float(np.mean((ratio < 0.5) | (ratio > 5.0)))
    assert 0 < outside < 1, "the fixture puts some tokens outside the band"
    _, aux = objective.loss(params, packed, step)
    assert float(aux.metrics["masked/band"]) == pytest.approx(outside)
    plain, _ = GRPOObjective(_model(), WIDTH - 1).loss(params, packed, step)
    banded, _ = objective.loss(params, packed, step)
    assert abs(float(plain.mean()[0]) - float(banded.mean()[0])) > 1e-4
    geometric = GRPOObjective(_model(), WIDTH - 1, geometric_mask=(0.99, 1.01))
    _, aux = geometric.loss(params, packed, step)
    assert float(aux.metrics["masked/geometric"]) == 1.0
    capped = GRPOObjective(_model(), WIDTH - 1, behavior_importance=2.0)
    with pytest.raises(ValueError, match="old_log_probs"):
        capped.loss(params, packed, step)


def test_mismatch_metrics_describe_every_trainable_token_whatever_a_mask_rejects():
    """verl computes IS weights and off-policy metrics over the response mask
    and applies rejection to the loss alone."""
    packed = pack(_rollouts(), WIDTH)
    model = _model()
    params = GRPOObjective(model, WIDTH - 1).init(jax.random.key(3))
    packed[OLD_LOG_PROBS_KEY] = np.asarray(GRPOObjective(model, WIDTH - 1).packed_log_probs(params, packed))
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    _, open_aux = GRPOObjective(model, WIDTH - 1, behavior_importance=(0.5, 5.0)).loss(params, packed, step)
    _, masked_aux = GRPOObjective(model, WIDTH - 1, behavior_importance=(0.5, 5.0),
                                  sequence_mask=(0.99, 1.01)).loss(params, packed, step)
    assert float(masked_aux.metrics["masked/sequence"]) == 1.0
    for key in ("mismatch/kl", "mismatch/k3_kl", "mismatch/ess", "masked/band"):
        assert float(masked_aux.metrics[key]) == pytest.approx(float(open_aux.metrics[key]))
    assert float(open_aux.metrics["mismatch/kl"]) > 0.1


def test_one_behavior_importance_option_takes_a_cap_or_a_band():
    with pytest.raises(ValueError, match="positive TIS cap"):
        GRPOObjective(_model(), WIDTH - 1, behavior_importance=0.0)
    with pytest.raises(ValueError, match="0 < low <= high"):
        GRPOObjective(_model(), WIDTH - 1, behavior_importance=(5.0, 0.5))


def test_the_loss_never_holds_the_logits_of_the_whole_batch():
    """GRPO scores through the chunked head: a large vocabulary adds tiles to
    the loss and gradient's temporaries, never a [rows, width, vocab] tensor."""
    vocab, rows, width = 65536, 8, 64
    model = CausalTransformer(vocab_size=vocab, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=width, dtype="float32", attention_impl="xla")
    objective = GRPOObjective(model, width - 1)
    params = jax.eval_shape(objective.init, jax.random.key(0))
    batch = {name: jax.ShapeDtypeStruct((rows, width), dtype) for name, dtype in (
        (IDS_KEY, jnp.int32), (SEGMENT_IDS_KEY, jnp.int32), (POSITIONS_KEY, jnp.int32),
        (RESPONSE_MASK_KEY, jnp.float32), (BEHAVIOR_LOG_PROBS_KEY, jnp.float32),
        (OLD_LOG_PROBS_KEY, jnp.float32), (ADVANTAGES_KEY, jnp.float32))}
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)

    def loss(variables, batch):
        return objective.loss(variables, batch, step)[0].mean()[0]

    compiled = jax.jit(jax.value_and_grad(loss)).lower(params, batch).compile()
    logits = rows * width * vocab * 4
    assert compiled.memory_analysis().temp_size_in_bytes < logits / 2
