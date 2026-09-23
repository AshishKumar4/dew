"""Strict prefix merge and packing of engine-sourced rollouts.

Rollouts are generated with a known chain structure: a call either appends
to the full history so far (it must merge) or rewrites some id of it (it
must split). The packed batch is then read back through its segment ids and
positions alone and compared with that structure.
"""

import numpy as np
import pytest

from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    CALL_INDEX_KEY,
    IDS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
    SESSION_INDEX_KEY,
    SESSION_WEIGHTS_KEY,
    VERSIONS_KEY,
    Call,
    Session,
    Status,
    advantages,
    chain_lengths,
    chains,
    pack,
    rows_needed,
    sampled_values,
    session_metrics,
)


def _ids(rng, count):
    return tuple(int(value) for value in rng.integers(1, 50, count))


def _call(rng, prompt, version):
    sampled = _ids(rng, int(rng.integers(1, 5)))
    return Call(prompt, sampled, tuple(float(-value) for value in rng.random(len(sampled))),
                "tool_calls", version)


def _rollout(rng, task="t", group="g", sample=0, reward=1.0, status=Status.COMPLETED):
    """A rollout and the expected chains, each a list of its call indices."""
    calls, expected = [], []
    history: tuple[int, ...] = ()
    for number in range(int(rng.integers(1, 6))):
        if not calls:
            prompt = _ids(rng, int(rng.integers(1, 6)))
            expected.append([number])
        elif rng.random() < 0.6:
            prompt = history + _ids(rng, int(rng.integers(0, 4)))
            expected[-1].append(number)
        else:
            changed = list(history)
            where = int(rng.integers(0, len(changed)))
            changed[where] = changed[where] % 49 + 1 + 50  # outside the drawn range, so it differs
            prompt = tuple(changed[:int(rng.integers(where + 1, len(changed) + 1))]) + _ids(rng, 2)
            expected.append([number])
        call = _call(rng, prompt, number)
        calls.append(call)
        history = call.prompt_ids + call.sampled_ids
    return Session(task, group, sample, 0, tuple(calls), status, reward), expected


def _read_chains(batch):
    """Every chain of the batch as (rollout, row, start, stop), from segment ids alone."""
    found = []
    segments = batch[SEGMENT_IDS_KEY]
    for row in range(segments.shape[0]):
        ids = segments[row]
        used = np.flatnonzero(ids)
        assert used.size == 0 or (used == np.arange(used.size)).all(), "chains are left-packed"
        labels = ids[used]
        assert (np.diff(labels) >= 0).all() and (labels.size == 0 or labels[0] == 1)
        for segment in np.unique(labels):
            span = np.flatnonzero(ids == segment)
            assert (np.diff(span) == 1).all(), "a chain is contiguous"
            start, stop = int(span[0]), int(span[-1]) + 1
            np.testing.assert_array_equal(batch[POSITIONS_KEY][row, start:stop], np.arange(stop - start))
            owners = np.unique(batch[SESSION_INDEX_KEY][row, start:stop])
            assert owners.size == 1
            found.append((int(owners[0]), row, start, stop))
    return found


@pytest.mark.parametrize("seed", range(40))
def test_strict_merge_follows_append_only_history_and_splits_on_any_rewrite(seed):
    rng = np.random.default_rng(seed)
    pairs = [_rollout(rng, group=str(index // 2), sample=index, reward=float(index % 3))
             for index in range(int(rng.integers(2, 7)))]
    rollouts = [rollout for rollout, _ in pairs]
    width = 64
    batch = pack(rollouts, width)
    found = _read_chains(batch)
    for index, (rollout, expected) in enumerate(pairs):
        mine = sorted((row, start, stop) for owner, row, start, stop in found if owner == index)
        assert len(mine) == len(expected)
        # Each expected chain is its last call's prompt plus that call's sampled ids.
        wanted = {rollout.calls[calls[-1]].prompt_ids + rollout.calls[calls[-1]].sampled_ids: calls
                  for calls in expected}
        for row, start, stop in mine:
            tokens = tuple(int(value) for value in batch[IDS_KEY][row, start:stop])
            assert tokens in wanted
            calls = wanted.pop(tokens)
            sampled = np.flatnonzero(batch[RESPONSE_MASK_KEY][row, start:stop]) + start
            expected_ids = [token for number in calls for token in rollout.calls[number].sampled_ids]
            np.testing.assert_array_equal(batch[IDS_KEY][row, sampled], expected_ids)
            np.testing.assert_array_equal(
                batch[BEHAVIOR_LOG_PROBS_KEY][row, sampled],
                np.asarray([p for number in calls for p in rollout.calls[number].behavior_log_probs], np.float32))
            np.testing.assert_array_equal(
                batch[VERSIONS_KEY][row, sampled],
                [number for number in calls for _ in rollout.calls[number].sampled_ids])
            np.testing.assert_array_equal(
                batch[CALL_INDEX_KEY][row, sampled],
                [number for number in calls for _ in rollout.calls[number].sampled_ids])
            others = np.setdiff1d(np.arange(start, stop), sampled)
            assert (batch[VERSIONS_KEY][row, others] == -1).all()
            assert (batch[BEHAVIOR_LOG_PROBS_KEY][row, others] == 0).all()
        assert not wanted
    pad = batch[SEGMENT_IDS_KEY] == 0
    assert (batch[SESSION_INDEX_KEY][pad] == -1).all() and (batch[RESPONSE_MASK_KEY][pad] == 0).all()
    # Session weights give each trainable rollout unit mass.
    for index in range(len(rollouts)):
        mine = batch[SESSION_INDEX_KEY] == index
        assert batch[SESSION_WEIGHTS_KEY][mine].sum() == pytest.approx(1.0)
        assert (batch[ADVANTAGES_KEY][mine] == advantages(rollouts)[index]).all()


def test_a_prompt_only_prefix_does_not_merge():
    """Section 6: the template dropped the sampled reasoning from history."""
    first = Call((1, 2, 3), (7, 8, 9), (-.1, -.2, -.3), "tool_calls", 0)
    rewritten = Call((1, 2, 3, 9, 4), (5,), (-.5,), "stop", 0)
    appended = Call((1, 2, 3, 7, 8, 9, 4), (5,), (-.5,), "stop", 0)
    split = pack([Session("t", "g", 0, 0, (first, rewritten), Status.COMPLETED, 1.0)], 16)
    merged = pack([Session("t", "g", 0, 0, (first, appended), Status.COMPLETED, 1.0)], 16)
    assert split[SEGMENT_IDS_KEY].max() == 2
    assert merged[SEGMENT_IDS_KEY].max() == 1
    np.testing.assert_array_equal(merged[IDS_KEY][0, :8], [1, 2, 3, 7, 8, 9, 4, 5])
    np.testing.assert_array_equal(merged[RESPONSE_MASK_KEY][0, :8], [0, 0, 0, 1, 1, 1, 0, 1])


def test_masked_rollouts_take_no_rows_and_no_baseline():
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    rollouts = [Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0),
                Session("t", "g", 1, 0, (call,), Status.AGENT_ERROR, 0.0),
                Session("t", "g", 2, 0, (call,), Status.TRUNCATED, 5.0),
                Session("t", "g", 3, 0, (call,), Status.INFRA_ERROR, None)]
    batch = pack(rollouts, 3)
    assert set(np.unique(batch[SESSION_INDEX_KEY])) == {0, 1}
    np.testing.assert_allclose(advantages(rollouts, "mean"), [.5, -.5, 0, 0])
    lonely = [rollouts[0], rollouts[2]]
    assert (advantages(lonely) == 0).all()


@pytest.mark.parametrize("truncation, expected", [
    ("mask", [.5, -.5, 0, 0]),       # the truncation takes no row and no baseline
    ("score", [-1, -2, 3, 0]),   # it trains on its reward 5: baseline 2
    ("zero", [2 / 3, -1 / 3, -1 / 3, 0]),  # it trains on 0: baseline 1/3
])
def test_the_truncation_policy_decides_whether_a_truncation_trains_and_on_what_reward(truncation, expected):
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    sessions = [Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0),
                Session("t", "g", 1, 0, (call,), Status.AGENT_ERROR, 0.0),
                Session("t", "g", 2, 0, (call,), Status.TRUNCATED, 5.0),
                Session("t", "g", 3, 0, (call,), Status.INFRA_ERROR, None)]
    np.testing.assert_allclose(advantages(sessions, "mean", truncation=truncation), expected, rtol=1e-6)
    batch = pack(sessions, 3, estimator="mean", truncation=truncation)
    trained = {0, 1} if truncation == "mask" else {0, 1, 2}
    assert set(np.unique(batch[SESSION_INDEX_KEY][batch[RESPONSE_MASK_KEY] > 0])) == trained
    assert rows_needed(chain_lengths(sessions, 3, truncation=truncation), 3) == batch[IDS_KEY].shape[0]
    metrics = session_metrics(sessions, batch, truncation=truncation)
    assert metrics["reward/mean"] == pytest.approx({"mask": .5, "score": 2.0, "zero": 1 / 3}[truncation])
    assert ("masked/truncated" in metrics) == (truncation == "mask")


def test_a_scored_truncation_needs_a_reward_and_an_unknown_policy_is_refused():
    call = Call((1, 2), (3,), (-.1,), "length", 0)
    unscored = [Session("t", "g", 0, 0, (call,), Status.TRUNCATED, None)]
    with pytest.raises(ValueError, match="reward"):
        pack(unscored, 3, truncation="score")
    assert pack(unscored, 3, truncation="zero")[RESPONSE_MASK_KEY].sum() == 1
    with pytest.raises(ValueError, match="truncation"):
        pack(unscored, 3, truncation="drop")


def test_rows_pad_to_a_fixed_count_and_refuse_overflow():
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    rollouts = [Session("t", "g", sample, 0, (call,), Status.COMPLETED, float(sample)) for sample in range(4)]
    assert pack(rollouts, 6)[IDS_KEY].shape == (2, 6)
    assert pack(rollouts, 6, rows=5)[IDS_KEY].shape == (5, 6)
    with pytest.raises(ValueError, match="more than the 1"):
        pack(rollouts, 6, rows=1)
    with pytest.raises(ValueError, match="wider than"):
        pack(rollouts, 2)


def test_sampled_values_scatter_per_call_values_in_call_order():
    rng = np.random.default_rng(3)
    rollouts = [_rollout(rng, sample=index, reward=float(index))[0] for index in range(4)]
    batch = pack(rollouts, 64)
    placed = sampled_values(batch, lambda index, number: rollouts[index].calls[number].behavior_log_probs)
    np.testing.assert_array_equal(placed, batch[BEHAVIOR_LOG_PROBS_KEY])


def test_session_metrics_report_merge_masking_reward_latency_and_lag():
    first = Call((1, 2), (3,), (-.1,), "tool_calls", 4)
    merged = Call((1, 2, 3, 7), (8, 9), (-.2, -.3), "stop", 5)
    rewritten = Call((1, 5), (6,), (-.4,), "stop", 5)
    rollouts = [
        Session("t", "g", 0, 0, (first, merged), Status.COMPLETED, 1.0, {"tests": 1.0}),
        Session("t", "g", 1, 0, (first, rewritten), Status.AGENT_ERROR, 0.0, {"tests": 0.0}),
        Session("t", "g", 2, 0, (first,), Status.TRUNCATED, None),
        Session("u", "g", 3, 0, (first, merged), Status.INFRA_ERROR, None),
    ]
    batch = pack(rollouts, 8)
    metrics = session_metrics(rollouts, batch, latencies=[1.0, 2.0, 3.0, 40.0], version=6)
    assert metrics["merge/calls_per_chain"] == pytest.approx(4 / 3)
    assert metrics["status/completed"] == metrics["status/truncated"] == .25
    assert metrics["masked/truncated"] == pytest.approx(1 / 9)
    assert metrics["masked/infra_error"] == pytest.approx(3 / 9)
    assert metrics["reward/mean"] == .5
    assert metrics["reward/component/tests"] == .5
    assert metrics["latency/max"] == 40.0 and metrics["latency/p50"] == 2.5
    assert metrics["lag/max"] == 2 and metrics["lag/mean"] == pytest.approx((2 + 1 + 1 + 2 + 1) / 5)


def test_each_sampled_id_weighs_one_over_its_rollouts_sampled_count():
    """A rollout split over two chains still weighs as one: each of its
    sampled ids carries 1 / (sampled ids of the rollout), not of its chain."""
    first = Call((1, 2), (3, 4), (-.1, -.2), "tool_calls", 0)
    rewritten = Call((1, 9), (5,), (-.3,), "stop", 0)
    lonely = Call((6,), (7, 8, 9), (-.1, -.1, -.1), "stop", 0)
    rollouts = [Session("t", "g", 0, 0, (first, rewritten), Status.COMPLETED, 1.0),
                Session("t", "g", 1, 0, (lonely,), Status.COMPLETED, 0.0)]
    batch = pack(rollouts, 4)
    assert batch[SEGMENT_IDS_KEY].shape[0] == 3, "the first rollout splits into two rows"
    sampled = batch[RESPONSE_MASK_KEY] != 0
    for index, count in ((0, 3), (1, 3)):
        mine = sampled & (batch[SESSION_INDEX_KEY] == index)
        np.testing.assert_allclose(batch[SESSION_WEIGHTS_KEY][mine], 1 / count)
    assert (batch[SESSION_WEIGHTS_KEY][~sampled] == 0).all()


def test_group_advantages_match_hand_computed_values_across_uneven_groups():
    """Two groups of different sizes, one with a masked member: the group
    estimator normalises by each group's own deviation (ddof 1)."""
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    rollouts = [Session("a", "g", 0, 0, (call,), Status.COMPLETED, 1.0),
                Session("a", "g", 1, 0, (call,), Status.COMPLETED, 0.0),
                Session("a", "g", 2, 0, (call,), Status.AGENT_ERROR, 1.0),
                Session("a", "g", 3, 0, (call,), Status.INFRA_ERROR, None),
                Session("b", "g", 0, 0, (call,), Status.COMPLETED, 0.0),
                Session("b", "g", 1, 0, (call,), Status.COMPLETED, 1.0)]
    expected = [0.57735, -1.15470, 0.57735, 0.0, -0.70711, 0.70711]
    np.testing.assert_allclose(advantages(rollouts), expected, atol=1e-4)
    batch = pack(rollouts, 3)
    for index, value in enumerate(expected):
        mine = batch[SESSION_INDEX_KEY] == index
        np.testing.assert_allclose(batch[ADVANTAGES_KEY][mine], value, atol=1e-4)


def test_chains_report_the_ids_pack_would_place():
    first = Call((1, 2), (3,), (-.1,), "tool_calls", 0)
    appended = Call((1, 2, 3, 4), (5,), (-.2,), "stop", 0)
    rewritten = Call((1, 9), (6,), (-.3,), "stop", 0)
    session = Session("t", "g", 0, 0, (first, appended, rewritten), Status.COMPLETED, 1.0)
    assert chains(session, 8) == ((1, 2, 3, 4, 5), (1, 9, 6))


@pytest.mark.parametrize("value", [None, "1.0", True, float("nan")])
def test_a_non_number_reward_or_likelihood_is_a_value_error(value):
    if value is not None:  # None is how an unscored session says it has no reward
        with pytest.raises(ValueError):
            Session("t", "g", 0, 0, (), Status.CANCELLED, value)
    with pytest.raises(ValueError):
        Call((1,), (2,), (value,), "stop", 0)


@pytest.mark.parametrize("value", [np.int64(1), np.int32(0), np.float32(-0.5), 2])
def test_numpy_and_python_numbers_are_accepted_as_rewards_and_likelihoods(value):
    session = Session("t", "g", 0, 0, (), Status.COMPLETED, value)
    assert type(session.reward) is float and session.reward == float(value)
    call = Call((1,), (2,), (value,), "stop", 0)
    assert type(call.behavior_log_probs[0]) is float
