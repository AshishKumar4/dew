"""Strict prefix merge and packing of engine-sourced rollouts.

Rollouts are generated with a known chain structure: a call either appends
to the full history so far (it must merge) or rewrites some id of it (it
must split). The packed batch is then read back through its segment ids and
positions alone and compared with that structure.
"""

import numpy as np
import pytest
from reference_error import assert_fp32_reduction_bound

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
    pack,
    packed_rows,
    planned,
    planned_rows,
    sampled_values,
    session_metrics,
)
from dew.rl.advantage import GROUP_EPS


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
        history = call.prompt + call.tokens
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
        wanted = {rollout.calls[calls[-1]].prompt + rollout.calls[calls[-1]].tokens: calls
                  for calls in expected}
        for row, start, stop in mine:
            tokens = tuple(int(value) for value in batch[IDS_KEY][row, start:stop])
            assert tokens in wanted
            calls = wanted.pop(tokens)
            sampled = np.flatnonzero(batch[RESPONSE_MASK_KEY][row, start:stop]) + start
            expected_ids = [token for number in calls for token in rollout.calls[number].tokens]
            np.testing.assert_array_equal(batch[IDS_KEY][row, sampled], expected_ids)
            np.testing.assert_array_equal(
                batch[BEHAVIOR_LOG_PROBS_KEY][row, sampled],
                np.asarray(
                    [p for number in calls for p in rollout.calls[number].behavior_log_probs], np.float32
                ),
            )
            np.testing.assert_array_equal(
                batch[VERSIONS_KEY][row, sampled],
                [number for number in calls for _ in rollout.calls[number].tokens])
            np.testing.assert_array_equal(
                batch[CALL_INDEX_KEY][row, sampled],
                [number for number in calls for _ in rollout.calls[number].tokens])
            others = np.setdiff1d(np.arange(start, stop), sampled)
            assert (batch[VERSIONS_KEY][row, others] == -1).all()
            assert (batch[BEHAVIOR_LOG_PROBS_KEY][row, others] == 0).all()
        assert not wanted
    pad = batch[SEGMENT_IDS_KEY] == 0
    assert (batch[SESSION_INDEX_KEY][pad] == -1).all() and (batch[RESPONSE_MASK_KEY][pad] == 0).all()
    # Session weights give each trainable rollout unit mass.
    for index in range(len(rollouts)):
        mine = batch[SESSION_INDEX_KEY] == index
        weights = batch[SESSION_WEIGHTS_KEY][mine]
        assert_fp32_reduction_bound(weights.sum(), 1.0, np.sum(np.abs(weights)), weights.size)
        # Per session, every call takes the session's advantage, as every id of its chains.
        [value] = set(advantages(rollouts)[index].tolist())
        assert (batch[ADVANTAGES_KEY][mine] == np.float32(value)).all()


def test_masked_rollouts_take_no_rows_and_no_baseline():
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    rollouts = [Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0),
                Session("t", "g", 1, 0, (call,), Status.AGENT_ERROR, 0.0),
                Session("t", "g", 2, 0, (call,), Status.TRUNCATED, 5.0),
                Session("t", "g", 3, 0, (call,), Status.INFRA_ERROR, None)]
    batch = pack(rollouts, 3)
    assert set(np.unique(batch[SESSION_INDEX_KEY])) == {0, 1}
    # One call a session: its advantage is the session's.
    np.testing.assert_array_equal(np.concatenate(advantages(rollouts, "mean")), [.5, -.5, 0, 0])
    lonely = [rollouts[0], rollouts[2]]
    assert (np.concatenate(advantages(lonely)) == 0).all()


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
    # Each trained reward less its group's float32 mean: a mean of n rewards and one difference, each
    # rounding within float32's reduction bound of every reward's magnitude.
    rewards = [1.0, 0.0, 5.0]
    assert_fp32_reduction_bound(np.concatenate(advantages(sessions, "mean", truncation=truncation)), expected,
                                2 * sum(map(abs, rewards)), len(rewards) + 1)
    batch = pack(sessions, 3, estimator="mean", truncation=truncation)
    trained = {0, 1} if truncation == "mask" else {0, 1, 2}
    assert set(np.unique(batch[SESSION_INDEX_KEY][batch[RESPONSE_MASK_KEY] > 0])) == trained
    assert packed_rows(sessions, 3, truncation=truncation) == batch[IDS_KEY].shape[0]
    metrics = session_metrics(sessions, batch, truncation=truncation)
    # The trained rewards' float64 mean, as their sum over their count: exactly these quotients.
    assert metrics["reward/mean"] == {"mask": 1 / 2, "score": 6 / 3, "zero": 1 / 3}[truncation]
    assert ("masked/truncated" in metrics) == (truncation == "mask")


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
    # Counts' quotients, which Python's float division rounds as the literal's does.
    assert metrics["merge/calls_per_chain"] == 4 / 3
    # A group's prompt laid once is no chain of its own.
    shared = session_metrics(rollouts, pack(rollouts, 8, share=True), latencies=[1.0, 2.0, 3.0, 40.0],
                             version=6)
    assert shared["merge/calls_per_chain"] == 4 / 3
    assert metrics["status/completed"] == metrics["status/truncated"] == .25
    assert metrics["masked/truncated"] == 1 / 9
    assert metrics["masked/infra_error"] == 3 / 9
    assert metrics["reward/mean"] == .5
    assert metrics["reward/component/tests"] == .5
    assert metrics["latency/max"] == 40.0 and metrics["latency/p50"] == 2.5
    assert metrics["lag/max"] == 2 and metrics["lag/mean"] == (2 + 1 + 1 + 2 + 1) / 5


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
    # Group a trains on 1, 0, 1: mean 2/3, deviation sqrt(1/3); group b on 0, 1: mean 1/2, deviation
    # sqrt(1/2); each over its deviation plus GROUP_EPS. Each value rounds in float32 through its group's
    # mean and variance (n terms each), a root, the epsilon's sum, a difference and a quotient: within
    # float32's bound over 2n + 4 roundings of its magnitude.
    a, b = np.sqrt(1 / 3) + GROUP_EPS, np.sqrt(1 / 2) + GROUP_EPS
    expected = np.asarray([(1 / 3) / a, (-2 / 3) / a, (1 / 3) / a, 0.0, (-1 / 2) / b, (1 / 2) / b])
    terms = np.asarray([10, 10, 10, 1, 8, 8])
    found = np.concatenate(advantages(rollouts))
    assert_fp32_reduction_bound(found, expected, np.abs(expected), terms)
    batch = pack(rollouts, 3)
    for index, value in enumerate(found):
        mine = batch[SESSION_INDEX_KEY] == index
        assert (batch[ADVANTAGES_KEY][mine] == value).all()


def test_groups_planned_apart_count_the_rows_their_batch_packs_in():
    """Two groups planned one at a time (`planned`, as a scheduler admits them) count the rows their
    sessions pack in together (`packed_rows`), their prompts shared: the second group's routing record lays
    routing out for the whole batch, so its unrecorded chain's prompt id agrees with the first group's
    unrecorded chain's, and the two share it, which planned apart under their own layouts they did not
    (7, 6 and 6 of 10 ids need three rows, where 6 and 4 twice fill two)."""
    def session(task, sample, prompt, tokens, routed=None):
        call = Call(prompt, tokens, tuple(-0.1 for _ in tokens), "stop", 0, routed_experts=routed)
        return Session(task, task, sample, 0, (call,), Status.COMPLETED, float(sample))

    first = [session("t", 0, (1,), (2, 3, 4)), session("t", 1, (9,), (10, 11, 12, 13, 14))]
    second = [session("u", 0, (1,), (5, 6, 7)),
              session("u", 1, (20,), (21, 22, 23, 24, 25), np.ones((5, 2, 2), np.uint8))]
    plans = [planned(first, 10, truncation="mask", share=True)]
    plans.append(planned(second, 10, truncation="mask", share=True, first=len(plans[0].lengths)))
    assert planned_rows(plans, 10) == packed_rows([*first, *second], 10, truncation="mask", share=True) == 3


def test_every_shared_pack_holds_its_rows_parents_whatever_they_share():
    """Packs that share their groups' prompts (`pack`'s `share`) hold the parents' table
    (`FORK_PARENTS_KEY`) whatever their rows share: a group whose chains share their prompt names its
    parents, and one whose chains share none and one that trains no session hold parents of none, so a
    rollout's shares pack one tree, whose documents' count the pool agrees (`agreed_records`)."""
    from dew.objectives.rl.sessions import FORK_PARENTS_KEY

    def session(group, sample, prompt, sampled, status=Status.COMPLETED, reward=1.0):
        call = Call(prompt, sampled, tuple(-0.1 for _ in sampled), "stop", 0)
        return Session("t", group, sample, 0, (call,), status, reward)

    shared = [session("g", sample, (1, 2, 3, 4), (5 + sample, 6)) for sample in range(2)]
    apart = [session("h", sample, (7 + sample, 8), (9, 10)) for sample in range(2)]
    untrained = [session("u", 0, (1, 2, 3, 4), (5,), Status.TRUNCATED, None)]
    packs = [pack(sessions, 16, rows=2, share=True) for sessions in (shared, apart, untrained)]
    assert [sorted(batch) for batch in packs[1:]] == [sorted(packs[0])] * 2
    assert [batch[FORK_PARENTS_KEY].shape[0] for batch in packs] == [2, 2, 2]
    assert [bool(np.any(batch[FORK_PARENTS_KEY] > 0)) for batch in packs] == [True, False, False]
