"""Strict prefix merge and packing of engine-sourced rollouts.

Rollouts are generated with a known chain structure: a call either appends
to the full history so far (it must merge) or rewrites some id of it (it
must split). The packed batch is then read back through its segment ids and
positions alone and compared with that structure.
"""

import numpy as np
import pytest

from dew.objectives.rl.rollout import ADVANTAGES_KEY, BEHAVIOR_LOG_PROBS_KEY, IDS_KEY, RESPONSE_MASK_KEY
from dew.objectives.rl.rollouts import (
    CALL_INDEX_KEY,
    POSITIONS_KEY,
    ROLLOUT_INDEX_KEY,
    ROLLOUT_WEIGHTS_KEY,
    SEGMENT_IDS_KEY,
    VERSIONS_KEY,
    Call,
    Rollout,
    Status,
    advantages,
    pack,
    sampled_values,
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
    return Rollout(task, group, sample, 0, tuple(calls), status, reward), expected


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
            owners = np.unique(batch[ROLLOUT_INDEX_KEY][row, start:stop])
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
    assert (batch[ROLLOUT_INDEX_KEY][pad] == -1).all() and (batch[RESPONSE_MASK_KEY][pad] == 0).all()
    # Rollout weights give each trainable rollout unit mass.
    for index in range(len(rollouts)):
        mine = batch[ROLLOUT_INDEX_KEY] == index
        assert batch[ROLLOUT_WEIGHTS_KEY][mine].sum() == pytest.approx(1.0)
        assert (batch[ADVANTAGES_KEY][mine] == advantages(rollouts)[index]).all()


def test_a_prompt_only_prefix_does_not_merge():
    """Section 6: the template dropped the sampled reasoning from history."""
    first = Call((1, 2, 3), (7, 8, 9), (-.1, -.2, -.3), "tool_calls", 0)
    rewritten = Call((1, 2, 3, 9, 4), (5,), (-.5,), "stop", 0)
    appended = Call((1, 2, 3, 7, 8, 9, 4), (5,), (-.5,), "stop", 0)
    split = pack([Rollout("t", "g", 0, 0, (first, rewritten), Status.COMPLETED, 1.0)], 16)
    merged = pack([Rollout("t", "g", 0, 0, (first, appended), Status.COMPLETED, 1.0)], 16)
    assert split[SEGMENT_IDS_KEY].max() == 2
    assert merged[SEGMENT_IDS_KEY].max() == 1
    np.testing.assert_array_equal(merged[IDS_KEY][0, :8], [1, 2, 3, 7, 8, 9, 4, 5])
    np.testing.assert_array_equal(merged[RESPONSE_MASK_KEY][0, :8], [0, 0, 0, 1, 1, 1, 0, 1])


def test_masked_rollouts_take_no_rows_and_no_baseline():
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    rollouts = [Rollout("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0),
                Rollout("t", "g", 1, 0, (call,), Status.AGENT_ERROR, 0.0),
                Rollout("t", "g", 2, 0, (call,), Status.TRUNCATED, 5.0),
                Rollout("t", "g", 3, 0, (call,), Status.INFRA_ERROR, None)]
    batch = pack(rollouts, 3)
    assert set(np.unique(batch[ROLLOUT_INDEX_KEY])) == {0, 1}
    np.testing.assert_allclose(advantages(rollouts, "mean"), [.5, -.5, 0, 0])
    lonely = [rollouts[0], rollouts[2]]
    assert (advantages(lonely) == 0).all()


def test_rows_pad_to_a_fixed_count_and_refuse_overflow():
    call = Call((1, 2), (3,), (-.1,), "stop", 0)
    rollouts = [Rollout("t", "g", sample, 0, (call,), Status.COMPLETED, float(sample)) for sample in range(4)]
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
    placed = sampled_values(batch, rollouts, lambda rollout, number: rollout.calls[number].behavior_log_probs)
    np.testing.assert_array_equal(placed, batch[BEHAVIOR_LOG_PROBS_KEY])
