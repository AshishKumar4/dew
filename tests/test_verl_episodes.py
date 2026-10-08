"""Collected tool episodes through verl's native rows and back.

An episode becomes a `Session` (`session_of`), which `to_verl` writes as one
`AgentLoopOutput` row per strict chain; the rows read back to the same
rollouts and pack to the same training batch.
"""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from episode_support import build, collect

from dew.objectives.rl import session_of
from dew.objectives.rl.sessions import pack
from dew.objectives.rl.verl import from_verl, to_verl


def rollouts_of(episodes):
    return [session_of(episode, group=str(episode.identity.task)) for episode in episodes]


def test_collected_episodes_survive_verl_rows_and_context_compaction():
    """A later call that saw a compacted context breaks the strict chain, so
    that rollout takes two verl rows; both come back to the same calls."""
    trainer, rollout = build()
    episodes = collect(rollout, trainer.initial_state())
    first = episodes[0]
    turn = first.transitions[1]
    compact = replace(turn, action=replace(turn.action, context=turn.action.context[-1:]))
    episodes = (replace(first, transitions=(first.transitions[0], compact)), *episodes[1:])
    rollouts = rollouts_of(episodes)
    rows = json.loads(json.dumps(to_verl(rollouts)))
    assert len(rows) == len(rollouts) + 1
    assert rows[0]["extra_fields"]["dew"]["chains"] == 2
    restored = [trajectory.session for trajectory in from_verl(rows)]
    assert restored == rollouts
    before, after = pack(rollouts, 64), pack(restored, 64)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


@pytest.mark.parametrize("field", ["context", "tokens"])
def test_action_construction_rejects_boolean_token_ids(field):
    trainer, rollout = build()
    action = collect(rollout, trainer.initial_state())[0].transitions[0].action
    ids = action.context if field == "context" else action.tokens
    with pytest.raises(ValueError, match="integer"):
        replace(action, **{field: (True, *ids[1:])})


def test_an_episode_record_reads_back_every_sampling_control():
    """A recorded action's policy comes back whole, its penalties and stop
    strings included, and an unset pad id stays the task's to fill."""
    from dew.objectives.rl.records import episode_from_record
    from dew.sampling import Sampling

    trainer, rollout = build()
    episode = collect(rollout, trainer.initial_state())[0]
    policy = Sampling(temperature=0.7, eos_token_ids=episode.transitions[0].action.sampling.eos_token_ids,
                      repetition_penalty=1.1, no_repeat_ngram_size=2, stop=("\n\n",))
    turns = tuple(replace(turn, action=replace(turn.action, sampling=policy)) for turn in episode.transitions)
    episode = replace(episode, transitions=turns)

    restored = episode_from_record(json.loads(json.dumps(asdict(episode))))

    assert restored == episode
    assert restored.transitions[0].action.sampling.pad_token_id is None
