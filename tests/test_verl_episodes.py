"""Collected tool episodes through verl's native rows and back.

An episode becomes a `Session` (`session_of`), which `to_verl` writes as one
`AgentLoopOutput` row per strict chain; the rows read back to the same
rollouts and pack to the same training batch.
"""

import json
from dataclasses import asdict, replace

import numpy as np
from episode_support import build, collect

from dew.objectives.rl import session_of
from dew.objectives.rl.sessions import pack
from dew.objectives.rl.verl import from_verl, to_verl


def rollouts_of(episodes):
    return [session_of(episode, group=str(episode.identity.task)) for episode in episodes]


def test_collected_episodes_survive_verl_rows_and_context_compaction():
    """A later call that saw a compacted context breaks the strict chain, so
    that rollout takes two verl rows; both come back to the same calls."""
    trainer, collector = build()
    episodes = collect(collector, trainer.initial_state())
    first = episodes[0]
    turn = first.transitions[1]
    compact = replace(turn, action=replace(turn.action, prompt=turn.action.prompt[-1:]))
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


def test_dews_record_of_a_verl_row_is_its_sessions_and_its_calls_own():
    """A row's `extra_fields.dew` holds its session's record and each call's, as their dataclasses declare
    them (`verl.DewRow`), in the bytes the hand-written record had, and reads back to the session."""
    from dew.objectives.rl.sessions import chains

    trainer, collector = build()
    session = rollouts_of(collect(collector, trainer.initial_state()))[0]
    built = chains(session, 0, 1 << 62)
    rows = json.loads(json.dumps(to_verl([session])))
    identity = {"task": session.task, "group": session.group, "sample": session.sample,
                "attempt": session.attempt, "status": session.status.value,
                "components": dict(session.components), "detail": session.detail, "source": session.source}

    def written(call, first, count):
        transcript = None if call.transcript is None else {
            "messages": [dict(message) for message in call.transcript.messages],
            "response": None if call.transcript.response is None else dict(call.transcript.response)}
        return {"start": first, "count": count, "finish_reason": call.finish_reason, "version": call.version,
                "raw_log_probs": None if call.raw_log_probs is None else list(call.raw_log_probs),
                "support": None if call.support is None else [list(kept) for kept in call.support],
                "reward": call.reward, "transcript": transcript}

    for number, (row, chain) in enumerate(zip(rows, built, strict=True)):
        assert json.dumps(row["extra_fields"]["dew"]) == json.dumps({
            "session": identity, "chain": number, "chains": len(built),
            "calls": [written(session.calls[index], first, count) for index, first, count in chain.members],
            "stamped": True})
    assert [trajectory.session for trajectory in from_verl(rows)] == [session]


def test_an_episode_record_reads_back_every_sampling_control():
    """A recorded episode's sampling policy comes back whole, its penalties and stop strings
    included, and an unset pad id stays the task's to fill."""
    from dew.objectives.rl.records import episode_from_record
    from dew.sampling import Sampling

    trainer, collector = build()
    episode = collect(collector, trainer.initial_state())[0]
    policy = Sampling(temperature=0.7, eos_token_ids=episode.sampling.eos_token_ids,
                      repetition_penalty=1.1, no_repeat_ngram_size=2, stop=("\n\n",))
    episode = replace(episode, sampling=policy)

    restored = episode_from_record(json.loads(json.dumps(asdict(episode))))

    assert restored == episode
    assert restored.sampling.pad_token_id is None


def test_an_episode_record_reads_back_the_engine_records_its_draws_carry():
    """A draw an engine served with its routing and its sampler's kept ids, as an environment
    source's episodes keep them, writes to JSON and reads back whole."""
    import numpy as np

    from dew.objectives.rl.records import episode_from_record, record_json

    trainer, collector = build()
    episode = collect(collector, trainer.initial_state())[0]
    turn = episode.transitions[0]
    draw = turn.action
    forwarded = len(draw.prompt) + len(draw.tokens) - 1
    routed = np.arange(forwarded * 2 * 2, dtype=np.int32).reshape(forwarded, 2, 2) % 4
    support = tuple((token, token + 1) for token in draw.tokens)
    served = replace(draw, routed_experts=routed, support=support)
    episode = replace(episode, transitions=(replace(turn, action=served), *episode.transitions[1:]))

    restored = episode_from_record(json.loads(record_json(episode)))

    assert restored == episode
    np.testing.assert_array_equal(restored.transitions[0].action.routed_experts, routed)
