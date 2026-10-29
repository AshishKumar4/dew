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


def test_an_episode_keeps_its_compaction_and_its_parsed_requests_through_its_record():
    """An episode whose first call's harness parsed a tool call from it and whose observation after it
    compacted the context reads back from its JSON record unchanged: the transition keeps its request
    and the observation its compaction. Its session gives the first call's transcript the task's
    messages and the request as its response, and the next call a transcript of the compacted
    observation's messages alone, the task's dropped with the history it replaced."""
    from dew.objectives.rl.records import episode_from_record, record_json

    trainer, collector = build()
    first = collect(collector, trainer.initial_state())[0]
    task = ({"role": "user", "content": "find it"},)
    request = {"role": "assistant", "tool_calls": [{"id": "call-0", "type": "function",
                                                    "function": {"name": "search", "arguments": "{}"}}]}
    summary = ({"role": "user", "content": "the search found it; the history so far, summarized"},)
    one = first.transitions[0]
    turns = (replace(one, request=request,
                     observation=replace(one.observation, messages=summary, compacted=True)),
             *first.transitions[1:])
    episode = replace(first, initial=replace(first.initial, messages=task), transitions=turns)
    assert episode_from_record(json.loads(record_json(episode))) == episode
    calls = session_of(episode, group="g").calls
    assert (calls[0].transcript.messages, calls[0].transcript.response) == (task, request)
    assert calls[1].transcript.messages == summary
