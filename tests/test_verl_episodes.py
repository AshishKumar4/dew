"""Lossless episode interchange through pinned verl's real rollout models."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from dew.objectives.rl.verl import from_verl, to_verl
from test_tool_episodes import build, collect

FIXTURE = Path(__file__).parent / "fixtures/rl/verl_episodes.json"


def test_verl_model_and_tensor_mapping_preserve_actions_and_likelihoods():
    """verl d040717: ids/masks exact and behavior-probability maximum difference 0."""
    reference = json.loads(FIXTURE.read_text())
    rows = [output["wire"] for output in reference["outputs"]]
    episodes = from_verl(rows)
    assert json.loads(json.dumps(to_verl(episodes))) == rows
    for episode in episodes:
        for turn in episode.transitions:
            assert turn.action.raw_log_probs != turn.action.behavior_log_probs
    for row, output in zip(rows, reference["outputs"], strict=True):
        tensors = output["tensors"]
        np.testing.assert_array_equal(row["prompt_ids"], tensors["prompts"])
        np.testing.assert_array_equal(row["response_ids"], tensors["responses"])
        np.testing.assert_array_equal(row["response_mask"], tensors["response_mask"])
        difference = np.max(np.abs(np.asarray(row["response_logprobs"]) - tensors["rollout_log_probs"]))
        assert difference == 0, f"verl behavior likelihood max difference {difference}"
        expected = np.zeros(len(row["response_ids"]))
        expected[-1] = row["reward_score"]
        np.testing.assert_array_equal(tensors["rm_scores"], expected)


def test_actual_rollout_survives_json_round_trip_and_context_compaction():
    trainer, rollout = build()
    episodes = collect(rollout, trainer.initial_state())
    # A later model call may see a compacted context. Concatenation would
    # score different states; independent verl call rows retain this input.
    first = episodes[0]
    turn = first.transitions[1]
    compact = replace(turn, action=replace(turn.action, context=turn.action.context[-1:]))
    episodes = (replace(first, transitions=(first.transitions[0], compact)), *episodes[1:])
    restored = from_verl(json.loads(json.dumps(to_verl(episodes))))
    assert restored == episodes
    before, after = rollout.project(episodes), rollout.project(restored)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("corruption", ["raw", "mask", "missing_turn", "order", "reward"])
def test_import_rejects_lost_provenance_and_partial_episodes(corruption):
    rows = [output["wire"] for output in json.loads(FIXTURE.read_text())["outputs"]]
    rows = deepcopy(rows)
    if corruption == "raw":
        del rows[0]["extra_fields"]["dew"]["action"]["raw_log_probs"]
    elif corruption == "mask":
        rows[0]["response_mask"][0] = 0
    elif corruption == "missing_turn":
        rows.pop(0)
    elif corruption == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["reward_score"] += 1
    with pytest.raises(ValueError):
        from_verl(rows)
