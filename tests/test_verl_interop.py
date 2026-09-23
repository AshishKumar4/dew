"""Native verl interop against fixtures verl 12ebe0c itself wrote.

`tools/verl_interop_reference.py` produced both files in an isolated env:
`verl_gsm8k_tool_agent.parquet` is the first rows of verl's
`gsm8k_tool_agent_loop.py` preprocessing, and `verl_native.json` holds two
`AgentLoopOutput` trajectories dumped as
verl dumps them (a tool loop with routing, a video single turn with its SGLang
processor payload), their `as_dict` tensors and the model's field names.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from dew.data.prompts import INFO_KEY, SOURCE_KEY, TRUTH_KEY, PromptSource
from dew.objectives.rl.sessions import (
    BEHAVIOR_LOG_PROBS_KEY,
    ROUTED_EXPERTS_KEY,
    ROUTED_KEY,
    SUPPORT_KEY,
    Call,
    Session,
    Status,
    pack,
)
from dew.objectives.rl.verl import FIELDS, from_verl, to_verl, verl_reward

FIXTURES = Path(__file__).parent / "fixtures/rl"
NATIVE = json.loads((FIXTURES / "verl_native.json").read_text())
PARQUET = FIXTURES / "verl_gsm8k_tool_agent.parquet"


def read(row):
    return bytes(row[row != 0].astype(np.uint8)).decode("utf-8")


def native_rows():
    return [output["wire"] for output in NATIVE["outputs"]]


def without_dew(row):
    extra = {key: value for key, value in row["extra_fields"].items() if key != "dew"}
    return {**row, "extra_fields": extra}


def test_dew_writes_exactly_the_agent_loop_output_fields():
    assert tuple(NATIVE["fields"]) == FIELDS


def test_native_rows_round_trip_through_rollouts():
    """Import, then export, gives verl's own rows back, and each rollout
    rebuilds the calls the tool loop made: two assistant turns, the tool
    response between them in the second call's prompt."""
    rows = native_rows()
    trajectories = from_verl(rows, media=True)
    assert [without_dew(row) for row in json.loads(json.dumps(to_verl(trajectories)))] == rows
    tool = trajectories[0].session
    first, second = tool.calls
    wire = rows[0]
    assert first.prompt_ids == tuple(wire["prompt_ids"])
    assert first.sampled_ids == tuple(wire["response_ids"][:5])
    assert second.prompt_ids == tuple(wire["prompt_ids"] + wire["response_ids"][:15])
    assert second.sampled_ids == tuple(wire["response_ids"][15:])
    assert (first.finish_reason, second.finish_reason) == ("tool_calls", "stop")
    assert first.version == second.version == 7
    assert trajectories[1].extras["mm_processor_output"] == rows[1]["mm_processor_output"]


def test_packed_native_rows_match_verls_tensor_mapping():
    """Packing the imported rollouts puts verl's `as_dict` tensors where the
    trainer reads them: the ids, the behavior likelihoods on sampled ids,
    and routing aligned to absolute positions (verl zero-fills the row the
    engine never forwarded; Dew marks it unrouted)."""
    trajectories = from_verl(native_rows(), media=True)
    for trajectory, output in zip(trajectories, NATIVE["outputs"], strict=True):
        tensors = output["tensors"]
        batch = pack([trajectory.session], 64)
        prompts, responses = tensors["prompts"], tensors["responses"]
        total = len(prompts) + len(responses)
        np.testing.assert_array_equal(batch["input_ids"][0, :total], prompts + responses)
        mask = batch["response_mask"][0, len(prompts):total]
        np.testing.assert_array_equal(mask, tensors["response_mask"])
        np.testing.assert_array_equal(
            batch[BEHAVIOR_LOG_PROBS_KEY][0, len(prompts):total],
            np.asarray(tensors["rollout_log_probs"], np.float32))
        np.testing.assert_array_equal(batch["advantages"].shape, (1, 64))
        if "routed_experts" in tensors:
            routed = np.asarray(tensors["routed_experts"])
            np.testing.assert_array_equal(batch[ROUTED_EXPERTS_KEY][0, :total - 1], routed[:total - 1])
            assert batch[ROUTED_KEY][0].tolist() == [True] * (total - 1) + [False] * (64 - total + 1)


def test_rollouts_round_trip_through_verl_rows_losslessly():
    """Dew's own metadata restores what verl's fields lack: identity,
    status, finish reasons, versions and the sampling support, across a
    rollout whose history broke the strict chain (two rows)."""
    routed = np.arange(4 * 2 * 2).reshape(4, 2, 2) % 5
    calls = (Call((1, 2), (3, 4), (-0.5, -0.25), "tool_calls", 3, routed_experts=routed[:3],
                  support=((3, 9), (4,))),
             Call((1, 2, 3, 4, 7), (8,), (-1.0,), "length", 4, routed_experts=routed[:, :, :].repeat(2, 0)[:5]),
             Call((5, 6), (2,), (-0.125,), "stop", 4))
    rollout = Session("task", "g", 1, 2, calls, Status.COMPLETED, 0.5, {"tests": 0.5}, "log://1")
    failed = Session("task", "g", 2, 0, (Call((1,), (2,), (-3.0,), "abort", 4),), Status.INFRA_ERROR, None)
    rows = json.loads(json.dumps(to_verl([rollout, failed])))
    assert len(rows) == 3
    assert rows[0]["extra_fields"]["min_global_steps"] == 3 and rows[0]["extra_fields"]["max_global_steps"] == 4
    restored = [trajectory.session for trajectory in from_verl(rows)]
    assert restored == [rollout, failed]
    for key, value in pack([rollout], 16).items():
        np.testing.assert_array_equal(pack(restored[:1], 16)[key], value, err_msg=key)
    assert SUPPORT_KEY in pack(restored[:1], 16)


@pytest.mark.parametrize("corruption", ["logprobs", "version", "reward", "chain", "field"])
def test_import_refuses_rows_that_lost_what_training_needs(corruption):
    rows = json.loads(json.dumps(native_rows()))
    if corruption == "logprobs":
        rows[0]["response_logprobs"] = None
    elif corruption == "version":
        del rows[0]["extra_fields"]["min_global_steps"]
    elif corruption == "reward":
        rows[0]["reward_score"] = None
    elif corruption == "chain":
        rows = to_verl([Session("t", "g", 0, 0, (Call((1,), (2,), (-1.0,), "stop", 0),
                                                 Call((5,), (6,), (-1.0,), "stop", 0)), Status.COMPLETED, 1.0)])
        rows.pop(0)
    else:
        rows[0]["teacher_ids"] = []
    with pytest.raises(ValueError):
        from_verl(rows, media=True)
    if corruption == "version":
        assert from_verl(rows, version=9, media=True)[0].session.calls[0].version == 9


def test_native_rows_group_as_verl_interleaves_its_samples():
    rows = native_rows() * 2
    rollouts = [trajectory.session for trajectory in from_verl(rows, samples=2, media=True)]
    assert [(rollout.task, rollout.sample) for rollout in rollouts] == [("0", 0), ("0", 1), ("1", 0), ("1", 1)]


def test_the_verl_rl_parquet_reads_its_nested_ground_truth():
    table = pq.read_table(PARQUET).to_pylist()
    rows = PromptSource.from_parquet(str(PARQUET), "tests/fixtures/tokenizers/tiny-tools", 512, 0)
    for index, expected in enumerate(table):
        batch = rows[index]
        assert read(batch[SOURCE_KEY]) == expected["data_source"] == "openai/gsm8k"
        assert read(batch[TRUTH_KEY]) == expected["reward_model"]["ground_truth"]
        info = json.loads(read(batch[INFO_KEY]))
        assert info["tools_kwargs"] == expected["extra_info"]["tools_kwargs"]


def test_calls_that_sampled_nothing_survive_the_round_trip():
    """An aborted call before its first token is a legal call; alone in its
    chain or between two others, it comes back where it was."""
    calls = (Call((1, 2), (3,), (-0.5,), "tool_calls", 1),
             Call((1, 2, 3, 4), (), (), "abort", 1),
             Call((1, 2, 3, 4, 5), (6,), (-0.25,), "stop", 2))
    between = Session("t", "g", 0, 0, calls, Status.INFRA_ERROR, None)
    alone = Session("t", "g", 1, 0, (Call((7,), (), (), "abort", 3),), Status.CANCELLED, None)
    rows = json.loads(json.dumps(to_verl([between, alone])))
    assert [trajectory.session for trajectory in from_verl(rows)] == [between, alone]


def test_a_native_rows_step_stamps_survive_two_round_trips():
    """verl reads `max_global_steps` for staleness; Dew keeps the stamps a
    native row came with and drops only the ones it wrote itself."""
    row = json.loads(json.dumps(native_rows()[0]))
    row["extra_fields"]["max_global_steps"] = 9
    once = json.loads(json.dumps(to_verl(from_verl([row]))))
    twice = json.loads(json.dumps(to_verl(from_verl(once))))
    for exported in (once, twice):
        assert (exported[0]["extra_fields"]["min_global_steps"], exported[0]["extra_fields"]["max_global_steps"]) == (7, 9)
    written = to_verl([Session("t", "g", 0, 0, (Call((1,), (2,), (-1.0,), "stop", 4),), Status.COMPLETED, 1.0)])
    assert "extra_fields" not in from_verl(json.loads(json.dumps(written)))[0].extras


def test_a_scorer_is_called_by_keyword_as_verl_calls_it():
    def compute_score(*, solution_str, data_source, extra_info=None, ground_truth):
        return {"score": float(solution_str == ground_truth and data_source == "src" and extra_info == {"k": 1})}

    assert verl_reward(compute_score)("src", "42", "42", '{"k": 1}') == 1.0


def test_media_rows_are_refused_for_training():
    rows = native_rows()
    with pytest.raises(ValueError, match="media"):
        from_verl(rows)


def test_text_rows_with_the_empty_media_verls_loops_write_are_training_rows():
    """ToolAgentLoop and SingleTurnAgentLoop write `multi_modal_data={}` and
    `mm_processor_kwargs={}` on every text-only trajectory (tool_agent_loop.py
    L74, L185-L199 at 12ebe0c); those rows carry no media and import for
    training, and the empty dicts come back out as they went in."""
    row = json.loads(json.dumps(native_rows()[0]))
    row.update(multi_modal_data={}, mm_processor_kwargs={})
    (trajectory,) = from_verl([row])
    assert trajectory.session.calls
    assert without_dew(json.loads(json.dumps(to_verl([trajectory])))[0]) == row
