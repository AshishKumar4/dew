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

from dew.data import Files, Loading, presets
from dew.data.compile import iterate
from dew.data.prompts import INFO_KEY, SOURCE_KEY, TRUTH_KEY, PromptRecord
from dew.objectives.rl.sessions import (
    BEHAVIOR_LOG_PROBS_KEY,
    ROUTED_EXPERTS_KEY,
    ROUTED_KEY,
    SUPPORT_KEY,
    Call,
    Session,
    Status,
    Transcript,
    pack,
)
from dew.objectives.rl.verl import compute_score, from_verl, to_verl, verl_reward

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
    assert first.prompt == tuple(wire["prompt_ids"])
    assert first.tokens == tuple(wire["response_ids"][:5])
    assert second.prompt == tuple(wire["prompt_ids"] + wire["response_ids"][:15])
    assert second.tokens == tuple(wire["response_ids"][15:])
    assert (first.finish_reason, second.finish_reason) == ("tool_calls", "stop")
    assert first.version == second.version == 7
    assert trajectories[1].extras["mm_processor_output"] == rows[1]["mm_processor_output"]


@pytest.mark.parametrize("count", [0, -1, 3])
def test_a_session_that_spans_no_rows_or_more_than_remain_is_refused(count):
    """A row Dew wrote names how many rows its session spans; a count of
    none, below none, or past the rows given is refused rather than read
    (a count of none once never advanced and read forever)."""
    rows = json.loads(json.dumps(to_verl(from_verl(native_rows(), media=True))))
    assert [row["extra_fields"]["dew"]["chains"] for row in rows] == [1, 1]
    rows[0]["extra_fields"]["dew"]["chains"] = count
    with pytest.raises(ValueError):
        from_verl(rows, media=True)


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
    status, finish reasons, versions, step rewards, transcripts and the sampling support,
    across a rollout whose token history or routing broke the strict chain
    (three rows), and packs to the same batch under either credit."""
    routed = np.arange(4 * 2 * 2).reshape(4, 2, 2) % 5
    calls = (
        Call(
            (1, 2), (3, 4), (-0.5, -0.25), "tool_calls", 3, routed_experts=routed[:3], support=((3, 9), (4,))
        ),
        Call((1, 2, 3, 4, 7), (8,), (-1.0,), "length", 4, routed_experts=routed[:, :, :].repeat(2, 0)[:5]),
        Call((5, 6), (2,), (-0.125,), "stop", 4, reward=0.75, transcript=Transcript(
            ({"role": "user", "content": "go on"},),
            {"role": "assistant", "content": "done", "tool_calls": [{"id": "a", "type": "function"}]})),
    )
    rollout = Session("task", "g", 1, 2, calls, Status.COMPLETED, 0.5, {"tests": 0.5}, "log://1")
    failed = Session("task", "g", 2, 0, (Call((1,), (2,), (-3.0,), "abort", 4),), Status.INFRA_ERROR, None)
    rows = json.loads(json.dumps(to_verl([rollout, failed])))
    assert len(rows) == 4
    assert (
        rows[0]["extra_fields"]["min_global_steps"] == 3 and rows[0]["extra_fields"]["max_global_steps"] == 3
    )
    assert rows[1]["extra_fields"]["min_global_steps"] == rows[1]["extra_fields"]["max_global_steps"] == 4
    restored = [trajectory.session for trajectory in from_verl(rows)]
    assert restored == [rollout, failed]
    other = Session("task", "g", 3, 0, (Call((1, 2), (3,), (-0.5,), "stop", 4, reward=0.25),),
                    Status.COMPLETED, 0.0)
    again = [trajectory.session for trajectory in from_verl(json.loads(json.dumps(to_verl([other]))))]
    assert again == [other]
    for credit in ("session", "turn"):
        packed = pack([rollout, other], 16, support_capacity=4, credit=credit)
        for key, value in pack([*restored[:1], *again], 16, support_capacity=4, credit=credit).items():
            np.testing.assert_array_equal(value, packed[key], err_msg=f"{credit} {key}")
        assert np.any(packed["advantages"] != 0)
    assert SUPPORT_KEY in pack(restored[:1], 16, support_capacity=4)


def test_the_verl_rl_parquet_reads_its_nested_ground_truth():
    table = pq.read_table(PARQUET).to_pylist()
    prompts = Files(str(PARQUET)).pipe(presets.prompts, "tests/fixtures/tokenizers/tiny-tools",
                                       max_prompt_len=512, batch=None, long="truncate")
    rows = list(iterate(prompts, read="memory", loading=Loading(workers=0)))
    for index, expected in enumerate(table):
        batch = rows[index]
        assert read(batch[SOURCE_KEY]) == expected["data_source"] == "openai/gsm8k"
        assert read(batch[TRUTH_KEY]) == expected["reward_model"]["ground_truth"]
        info = json.loads(read(batch[INFO_KEY]))
        assert info["tools_kwargs"] == expected["extra_info"]["tools_kwargs"]
        # The row's task family reaches the reward with the rest of its context (P92).
        assert info["ability"] == expected["ability"]


def test_a_prompt_rows_ability_rides_beside_context_that_is_no_mapping():
    """A row's `ability` joins a mapping `extra_info` as a field of it, and rides beside one that is
    no mapping (a string) in an envelope that keeps both; a mapping that names another ability is
    refused, since keeping either would lose the other."""
    from dew.data.prompts import _extra_info

    assert _extra_info({"extra_info": {"split": "train"}, "ability": "math"}) == {
        "split": "train", "ability": "math"}
    assert _extra_info({"extra_info": "suite-A", "ability": "math"}) == {
        "extra_info": "suite-A", "ability": "math"}
    assert _extra_info({"ability": "math"}) == {"ability": "math"}
    with pytest.raises(ValueError):
        _extra_info({"extra_info": {"ability": "code"}, "ability": "math"})


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


def test_a_reward_crosses_the_verl_boundary_with_the_records_three_fields():
    """verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` scores in Dew
    as a `Reward` of the completion and its record, each field passed as verl names it, and a Dew
    reward handed back to verl reads what verl passed it."""
    calls = []

    def score(data_source, solution_str, ground_truth, extra_info):
        calls.append((data_source, solution_str, ground_truth, extra_info))
        return float(solution_str == ground_truth)

    record = PromptRecord("gsm8k", "72", '{"index": 3}')
    assert verl_reward(score)("72", record) == 1.0
    assert calls == [("gsm8k", "72", "72", '{"index": 3}')]
    seen = []

    def reward(completion, given):
        seen.append((completion, given))
        return 0.5

    assert compute_score(reward)("gsm8k", "71", "72", "") == 0.5
    # verl may pass extra info that is not text; the record holds it as a prompt batch would.
    assert compute_score(reward)("gsm8k", "71", ["72"], {"split": "test", "index": 3}) == 0.5
    assert seen == [("71", PromptRecord("gsm8k", "72", "")),
                    ("71", PromptRecord("gsm8k", '["72"]', '{"index": 3, "split": "test"}'))]

