"""Write the verl-format fixtures Dew's native interop is tested against.

Run in an isolated env with verl installed from the revision below (plus
torch and datasets, which verl imports), from the repository root:

    python tools/verl_interop_reference.py /path/to/gsm8k_tool_agent_loop/test.parquet

The parquet argument is the output of verl's own
`examples/data_preprocess/gsm8k_tool_agent_loop.py`; its first four rows are
copied with verl's schema to `tests/fixtures/rl/verl_gsm8k_tool_agent.parquet`.
`tests/fixtures/rl/verl_native.json` then holds, all produced by verl code:
two `AgentLoopOutput` trajectories as a tool agent loop and a video single-turn
loop emit them (the video payload from `AgentLoopBase.build_sglang_video_payload`),
dumped as JSON, with the tensors `as_dict` maps them to; and the model's
field names. Dew never imports verl.
"""

import json
import sys
from importlib.metadata import distribution
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "12ebe0cb4d300c58449fb6c675379e8700015c51"
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/rl"


def trajectories():
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

    # Tool loop: assistant turn, tool response (mask 0, logprob 0.0), assistant turn.
    prompt = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
    first, tool, second = [40, 1184, 311, 1618, 151645], [198, 151644, 872, 198, 19, 151645, 198, 151644, 77091, 198], [
        785, 4226, 374, 220, 19, 151645]
    response = first + tool + second
    mask = [1] * len(first) + [0] * len(tool) + [1] * len(second)
    logprobs = [-0.25, -1.5, -0.03, -0.7, -0.01] + [0.0] * len(tool) + [-0.4, -0.02, -0.9, -0.05, -0.6, -0.001]
    total = len(prompt) + len(response)
    routed = [[[(3 * position + layer) % 8, (5 * position + layer + 1) % 8] for layer in range(3)]
              for position in range(total - 1)]
    tool_loop = AgentLoopOutput(
        prompt_ids=prompt, response_ids=response, response_mask=mask, response_logprobs=logprobs,
        routed_experts=routed, reward_score=1.0, num_turns=4, metrics=AgentLoopMetrics(generate_sequences=0.5),
        extra_fields={"turn_scores": [], "tool_rewards": [1.0], "min_global_steps": 7, "max_global_steps": 8})
    video = AgentLoopBase.build_sglang_video_payload(
        [["frame0", "frame1"]],
        {"pixel_values_videos": [[0.25, -0.5, 1.0, 0.0], [0.125, 0.5, -1.0, 2.0]], "video_grid_thw": [[1, 2, 2]]})
    single = AgentLoopOutput(
        prompt_ids=[151644, 872, 198, 151656, 151656, 151645, 198, 151644, 77091, 198],
        response_ids=[32, 8251, 151645], response_mask=[1, 1, 1], response_logprobs=[-0.3, -2.25, -0.0625],
        reward_score=0.0, num_turns=2, metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [], "tool_rewards": [], "min_global_steps": 8, "max_global_steps": 8},
        multi_modal_data={"videos": ["frames/0.mp4"]}, mm_processor_kwargs={"fps": 2.0},
        mm_processor_output=video)
    outputs = []
    for model in (tool_loop, single):
        mapped = model.as_dict()
        tensors = {key: value.tolist() for key, value in mapped.items() if hasattr(value, "tolist")}
        outputs.append({"wire": model.model_dump(mode="json"), "tensors": tensors})
    fields = {name: info.is_required() for name, info in AgentLoopOutput.model_fields.items()}
    return outputs, fields


def main(parquet: str) -> None:
    direct = distribution("verl").read_text("direct_url.json")
    if direct is None or json.loads(direct)["vcs_info"]["commit_id"] != REVISION:
        raise RuntimeError(f"install verl from git revision {REVISION}")
    table = pq.read_table(parquet).slice(0, 4)
    pq.write_table(table, FIXTURES / "verl_gsm8k_tool_agent.parquet")
    outputs, fields = trajectories()
    (FIXTURES / "verl_native.json").write_text(json.dumps(
        {"revision": REVISION, "fields": fields, "outputs": outputs}, indent=1) + "\n")
    print(f"{len(table)} parquet rows, {len(outputs)} trajectories")


if __name__ == "__main__":
    main(sys.argv[1])
