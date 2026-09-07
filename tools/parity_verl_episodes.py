"""Validate native sampled episode rows with pinned verl's real AgentLoopOutput.

Run with PYTHONPATH=src:tests JAX_PLATFORMS=cpu in the isolated reference
venv used by parity_behavior.py. The fixture includes verl's JSON model dump
and its actual tensor mapping, including behavior log probabilities.
"""

from dataclasses import replace
from importlib.metadata import distribution
from importlib import import_module
import json
from pathlib import Path

from dew.objectives.rl.verl import to_verl

REVISION = "d040717b21af2e23e8e789a3e354cff2394ae2de"


def main() -> None:
    import torch
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    direct = distribution("verl").read_text("direct_url.json")
    if direct is None or json.loads(direct)["vcs_info"]["commit_id"] != REVISION:
        raise RuntimeError(f"install verl from git revision {REVISION}")
    torch.set_num_threads(1)
    native = import_module("test_tool_episodes")
    trainer, rollout = native.build()
    episodes = native.collect(rollout, trainer.initial_state())
    episodes = tuple(replace(episode, _binding_id="fixture", transitions=tuple(
        replace(turn, action=replace(turn.action, _binding_id="fixture")) for turn in episode.transitions))
        for episode in episodes)
    outputs = []
    for row in to_verl(episodes):
        model = AgentLoopOutput.model_validate(row)
        mapped = model.as_dict()
        tensors = {key: mapped[key].tolist() for key in
                   ("prompts", "responses", "response_mask", "rollout_log_probs", "rm_scores")}
        outputs.append({"wire": model.model_dump(mode="json", exclude_unset=True), "tensors": tensors})
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/verl_episodes.json"
    path.write_text(json.dumps({"revision": REVISION, "outputs": outputs}, indent=2) + "\n")
    print(f"{len(outputs)} actual model calls validated by verl; wrote {path}")


if __name__ == "__main__":
    main()
