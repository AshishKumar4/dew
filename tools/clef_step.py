"""The Clef recipe's training step on a tiny Qwen 3.5 hybrid that fla's port runs, lowered for a
CUDA generation on any host or trained for a few steps on this one's devices.

    python tools/clef_step.py --lower-for sm80    # a CPU will do: every Pallas-Triton kernel lowers
    python tools/clef_step.py --steps 3           # on the GPU: the step compiles and runs

c58 (an A100) and c57 (a G4) passed every CPU test and the port's own GPU
tests, then died in their first compile of the decision step, in a
Pallas-Triton lowering. A CPU interprets Dew's Pallas kernels, so only a
lowering for the GPU meets that failure, and `--trainer.lower-for` lowers
the run's real step for one (`dew.training.lowering`). The step is
recipes/decision/clef.py's own run at its defaults, bfloat16 compute, LoRA
on every projection and Clef's joint head. The backbone is the size of a
test fixture: the tiny Qwen 3.5's three Gated DeltaNet layers and one
full-attention layer, with 64-wide key and value heads, the narrowest the
port takes (`dew.nn.linear.fla_refusal`), so `KERNELS` chooses on it as on
the 0.8B.

The last line printed is a JSON record: the Pallas-Triton kernels the
lowered step calls, by name (`dew.training.lowering.triton_kernels`), or the
loss of each trained step.
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = Path("tests") / "fixtures" / "hf" / "qwen38-dense-tiny"
"""The tiny Qwen 3.5 whose text config and tokenizer the backbone takes, in a Dew checkout."""

WIDTHS = {"linear_key_head_dim": 64, "linear_value_head_dim": 64, "linear_num_value_heads": 2}
"""The Gated DeltaNet heads the port tiles: keys of 64, 128 or 256 and values by 64."""


def recipe(tree: Path, name: str):
    """A module of `tree`'s recipes/decision, which imports its siblings by name."""
    recipes = tree / "recipes" / "decision"
    sys.path.insert(0, str(recipes))
    spec = importlib.util.spec_from_file_location(f"decision_{name}", recipes / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def backbone(tree: Path, directory: Path) -> Path:
    """The tiny Qwen 3.5's decoder at the port's widths, drawn by transformers' own initializer and
    saved in its layout with the tiny model's tokenizer: Dew writes no Gated DeltaNet hybrid's
    checkpoint of its own (`refuse_lossy_export`). The multi-token-prediction head stays off, as
    transformers' causal LM builds none."""
    import torch
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    qwen = tree / QWEN
    text = json.loads((qwen / "config.json").read_text())["text_config"]
    text = {name: value for name, value in text.items() if not name.startswith("mtp_")}
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_dict({**text, **WIDTHS})).save_pretrained(directory)
    for asset in [*qwen.glob("tokenizer*"), qwen / "chat_template.jinja"]:
        (directory / asset.name).write_bytes(asset.read_bytes())
    return directory


def mixture(tree: Path, directory: Path) -> Path:
    """Twenty-four tickets, each asking a team and whether it is urgent, written as a mixture's one set by
    recipes/decision/sources.py, with nothing decontaminated."""
    sources, contamination = recipe(tree, "sources"), recipe(tree, "contamination")
    team = {"type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "payments", "technical": "outages", "sales": "pricing"}}
    urgent = {"type": "noul", "instructions": "Is it urgent?"}
    rows = [{"state": f"ticket {index}: charged twice for plan {index % 5}",
             "questions": {"team": team, "urgent": urgent}, "targets": {"team": [0.7, 0.2, 0.1]},
             "answers": {"urgent": "true" if index % 2 else "false"}} for index in range(24)]
    directory.mkdir(parents=True, exist_ok=True)
    written = directory / "rows.jsonl"
    written.write_text("".join(json.dumps(row) + "\n" for row in rows))
    default = sources.Mixture()
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
           for entry in dataclasses.fields(default)}
    chosen = sources.Mixture(**{**off, "rows": sources.Rows(weight=1.0, held=8, path=str(written))})
    sources.write(chosen, directory / "mixture", contamination.Overlaps.of([]))
    return directory / "mixture"


def run(tree: Path, work: Path, steps: int, generation: str | None):
    """`tree`'s Clef recipe's run over the tiny backbone and mixture in `work`, `steps` long, lowered
    for `generation` instead where one is named."""
    from dew.decision.config import DecisionRunConfig

    flags = ["--pretrained", str(backbone(tree, work / "backbone")), "--revision", "None",
             "--mixture.root", str(mixture(tree, work / "data")), "--max-len", "1024",
             "--lora.rank", "4", "--head.width", "24", "--head.heads", "2", "--head.feedforward", "40",
             "--head.layers", "1", "--head.routing-layers", "1",
             "--trainer.name", "clef-step", "--trainer.checkpoint-dir", str(work / "runs"),
             "--trainer.batch-size", "8", "--trainer.steps", str(steps), "--trainer.log-every", "1",
             "--trainer.eval-every", str(steps), "--trainer.checkpoint-every", str(steps),
             "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
             "--data.loading.workers", "0", "--optim.schedule.warmup-steps", "1", "--no-calibrate"]
    if generation is not None:
        flags += ["--trainer.lower-for", generation]
    return DecisionRunConfig.cli(flags, default=recipe(tree, "clef").run_config()).run()


def host_devices(count: int) -> None:
    """XLA_FLAGS with `count` host devices, as many as the GPUs the step lowers for, before JAX reads
    them: a mesh over more splits the batch, and the kernels, which see whole arrays, refuse it
    (`seen_whole`), so the devices this host happens to simulate (the suite's eight) must not
    decide the program."""
    kept = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
            if not flag.startswith("--xla_force_host_platform_device_count")]
    os.environ["XLA_FLAGS"] = " ".join([*kept, f"--xla_force_host_platform_device_count={count}"])


def lowered(tree: Path, work: Path, generation: str) -> dict:
    """The kernels the step calls once lowered for `generation`."""
    from dew.training.lowering import Lowered

    try:
        # Two steps: the schedule decays after its one warmup step.
        run(tree, work, 2, generation)
    except Lowered as done:
        return {"generation": done.generation, "kernels": done.kernels}
    raise RuntimeError(f"the run trained rather than lowering for {generation}")


def trained(tree: Path, work: Path, steps: int) -> dict:
    """The loss of each of `steps` steps on this host's devices, as the run's journal holds them."""
    import jax

    run(tree, work, steps, None)
    journal = work / "runs" / "clef-step" / "tracking" / "scalars.jsonl"
    scalars = [json.loads(line)["scalars"] for line in journal.read_text().splitlines()]
    return {"device_kind": jax.devices()[0].device_kind,
            "losses": [entry["train/loss"] for entry in scalars if "train/loss" in entry]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lower-for", help="a CUDA generation ('sm80', 'sm120') to lower the step for")
    parser.add_argument("--devices", type=int, default=1, help="the GPUs the lowered step spreads over")
    parser.add_argument("--steps", type=int, default=3, help="the steps to train, where nothing is lowered")
    parser.add_argument("--work", type=Path, help="where the backbone, the data and the run go; a temporary "
                        "directory by default")
    parser.add_argument("--tree", type=Path, default=ROOT, help="the Dew checkout whose recipe and tiny "
                        "model run, with its src on the path; this one by default")
    args = parser.parse_args()
    if args.lower_for:
        host_devices(args.devices)
    with tempfile.TemporaryDirectory() as scratch:
        work = args.work or Path(scratch)
        result = (lowered(args.tree, work, args.lower_for) if args.lower_for else
                  trained(args.tree, work, args.steps))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
