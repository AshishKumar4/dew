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
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes" / "decision"
QWEN = ROOT / "tests" / "fixtures" / "hf" / "qwen38-dense-tiny"
"""The tiny Qwen 3.5 whose text config and tokenizer the backbone takes."""

WIDTHS = {"linear_key_head_dim": 64, "linear_value_head_dim": 64, "linear_num_value_heads": 2}
"""The Gated DeltaNet heads the port tiles: keys of 64, 128 or 256 and values by 64."""


def recipe(name: str):
    """A module of recipes/decision, which imports its siblings by name."""
    sys.path.insert(0, str(RECIPES))
    spec = importlib.util.spec_from_file_location(f"decision_{name}", RECIPES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def backbone(directory: Path) -> Path:
    """The tiny Qwen 3.5's decoder at the port's widths, freshly initialized, written in its Hugging
    Face layout with its tokenizer. The text config is written as it is: Dew derives none for a
    Gated DeltaNet hybrid (`refuse_lossy_export`)."""
    import jax
    import numpy as np

    from dew.interop.hf_decoders import export_decoder_weights, translate_config
    from dew.interop.safetensors_io import save_hf_layout
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import from_record

    text = {**json.loads((QWEN / "config.json").read_text())["text_config"], **WIDTHS,
            "architectures": ["Qwen3_5ForCausalLM"]}
    model = from_record(CausalTransformer, {**translate_config(text), "dtype": "float32"})
    variables = model.init(jax.random.key(0), np.zeros((1, 4), np.int32))
    save_hf_layout(export_decoder_weights(model, variables, text), text, directory)
    for asset in [*QWEN.glob("tokenizer*"), QWEN / "chat_template.jinja"]:
        (directory / asset.name).write_bytes(asset.read_bytes())
    return directory


def mixture(directory: Path) -> Path:
    """Twenty-four tickets, each asking a team and whether it is urgent, written as a mixture's one set by
    recipes/decision/sources.py, with nothing decontaminated."""
    sources, contamination = recipe("sources"), recipe("contamination")
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


def run(work: Path, steps: int, generation: str | None):
    """The Clef recipe's run over the tiny backbone and mixture in `work`, `steps` long, lowered for
    `generation` instead where one is named."""
    from dew.decision.config import DecisionRunConfig

    flags = ["--pretrained", str(backbone(work / "backbone")), "--revision", "None",
             "--mixture.root", str(mixture(work / "data")), "--max-len", "1024",
             "--lora.rank", "4", "--head.width", "24", "--head.heads", "2", "--head.feedforward", "40",
             "--head.layers", "1", "--head.routing-layers", "1",
             "--trainer.name", "clef-step", "--trainer.checkpoint-dir", str(work / "runs"),
             "--trainer.batch-size", "8", "--trainer.steps", str(steps), "--trainer.log-every", "1",
             "--trainer.eval-every", str(steps), "--trainer.checkpoint-every", str(steps),
             "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
             "--data.loading.workers", "0", "--optim.schedule.warmup-steps", "1", "--no-calibrate"]
    if generation is not None:
        flags += ["--trainer.lower-for", generation]
    return DecisionRunConfig.cli(flags, default=recipe("clef").run_config()).run()


def lowered(work: Path, generation: str) -> dict:
    """The kernels the step calls once lowered for `generation`."""
    from dew.training.lowering import Lowered

    try:
        run(work, 1, generation)
    except Lowered as done:
        return {"generation": done.generation, "kernels": done.kernels}
    raise RuntimeError(f"the run trained rather than lowering for {generation}")


def trained(work: Path, steps: int) -> dict:
    """The loss of each of `steps` steps on this host's devices, as the run's journal holds them."""
    import jax

    run(work, steps, None)
    journal = work / "runs" / "clef-step" / "tracking" / "scalars.jsonl"
    scalars = [json.loads(line)["scalars"] for line in journal.read_text().splitlines()]
    return {"device_kind": jax.devices()[0].device_kind,
            "losses": [entry["train/loss"] for entry in scalars if "train/loss" in entry]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lower-for", help="a CUDA generation ('sm80', 'sm120') to lower the step for")
    parser.add_argument("--steps", type=int, default=3, help="the steps to train, where nothing is lowered")
    parser.add_argument("--work", type=Path, help="where the backbone, the data and the run go; a temporary "
                        "directory by default")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        work = args.work or Path(scratch)
        result = lowered(work, args.lower_for) if args.lower_for else trained(work, args.steps)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
