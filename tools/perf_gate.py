#!/usr/bin/env python3
"""The performance gate: one fixed battery, two Dew trees, one machine.

    python tools/perf_gate.py run --base main=/content/main --head head=/content/head \\
        --rounds 3 --model /content/hf/Qwen3-0.6B --flowers /content/flowers --out gate.json
    python tools/perf_gate.py report gate.json --table gate.md

`run` measures every row of `BATTERY` in each tree, in alternating rounds (base
then head, then head then base), each row a process of its own. A row runs a
tree's own copy of its tool where the tree has one and the head's otherwise,
with the tree's `src` first on the path, so a tree from before a tool existed
is measured by the newer tool, and a row a tree cannot run is reported, not
gated. `report` gates each row by its own noise: a row regresses when the
head's median is worse than the base's by more than the larger of the two
trees' spreads across rounds (and at least `FLOOR`), and the head's best
sample is worse than the base's worst. It writes the table and exits 1 on a
regression.

The battery is the training step (a dense and an MoE decoder, a DiT, an
MM-DiT, FLUX.2 and Wan), the forward pass of three vision towers and the SD
VAE's decoder, LM serving at 32 and 128 slots, attention forward and
backward, the image input pipeline and a checkpoint's save and restore.
docs/performance.md, "The performance gate", says how it runs before main
takes a commit.
"""

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

FLOOR = 0.02
"""The smallest band a row is gated by, as a fraction of the base's median."""

DENSE = {"vocab_size": 50304, "emb_features": 1024, "num_layers": 24, "num_heads": 16,
         "mlp_features": 2816, "max_seq_len": 1024}
MOE = {"vocab_size": 50304, "emb_features": 768, "num_layers": 12, "num_heads": 12, "mlp_features": 2048,
       "max_seq_len": 1024,
       "mixture": {"experts": 8, "top_k": 2, "layers": [1, 3, 5, 7, 9, 11], "dispatch": "global"}}
DIT = {"patch_size": 4, "emb_features": 768, "num_layers": 12, "num_heads": 12, "mlp_ratio": 4}
# FLUX.2 and Wan 2.1 1.3B at their head width over a few of their blocks: the
# rotary, the joint attention and the modulated feed-forwards of the hot path.
FLUX2 = {"num_layers": 2, "num_single_layers": 4, "heads": 12, "head_dim": 128,
         "joint_attention_dim": 3 * 768, "axes_dims_rope": [32, 32, 32, 32]}
WAN = {"num_attention_heads": 12, "attention_head_dim": 128, "text_dim": 768, "ffn_dim": 8960,
       "num_layers": 4}
STEP_CASES = {
    "step lm-dense 4x1024": {"architecture": "causal_transformer", "config": DENSE, "dtype": "bfloat16",
                             "batch_size": 4, "seq_len": 1024},
    "step lm-moe 4x1024": {"architecture": "causal_transformer", "config": MOE, "dtype": "bfloat16",
                           "batch_size": 4, "seq_len": 1024},
    "step simple-dit-b 32x64px": {"architecture": "simple_dit", "config": DIT, "dtype": "bfloat16",
                                  "batch_size": 32, "image_size": 64},
    "step simple-mmdit-b 32x64px": {"architecture": "simple_mmdit", "config": DIT, "dtype": "bfloat16",
                                    "batch_size": 32, "image_size": 64},
    "step flux2 4x32px": {"architecture": "flux2_transformer", "config": FLUX2, "dtype": "bfloat16",
                          "batch_size": 4, "image_size": 32, "channels": 128},
    "step wan 2x5x32px": {"architecture": "wan_transformer", "config": WAN, "dtype": "bfloat16",
                          "batch_size": 2, "image_size": 32, "channels": 16, "frames": 5},
}
# Forward passes of inference modules at their published sizes, random weights.
FORWARD_CASES = ("vision siglip-400m 384px b8", "vision qwen3.5 448px b8", "vision gemma4 672px b4",
                 "vae decode sd 512px b4")

# A checkpoint of lm-dense's whole training state, saved and then restored onto its shardings.
CHECKPOINT = """
import json, sys, tempfile, time
import jax
import benchmark_step as bench
from dew.checkpoints import Checkpoints
case = bench.Case(**json.loads(sys.argv[1]))
trainer = bench.build_trainer(case)
abstract = jax.eval_shape(trainer.initial_state)
shardings = trainer.shardings(abstract)
state = jax.block_until_ready(jax.jit(trainer.initial_state, out_shardings=shardings)())
template = jax.tree.map(lambda leaf, where: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where),
                        abstract, shardings)
saves, restores = [], []
# A train state is checkpointed under its own step, which a restore checks.
step = int(state.step)
for _ in range(3):
    with tempfile.TemporaryDirectory() as directory:
        checkpoints = Checkpoints(directory)
        started = time.perf_counter()
        checkpoints.save(step, state, None)
        checkpoints.wait()
        saves.append(time.perf_counter() - started)
        started = time.perf_counter()
        jax.block_until_ready(checkpoints.restore(template, step)[0])
        restores.append(time.perf_counter() - started)
print(json.dumps({"save_ms": 1e3 * sorted(saves)[1], "restore_ms": 1e3 * sorted(restores)[1]}))
"""


@dataclass(frozen=True)
class Row:
    name: str
    unit: str
    """'ms', lower is better, or 'items/s', higher is better."""
    tool: str
    """The tools/ file the row runs, or '' for a script of this file's."""
    gated: bool = True
    """Whether a regression of this row fails the gate, or is only shown."""


BATTERY = [
    *[Row(name, "ms", "benchmark_step.py") for name in STEP_CASES],
    *[Row(name, "ms", "benchmark_forward.py") for name in FORWARD_CASES],
    Row("serve qwen3-0.6b 32 slots", "items/s", "benchmark_lm_serving.py"),
    Row("serve qwen3-0.6b 128 slots", "items/s", "benchmark_lm_serving.py"),
    # The median of the same repeats: the stalls best-of-five hides (a recompile, a pause), shown.
    Row("serve qwen3-0.6b 32 slots, median repeat", "items/s", "benchmark_lm_serving.py", gated=False),
    Row("serve qwen3-0.6b 128 slots, median repeat", "items/s", "benchmark_lm_serving.py", gated=False),
    Row("attention cudnn S=2048 D=128 causal fwd+bwd", "ms", "benchmark_attention.py"),
    Row("attention xla S=2048 D=128 causal fwd+bwd", "ms", "benchmark_attention.py"),
    Row("image pipeline flowers 128px, device crop+flip+jitter", "items/s", "benchmark_image_pipeline.py"),
    Row("checkpoint lm-dense save", "ms", ""),
    Row("checkpoint lm-dense restore", "ms", ""),
]


def _tools(tree: Path, tool: str, newer: Path) -> Path:
    """The tree's own tools/ where it has `tool`, else the head's (`newer`), else this file's own
    directory, where a gate run outside a checkout keeps the tools no tree has yet."""
    for tools in (tree / "tools", newer, Path(__file__).resolve().parent):
        if (tools / tool).is_file():
            return tools
    return newer


def _run(tree: Path, tool: str, argv: list[str], newer: Path, timeout: float) -> str:
    tools = _tools(tree, tool or "benchmark_step.py", newer)
    env = {**os.environ, "PYTHONPATH": f"{tree / 'src'}:{tools}"}
    command = [sys.executable, *([str(tools / tool)] if tool else []), *argv]
    done = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:3])} exited {done.returncode}: {done.stderr[-1500:]}")
    return done.stdout


def _probes(args: argparse.Namespace, out: Path):
    """Each probe: the rows it measures, its tool, its arguments, and how its output reads into them."""
    for name, case in STEP_CASES.items():
        yield ([name], "benchmark_step.py",
               ["--cases", json.dumps([case]), "--warmup", "10", "--steps", "30", "--json-out", str(out)],
               lambda _, name=name: {name: json.loads(out.read_text())[0]["ms_per_step"]})
    # A process for each forward, so a module one tree lacks costs that row alone.
    for name in FORWARD_CASES:
        yield ([name], "benchmark_forward.py", ["--cases", name, "--json-out", str(out)],
               lambda _, name=name: {name: json.loads(out.read_text())[name]})
    yield ([row.name for row in BATTERY if row.name.startswith("serve")], "benchmark_lm_serving.py",
           ["--backend", "dew", "--model", args.model, "--slots", "32,128", "--repeats", "5",
            "--out", str(out)],
           # The best repeat: a host-bound row's noise only ever slows it (32 slots spread 33% by median).
           lambda _: {name: value for entry in json.loads(out.read_text())["sweep"]
                      for name, value in (
                          (f"serve qwen3-0.6b {entry['slots']} slots",
                           max(r["output_tokens_per_second"] for r in entry["repeats"])),
                          (f"serve qwen3-0.6b {entry['slots']} slots, median repeat",
                           statistics.median(r["output_tokens_per_second"] for r in entry["repeats"])))})
    yield ([row.name for row in BATTERY if row.name.startswith("attention")], "benchmark_attention.py",
           ["--implementations", "cudnn", "xla", "--sequence-lengths", "2048", "--head-dims", "128",
            "--causal", "True", "--json-out", str(out)],
           lambda _: {f"attention {row['implementation']} S=2048 D=128 causal fwd+bwd":
                      row["forward_backward_ms"]
                      for row in json.loads(out.read_text()) if "forward_backward_ms" in row})
    pipeline = "image pipeline flowers 128px, device crop+flip+jitter"
    yield ([pipeline], "benchmark_image_pipeline.py",
           ["--flowers", args.flowers, "--images", "512", "--repeats", "3", "--no-training",
            "--out", str(out)],
           lambda _: {pipeline: json.loads(out.read_text())["pipeline"]["device_crop_flip_jitter"]
                      ["images_per_second"]})
    case = {**STEP_CASES["step lm-dense 4x1024"], "batch_size": 1}
    yield (["checkpoint lm-dense save", "checkpoint lm-dense restore"], "",
           ["-c", CHECKPOINT, json.dumps(case)],
           lambda printed: {f"checkpoint lm-dense {kind}":
                            json.loads(printed.strip().splitlines()[-1])[f"{kind}_ms"]
                            for kind in ("save", "restore")})


def measure(tree: Path, args: argparse.Namespace, newer: Path) -> dict[str, float | str]:
    """One sample of every row in `tree`: a number, or why the row could not run there."""
    sample: dict[str, float | str] = {}
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "out.json"
        for names, tool, argv, read in _probes(args, out):
            if args.only and not any(word in name for name in names for word in args.only):
                continue
            try:
                values = read(_run(tree, tool, argv, newer, 3600))
                sample.update({name: float(values[name]) for name in names})
            except Exception as error:  # a row that cannot run in this tree is reported, not gated
                sample.update(dict.fromkeys(names, f"not run: {str(error)[-400:]}"))
            out.unlink(missing_ok=True)
    return sample


def run(args: argparse.Namespace) -> int:
    trees = dict(spec.split("=", 1) for spec in (args.base, args.head))
    (base, base_path), (head, head_path) = ((name, Path(path)) for name, path in trees.items())
    rows = [row for row in BATTERY if not args.only or any(word in row.name for word in args.only)]
    results = {"base": base, "head": head, "rows": [row.__dict__ for row in rows],
               "samples": {base: [], head: []}}
    for round_ in range(args.rounds):
        order = [(base, base_path), (head, head_path)]
        for name, path in order if round_ % 2 == 0 else order[::-1]:
            results["samples"][name].append(measure(path, args, head_path / "tools"))
            Path(args.out).write_text(json.dumps(results, indent=2))
            print(f"round {round_ + 1}: {name} measured", flush=True)
    return 0


def verdict(row: Row, base: list, head: list) -> tuple[str, dict]:
    """Whether `row` regressed from the `base` samples to the `head` ones, and the numbers that say so.

    A row moves when its median moves by more than the larger of the two trees' spreads (at
    least FLOOR) and the two trees' samples do not overlap at all. With three rounds each, the
    samples separate by chance 1 time in 20 when nothing changed (1 / C(6, 3)), and the band
    lowers that further; it is also the smallest change the row can see. A row the base cannot
    run (a tree older than its tool) is not compared; one the base runs and the head does not is
    broken; one neither runs leaves the battery incomplete."""
    base, head = [x for x in base if isinstance(x, float)], [x for x in head if isinstance(x, float)]
    if not base and not head:
        return "incomplete", {}
    if not base:
        return "not compared", {}
    if not head:
        return "broken", {}
    worse = 1.0 if row.unit == "ms" else -1.0

    def spread(xs):
        return (max(xs) - min(xs)) / statistics.median(xs)
    band = max(spread(base), spread(head), FLOOR)
    change = worse * (statistics.median(head) - statistics.median(base)) / statistics.median(base)
    separated = (min(head) > max(base)) if worse > 0 else (max(head) < min(base))
    faster = (max(head) < min(base)) if worse > 0 else (min(head) > max(base))
    state = "regressed" if change > band and separated else "faster" if -change > band and faster else "level"
    return state, {"base": statistics.median(base), "head": statistics.median(head), "change": change,
                   "band": band}


def report(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text())
    base, head = results["base"], results["head"]
    lines = [f"| row | unit | {base} | {head} | change | band | verdict |",
             "|---|---|---:|---:|---:|---:|---|"]
    failed = []
    for spec in results["rows"]:
        row = Row(**spec)
        state, numbers = verdict(row, [s.get(row.name) for s in results["samples"][base]],
                                 [s.get(row.name) for s in results["samples"][head]])
        if state in ("broken", "incomplete") or (state == "regressed" and row.gated):
            failed.append(f"{row.name} ({state})")
        if not row.gated:
            state = f"{state}, not gated"
        cells = (["-"] * 4 if not numbers else
                 [f"{numbers['base']:.2f}", f"{numbers['head']:.2f}", f"{numbers['change']:+.1%}",
                  f"{numbers['band']:.1%}"])
        lines.append(f"| {row.name} | {row.unit} | " + " | ".join(cells) + f" | {state} |")
    table = "\n".join(lines) + "\n"
    print(table)
    if args.table:
        Path(args.table).write_text(table)
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/perf_gate.py")
    operations = parser.add_subparsers(dest="operation", required=True)
    running = operations.add_parser("run")
    running.add_argument("--base", required=True, help="name=path of the tree to compare against")
    running.add_argument("--head", required=True, help="name=path of the tree under test")
    running.add_argument("--rounds", type=int, default=3)
    running.add_argument("--model", required=True, help="a local Qwen3-0.6B snapshot")
    running.add_argument("--flowers", required=True, help="the TFDS oxford_flowers102 directory")
    running.add_argument("--out", required=True)
    running.add_argument("--only", nargs="*", default=[],
                         help="measure only the rows whose names hold one of these words")
    reporting = operations.add_parser("report")
    reporting.add_argument("results")
    reporting.add_argument("--table")
    args = parser.parse_args()
    with contextlib.suppress(BrokenPipeError):
        return run(args) if args.operation == "run" else report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
