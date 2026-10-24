#!/usr/bin/env python3
"""The performance gate: one fixed battery, two Dew trees, one machine.

    python tools/perf_gate.py run --base main=/content/main --head head=/content/head \\
        --rounds 3 --model /content/hf/Qwen3-0.6B --flowers /content/flowers --out gate.json
    python tools/perf_gate.py report gate.json --table gate.md

`run` measures every row of `BATTERY` in each tree, in alternating rounds (base
then head, then head then base), each row a process of its own in the tree's
own locked environment (tools/environments.py), so a change of a locked
wheel is measured with the code. A row runs a tree's own copy of its tool
where the tree has one and the head's otherwise, with the tree's `src` first
on the path, so a tree from before a tool existed is measured by the newer
tool, and a row a tree cannot run (a tree older than its tool, or than the
locks) is reported, not gated. `report` gates each row by its own noise: a
row regresses when the head's median is worse than the base's by more than
the larger of the two trees' spreads across rounds (and at least `FLOOR`), and
the head's best sample is worse than the base's worst. It writes the table and exits 1 on a
regression.

The battery is the training step (a dense and an MoE decoder, a DiT, an
MM-DiT, FLUX.2 and Wan), the forward pass of three vision towers and the SD
VAE's decoder, LM serving at 32 and 128 slots, a full-depth Dory server's warm compile,
attention forward and backward, the image input pipeline and a checkpoint's save and restore.
docs/performance.md, "The performance gate", says how it runs before main
takes a commit.
"""

import argparse
import contextlib
import functools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_cases import DIT_B, FORWARD, LM_DENSE, LM_MOE
from environments import CACHE, Unlocked, environment, machine_lock

FLOOR = 0.02
"""The smallest band a row is gated by, as a fraction of the base's median."""
COMPILE_FLOOR = 0.25
"""The least growth of a step's compile time the gate fails a head for: not a noise allowance (the rounds'
spread still widens the band as for any row) but the materiality of the project's rule that compile time is
performance and a head that grows it materially gives the reason: a quarter."""

# FLUX.2 and Wan 2.1 1.3B at their head width over a few of their blocks: the
# rotary, the joint attention and the modulated feed-forwards of the hot path.
FLUX2 = {"num_layers": 2, "num_single_layers": 4, "heads": 12, "head_dim": 128,
         "joint_attention_dim": 3 * 768, "axes_dims_rope": [32, 32, 32, 32]}
WAN = {"num_attention_heads": 12, "attention_head_dim": 128, "text_dim": 768, "ffn_dim": 8960,
       "num_layers": 4}
STEP_CASES = {
    "step lm-dense 4x1024": {"architecture": "causal_transformer", "config": LM_DENSE, "dtype": "bfloat16",
                             "batch_size": 4, "seq_len": 1024},
    "step lm-moe 4x1024": {"architecture": "causal_transformer", "config": LM_MOE, "dtype": "bfloat16",
                           "batch_size": 4, "seq_len": 1024},
    "step simple-dit-b 32x64px": {"architecture": "simple_dit", "config": DIT_B, "dtype": "bfloat16",
                                  "batch_size": 32, "image_size": 64},
    "step simple-mmdit-b 32x64px": {"architecture": "simple_mmdit", "config": DIT_B, "dtype": "bfloat16",
                                    "batch_size": 32, "image_size": 64},
    "step flux2 4x32px": {"architecture": "flux2_transformer", "config": FLUX2, "dtype": "bfloat16",
                          "batch_size": 4, "image_size": 32, "channels": 128},
    "step wan 2x5x32px": {"architecture": "wan_transformer", "config": WAN, "dtype": "bfloat16",
                          "batch_size": 2, "image_size": 32, "channels": 16, "frames": 5},
}
SLOTS = (32, 128)
"""The serving rows' slot counts."""
ATTENTION = ("cudnn", "xla")
"""The attention rows' implementations."""

# A checkpoint of lm-dense's whole training state, saved and then restored onto its shardings, and the
# measured peak of a save and restore in a process of their own (tools/peak_memory.py), taken before this
# one opens a backend.
CHECKPOINT = """
import importlib.util, json, os, sys, tempfile, time
if sys.argv[2:] != ["--peak-run"]:
    # The gate's own helper, by its path (the script's second argument), whatever tools/ the tree holds;
    # the workload's modules still come from the tree's, as its peak runs' do.
    spec = importlib.util.spec_from_file_location("peak_memory", sys.argv[2])
    peak_memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peak_memory)
    peaks = peak_memory.measured([os.path.abspath(__file__), sys.argv[1], "--peak-run"])
import jax
from benchmark_cases import Case
from benchmark_models import build_trainer
from dew.checkpoints import Checkpoints
case = Case(**json.loads(sys.argv[1]))
trainer = build_trainer(case)
state, shardings, _ = trainer.place()
state = jax.block_until_ready(state)
template = jax.tree.map(lambda leaf, where: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where),
                        state, shardings)
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
if sys.argv[2:] != ["--peak-run"]:
    print(json.dumps({"save_ms": 1e3 * sorted(saves)[1], "restore_ms": 1e3 * sorted(restores)[1],
                      "peak_device_bytes": None if None in peaks else max(peaks)}))
"""


@dataclass(frozen=True)
class Row:
    name: str
    unit: str
    """'ms', 's' or 'GiB' (a measured peak), lower is better, or 'items/s', higher is better."""
    tool: str
    """The tools/ file the row runs, or '' for a script of this file's."""
    gated: bool = True
    """Whether a regression of this row fails the gate, or is only shown."""
    hardware: str | None = None
    """Device-kind substring required to measure this row; an absent device is not a pass."""
    head_only: bool = False
    """Compare the head with an external reference, not with the base tree."""
    floor: float = FLOOR
    """The smallest band the row is gated by (`FLOOR`, a compile's `COMPILE_FLOOR`)."""
    first_round: bool = False
    """Measured in the first round alone: a compile, cold there, which later rounds load from the tree's
    cache."""


@dataclass(frozen=True)
class Probe:
    """One process of the battery: the rows it measures, the arguments of the
    tool they name, and how that tool's output reads into their values."""

    rows: tuple[Row, ...]
    arguments: Callable[[argparse.Namespace, Path], list[str]]
    """The tool's arguments, from the run's and the file the tool writes."""
    read: Callable[[str, Path], tuple]
    """Each row's value in the rows' order, from the tool's standard output
    and the file it wrote."""

    @property
    def tool(self) -> str:
        """The tools/ file the probe's rows run."""
        return self.rows[0].tool


GIB = 2 ** 30


def peak(record: dict) -> float | None:
    """A tool's record's measured peak device memory in GiB (`peak_device_bytes`, tools/peak_memory.py), or
    None where it measured none: a tree whose tool predates it, or a device whose allocator keeps no
    high-water, which the gate shows unmeasured, never as zero."""
    measured = record.get("peak_device_bytes")
    return None if measured is None else measured / GIB


def _stepped(out: Path) -> tuple:
    """A step's time, the time its program took to compile, and its measured peak."""
    [step] = json.loads(out.read_text())
    return step["ms_per_step"], step["compile_seconds"], peak(step)


def _forwarded(out: Path, name: str) -> tuple:
    """A forward's time and its measured peak; a tree whose tool predates the peak wrote the time alone."""
    entry = json.loads(out.read_text())[name]
    return (entry, None) if isinstance(entry, float) else (entry["ms"], peak(entry))


def _served(out: Path) -> tuple:
    """Each slot count's best repeat, then its median repeat, then its measured peak. A host-bound
    row's noise only ever slows it (32 slots spread 33% by median)."""
    sweep = {entry["slots"]: entry for entry in json.loads(out.read_text())["sweep"]}
    repeats = {slots: [repeat["output_tokens_per_second"] for repeat in entry["repeats"]]
               for slots, entry in sweep.items()}
    return (*[max(repeats[slots]) for slots in SLOTS],
            *[statistics.median(repeats[slots]) for slots in SLOTS],
            *[peak(sweep[slots]) for slots in SLOTS])


def _attended(out: Path) -> tuple:
    """Each implementation's forward and backward time, then each one's measured peak."""
    rows = {row["implementation"]: row for row in json.loads(out.read_text()) if "forward_backward_ms" in row}
    return (*(rows[implementation]["forward_backward_ms"] for implementation in ATTENTION),
            *(peak(rows[implementation]) for implementation in ATTENTION))


def _qwen35_comparison(out: Path) -> tuple:
    """Translate the paired tool's record into an external-reference row's result."""
    record = json.loads(out.read_text())
    comparison = record["comparison"]
    return ({"value": comparison["dew_tokens_per_s"], "ratio": comparison["dew_over_torch"],
             "tolerance": comparison["tolerance"], "passed": comparison["performance_claim_passed"]}
            if comparison is not None else {"error": record.get("error", "incomplete pair")},)


PROBES = [
    # Each step's time, and the time its program took to compile in that process (`COMPILE_FLOOR`).
    *[Probe((Row(name, "ms", "benchmark_step.py"),
             Row(f"compile {name.removeprefix('step ')}", "s", "benchmark_step.py", floor=COMPILE_FLOOR,
                 first_round=True),
             Row(f"peak {name}", "GiB", "benchmark_step.py")),
            lambda _, out, case=case: ["--cases", json.dumps([case]), "--warmup", "10", "--steps", "30",
                                       "--json-out", str(out)],
            lambda _, out: _stepped(out))
      for name, case in STEP_CASES.items()],
    # A process for each forward, so a module one tree lacks costs that row alone.
    *[Probe((Row(name, "ms", "benchmark_forward.py"), Row(f"peak {name}", "GiB", "benchmark_forward.py")),
            lambda _, out, name=name: ["--cases", name, "--json-out", str(out)],
            lambda _, out, name=name: _forwarded(out, name))
      for name in FORWARD],
    Probe((*[Row(f"serve qwen3-0.6b {slots} slots", "items/s", "benchmark_lm_serving.py") for slots in SLOTS],
           # The median of the same repeats: the stalls best-of-five hides (a recompile, a pause), shown.
           *[Row(f"serve qwen3-0.6b {slots} slots, median repeat", "items/s", "benchmark_lm_serving.py",
                 gated=False) for slots in SLOTS],
           *[Row(f"peak serve qwen3-0.6b {slots} slots", "GiB", "benchmark_lm_serving.py")
             for slots in SLOTS]),
          lambda args, out: ["--backend", "dew", "--model", args.model, "--slots", ",".join(map(str, SLOTS)),
                             "--repeats", "5", "--out", str(out)],
          lambda _, out: _served(out)),
    # The Server's warm compiles over Dory 1.0 4B as released (random weights), program by program: its
    # construction's decode step, then each admission width's prefill. XLA's fusion of nGPT's per-layer
    # normalization grew the prefill's compile quadratically with depth at the released width, a warm of
    # 3012 s on an A100, where a narrow width showed none of it (ledger 635); the whole warm is shown
    # beside them.
    Probe((*(Row(f"compile serve dory-1.0-4b {program}", "s", "benchmark_serving_compile.py",
                 floor=COMPILE_FLOOR, first_round=True)
             for program in ("construction", *(f"admission {rows}" for rows in (1, 2, 4, 8)))),
           Row("compile serve dory-1.0-4b warm 16 slots", "s", "benchmark_serving_compile.py",
               gated=False, first_round=True)),
          lambda _, out: ["--family", "dory-1.0-4b", "--slots", "16", "--json-out", str(out)],
          lambda _, out: (lambda record: (record["construction_seconds"],
                                          *record["by_admission_width"].values(),
                                          record["warm_seconds"]))(json.loads(out.read_text()))),
    Probe((*(Row(f"attention {implementation} S=2048 D=128 causal fwd+bwd", "ms", "benchmark_attention.py")
             for implementation in ATTENTION),
           *(Row(f"peak attention {implementation} S=2048 D=128 causal fwd+bwd", "GiB",
                 "benchmark_attention.py") for implementation in ATTENTION)),
          lambda _, out: ["--implementations", *ATTENTION, "--sequence-lengths", "2048", "--head-dims", "128",
                          "--causal", "True", "--json-out", str(out)],
          lambda _, out: _attended(out)),
    Probe((Row("image pipeline flowers 128px, device crop+flip+jitter", "items/s",
               "benchmark_image_pipeline.py"),
           Row("peak image pipeline flowers 128px, device crop+flip+jitter", "GiB",
               "benchmark_image_pipeline.py")),
          lambda args, out: ["--flowers", args.flowers, "--images", "512", "--repeats", "3", "--no-training",
                             "--out", str(out)],
          lambda _, out: (lambda pipeline: (pipeline["images_per_second"], peak(pipeline)))(
              json.loads(out.read_text())["pipeline"]["device_crop_flip_jitter"])),
    Probe((Row("checkpoint lm-dense save", "ms", ""), Row("checkpoint lm-dense restore", "ms", ""),
           Row("peak checkpoint lm-dense save and restore", "GiB", "")),
          lambda _, out: [str(_script(out, CHECKPOINT)),
                          json.dumps({**STEP_CASES["step lm-dense 4x1024"], "batch_size": 1}),
                          str(Path(__file__).resolve().with_name("peak_memory.py"))],
          lambda printed, _: (lambda record: (record["save_ms"], record["restore_ms"], peak(record)))(
              json.loads(printed.strip().splitlines()[-1]))),
    Probe((Row("reference qwen3.5-0.8b 8x3072 vs torch", "items/s", "reference_runs/qwen35_pair.py",
               hardware="A100-SXM4-80GB", head_only=True),),
          lambda args, out: ["--torch-python", args.torch_python, "--out", str(out), "--claim"],
          lambda _, out: _qwen35_comparison(out)),
]
BATTERY = [row for probe in PROBES for row in probe.rows]


def _script(out: Path, source: str) -> Path:
    """`source` as a file beside the probe's output, which a measured peak runs again in a process of its own
    (`python <file>`, as tools/peak_memory.py runs a program)."""
    script = out.with_suffix(".py")
    script.write_text(source)
    return script


def _tools(tree: Path, tool: str, newer: Path) -> Path:
    """The tree's own tools/ where it has `tool`, else the head's (`newer`), else this file's own
    directory, where a gate run outside a checkout keeps the tools no tree has yet."""
    for tools in (tree / "tools", newer, Path(__file__).resolve().parent):
        if (tools / tool).is_file():
            return tools
    return newer


def _run(tree: Path, python: Path, tool: str, argv: list[str], newer: Path, timeout: float,
         cache: Path | None = None, settings: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    tools = _tools(tree, tool or "benchmark_step.py", newer)
    env = {**os.environ, **(settings or {}), "PYTHONPATH": f"{tree / 'src'}:{tools}"}
    if cache is not None:
        # The tree's own compile cache, empty at the gate's start: its first round compiles cold (the compile
        # rows), its later rounds load what that round compiled, and neither tree loads the other's programs.
        # Every program is kept, however quickly it compiled or small it is (the cache's write thresholds,
        # which Dew's own cache setup clears too); a later round still traces and lowers, then loads.
        env.update(JAX_ENABLE_COMPILATION_CACHE="true", JAX_COMPILATION_CACHE_DIR=str(cache),
                   JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
                   JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="-1")
    command = [str(python), *([str(tools / tool)] if tool else []), *argv]
    return subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)


def hardware_status(row: Row, device_kind: str) -> str | None:
    """A hardware-limited row is unmeasured off its named device, never silently passed."""
    if row.hardware is not None and row.hardware not in device_kind:
        return f"not measured on this hardware: requires {row.hardware}, found {device_kind}"
    return None


def measure(tree: Path, python: Path, args: argparse.Namespace, newer: Path, is_head: bool = True,
            cache: Path | None = None, first: bool = True, settings: dict[str, str] | None = None) -> dict:
    """One sample of every row in `tree`, run by `python` with `settings` in its environment: a number, or
    why the row could not run there."""
    sample = {}

    @functools.cache
    def hardware() -> str:
        """This machine's device kind, read once and only when a row names hardware."""
        return subprocess.run([str(python), "-c", "import jax; print(jax.devices()[0].device_kind)"],
                              env={**os.environ, **(settings or {})}, capture_output=True, text=True,
                              check=True).stdout.strip()

    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "out.json"
        for index, probe in enumerate(PROBES):
            names = [row.name for row in probe.rows]
            external = all(row.head_only for row in probe.rows)
            if external and not is_head:
                continue
            claiming = args.claim and any(row.hardware for row in probe.rows)
            if args.only and not claiming and not any(word in name for name in names for word in args.only):
                continue
            if not first and all(row.first_round for row in probe.rows):
                continue  # every row it measures is a first round's compile
            done = None
            try:
                unsupported = {row.name: hardware_status(row, hardware()) for row in probe.rows
                               if row.hardware is not None}
                if any(unsupported.values()):
                    sample.update({name: reason for name, reason in unsupported.items()
                                   if reason is not None})
                    continue
                done = _run(tree, python, probe.tool, probe.arguments(args, out), newer, 3600, cache,
                            settings)
                if done.returncode != 0 and not external:
                    raise RuntimeError(f"{probe.tool} exited {done.returncode}: {done.stderr[-1500:]}")
                values = probe.read(done.stdout, out)
                # A later round's compile is a load from the tree's cache, not a compile: not a sample.
                # A peak the tree's tool did not measure (`peak`) is unmeasured, its row's sample None,
                # beside the rows it did measure.
                sample.update({row.name: {**value, "exit_code": done.returncode} if external
                               else None if value is None else float(value)
                               for row, value in zip(probe.rows, values, strict=True)
                               if first or not row.first_round})
            except Exception as error:  # a row that cannot run in this tree is reported, not gated
                failure = ({"error": str(error)[-400:],
                            "exit_code": None if done is None else done.returncode}
                           if external else f"not run: {str(error)[-400:]}")
                sample.update(dict.fromkeys(names, failure))
            if out.is_file():
                evidence = Path(args.out).resolve().with_suffix("")
                evidence.mkdir(parents=True, exist_ok=True)
                target = evidence / f"{tree.name}-{Path(scratch).name}-{index}.json"
                shutil.copyfile(out, target)
                sample.update({f"evidence {name}": str(target) for name in names})
            out.unlink(missing_ok=True)
    return sample


def run(args: argparse.Namespace) -> int:
    from dew.files import write_atomically

    trees = dict(spec.split("=", 1) for spec in (args.base, args.head))
    (base, base_path), (head, head_path) = ((name, Path(path)) for name, path in trees.items())
    settings = dict(setting.split("=", 1) for setting in getattr(args, "head_env", None) or [])
    # Each tree in the environment its own lock names, so a change of a locked wheel is measured too. A
    # tree that holds no such lock (one older than the locks) cannot run a row, which its samples say.
    lock, pythons, unbuilt = machine_lock(), {}, {}
    for name, path in ((base, base_path), (head, head_path)):
        try:
            pythons[name] = environment(path, lock, Path(args.environments or CACHE))
        except Unlocked as missing:
            unbuilt[name] = f"not run: {missing}"
    claiming = getattr(args, "claim", False)
    rows = [row for row in BATTERY if not args.only or any(word in row.name for word in args.only)
            or (claiming and row.hardware)]
    results = {"base": base, "head": head, "rounds": args.rounds, "rows": [row.__dict__ for row in rows],
               "samples": {base: [], head: []}, "claim": claiming, "head settings": settings,
               "environments": {name: str(python.parent.parent) for name, python in pythons.items()},
               "seconds": {"first round": 0.0, "later rounds": 0.0}}
    # Each tree's compile cache, for this run alone: removed at its end, so no run loads another's programs.
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    caching = tempfile.TemporaryDirectory(prefix="perf-gate-cache-", dir=Path(args.out).resolve().parent)
    with caching as caches:
        for round_ in range(args.rounds):
            order = [(base, base_path), (head, head_path)]
            began = time.perf_counter()
            for name, path in order if round_ % 2 == 0 else order[::-1]:
                kept = (row.name for row in rows if not row.head_only or name == head)
                results["samples"][name].append(
                    dict.fromkeys(kept, unbuilt[name]) if name in unbuilt
                    else measure(path, pythons[name], args, head_path / "tools", name == head,
                                 Path(caches) / ("head" if name == head else "base"), first=round_ == 0,
                                 settings=settings if name == head else None))
                write_atomically(args.out, json.dumps(results, indent=2))
                print(f"round {round_ + 1}: {name} measured", flush=True)
            spent = time.perf_counter() - began
            results["seconds"]["first round" if round_ == 0 else "later rounds"] += spent
            write_atomically(args.out, json.dumps(results, indent=2))
    return report(argparse.Namespace(results=args.out, table=None, claim=True)) if claiming else 0


def measured(sample: object) -> bool:
    """Whether a round's sample of a row is a measurement: a finite positive number. A row a
    round could not run holds why (`not run: ...`), and a round that wrote no sample holds None."""
    return isinstance(sample, float) and math.isfinite(sample) and sample > 0


def reference_verdict(head: list) -> tuple[str, dict]:
    """A reference row keeps its own exit status and ratio, even when it fails.

    No base sample is needed: a head that reaches the reference can qualify
    while its parent still trails it. A nonzero child exit is a measured
    failure, not an unmeasured row. Missing rounds never qualify a claim.
    """
    unmeasured = [isinstance(sample, str) and sample.startswith("not measured on this hardware")
                  for sample in head]
    taken = [sample for sample in head if isinstance(sample, dict)]
    codes = [sample.get("exit_code") for sample in taken]
    ratios = [sample.get("ratio") for sample in taken]
    values = [sample.get("value") for sample in taken]
    bands = [sample.get("tolerance") for sample in taken]
    numbers = {"exit_codes": codes}
    if taken and all(map(measured, ratios)):
        numbers["ratio"] = statistics.median(ratios)
    if taken and all(map(measured, values)):
        numbers["head"] = statistics.median(values)
    if taken and all(isinstance(band, float) and math.isfinite(band) and band >= 0 for band in bands):
        numbers["band"] = max(bands)
    if any(code != 0 for code in codes) or any(sample.get("passed") is False for sample in taken):
        return "failed", numbers
    if head and all(unmeasured):
        return "not measured on this hardware", {}
    if (not head or len(taken) != len(head) or not all(sample.get("passed") is True for sample in taken)
            or not {"ratio", "head", "band"} <= numbers.keys()):
        return "incomplete", numbers
    return "passed", numbers


def verdict(row: Row, base: list, head: list) -> tuple[str, dict]:
    """Whether `row` regressed from the `base` samples to the `head` ones, and the numbers that say so.

    A row moves when its median moves by more than the larger of the two trees' spreads (at
    least FLOOR) and the two trees' samples do not overlap at all. With three rounds each, the
    samples separate by chance 1 time in 20 when nothing changed (1 / C(6, 3)), and the band
    lowers that further; it is also the smallest change the row can see. A row the base cannot
    run in any round (a tree older than its tool) is not compared; one the base runs and the head
    runs in no round is broken; one neither runs leaves the battery incomplete. A row either tree
    measured in some rounds and not in others, or measured as NaN, an infinity or zero, is
    partial: the rounds that ran would be judged as if they were all, so it is not judged."""
    if row.head_only:
        return reference_verdict(head)
    if row.hardware and any(isinstance(x, str) and x.startswith("not measured on this hardware")
                            for x in [*base, *head]):
        return "not measured on this hardware", {}
    if not any(measured(x) for x in base) and not any(measured(x) for x in head):
        return "incomplete", {}
    if not any(measured(x) for x in base):
        return "not compared", {}
    if not any(measured(x) for x in head):
        return "broken", {}
    missing = {name: sum(not measured(x) for x in samples)
               for name, samples in (("base", base), ("head", head)) if not all(map(measured, samples))}
    if missing:
        return "partial", {"unmeasured rounds": missing}
    worse = -1.0 if row.unit == "items/s" else 1.0

    def spread(xs):
        return (max(xs) - min(xs)) / statistics.median(xs)
    band = max(spread(base), spread(head), row.floor)
    change = worse * (statistics.median(head) - statistics.median(base)) / statistics.median(base)
    separated = (min(head) > max(base)) if worse > 0 else (max(head) < min(base))
    faster = (max(head) < min(base)) if worse > 0 else (min(head) > max(base))
    state = "regressed" if change > band and separated else "faster" if -change > band and faster else "level"
    return state, {"base": statistics.median(base), "head": statistics.median(head), "change": change,
                   "band": band}


def report(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text())
    base, head = results["base"], results["head"]
    # A round a run never reached counts as unmeasured, so an interrupted run is partial.
    rounds = results.get("rounds", max(map(len, results["samples"].values())))
    samples = {name: [*taken, *[{}] * (rounds - len(taken))] for name, taken in results["samples"].items()}
    lines = [f"| row | unit | {base} | {head} | change | band | verdict |",
             "|---|---|---:|---:|---:|---:|---|"]
    failed = []
    claiming = getattr(args, "claim", False) or results.get("claim", False)
    if claiming:
        included = {row["name"] for row in results["rows"]}
        failed.extend(f"{row.name} (required hardware row omitted)" for row in BATTERY
                      if row.hardware and row.name not in included)
    for spec in results["rows"]:
        row = Row(**spec)
        taken = 1 if row.first_round else rounds
        state, numbers = verdict(row, [s.get(row.name) for s in samples[base][:taken]],
                                 [s.get(row.name) for s in samples[head][:taken]])
        if (state in ("broken", "incomplete", "partial", "failed") or (state == "regressed" and row.gated)
                or (claiming and row.hardware and state in
                    ("not measured on this hardware", "not compared"))):
            failed.append(f"{row.name} ({state})")
        if not row.gated:
            state = f"{state}, not gated"
        if state.startswith("partial"):
            state += " (" + ", ".join(f"{tree} {count} of {rounds} rounds unmeasured"
                                      for tree, count in numbers["unmeasured rounds"].items()) + ")"
        cells = (["-"] * 4 if "base" not in numbers else
                 [f"{numbers['base']:.2f}", f"{numbers['head']:.2f}", f"{numbers['change']:+.1%}",
                  f"{numbers['band']:.1%}"])
        if row.head_only and "exit_codes" in numbers:
            state += " (exit codes " + ", ".join(str(code) for code in numbers["exit_codes"]) + ")"
            cells = ["-", f"{numbers['head']:.2f}" if "head" in numbers else "-",
                     f"{numbers['ratio']:.3f}x reference" if "ratio" in numbers else "-",
                     f"{numbers['band']:.1%}" if "band" in numbers else "-"]
        lines.append(f"| {row.name} | {row.unit} | " + " | ".join(cells) + f" | {state} |")
    if "seconds" in results:
        # The two budgets apart: the first round's cold compiles, the later rounds' timing on cached programs.
        lines.append("")
        lines.append("Rounds: " + ", ".join(f"{part} {spent:.0f} s"
                                            for part, spent in results["seconds"].items()))
    if results.get("head settings"):
        lines.append("")
        lines.append(f"{head} ran with " + ", ".join(f"{name}={value!r}"
                                                     for name, value in results["head settings"].items()))
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
    running.add_argument("--environments", default="",
                         help="where each tree's locked environment is built (tools/environments.py's cache)")
    running.add_argument("--torch-python", default="",
                         help="pinned reference interpreter for the Qwen3.5 row")
    running.add_argument("--claim", action="store_true",
                         help="require every hardware row, including under --only")
    running.add_argument("--only", nargs="*", default=[],
                         help="measure only the rows whose names hold one of these words")
    running.add_argument("--head-env", action="append", default=[], metavar="NAME=VALUE",
                         help="set in the head's processes alone; with one tree as base and head, an A/B "
                              "of that setting (an XLA flag) on every row")
    reporting = operations.add_parser("report")
    reporting.add_argument("results")
    reporting.add_argument("--table")
    reporting.add_argument("--claim", action="store_true",
                           help="fail if a required hardware row was unmeasured")
    args = parser.parse_args()
    with contextlib.suppress(BrokenPipeError):
        return run(args) if args.operation == "run" else report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
