#!/usr/bin/env python3
"""Run Dew and the pinned torch reference consecutively in this session.

Each round starts fresh processes from this checkout on the same device, with
the same weights/seed and a 20-warmup/200-timed-step window: Dew's, torch's, and
torch's again (`torch~aa`, its A/A), in an order drawn at random each round and
kept. Only the window's endpoint-synchronized wall throughput decides the gate:
each round's Dew over its torch, judged against each round's torch~aa over its
torch by Dew's one judgment of a time against another
(`dew.telemetry.benchmark.judged`, `performance_verdict`); per-step
event/synchronized diagnostics never enter the ratio. Dew slower, the judgment
differing below 1, fails; level means no difference found, not parity proven.

Use --torch-python /path/to/pinned/bin/python; the reference validates its
complete lock before running. The caller's interpreter supplies Dew unless
--dew-python names another. --smoke exercises both real CPU steps, but cannot
pass a performance claim. A pairing judges speed alone: whether the two compute
the same loss is check 1's (tests/reference_error.py). The record contains both
source commits, full environments, input hashes, timing windows and scoped peaks.
"""

import argparse
import json
import math
import os
import secrets
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


ROUNDS = 6
"""The pairing's rounds unless it names its own: the fewest whose ratios reach an interval at `LEVEL`."""
LEVEL = 0.95
ARMS = ("dew", "torch", "torch~aa")
"""The pairing's arms: Dew's step, torch's, and torch's again, its A/A, each round."""


def framework_of(arm: str) -> str:
    """The framework an arm runs: `torch~aa` runs torch's step, as torch, and is validated as torch's."""
    return arm.removesuffix("~aa")


def performance_verdict(dew: list[float], torch: list[float], alone: list[float]) -> dict:
    """Judge repeated-window throughputs, a round's each: each round's Dew over its torch, against each
    round's torch~aa over its torch (`dew.telemetry.benchmark.judged` at `LEVEL`). Dew is slower where the
    judgment differs below 1, faster where it differs above it, and level otherwise: no difference found,
    not parity proven. Missing or invalid samples never pass; too few rounds reach no interval and raise."""
    from dew.telemetry import benchmark

    if not len(dew) == len(torch) == len(alone) or any(
            not math.isfinite(value) or value <= 0 for samples in (dew, torch, alone) for value in samples):
        raise ValueError("need equal counts of finite positive window throughputs, one an arm a round")
    judged = benchmark.judged([mine / theirs for mine, theirs in zip(dew, torch, strict=True)],
                              [again / theirs for again, theirs in zip(alone, torch, strict=True)],
                              level=LEVEL)
    state = "level" if not judged.differs else "slower" if judged.interval[1] < 1 else "faster"
    return {"verdict": state, "passed": state != "slower", "dew_over_torch": judged.ratio,
            "interval": list(judged.interval), "alone": list(judged.alone), "differs": judged.differs,
            "rounds": len(dew), "dew_tokens_per_s": statistics.median(dew),
            "torch_tokens_per_s": statistics.median(torch),
            "spread_scope": "each round's endpoint-synchronized window throughputs, paired"}


def compare(records: dict[str, list[dict]], session: str, smoke: bool) -> dict:
    """Refuse mismatched workloads or sessions before considering speed."""
    from qwen35 import WHOLE_PEAK_SCOPE, workload

    reference = records["torch"][0]
    config = reference["config"]
    geometry = tuple(config[key] for key in ("vocab", "hidden", "batch", "sequence", "warmup", "steps"))
    if not smoke and geometry != (248320, 1024, 8, 3072, 20, 200):
        raise ValueError("the claim requires the full 8x3072, 20-warmup/200-step Qwen3.5 workload")
    expected = {**workload(config["seed"], config["vocab"], config["hidden"], config["batch"],
                          config["sequence"], smoke, config["warmup"], config["steps"]),
                "work": reference["work"]}
    for arm, rounds in records.items():
        for record in rounds:
            if (record["framework"] != framework_of(arm) or record["session"] != session
                    or record["smoke"] != smoke):
                raise ValueError("record belongs to a different framework, session or smoke mode")
            if record["commit"] != reference["commit"]:
                raise ValueError("checkout changed during the paired session")
            if record["environment"]["packages"] != rounds[0]["environment"]["packages"]:
                raise ValueError("framework environment changed between rounds")
            if record["host"]["hostname"] != reference["host"]["hostname"]:
                raise ValueError("records did not run on the same host")
            if record["host"]["cuda_visible_devices"] != reference["host"]["cuda_visible_devices"]:
                raise ValueError("records did not use the same device selection")
            if record["environment"]["device"] != reference["environment"]["device"]:
                raise ValueError("records did not run on the same device kind")
            for field, value in expected.items():
                if {**record["config"], "work": record["work"]}.get(field) != value:
                    raise ValueError(f"unmatched workload field {field}")
            if record["config"]["parameter_count"] != config["parameter_count"]:
                raise ValueError("unmatched backbone parameter count")
            if record["memory"]["whole_peak_scope"] != WHOLE_PEAK_SCOPE:
                raise ValueError("unmatched peak scope")
            if not smoke and (record["memory"]["whole_peak_bytes"] is None or
                              record["memory"]["whole_peak_bytes"] <= 0):
                raise ValueError("missing whole-run peak")
            if not record["checks"]["finite_losses"] or not record["checks"]["no_timing_window_compilation"]:
                raise ValueError("non-finite loss or compilation in timing window")
    rates = {arm: [record["steady"]["window_tokens_per_s"] for record in rounds]
             for arm, rounds in records.items()}
    decision = performance_verdict(rates["dew"], rates["torch"], rates["torch~aa"])
    return {**decision, "performance_claim_passed": decision["passed"] and not smoke,
            "speedup_claim_passed": decision["verdict"] == "faster" and not smoke,
            "window_tokens_per_s": rates,
            "mfu": {key: [row["steady"]["window_mfu"] for row in rows] for key, rows in records.items()},
            "whole_peak_bytes": {key: [row["memory"]["whole_peak_bytes"] for row in rows]
                                  for key, rows in records.items()},
            "peak_scope": WHOLE_PEAK_SCOPE,
            "peak_caveat": "same lifetime scope; torch and XLA allocator accounting differ"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--dew-python", default=sys.executable)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    from dew.telemetry import benchmark

    try:
        # The judgment's own word on whether these rounds can reach an interval, before any runs.
        benchmark.judged([1.0] * args.rounds, [1.0] * args.rounds, level=LEVEL)
    except ValueError as few:
        parser.error(f"--rounds {args.rounds}: {few}")
    if args.claim and args.smoke:
        parser.error("CPU smoke cannot qualify a performance claim")
    return args


def run(args) -> int:
    """Fresh child processes release their device memory before their partner starts."""
    from common import host, write_record
    from reference_source import git_head

    root = Path.cwd().resolve()
    args.out = args.out.resolve()
    directory = args.out.with_suffix("")
    directory.mkdir(parents=True, exist_ok=False)
    session = str(uuid.uuid4())
    records: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    result = {"schema": "dew.qwen35-pair.v1", "session": session, "smoke": args.smoke,
              "commit": git_head(root), "host": host(), "started_ns": time.time_ns(),
              "records": records, "commands": [], "comparison": None}
    write_record(args.out, result)
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    work = directory / "work.json"
    command = [args.dew_python, str(Path(__file__).resolve().with_name("work.py")), "qwen35",
               "--out", str(work)]
    if args.smoke:
        command += ["--smoke", "--dtype", "float32"]
    counted = subprocess.run(command, cwd=root, env=environment, check=True)
    result["commands"].append({"argv": command, "returncode": counted.returncode})
    fixture = directory / "tiny"

    def step(arm: str, out: Path, *extra: str) -> int:
        """`arm`'s step in a fresh process, its record at `out`, as its framework runs it (`framework_of`)."""
        framework = framework_of(arm)
        script = root / "tools/reference_runs" / f"{framework}_qwen35_step.py"
        if not script.is_file():
            script = Path(__file__).resolve().with_name(script.name)
        command = [args.torch_python if framework == "torch" else args.dew_python, str(script),
                   "--out", str(out), "--work", str(work), "--seed", str(args.seed), "--session", session,
                   *(["--smoke", "--warmup", "3", "--steps", "4"] if args.smoke else []), *extra]
        done = subprocess.run(command, cwd=root, env=environment, check=False)
        result["commands"].append({"argv": command, "returncode": done.returncode})
        return done.returncode

    # A smoke's tiny checkpoint, seeded, exported by torch's step before the session, so that whichever arm a
    # round draws first, Dew's step loads the weights torch's start from.
    if args.smoke and step("torch", directory / "fixture.json", "--smoke-fixture", str(fixture)):
        result["error"] = "torch's step did not export the smoke's checkpoint"
        write_record(args.out, result)
        return 1
    places = secrets.SystemRandom()
    result["orders"] = []
    for round_ in range(args.rounds):
        order = places.sample(ARMS, len(ARMS))
        result["orders"].append(order)
        for arm in order:
            out = directory / f"{round_}-{arm}.json"
            if step(arm, out, *(["--smoke-fixture", str(fixture)] if args.smoke and arm == "dew" else [])):
                result["error"] = f"{arm} round {round_} exited {result['commands'][-1]['returncode']}"
                write_record(args.out, result)
                return 1
            records[arm].append(json.loads(out.read_text()))
            write_record(args.out, result)
    result["finished_ns"] = time.time_ns()
    try:
        result["comparison"] = compare(records, session, args.smoke)
    except (ValueError, KeyError) as error:
        result["error"] = str(error)
        write_record(args.out, result)
        return 1
    write_record(args.out, result)
    print(json.dumps(result["comparison"], indent=2))
    return 0 if args.smoke or result["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
