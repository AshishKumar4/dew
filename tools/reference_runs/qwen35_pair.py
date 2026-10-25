#!/usr/bin/env python3
"""Run Dew and the pinned torch reference consecutively in this session.

Each round starts fresh processes from this checkout on the same device, with
the same weights/seed and a 20-warmup/200-timed-step window. Only the window's
endpoint-synchronized wall throughput decides the gate. Two or more rounds
measure min-max throughput spread; per-step event/synchronized diagnostics
never enter the ratio. A deficit beyond the larger relative spread fails.
Within-spread differences mean level, not evidence of a speedup.

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
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def performance_verdict(dew: list[float], torch: list[float]) -> dict:
    """Judge repeated-window throughputs; missing/invalid samples never pass.

    The band is the larger framework's full observed throughput range divided
    by its median. There is no arbitrary percentage floor and no per-step
    noise substituted for window noise. At least two windows on each side are
    needed to observe a spread. A level verdict allows statistical parity,
    not the assertion that Dew is faster.
    """
    if len(dew) != len(torch) or any(len(samples) < 2 or any(
            not math.isfinite(value) or value <= 0 for value in samples) for samples in (dew, torch)):
        raise ValueError("need equal counts of at least two finite positive window throughputs")
    medians = [statistics.median(samples) for samples in (dew, torch)]
    band = max((max(samples) - min(samples)) / median
               for samples, median in zip((dew, torch), medians, strict=True))
    ratio = medians[0] / medians[1]
    state = "slower" if ratio < 1 - band else "faster" if ratio > 1 + band else "level"
    return {"verdict": state, "passed": state != "slower", "dew_over_torch": ratio,
            "tolerance": band, "dew_tokens_per_s": medians[0], "torch_tokens_per_s": medians[1],
            "spread_scope": "min-max of repeated endpoint-synchronized window throughputs"}


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
    for framework, rounds in records.items():
        for record in rounds:
            if record["framework"] != framework or record["session"] != session or record["smoke"] != smoke:
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
    rates = {framework: [record["steady"]["window_tokens_per_s"] for record in rounds]
             for framework, rounds in records.items()}
    decision = performance_verdict(rates["dew"], rates["torch"])
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
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.rounds < 2:
        parser.error("at least two windows per framework are required to measure spread")
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
    records = {"torch": [], "dew": []}
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
    for round_ in range(args.rounds):
        for framework in ("torch", "dew"):
            out = directory / f"{round_}-{framework}.json"
            script = root / "tools/reference_runs" / f"{framework}_qwen35_step.py"
            if not script.is_file():
                script = Path(__file__).resolve().with_name(script.name)
            command = [args.torch_python if framework == "torch" else args.dew_python,
                       str(script),
                       "--out", str(out), "--work", str(work), "--seed", str(args.seed), "--session", session]
            if args.smoke:
                command += ["--smoke", "--smoke-fixture", str(directory / f"tiny-{round_}"),
                            "--warmup", "3", "--steps", "4"]
            done = subprocess.run(command, cwd=root, env=environment, check=False)
            result["commands"].append({"argv": command, "returncode": done.returncode})
            if done.returncode:
                result["error"] = f"{framework} round {round_} exited {done.returncode}"
                write_record(args.out, result)
                return 1
            records[framework].append(json.loads(out.read_text()))
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
