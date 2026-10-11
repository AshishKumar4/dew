#!/usr/bin/env python3
"""Run the approved A100 head-to-head and attribute both arms on the same VM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ATTRIBUTOR = "b420100f20856822ae19d7f314a9a22ba24e26df"
BENCHMARK = Path(__file__).with_name("benchmark_decision_budgets.py")


def run_arm(args: argparse.Namespace, arm: str, timeout: int, *, profile: bool = False) -> Path:
    output = args.output / (arm + ("-profile" if profile else ""))
    output.mkdir(parents=True, exist_ok=True)
    flags = [sys.executable, str(BENCHMARK), "--arm", arm, "--output", str(output / "entry.json")]
    if args.smoke:
        flags.extend(["--smoke", "--warmup-updates", "1", "--timed-updates", "2", "--windows", "1"])
    env = {**os.environ, "PYTHONPATH": "src", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    if profile:
        flags.extend(["--warmup-updates", "32", "--timed-updates", "5", "--windows", "1",
                      "--profile-directory", str(output / "trace")])
        env["XLA_FLAGS"] = " ".join(filter(None, (os.environ.get("XLA_FLAGS"),
            "--xla_gpu_enable_command_buffer=", f"--xla_dump_to={output / 'hlo'}",
            "--xla_dump_hlo_as_text", "--xla_dump_hlo_module_re=jit_step")))
    with (output / "run.log").open("w") as log:
        subprocess.run(flags, check=True, env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout)
    return output


def attribute(output: Path, side: Path, timeout: int) -> None:
    env = {**os.environ, "PYTHONPATH": str(side / "src")}
    subprocess.run([sys.executable, str(side / "tools/attribute_step.py"),
                    "--trace", str(output / "trace"), "--hlo", str(output / "hlo"),
                    "--steps", "10", "--entry", str(output / "entry.json"),
                    "--out", str(output / "attribution.json")], check=True, env=env, timeout=timeout)


def remaining(args: argparse.Namespace, started: float) -> int:
    seconds = int(args.hard_cap_minutes * 60 - (time.monotonic() - started) - 180)
    if seconds < 1:
        raise TimeoutError("hard-cap guard leaves less than three minutes for cleanup")
    return min(seconds, args.command_timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-cap-minutes", type=int, default=150)
    parser.add_argument("--command-timeout", type=int, default=3600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    reports = {}
    for arm in ("rows", "tokens"):
        output = run_arm(args, arm, remaining(args, started))
        report = json.loads((output / "entry.json").read_text())
        reports[arm] = {name: report[name] for name in (
            "windows", "compile_count", "whole_run_compile_seconds", "live_peak_bytes", "checks")}
    if not args.smoke:
        with tempfile.TemporaryDirectory(prefix="dew-attribution-") as temporary:
            side = Path(temporary) / "dew"
            subprocess.run(["git", "fetch", "origin", ATTRIBUTOR], check=True, timeout=120)
            subprocess.run(["git", "worktree", "add", "--detach", str(side), ATTRIBUTOR], check=True)
            try:
                for arm in ("rows", "tokens"):
                    if remaining(args, started) < 600:
                        raise TimeoutError("less than ten minutes remain for required attribution")
                    output = run_arm(args, arm, remaining(args, started), profile=True)
                    attribute(output, side, remaining(args, started))
                    reports[arm]["attribution"] = json.loads((output / "attribution.json").read_text())
            finally:
                subprocess.run(["git", "worktree", "remove", str(side)], check=True)
    summary = {"commit": sha, "smoke": args.smoke, "elapsed_seconds": time.monotonic() - started,
               "arms": reports, "artifact_policy": "retain raw profiles/HLO on this VM; export summary only"}
    (args.output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"commit": sha, "smoke": args.smoke,
                      "seconds": summary["elapsed_seconds"], "checks": {arm: reports[arm]["checks"]
                                                                          for arm in reports}}))


if __name__ == "__main__":
    main()
