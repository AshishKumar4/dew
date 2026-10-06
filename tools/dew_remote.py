#!/usr/bin/env python3
"""Run a pushed Dew revision on the authenticated Cloudflare CPU fleet.

    dew-remote run <revision> [--python 3.14] -- <command...>
    dew-remote suite <revision> [--python 3.14] [--jobs N]

`suite` runs CI's non-network suite as N concurrent jobs and prints one
report of every job's outcome; it exits nonzero when any job fails or ends
without its summary. Each job runs whole test files, packed by the seconds
tests/test_durations.json records at that revision, so it imports only its
own modules (collecting the whole suite costs a job two minutes); a file
heavier than a job's share runs as that many pytest-split groups of itself.
"""

import argparse
import heapq
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUITE = ["python", "-m", "pytest", "-q", "-m", "not network", "-rfE", "--tb=line"]
"""CI's selection (.github/workflows/ci.yml), reported briefly; the files follow."""
SKIPPED = {"tests/test_gen_api.py"}
"""CI runs it in the lint job, under the griffe it pins."""
SPLIT = ["--splitting-algorithm", "duration_based_chunks", "--durations-path", "tests/test_durations.json"]
SUMMARY = re.compile(r"^=*\s*(?:\d+ \w+(?:, )?)+ in [\d.]+s")
COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)")


def stream(revision: str, python: str, command: list[str], echo: bool = True) -> tuple[int, str]:
    """Run `command` at `revision` and return its exit code and output; the
    output is echoed as it arrives where `echo`, and logged under ~/.cache/dew/remote."""
    config = json.loads((Path.home() / ".config/dew-remote.json").read_text())
    body = json.dumps({"revision": revision, "python": python, "command": command}).encode()
    url = config["endpoint"].rstrip("/") + "/v1/remote/run"
    headers = {"Authorization": "Bearer " + config["token"], "Content-Type": "application/json",
               "User-Agent": "Dew-Gateway-Operator/1.0"}
    deadline = time.monotonic() + 40 * 60
    while True:
        try:
            request = urllib.request.Request(url, data=body, headers=headers)
            response = urllib.request.urlopen(request, timeout=2500)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503) or time.monotonic() >= deadline:
                message = error.read().decode()
                print(message, file=sys.stderr)
                return 1, message
            outcome = json.loads(error.read())
            if echo:
                print(outcome.get("message", "Waiting for a runner"), file=sys.stderr, flush=True)
            time.sleep(min(int(error.headers.get("Retry-After", "15")), 30))
    logs = Path.home() / ".cache/dew/remote"
    logs.mkdir(parents=True, exist_ok=True)
    exit_code = 1
    log = None
    output: list[str] = []
    try:
        with response:
            for line in response:
                event = json.loads(line)
                if event["type"] == "job":
                    path = logs / f"{event['id']}.log"
                    log = path.open("w")
                    if echo:
                        print(f"Job {event['id']}; log: {path}", file=sys.stderr, flush=True)
                elif event["type"] in ("stdout", "stderr"):
                    output.append(event["text"])
                    if echo:
                        target = sys.stdout if event["type"] == "stdout" else sys.stderr
                        print(event["text"], end="", file=target, flush=True)
                    if log:
                        log.write(event["text"])
                        log.flush()
                elif event["type"] == "exit":
                    exit_code = event["code"]
                elif event["type"] == "error":
                    output.append(event["message"] + "\n")
                    if echo:
                        print(event["message"], file=sys.stderr, flush=True)
    except (OSError, ValueError) as error:
        output.append(f"the stream ended early: {error!r}\n")
    finally:
        if log:
            log.close()
    return exit_code, "".join(output)


def plan(weights: dict[str, float], jobs: int) -> list[list[str]]:
    """The arguments of about `jobs` pytest commands that run every file in
    `weights` (its recorded seconds) once, each command near an equal share.

    A file heavier than a share runs as that many pytest-split groups of
    itself; the rest are packed heaviest first into the lightest command.
    """
    share = sum(weights.values()) / jobs
    commands = []
    for name, weight in weights.items():
        if weight > share:
            groups = math.ceil(weight / share)
            commands += [[name, "--splits", str(groups), "--group", str(group), *SPLIT]
                         for group in range(1, groups + 1)]
    remaining = max(jobs - len(commands), 1)
    bins: list[tuple[float, int, list[str]]] = [(0.0, index, []) for index in range(remaining)]
    for name, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        if weight <= share:
            held, index, names = heapq.heappop(bins)
            heapq.heappush(bins, (held + weight, index, [*names, name]))
    return commands + [sorted(names) for _, _, names in sorted(bins, key=lambda bin_: bin_[1]) if names]


def weights_at(revision: str, checkout: Path) -> dict[str, float]:
    """Every test file of `revision` in `checkout`, with the seconds its
    recorded tests took; a file the record does not name weighs the mean."""
    git = ["git", "-C", str(checkout)]
    subprocess.run([*git, "fetch", "-q", "origin"], check=False)
    commit = next(candidate for candidate in (revision, f"origin/{revision}") if subprocess.run(
        [*git, "rev-parse", "-q", "--verify", f"{candidate}^{{commit}}"], capture_output=True
    ).returncode == 0)
    listed = subprocess.run([*git, "ls-tree", "-r", "--name-only", commit, "tests/"],
                            capture_output=True, text=True, check=True).stdout.split()
    files = [name for name in listed if re.fullmatch(r"tests/test_[^/]*\.py", name) and name not in SKIPPED]
    recorded = json.loads(subprocess.run([*git, "show", f"{commit}:tests/test_durations.json"],
                                         capture_output=True, text=True, check=True).stdout)
    seconds: dict[str, float] = {}
    for node, weight in recorded.items():
        seconds[node.split("::")[0]] = seconds.get(node.split("::")[0], 0.0) + weight
    known = [seconds[name] for name in files if name in seconds]
    mean = sum(known) / len(known) if known else 1.0
    return {name: seconds.get(name, mean) for name in files}


CHECKOUT = Path(__file__).resolve().parents[1]
"""The repository this tool sits in, which reads a revision's test files and their record."""


def suite(revision: str, python: str, jobs: int, checkout: Path = CHECKOUT) -> int:
    """Run the suite at `revision` as about `jobs` concurrent commands and report them as one run.

    A command whose output holds no pytest summary lost its stream or its
    runner, not a test, so it runs once more before it counts as failed.
    """
    commands = plan(weights_at(revision, checkout), jobs)

    def group(index: int) -> tuple[int, int, str]:
        command = [*SUITE, *commands[index - 1]]
        for _ in range(2):
            code, output = stream(revision, python, command, echo=False)
            if any(SUMMARY.match(line) for line in output.splitlines()):
                break
        return index, code, output

    began = time.monotonic()
    with ThreadPoolExecutor(len(commands)) as pool:
        groups = sorted(pool.map(group, range(1, len(commands) + 1)))
    totals: dict[str, int] = {}
    failures, unfinished = [], []
    for index, code, output in groups:
        summary = [line for line in output.splitlines() if SUMMARY.match(line)]
        if not summary:
            unfinished.append(f"group {index}: exit {code}, no pytest summary; {output.strip()[-300:]}")
            continue
        for count, kind in COUNT.findall(summary[-1]):
            kind = "errors" if kind.startswith("error") else kind
            totals[kind] = totals.get(kind, 0) + int(count)
        failures += [f"group {index}: {line}" for line in output.splitlines()
                     if line.startswith(("FAILED ", "ERROR "))]
        # pytest exits 5 for a pytest-split group of a file that selects nothing.
        if code not in (0, 5) and not any(line.startswith(f"group {index}: ") for line in failures):
            failures.append(f"group {index}: exit {code}")
    minutes = (time.monotonic() - began) / 60
    report = [f"{revision} on Python {python}: {len(commands)} groups in {minutes:.1f} min",
              ", ".join(f"{count} {kind}" for kind, count in sorted(totals.items())) or "no tests ran",
              *failures, *unfinished]
    path = Path.home() / f".cache/dew/remote/suite-{revision}-{python}-{int(time.time())}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"report: {path}", file=sys.stderr)
    return 1 if failures or unfinished else 0


def main():
    parser = argparse.ArgumentParser(prog="dew-remote")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("revision")
    run.add_argument("--python", choices=("3.12", "3.14"), default="3.12")
    whole = subparsers.add_parser("suite")
    whole.add_argument("revision")
    whole.add_argument("--python", choices=("3.12", "3.14"), default="3.12")
    whole.add_argument("--jobs", type=int, default=100,
                       help="groups to split the suite into; 100 runs about 3 minutes each")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    if args.operation == "suite":
        return suite(args.revision, args.python, args.jobs)
    command = arguments[separator + 1:]
    if not command:
        parser.error("run needs a command after --")
    return stream(args.revision, args.python, command)[0]


if __name__ == "__main__":
    raise SystemExit(main())
