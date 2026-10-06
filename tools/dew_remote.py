#!/usr/bin/env python3
"""Run a pushed Dew revision on the authenticated Cloudflare CPU fleet.

    dew-remote run <revision> [--python 3.14] -- <command...>
    dew-remote suite <revision> [--python 3.14] [--jobs N]

`suite` runs CI's non-network suite as N concurrent jobs, each one group of
pytest-split's duration_based_chunks over tests/test_durations.json (as the
CI shards are, but more of them), and prints one report of every job's
outcome; it exits nonzero when any job fails or ends without its summary.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUITE = ["python", "-m", "pytest", "-q", "-m", "not network", "--ignore=tests/test_gen_api.py", "-rfE",
         "--tb=line", "--splitting-algorithm", "duration_based_chunks",
         "--durations-path", "tests/test_durations.json"]
"""CI's selection (.github/workflows/ci.yml), reported briefly; `--splits N --group i` follows."""
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


def suite(revision: str, python: str, jobs: int) -> int:
    """Run the suite's `jobs` groups concurrently and report them as one run.

    A group whose output holds no pytest summary lost its stream or its
    runner, not a test, so it runs once more before it counts as failed.
    """
    def group(index: int) -> tuple[int, int, str]:
        command = [*SUITE, "--splits", str(jobs), "--group", str(index)]
        for _ in range(2):
            code, output = stream(revision, python, command, echo=False)
            if any(SUMMARY.match(line) for line in output.splitlines()):
                break
        return index, code, output

    began = time.monotonic()
    with ThreadPoolExecutor(jobs) as pool:
        groups = sorted(pool.map(group, range(1, jobs + 1)))
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
        # pytest exits 5 for a group the chunks left empty.
        if code not in (0, 5) and not any(line.startswith(f"group {index}: ") for line in failures):
            failures.append(f"group {index}: exit {code}")
    report = [f"{revision} on Python {python}: {jobs} groups in {(time.monotonic() - began) / 60:.1f} min",
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
