#!/usr/bin/env python3
"""Run a pushed Dew revision on the authenticated Cloudflare CPU fleet."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="dew-remote")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("revision")
    run.add_argument("--python", choices=("3.12", "3.14"), default="3.12")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    if not command:
        parser.error("run needs a command after --")
    config = json.loads((Path.home() / ".config/dew-remote.json").read_text())
    body = json.dumps({"revision": args.revision, "python": args.python, "command": command}).encode()
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
                print(error.read().decode(), file=sys.stderr)
                return 1
            outcome = json.loads(error.read())
            print(outcome.get("message", "Waiting for a runner"), file=sys.stderr, flush=True)
            time.sleep(min(int(error.headers.get("Retry-After", "15")), 30))
    directory = Path.home() / ".cache/dew/remote"
    directory.mkdir(parents=True, exist_ok=True)
    exit_code = 1
    log = None
    try:
        with response:
            for line in response:
                event = json.loads(line)
                if event["type"] == "job":
                    path = directory / f"{event['id']}.log"
                    log = path.open("w")
                    print(f"Job {event['id']}; log: {path}", file=sys.stderr, flush=True)
                elif event["type"] in ("stdout", "stderr"):
                    output = sys.stdout if event["type"] == "stdout" else sys.stderr
                    print(event["text"], end="", file=output, flush=True)
                    if log:
                        log.write(event["text"])
                        log.flush()
                elif event["type"] == "exit":
                    exit_code = event["code"]
                elif event["type"] == "error":
                    print(event["message"], file=sys.stderr, flush=True)
    finally:
        if log:
            log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
