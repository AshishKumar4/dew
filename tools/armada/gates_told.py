"""gates_told.py OUTCOMES: CI's lint job's gates (tools/check_gates' names, which `armada map --json` ran one
a task and wrote OUTCOMES of, each naming its item) said one a line in check_gates' order, each with its exit
and seconds, and under a red one the last `SHOWN` lines of what it printed, so a gate's findings are in the
job's log, where the land queue quotes them. Exits 1 where a gate is red or has no outcome, 0 where every one
is green."""

import json
import sys
from pathlib import Path

SHOWN = 60
"""The last lines of a red gate's output said under it: a duplicate's two locations and its text, a type
error's file, line and message."""


def told(path: str) -> int:
    gates = [line.split("\t")[0] for line in Path("tools/check_gates").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    found = {}
    with open(path, errors="replace") as lines:
        for line in lines:
            if line.startswith("{") and "index" in (outcome := json.loads(line)):
                # Each outcome named by the item armada ran (its stdin's line), never by its place among them.
                found[str(outcome.get("item", "")).strip()] = outcome
            elif not line.startswith("{"):
                print(line, end="")
    red = 0
    for gate in gates:
        outcome = found.get(gate)
        if outcome is None:
            print(f"{gate}: no outcome")
            red = 1
            continue
        code = outcome.get("exitCode")
        print(f"{gate}: exit {code} in {outcome.get('seconds', 0):.0f} s")
        if code != 0:
            red = 1
            for text in str(outcome.get("tail", "")).splitlines()[-SHOWN:]:
                print(f"  | {text}")
    return red


if __name__ == "__main__":
    sys.exit(told(sys.argv[1]))
