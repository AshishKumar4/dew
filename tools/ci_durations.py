#!/usr/bin/env python3
"""Write the test durations CI splits its shards by, from a run's JUnit reports.

pytest-split balances the shards on `tests/test_durations.json`, a map from
each test's node id to its seconds. A test the file does not name counts as
the average, so the file only needs refreshing when the suite's weight moves:

    gh run download <run id> --repo AshishKumar4/dew --dir /tmp/ci-run
    python tools/ci_durations.py /tmp/ci-run/pytest-report-3.14-*/pytest-report.xml
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "test_durations.json"


def node_id(classname: str, name: str) -> str:
    """`tests.test_x.Case` and `test_y[a]` as pytest's `tests/test_x.py::Case::test_y[a]`."""
    parts = classname.split(".")
    module = next(index for index, part in enumerate(parts) if part.startswith("test_"))
    return "::".join(["/".join(parts[:module + 1]) + ".py", *parts[module + 1:], name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    durations = {}
    for report in args.reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            # A failure's or a skip's time is not the test's weight: a timeout
            # would outweigh a shard, and a skip weighs nothing on this lane.
            if any(case.find(kind) is not None for kind in ("failure", "error", "skipped")):
                continue
            durations[node_id(case.get("classname", ""), case.get("name", ""))] = float(case.get("time", 0))
    OUT.write_text(json.dumps(dict(sorted(durations.items())), indent=0) + "\n")
    print(f"{len(durations)} tests, {sum(durations.values()):.0f} s, written to {OUT}")


if __name__ == "__main__":
    main()
