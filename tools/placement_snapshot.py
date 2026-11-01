"""Write where Dew places every parameter of every model it builds (tests/placement_snapshot.py) at this
checkout, to tests/fixtures/placement/snapshot.tsv, which tests/test_placement_snapshot.py holds each case
to: one line a leaf, so a change to how axes are declared shows each placement it moves.

    python tools/placement_snapshot.py

Run it on CPU under the suite's environment, from the checkout whose placements are the contract: a change
that moves a placement on purpose writes the snapshot again in its own head, its diff the moves."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
SNAPSHOT = ROOT / "tests" / "fixtures" / "placement" / "snapshot.tsv"


def main() -> None:
    import os

    from lane_environment import configure_lane

    # The suite's environment: its eight simulated devices among it.
    configure_lane(os.environ)
    from placement_snapshot import COLUMNS, cases, snapshot

    lines = ["\t".join(COLUMNS)]
    for name, build in sorted(cases().items()):
        lines += snapshot(name, build)
        print(f"{name}: placed", flush=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text("\n".join(lines) + "\n")
    print(f"{SNAPSHOT}: {len(lines) - 1} leaves")


if __name__ == "__main__":
    main()
