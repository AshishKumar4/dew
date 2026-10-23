"""Give a recorded program's capture the digest of the program as the page now shows it.

A recording names the sha256 of the program it ran (site/snippets/cells.py declares which capture names which
program's, and under which key), and the page refuses one of other code (src/data/recordings.mjs). A change
that cannot alter what a program shows, a rename among them, keeps its recording: the change's review says so,
and the cell's CPU smoke runs it. This writes each named program's new digest into its capture and changes
nothing else there, byte for byte. A change that alters what a program shows records it again instead
(site/scripts/capture_snippets.py).

    python site/scripts/digest.py lm hero hero.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
import cells


def recorded(capture: dict, key: str) -> str:
    """The digest `capture` names under `key` (dotted: meta.cell_sha256)."""
    for part in key.split("."):
        capture = capture[part]
    return capture


def main() -> None:
    recordings = cells.recordings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", choices=sorted(recordings), help="the recorded programs")
    options = parser.parse_args()
    programs = cells.programs()
    for name in options.names:
        path, key = recordings[name]
        capture = cells.SITE / path
        text = capture.read_bytes()
        old, new = recorded(json.loads(text), key), cells.digest(programs[name])
        if text.count(old.encode()) != 1:
            raise SystemExit(f"{name}: {path} names {old} {text.count(old.encode())} times, not once, "
                             f"under {key}")
        capture.write_bytes(text.replace(old.encode(), new.encode()))
        print(f"{name}: {path} {key} {old[:12]} -> {new[:12]}")


if __name__ == "__main__":
    main()
