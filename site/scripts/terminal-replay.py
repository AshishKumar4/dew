"""Prepare text frames from an asciinema v2 recording, without running training.

    python site/scripts/terminal-replay.py lm.cast --meta meta.json --script demo.py

meta.json contains the same machine/runtime fields as capture.json's meta.
The original cast is kept alongside the sampled frames for reproducibility.
"""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pyte
from capture_snippets import DATA, DimScreen, terminal_cells


def frames_from_cast(cast: str, count: int = 97) -> dict:
    records = [json.loads(line) for line in cast.splitlines() if line.strip()]
    header, events = records[0], records[1:]
    if header.get("version") != 2 or not events:
        raise ValueError("expected an asciinema v2 recording with events")
    seconds = events[-1][0]
    if seconds <= 0 or any(a[0] > b[0] for a, b in pairwise(events)):
        raise ValueError("recording times must increase and end after zero")
    screen = DimScreen(header["width"], header["height"])
    stream = pyte.Stream(screen)
    frames = []
    index = 0
    for position in range(count):
        timestamp = seconds * position / (count - 1)
        while index < len(events) and events[index][0] <= timestamp:
            _, kind, payload = events[index]
            if kind == "o":
                stream.feed(payload)
            index += 1
        rows = terminal_cells(screen, header["width"])
        # Identical screens need no new frame, but keep the final timestamp.
        if not frames or rows != frames[-1]["screen"] or position == count - 1:
            frames.append({"time": timestamp, "screen": rows})
    return {"columns": header["width"], "seconds": seconds,
            "speed": max(1, seconds / 24), "frames": frames,
            "sha256": hashlib.sha256(cast.encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cast", type=Path)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    options = parser.parse_args()
    meta = json.loads(options.meta.read_text())
    if any(not meta.get(key) for key in ("where", "jax", "dew", "date")):
        raise ValueError("recording metadata needs where, jax, dew and date")
    cast = options.cast.read_text()
    replay = frames_from_cast(cast)
    output = DATA.parents[1] / "public/hero"
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.cast").write_text(cast)
    (output / "train.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
    capture = {"about": "Terminal frames from public/hero/train.cast, prepared by scripts/terminal-replay.py.",
               "meta": meta, "hero": {"returncode": 0, "seconds": replay["seconds"],
                                      "columns": replay["columns"], "screen": replay["frames"][-1]["screen"],
                                      "rows": max(len(frame["screen"]) for frame in replay["frames"]),
                                      "replay": "/hero/train.json"}}
    (DATA / "capture.json").write_text(json.dumps(capture, indent="\t") + "\n")
    (DATA / "hero.py").write_text(options.script.read_text())
    print(f"prepared {len(replay['frames'])} frames from {replay['seconds']:.1f} s of recording")  # noqa: T201


if __name__ == "__main__":
    main()
