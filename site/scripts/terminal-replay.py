"""Prepare text frames from an asciinema v2 recording, without running training.

    python site/scripts/terminal-replay.py lm.cast --meta meta.json --script demo.py

meta.json contains the same machine/runtime fields as capture.json's meta.
The original cast is kept alongside the sampled frames for reproducibility.
"""

import argparse
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

import pyte
from capture_snippets import DATA, DimScreen, terminal_cells


def frames_from_cast(cast: str, count: int = 41) -> dict:
    records = [json.loads(line) for line in cast.splitlines() if line.strip()]
    header, events = records[0], records[1:]
    if header.get("version") != 2 or not events:
        raise ValueError("expected an asciinema v2 recording with events")
    seconds = events[-1][0]
    if seconds <= 0 or any(a[0] > b[0] for a, b in pairwise(events)):
        raise ValueError("recording times must increase and end after zero")
    source = DimScreen(header["width"], header["height"])
    source_stream = pyte.Stream(source)
    screen = DimScreen(header["width"], max(200, header["height"]))
    stream = pyte.Stream(screen)
    frames = []
    index = 0
    previous_key = None
    repeated = False
    while index < len(events):
        while index < len(events):
            timestamp, kind, payload = events[index]
            if kind == "o":
                stream.feed(payload)
                source_stream.feed(payload)
            index += 1
            # A PTY may split one refresh over several reads. Snapshot after
            # that burst, not halfway through the panel being repainted.
            if index == len(events) or events[index][0] - timestamp > 0.002:
                break
        rows = terminal_cells(screen, header["width"])
        # A terminal can retain a fragment of an earlier, shorter live panel
        # above the current one. The viewport starts at the latest Dew panel;
        # the original cast still contains the uncut terminal output.
        panels = [i for i, row in enumerate(rows)
                  if "dew ·" in "".join(run["text"] for run in row)]
        if panels:
            rows = rows[panels[-1]:]
        if not rows:
            continue
        text = "\n".join("".join(run["text"] for run in row) for row in rows)
        progress = re.search(r"\b\d+/\d+\b", text)
        ratio = progress[0] if progress else None
        key = (ratio, 0 if ratio and ratio.startswith("0/") else len(rows))
        frame = {"time": timestamp, "screen": rows}
        # Keep the first and last screen at a step, omitting spinner/clock-only
        # refreshes between them. Every new step and extra output row survives.
        if key == previous_key and repeated:
            frames[-1] = frame
        else:
            frames.append(frame)
            repeated = key == previous_key
        previous_key = key
    if not frames:
        raise ValueError("recording has no visible output")
    if len(frames) > count:
        frames = [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]
    frames[-1]["time"] = seconds
    # Keeping scrollback must not alter the source viewport, even its blank
    # cells and styles. Check the actual terminal cells before serializing.
    offset = screen.cursor.y - source.cursor.y
    for y in range(header["height"]):
        for x in range(header["width"]):
            if screen.buffer[offset + y][x] != source.buffer[y][x]:
                raise ValueError(f"expanded viewport changes recorded cell ({x}, {y})")
    source_rows = terminal_cells(source, header["width"])
    source_rows += [[] for _ in range(header["height"] - len(source_rows))]
    elapsed = 0
    for previous, frame in zip([None, *frames[:-1]], frames, strict=True):
        if previous is not None:
            elapsed += min(0.25, max(0.15, (frame["time"] - previous["time"]) / 2))
        frame["at"] = elapsed
    return {"columns": header["width"], "seconds": seconds,
            "duration": elapsed, "frames": frames,
            "source_rows": header["height"], "viewer_rows": screen.lines,
            "final_cursor_row": screen.cursor.y - (panels[-1] if panels else 0), "source_final_screen": source_rows,
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
               "meta": meta, "hero": {"returncode": 0, "seconds": meta.get("wall_seconds", replay["seconds"]),
                                      "columns": replay["columns"], "screen": replay["frames"][-1]["screen"],
                                      "rows": max(len(frame["screen"]) for frame in replay["frames"]),
                                      "source_rows": replay["source_rows"], "viewer_rows": replay["viewer_rows"],
                                      "replay": "/hero/train.json"}}
    (DATA / "capture.json").write_text(json.dumps(capture, indent="\t") + "\n")
    (DATA / "hero.py").write_text(options.script.read_text())
    print(f"prepared {len(replay['frames'])} frames from {replay['seconds']:.1f} s of recording")  # noqa: T201


if __name__ == "__main__":
    main()
