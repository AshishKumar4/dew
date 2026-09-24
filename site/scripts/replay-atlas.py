"""Pack a text-to-image sampling trajectory for the landing page's replay section.

    python scripts/replay-atlas.py SRC_DIR NAME --provenance FILE [--frames 32] [--size 256]

SRC_DIR holds x0_NNN.png, the model's predicted image at each sampler step
(000 is the first step, from pure noise), and trajectory.json with the prompt,
the seed, the sampler, the step count and the guidance scale. FILE is a JSON
record of the model: its name, parameter count, and the Dew commit and device
that sampled it. This writes public/hero/replay/NAME/: x0.webp, `frames` of
those steps in a grid (8 a row, always the first and the last), final.webp,
the last step at full size, and meta.json, which the landing page's replay
section reads.
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = 8


def arg(name: str, default: str | None = None) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    if default is None:
        raise SystemExit(f"{name} is required")
    return default


def main() -> None:
    src, name = Path(sys.argv[1]), sys.argv[2]
    frames, size = int(arg("--frames", "32")), int(arg("--size", "256"))
    trajectory = json.loads((src / "trajectory.json").read_text())
    provenance = json.loads(Path(arg("--provenance")).read_text())
    for record, keys, where in ((trajectory, ("prompt", "seed", "sampler", "steps"), "trajectory.json"),
                                (provenance, ("model", "parameters", "dew_commit", "device"), "--provenance")):
        for key in keys:
            if record.get(key) in (None, ""):
                raise SystemExit(f"{where}: no {key}")
    steps = sorted(int(path.stem.split("_")[1]) for path in src.glob("x0_*.png"))
    if not steps or steps != list(range(len(steps))):
        raise SystemExit(f"{src}: x0_NNN.png must run from 000 without gaps; found {steps[:3]}...{steps[-3:]}")
    last = steps[-1]
    keep = sorted({round(i * last / (frames - 1)) for i in range(frames)})

    out = ROOT / "public" / "hero" / "replay" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(keep) / COLUMNS)
    atlas = Image.new("RGB", (COLUMNS * size, rows * size))
    for index, step in enumerate(keep):
        frame = Image.open(src / f"x0_{step:03d}.png").convert("RGB").resize((size, size), Image.LANCZOS)
        atlas.paste(frame, ((index % COLUMNS) * size, (index // COLUMNS) * size))
    atlas.save(out / "x0.webp", quality=80, method=6)
    Image.open(src / f"x0_{last:03d}.png").convert("RGB").save(out / "final.webp", quality=86, method=6)
    meta = {
        "prompt": trajectory["prompt"],
        "seed": trajectory["seed"],
        "sampler": trajectory["sampler"],
        "steps": trajectory["steps"],
        "cfg": trajectory.get("cfg", trajectory.get("cfg_scale")),
        "model": {key: provenance[key] for key in ("model", "parameters", "dew_commit", "device")},
        "frames": len(keep),
        "kept_steps": keep,
        "columns": COLUMNS,
        "size": size,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(out, len(keep), "frames;", {path.name: path.stat().st_size for path in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
