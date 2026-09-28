"""Pack a text-to-image sampling trajectory for the landing page's replay section.

    python scripts/replay-atlas.py SRC_DIR NAME --provenance FILE [--frames 32]

SRC_DIR holds x0_NNN.png, the model's predicted image at each sampler step
(000 is the first step, from pure noise), and trajectory.json with the prompt,
the seed, the sampler, the step count and the guidance scale. FILE is a JSON
record of the model: its name, parameter count, and the Dew commit and device
that sampled it. This writes public/hero/replay/NAME/: x0.webp, `frames` of
those steps in a grid (8 a row, always the first and the last), final.webp,
the last step, and meta.json, which the landing page's replay section reads.

A model trained on LAION draws the flat white or black bands its letterboxed
training images carry. Every frame is cropped to the box inside the last
frame's bands, so the replay shows the picture the model drew at its own
aspect; meta.json records the box.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = 8


def arg(name: str, default: str | None = None) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    if default is None:
        raise SystemExit(f"{name} is required")
    return default


def band(lines: np.ndarray) -> int:
    """How many leading lines are letterbox padding: flat near-white or near-black, then
    at most two lines of the blurred edge the VAE decodes between the band and the picture."""
    count = 0
    for line in lines:
        mean, spread = float(line.mean()), float(line.std())
        if spread >= 8.0 or 30.0 <= mean <= 225.0:
            break
        count += 1
    if count:
        white = float(lines[0].mean()) > 225.0
        for line in lines[count:count + 2]:
            mean = float(line.mean())
            if (mean <= 200.0) if white else (mean >= 55.0):
                break
            count += 1
    return count


def content_box(image: Image.Image) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the image inside its letterbox bands."""
    pixels = np.asarray(image.convert("RGB"), np.float32)
    columns = pixels.transpose(1, 0, 2)
    height, width = pixels.shape[:2]
    top, bottom = band(pixels), band(pixels[::-1])
    left, right = band(columns), band(columns[::-1])
    if top + bottom >= height or left + right >= width:
        raise SystemExit("the last frame is one flat colour")
    return left, top, width - right, height - bottom


def main() -> None:
    src, name = Path(sys.argv[1]), sys.argv[2]
    frames = int(arg("--frames", "32"))
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
    final = Image.open(src / f"x0_{last:03d}.png").convert("RGB")
    box = content_box(final)
    width, height = box[2] - box[0], box[3] - box[1]

    out = ROOT / "public" / "hero" / "replay" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(keep) / COLUMNS)
    atlas = Image.new("RGB", (COLUMNS * width, rows * height))
    for index, step in enumerate(keep):
        frame = Image.open(src / f"x0_{step:03d}.png").convert("RGB").crop(box)
        atlas.paste(frame, ((index % COLUMNS) * width, (index // COLUMNS) * height))
    atlas.save(out / "x0.webp", quality=80, method=6)
    final.crop(box).save(out / "final.webp", quality=86, method=6)
    meta = {
        "prompt": trajectory["prompt"],
        "seed": trajectory["seed"],
        "sampler": trajectory["sampler"],
        "steps": trajectory["steps"],
        "cfg": trajectory.get("cfg", trajectory.get("cfg_scale")),
        "guidance_interval": trajectory.get("guidance_interval", "every step"),
        "negative_prompt": trajectory.get("negative_prompt"),
        "model": {key: provenance[key] for key in ("model", "parameters", "dew_commit", "device")},
        "frames": len(keep),
        "kept_steps": keep,
        "columns": COLUMNS,
        "crop": list(box),
        "width": width,
        "height": height,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(out, len(keep), "frames;", f"crop {box};", {path.name: path.stat().st_size for path in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
