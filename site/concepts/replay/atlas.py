"""Pack a trajectory's frames into two atlases for concept C.

    python atlas.py SRC_DIR OUT_DIR [--frames 32] [--size 256]

SRC_DIR holds xt_NNN.png and x0_NNN.png (step 000 is pure noise) and
trajectory.json. OUT_DIR gets xt.webp, x0.webp (frames left to right, top to
bottom, 8 per row) and meta.json with the prompt, the kept steps and sigmas.
"""
import json
import math
import sys
from pathlib import Path

from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 32
size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 256
out.mkdir(parents=True, exist_ok=True)
trajectory = json.loads((src / "trajectory.json").read_text())
available = sorted(int(p.stem.split("_")[1]) for p in src.glob("xt_*.png"))
last = available[-1]
# Keep `frames` steps, always the first (noise) and the last (the sample).
keep = sorted({round(i * last / (frames - 1)) for i in range(frames)})
columns = 8
rows = math.ceil(len(keep) / columns)
for kind in ("xt", "x0"):
    atlas = Image.new("RGB", (columns * size, rows * size))
    for n, step in enumerate(keep):
        frame = Image.open(src / f"{kind}_{step:03d}.png").convert("RGB").resize((size, size), Image.LANCZOS)
        atlas.paste(frame, ((n % columns) * size, (n // columns) * size))
    atlas.save(out / f"{kind}.webp", quality=80 if kind == "x0" else 70, method=6)
# The sample itself, full size, for the page's no-WebGL fallback and first paint.
Image.open(src / f"x0_{last:03d}.png").convert("RGB").save(out / "final.webp", quality=86, method=6)
sigmas = trajectory.get("sigmas") or trajectory.get("sigma")
meta = {
    "prompt": trajectory.get("prompt"),
    "seed": trajectory.get("seed"),
    "sampler": trajectory.get("sampler"),
    "steps": trajectory.get("steps"),
    "cfg": trajectory.get("cfg", trajectory.get("cfg_scale")),
    "frames": len(keep),
    "kept_steps": keep,
    "sigmas": [sigmas[s] for s in keep] if isinstance(sigmas, list) and len(sigmas) > last else None,
    "columns": columns,
    "size": size,
    "placeholder": bool(trajectory.get("placeholder", False)),
}
(out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(out, meta["frames"], "frames;", {k: (out / f"{k}.webp").stat().st_size for k in ("xt", "x0")})
