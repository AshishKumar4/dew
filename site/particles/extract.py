"""Pull the manifest, weights and sample picture out of a training log.

    python particles/extract.py NAME LOG

writes public/hero/particles/NAME.json and NAME.bin, which the landing page's
hero loads, and particles/samples/NAME.png, the model's samples against the mask.
"""
import base64
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
name, log = sys.argv[1], Path(sys.argv[2]).read_text(errors="replace")
out = here.parent / "public" / "hero" / "particles"
out.mkdir(parents=True, exist_ok=True)
(here / "samples").mkdir(exist_ok=True)


def after(marker: str) -> str:
    line = next(line for line in log.splitlines() if line.startswith(marker))
    return line[len(marker) :].strip()


manifest = json.loads(after("MANIFEST_JSON"))
weights = base64.b64decode(after("WEIGHTS_B64"))
(out / f"{name}.json").write_text(json.dumps(manifest, indent=1) + "\n")
(out / f"{name}.bin").write_bytes(weights)
(here / "samples" / f"{name}.png").write_bytes(base64.b64decode(after("SAMPLES_PNG_B64")))
print(name, manifest["parameters"], "parameters;", len(weights) // 4, "floats;", manifest["samples_inside"], "inside")
