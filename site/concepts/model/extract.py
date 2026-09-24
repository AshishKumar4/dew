"""Pull the manifest, weights and sample picture out of a training log into site/concepts/public/b/."""
import base64, json, sys
from pathlib import Path
log = Path(sys.argv[1]).read_text(errors="replace")
out = Path(__file__).resolve().parents[1] / "public/b"
out.mkdir(parents=True, exist_ok=True)
def after(marker):
    line = next(l for l in log.splitlines() if l.startswith(marker))
    return line[len(marker):].strip()
manifest = json.loads(after("MANIFEST_JSON"))
weights = base64.b64decode(after("WEIGHTS_B64"))
(out / "model.json").write_text(json.dumps(manifest, indent=1) + "\n")
(out / "model.bin").write_bytes(weights)
Path("samples.png").write_bytes(base64.b64decode(after("SAMPLES_PNG_B64")))
print(manifest["parameters"], len(weights) // 4, "floats;", manifest["samples_inside"], "inside")
