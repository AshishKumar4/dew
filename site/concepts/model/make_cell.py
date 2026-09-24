"""Write train_cell.py: the training script with the mask embedded, and the step count set."""
import base64, sys
from pathlib import Path
steps = sys.argv[1] if len(sys.argv) > 1 else "30000"
source = Path("train_particles.py").read_text()
mask = base64.b64encode(Path("dew-mask.png").read_bytes()).decode()
cell = f'import os\nos.environ["PARTICLE_STEPS"] = "{steps}"\n' + source.replace("__MASK_PNG__", mask)
Path(f"train_cell_{steps}.py").write_text(cell)
print(f"train_cell_{steps}.py", len(cell))
