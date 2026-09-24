"""Write a Colab cell: train_particles.py with one mask embedded and the step count set.

    python particles/make_cell.py NAME [STEPS]

reads particles/masks/NAME.png and writes particles/cells/NAME-STEPS.py.
"""
import base64
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
name = sys.argv[1]
steps = sys.argv[2] if len(sys.argv) > 2 else "30000"
source = (here / "train_particles.py").read_text()
mask = base64.b64encode((here / "masks" / f"{name}.png").read_bytes()).decode()
cell = f'import os\nos.environ["PARTICLE_STEPS"] = "{steps}"\n' + source.replace("__MASK_PNG__", mask)
out = here / "cells" / f"{name}-{steps}.py"
out.parent.mkdir(exist_ok=True)
out.write_text(cell)
print(out, len(cell))
