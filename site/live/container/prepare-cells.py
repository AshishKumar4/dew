"""Fetch what the page's pool cells read, by running each once, capped, online.

A training context (gateway_manager.py) runs offline over a private overlay of
/opt/train, so the models, datasets and tokenized corpora each pool cell reads
must be there already. Running the cells puts there exactly the files each
reads. setup-managed.sh makes /opt/train read-only afterwards.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/opt/live/cells")
import cells

programs = {name: cells.cell(name) for name in cells.CELLS["pool"]}
programs["hero"] = cells.training_example(Path("/opt/live/cells/hero.py").read_text())
env = {**os.environ, "HF_HOME": "/opt/train/hf", "XDG_CACHE_HOME": "/opt/train/cache",
       "PYTHONPATH": "/opt/live", "JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
# Compiled programs stay out of the model process's cache.
for name in ("JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
             "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "XLA_FLAGS"):
    env.pop(name, None)
for name, code in programs.items():
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as work:
        subprocess.run([sys.executable, "-c", "import live_training\nlive_training.install()\n" + code],
                       cwd=work, env=env, check=True, timeout=900, stdout=subprocess.DEVNULL)
    print(f"prepared {name} in {time.monotonic() - started:.0f} s", flush=True)
