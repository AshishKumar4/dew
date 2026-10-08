"""Fetch what the page's pool cells read, by running each, online, up to its training.

A training context (gateway_manager.py) runs offline over a private overlay of
/opt/train, so the models, datasets and tokenized corpora each pool cell reads
must be there already. Running a cell puts there exactly the files it reads;
every cell reads them before `Trainer.fit`, so the run stops there (`FETCH`),
and the preparation stays inside the registry's 15-minute alarm. The context
smoke then runs each cell whole (benchmark_gateway.py). setup-managed.sh makes
/opt/train read-only afterwards.
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
# A cell's program, run until its first Trainer.fit.
FETCH = """import sys
from dew.training import Trainer
class Fetched(Exception):
    pass
def fit(*args, **options):
    raise Fetched
Trainer.fit = fit
try:
    exec(compile(sys.stdin.read(), "cell", "exec"), {"__name__": "__main__"})
except Fetched:
    pass
"""
for name, code in programs.items():
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as work:
        subprocess.run([sys.executable, "-c", FETCH], input=code, text=True, cwd=work, env=env, check=True,
                       timeout=600, stdout=subprocess.DEVNULL)
    print(f"prepared {name} in {time.monotonic() - started:.0f} s", flush=True)
