"""Run the landing page's sampling cells once, while the image builds.

The Dockerfile has put the model at /opt/models/text-to-image. Running
sampler_setup.py and then sampler.py, the cells the page sends (deploy.mjs
copies them from site/src/data), leaves the tokenizer and VAE configs in
HF_HOME and the programs the default cell compiles in
JAX_COMPILATION_CACHE_DIR, so a visitor's first run reads them from disk.
"""

import time
from pathlib import Path

HERE = Path(__file__).parent
scope: dict = {}
for cell in ("sampler_setup.py", "sampler.py"):
    started = time.perf_counter()
    exec((HERE / cell).read_text(), scope)
    print(f"warm: {cell} ran in {time.perf_counter() - started:.1f} s", flush=True)
