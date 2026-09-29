"""Compile the landing page's sampling cells once, while the image builds.

The Dockerfile has put the model at /opt/models/text-to-image. This runs what
a kernel runs before its page connects (server.py's PRELOAD: sampler_setup.py,
which deploy.mjs copies from site/src/data, then preload.py), then
the page's sampling cell with each sampler its setup cell imports, at the step
counts the page suggests. That leaves the tokenizer and VAE configs in HF_HOME
and every program those cells compile in JAX_COMPILATION_CACHE_DIR, so a
visitor's run reads them from disk. Another guidance scale, step count or
batch compiles on the visitor's kernel.
"""

import re
import time
from pathlib import Path

HERE = Path(__file__).parent
SAMPLERS = ("DPMSolverMultistep", "EulerAncestral", "Heun")
STEPS = (15, 30)

scope: dict = {}
started = time.perf_counter()
exec("\n".join((HERE / cell).read_text() for cell in ("sampler_setup.py", "preload.py")), scope)
print(f"warm: the setup cell ran in {time.perf_counter() - started:.1f} s", flush=True)
cell = (HERE / "sampler.py").read_text()
for sampler in SAMPLERS:
    for steps in STEPS:
        variant = re.sub(r"steps=\d+", f"steps={steps}", re.sub(r"sampler=\w+\(\)", f"sampler={sampler}()", cell))
        if variant.count(f"sampler={sampler}()") != 1 or variant.count(f"steps={steps}") != 1:
            raise ValueError("sampler.py no longer passes steps= and sampler= the way warm.py rewrites them")
        started = time.perf_counter()
        exec(variant, scope)
        print(f"warm: {sampler}, {steps} steps ran in {time.perf_counter() - started:.1f} s", flush=True)
