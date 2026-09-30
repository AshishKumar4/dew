"""Run the landing page's setup cell once, while the image builds.

The Dockerfile has put the model at /opt/models/text-to-image. Running
sampler_setup.py, which deploy.mjs copies from site/src/data, leaves the
tokenizer and VAE configs it reads in HF_HOME, so the running container, which
has no network, finds them. Nothing compiled here is kept: JAX keys a CPU
program by the host's CPU, and the build machine is never a Cloudflare host.
"""

import time
from pathlib import Path

started = time.perf_counter()
exec((Path(__file__).parent / "sampler_setup.py").read_text(), {})
print(f"warm: the setup cell ran in {time.perf_counter() - started:.1f} s", flush=True)
