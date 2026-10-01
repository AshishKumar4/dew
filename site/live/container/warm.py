"""Run the landing page's setup cell once, while the image builds.

The Dockerfile has put the text-to-image model in the hub cache. Running
sampler_setup.py, which deploy.mjs copies from site/src/data, leaves the
tokenizer and VAE configs it reads in HF_HOME, so the running container, which
has no network, finds them. The setup cell names the model's repo without a
revision; with the network, the hub resolves `main` again, so this checks that
`main` still names the pinned revision. Nothing compiled here is kept: JAX keys
a CPU program by the host's CPU, and the build machine is never a Cloudflare
host.
"""

import time
from pathlib import Path

from huggingface_hub import snapshot_download

started = time.perf_counter()
exec((Path(__file__).parent / "sampler_setup.py").read_text(), {})
print(f"warm: the setup cell ran in {time.perf_counter() - started:.1f} s", flush=True)

repo, revision = (Path(__file__).parent / "text-to-image").read_text().strip().split("@")
if Path(snapshot_download(repo_id=repo, local_files_only=True)).name != revision:
    raise SystemExit(f"{repo}'s main has moved past {revision}; pin the new revision in text-to-image")
