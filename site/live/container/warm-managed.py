"""Download the pinned models and warm them as the live service serves them, before the managed
filesystem snapshot is captured: `model_service.NativeModels` loads and warms every model it
serves, so the snapshot's compilation cache holds the programs a restored host runs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
from huggingface_hub import snapshot_download
from model_service import NativeModels

started = time.perf_counter()
root = Path('/opt/live')
repo, revision = (root / 'text-to-image').read_text().strip().split('@')
snapshot_download(repo_id=repo, revision=revision)
for line in (root / 'text-models').read_text().split():
    name, version = line.split('@')
    snapshot_download(repo_id=name, revision=version, local_dir=f'/opt/models/{name}',
                      allow_patterns=['*.json', '*.safetensors', '*.txt'])
NativeModels(root)
report = {'os_release': Path('/etc/os-release').read_text(),
          'base_image': 'cloudflare/debian-trixie',
          'apt_packages': subprocess.check_output(
              ['dpkg-query', '-W', '-f=${binary:Package}=${Version}\n'], text=True).splitlines(),
          'pip_freeze': subprocess.check_output(
              [sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
          'dew': (root / 'dew-commit').read_text().strip(), 'jax': jax.__version__,
          'warm_seconds': time.perf_counter() - started, 'xla_flags': os.environ.get('XLA_FLAGS'),
          'cache_bytes': sum(p.stat().st_size for p in Path('/opt/xla').rglob('*') if p.is_file())}
(root / 'prepared.json').write_text(json.dumps(report) + '\n')
print(json.dumps(report), flush=True)
