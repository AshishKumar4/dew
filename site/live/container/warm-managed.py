"""Warm the pinned models before capturing the managed filesystem snapshot."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download

from dew.inference import Server
from dew.interop import PretrainedDecoder
from dew.sampling import CFG, DPMSolverMultistep, Sampling, TextToImage

started = time.perf_counter()
root = Path('/opt/live')
repo, revision = (root / 'text-to-image').read_text().strip().split('@')
snapshot_download(repo_id=repo, revision=revision)
for line in (root / 'text-models').read_text().split():
    name, version = line.split('@')
    snapshot_download(repo_id=name, revision=version, local_dir=f'/opt/models/{name}',
                      allow_patterns=['*.json', '*.safetensors', '*.txt'])
pipe = TextToImage.from_pretrained(repo, revision=revision)
negative = ('letterbox, white border, black border, frame, text, watermark, collage, blurry, lowres, '
            'low quality, dull colors, washed out, low contrast, grainy')
for steps in (15, 30):
    inputs = pipe.prepare(['the northern lights over a frozen lake at night'], key=3,
                          steps=steps, unconditional=negative)
    pipe(inputs, key=3, steps=steps, solver=DPMSolverMultistep(),
         guidance=CFG(6.0, interval=(0.15, 0.9))).host()
bundle = PretrainedDecoder.load('/opt/models/HuggingFaceTB/SmolLM2-135M-Instruct',
                                dtype=jnp.float32, max_seq_len=256)
server = Server.from_task(bundle.text_generation(sampling=Sampling(temperature=0)), slots=8, capacity=256)
ids = bundle.processor('The capital of France is').tokens[0]
tickets = [server.submit(ids, 24, key=0) for _ in range(8)]
server.run()
for ticket in tickets:
    if not ticket.result().text[0]:
        raise RuntimeError('the pinned text task returned no text')
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
