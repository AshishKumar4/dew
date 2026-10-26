# A context's text request and what it used, one line of JSON, run by benchmark_gateway.py's measured().
from dew.interop import PretrainedDecoder
from dew.sampling import Sampling

model = PretrainedDecoder.load('HuggingFaceTB/SmolLM2-135M-Instruct', dtype='float32', max_seq_len=256)
assert model.text_generation(sampling=Sampling(temperature=0))('The capital of France is', 24, key=0).text[0]
import json
import os
from pathlib import Path

pss = next(line for line in Path('/proc/self/smaps_rollup').read_text().splitlines() if line.startswith('Pss:'))
print(json.dumps({'uid': os.getuid(), 'fds': len(os.listdir('/proc/self/fd')), 'pss_bytes': int(pss.split()[1]) * 1024}))
