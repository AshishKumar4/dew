import os,sys,json,contextlib
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),str(Path.cwd()/'tests'),str(Path.cwd()/'tools')]
import numpy as np
from huggingface_hub import hf_hub_download
pin=json.loads(Path('tests/fixtures/llama_cpp/recording.npz.fetched.json').read_text())
f=hf_hub_download(pin['repo'],pin['path'],repo_type='dataset',revision=pin['revision'])
st=dict(np.load(f))
import jax
import reference_runs,reference_error
reference_runs.recording=lambda _:st
os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
mode=sys.argv[1]
if mode=='wide':
 reference_error.summed_as_run=lambda:jax.default_matmul_precision('highest')
from test_llama_cpp_export import test_llama_cpp_reads_the_decoder_dew_exports_at_dews_logits
try:
 test_llama_cpp_reads_the_decoder_dew_exports_at_dews_logits()
 print('K gate',mode,'PASS',flush=True)
except Exception as e:
 print('K gate',mode,type(e).__name__,str(e).replace('\n',' / '),flush=True)
