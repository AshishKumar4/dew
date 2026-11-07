import os,sys,json,time,contextlib
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'tests')]
from lane_environment import configure_lane
configure_lane(os.environ)
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.numpy import load_file
from test_block_diffusion import WORKFLOW,transformers_model,transformers_canvas,dew_canvas
from interop_support import transformers_arms
from reference_error import assert_as_exact,as_given,seeds_for
from dew.interop import diffusion_gemma as adapter
from tools.torch_precision import replaced,dtype_widened
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
kind=sys.argv[1]; seeds=int(sys.argv[2]); fix=int(sys.argv[3])
source=transformers_model(WORKFLOW)
narrow=DiffusionGemmaForBlockDiffusion.from_pretrained(str(WORKFLOW),dtype=torch.bfloat16,local_files_only=True,experts_implementation='eager').eval()
narrow.set_attn_implementation('eager')
reference,truth=transformers_arms(narrow,transformers_canvas,source=source)
original=truth
@as_given
def corrected(*inputs):
 with replaced((torch.Tensor,'softmax',dtype_widened(torch.Tensor.softmax))): return original(*inputs)
if fix: truth=corrected
config=json.loads((WORKFLOW/'config.json').read_text())
model=adapter.build(config,dtype='bfloat16',attention_impl='xla')
variables,_=adapter.translate_weights(load_file(str(WORKFLOW/'model.safetensors')),config)
variables=jax.tree.map(jnp.asarray,variables)
ours=jax.jit(lambda *inputs:dew_canvas(model,variables)(*inputs))
def draw(seed):
 rng=np.random.default_rng([198,seed])
 prompt=rng.integers(2,60,(2,5)); image=prompt.copy(); image[:,2]=60
 canvas=rng.integers(2,60,(2,4)); previous=rng.standard_normal((2,4,64)).astype(np.float32); pixels=rng.uniform(0,1,(2,1,3,4,4)).astype(np.float32)
 return (image,canvas,pixels) if kind=='image' else ((prompt,canvas,previous) if kind=='conditioned' else (prompt,canvas))
start=time.monotonic()
try:
 if seeds==0:
  seeds=seeds_for(reference,draw,'dgemma '+kind,truth_fn=truth,dew_fn=as_given(ours))
  print('K sized',kind,'fixed',fix,'seeds',seeds,flush=True)
 assert_as_exact(as_given(ours),reference,draw,'dgemma '+kind,seeds=seeds,truth_fn=truth)
 print('K gate',kind,'seeds',seeds,'fixed',fix,'PASS','seconds',time.monotonic()-start)
except Exception as e:
 print('K gate',kind,'seeds',seeds,'fixed',fix,type(e).__name__,str(e).replace('\n',' ')[:5000],'seconds',time.monotonic()-start)
