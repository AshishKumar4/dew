import os, sys, contextlib, json
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd()/'tests')]
from lane_environment import configure_lane
configure_lane(os.environ)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.numpy import load_file
from test_block_diffusion import WORKFLOW, transformers_model, transformers_canvas, dew_canvas
from dew.interop import diffusion_gemma as adapter
from tools.torch_precision import float64, float64_twin, replaced, dtype_widened
from flax import linen as nn
source=transformers_model(WORKFLOW)
wide=float64_twin(source)
config=json.loads((WORKFLOW/'config.json').read_text())
model=adapter.build(config,dtype='float64',attention_impl='xla')
variables,_=adapter.translate_weights(load_file(str(WORKFLOW/'model.safetensors')),config,param_dtype='float64')
variables=jax.tree.map(jnp.asarray,variables)
kind=sys.argv[1]
patch=int(sys.argv[2])
seed=int(sys.argv[3])
rng=np.random.default_rng([198,seed])
prompt=rng.integers(2,60,(2,5)); image=prompt.copy(); image[:,2]=60
canvas=rng.integers(2,60,(2,4)); previous=rng.standard_normal((2,4,64)).astype(np.float32); pixels=rng.uniform(0,1,(2,1,3,4,4)).astype(np.float32)
inputs=(image if kind=='image' else prompt,canvas)
if kind=='conditioned': inputs+=(previous.astype(np.float64),)
if kind=='image': inputs+=(pixels.astype(np.float64),)
convert=lambda x: torch.from_numpy(x.astype(np.int64) if x.dtype.kind in 'iu' else x)
captures={}; before={}
names=['model.encoder.language_model.layers.0','model.encoder.language_model.layers.1','model.decoder.self_conditioning.pre_norm','model.decoder.self_conditioning','model.decoder.layers.0','model.decoder.layers.1','model.decoder.norm']
for n,m in wide.named_modules():
 if n not in names: continue
 def hook(m,a,o,n=n):
  if isinstance(o,torch.Tensor): captures[n]=o.detach().numpy(); before[n]=a[0].detach().numpy()
 m.register_forward_hook(hook)
ctx=replaced((torch.Tensor,'softmax',dtype_widened(torch.Tensor.softmax))) if patch else contextlib.nullcontext()
with torch.no_grad(),float64(),ctx:
 out=transformers_canvas(wide,*map(convert,inputs))['logits'].numpy()
jc={}; ji={}
def interceptor(next_fun,args,kwargs,ctx):
 o=next_fun(*args,**kwargs)
 n='/'.join(ctx.module.scope.path)+':'+ctx.method_name
 if isinstance(o,jax.Array): jc[n]=np.asarray(o)
 if args and isinstance(args[0],jax.Array): ji[n]=np.asarray(args[0])
 return o
with nn.intercept_methods(interceptor): ours=dew_canvas(model,variables)(*inputs)['logits']
compiled=jax.jit(lambda *i: dew_canvas(model,variables)(*i)['logits'])(*inputs)
for jn,tn in [('text/layers_0:__call__','model.decoder.layers.0'),('text/layers_1:__call__','model.decoder.layers.1'),('text/norm:__call__','model.decoder.norm'),('self_conditioning/pre_norm:__call__','model.decoder.self_conditioning.pre_norm'),('self_conditioning:__call__','model.decoder.self_conditioning')]:
 if jn in jc and tn in captures: print('K layer',kind,patch,seed,jn,'input',np.max(np.abs(ji[jn]-before[tn])),'output',np.max(np.abs(jc[jn]-captures[tn])))
if kind=='conditioned':
 probs=previous.astype(np.float64)
 with float64():
  old=torch.tensor(probs).softmax(-1,dtype=torch.float32)
  new=torch.tensor(probs).softmax(-1,dtype=torch.float64)
 print('K softmax',kind,patch,seed,old.dtype,new.dtype,'delta',torch.max(torch.abs(old-new)).item())
print('K final',kind,'patch',patch,'seed',seed,'raw',np.max(np.abs(ours-out)),'jit',np.max(np.abs(compiled-out)),'original_cast',np.max(np.abs(np.asarray(compiled).astype(np.float32)-out)),'dtype',out.dtype,ours.dtype)
