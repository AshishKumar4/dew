import os, sys
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd()/'tests')]
from lane_environment import configure_lane
configure_lane(os.environ)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import torch, inspect
from safetensors.numpy import load_file
from test_block_diffusion import WORKFLOW, transformers_model, transformers_canvas, dew_canvas
from dew.interop import diffusion_gemma as adapter
from tools.torch_precision import float64, float64_twin
import json
from flax.traverse_util import flatten_dict
source=transformers_model(WORKFLOW)
wide=float64_twin(source)
config=json.loads((WORKFLOW/'config.json').read_text())
model=adapter.build(config,dtype='float64',attention_impl='xla')
variables,_=adapter.translate_weights(load_file(str(WORKFLOW/'model.safetensors')),config,param_dtype='float64')
with np.load(WORKFLOW/'reference.npz') as f: data=dict(f)
mode=sys.argv[1]
if mode=='source':
 import transformers.models.diffusion_gemma.modeling_diffusion_gemma as mod
 for n in ['DiffusionGemmaTextRouter','DiffusionGemmaForBlockDiffusion','DiffusionGemmaTextScaledWordEmbedding','DiffusionGemmaSelfConditioning']:
  for i,line in enumerate(inspect.getsource(getattr(mod,n)).splitlines()): print('K source',n,i,line)
 print('K modules',[(n,type(m).__name__) for n,m in wide.named_modules()])
else:
 captures={}
 for n,m in wide.named_modules():
  def hook(m,a,o,n=n):
   if isinstance(o,torch.Tensor): captures.setdefault(n,[]).append(o.detach().numpy())
  m.register_forward_hook(hook)
 inputs=(data['prompt'],data['canvas'])
 if mode=='conditioned': inputs+= (data['previous'],)
 with torch.no_grad(),float64():
  out=transformers_canvas(wide,*[torch.from_numpy(np.array(x)).long() if x.dtype.kind in 'iu' else torch.from_numpy(x).double() for x in inputs])['logits'].numpy()
 ours=dew_canvas(model,variables)(*inputs)['logits']
 print('K final',mode,'dtype',out.dtype,ours.dtype,'max',np.max(np.abs(ours-out)),'rms',np.sqrt(np.mean((ours-out)**2)))
 # intercept every flax module call without adding mutable state
 jc={}
 from flax import linen as nn
 def interceptor(next_fun,args,kwargs,ctx):
  o=next_fun(*args,**kwargs)
  if isinstance(o,jax.Array): jc.setdefault('/'.join(ctx.module.scope.path)+':'+ctx.method_name,[]).append(np.asarray(o))
  return o
 with nn.intercept_methods(interceptor): dew_canvas(model,variables)(*inputs)
 for n,values in jc.items():
  for i,v in enumerate(values):
   candidates=[]
   for tn,tvs in captures.items():
    for ti,t in enumerate(tvs):
     if t.shape==v.shape: candidates.append((float(np.max(np.abs(t-v))),tn,ti))
   if candidates: print('K match',n,i,str(v.shape),v.dtype,sorted(candidates)[:2])
 print('K embedscale',[(n,b.item()) for n,b in wide.named_buffers() if 'embed_scale' in n])
