import os,sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),str(Path.cwd()/'tests'),str(Path.cwd()/'tools')]
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from test_llama_cpp_export import draw,exported
from dew.nn.precision import FP32_SUMS
from dew.nn.dense import Dense,DenseGeneral
from huggingface_hub import hf_hub_download
pin=json.loads(Path('tests/fixtures/llama_cpp/recording.npz.fetched.json').read_text())
st=dict(np.load(hf_hub_download(pin['repo'],pin['path'],repo_type='dataset',revision=pin['revision'])))
model,vs=exported()
os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
mode=sys.argv[1]
paths={'qk':('q_proj','k_proj'),'v':('v_proj',),'out':('o_proj',),'gate':('gate_proj',),'up':('up_proj',),'down':('down_proj',),'mlp':('gate_proj','up_proj','down_proj'),'head':('lm_head',)}
selected=[]
def interceptor(next_fun,args,kwargs,context):
 y=next_fun(*args,**kwargs)
 if context.method_name=='__call__' and isinstance(context.module,(Dense,DenseGeneral)):
  path=context.module.scope.path_text
  chosen=(path.endswith(tuple('/'+p for p in paths[mode])) if mode in paths else path.startswith('/'+mode+'/'))
  if chosen:
   if path not in selected:selected.append(path)
   x=args[0];k=context.module.get_variable('params','kernel');k=k.value if hasattr(k,'value') else k
   with jax.enable_x64():return jnp.matmul(x.astype(jnp.float64),jnp.asarray(k,jnp.float64),precision='highest').astype(y.dtype)
 return y
errs=[];refs=[]
with jax.default_matmul_precision(FP32_SUMS),nn.intercept_methods(interceptor):
 for seed in range(40):
  truth=st['llama-cpp.truth.logits'][seed];ref=st['llama-cpp.reference.logits'][seed]
  y=np.asarray(model.apply(vs,jnp.asarray(draw(seed)['ids'])))
  errs.append(np.mean((y.astype('float64')-truth)**2));refs.append(np.mean((ref.astype('float64')-truth)**2))
print('K result',mode,'rms',np.sqrt(np.mean(errs)),'reference',np.sqrt(np.mean(refs)),'ratio',np.sqrt(np.mean(errs)/np.mean(refs)),'selected',selected,flush=True)
