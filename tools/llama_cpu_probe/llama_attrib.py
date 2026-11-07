import os,sys,json,time
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),str(Path.cwd()/'tests'),str(Path.cwd()/'tools')]
import numpy as np
from huggingface_hub import hf_hub_download
pin=json.loads(Path('tests/fixtures/llama_cpp/recording.npz.fetched.json').read_text())
f=hf_hub_download(pin['repo'],pin['path'],repo_type='dataset',revision=pin['revision'])
st=dict(np.load(f))
import jax,jax.numpy as jnp
from test_llama_cpp_export import draw,exported
from dew.nn.precision import FP32_SUMS
os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
model,vs=exported()
print('K ENV',os.environ.get('XLA_FLAGS'),jax.__version__,jax.devices(),flush=True)
print('K keys',[(k,np.shape(v)) for k,v in st.items()],flush=True)
print('K model',model,flush=True)
mode=sys.argv[1]
if mode=='inspect':
 with jax.default_matmul_precision(FP32_SUMS):
  out,inter=model.apply(vs,jnp.asarray(draw(0)['ids']),capture_intermediates=True,mutable=['intermediates'])
 def walk(t,p=''):
  if isinstance(t,dict):
   for k,v in t.items():walk(v,p+'/'+k)
  else:print('K leaf',p,jax.tree.map(lambda x:(x.shape,str(x.dtype)),t),flush=True)
 walk(vs);walk(inter)
else:
 if mode in ('plain','highest_plain'):os.environ.pop('DEW_WIDE_CPU_ARITHMETIC',None)
 if mode=='float64':
  jax.config.update('jax_enable_x64',True)
  model=model.clone(dtype=jnp.float64)
  vs=jax.tree.map(lambda x:np.asarray(x,dtype=np.float64),vs)
 precision=FP32_SUMS if mode in ('fp32','plain') else 'highest'
 errs=[];refs=[]
 with jax.default_matmul_precision(precision):
  for seed in range(40):
   truth=st['llama-cpp.truth.logits'][seed]
   ref=st['llama-cpp.reference.logits'][seed]
   result=np.asarray(model.apply(vs,jnp.asarray(draw(seed)['ids'])))
   errs.append(np.mean((result.astype('float64')-truth)**2));refs.append(np.mean((ref.astype('float64')-truth)**2))
 print('K result',mode,'rms',float(np.sqrt(np.mean(errs))),'ref',float(np.sqrt(np.mean(refs))),'ratio',float(np.sqrt(np.mean(errs)/np.mean(refs))),'perseed',np.sqrt(np.array(errs)/np.array(refs)).tolist(),flush=True)
