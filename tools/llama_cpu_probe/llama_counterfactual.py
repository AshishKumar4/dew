import os,sys,json,functools,collections
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
mode=sys.argv[1]
from jax._src.lax.lax import dot_general_p
from jax._src import source_info_util
raw=dot_general_p.bind
counts=collections.Counter()
def patched(*args,**params):
 a,b=args
 if a.dtype==jnp.float32 and b.dtype==jnp.float32:
  dims=params['dimension_numbers']; attention=bool(dims[1][0])
  chosen=(mode in ('dots','audit') or (mode=='dense' and not attention) or (mode=='attention' and attention))
  if chosen and mode!='audit':
   label=(str(a.shape),str(b.shape),str(dims))
   counts[label]+=1
   with jax.enable_x64():
    return raw(a.astype(jnp.float64),b.astype(jnp.float64),**{**params,'preferred_element_type':None,'precision':(jax.lax.Precision.HIGHEST,jax.lax.Precision.HIGHEST)}).astype(jnp.float32)
 return raw(*args,**params)
dot_general_p.bind=patched
if mode=='norm':
 import dew.nn.norms as norms
 raw_norm=norms._xla_rms_normalized
 def norm(x,scale,epsilon,dtype,scale_offset,scale_after_cast,fp32_statistics,activation=None):
  with jax.enable_x64():
   return raw_norm(x.astype(jnp.float64),None if scale is None else scale.astype(jnp.float64),epsilon,jnp.float64,scale_offset,False,True,activation).astype(dtype)
 norms._xla_rms_normalized=norm
if mode=='rotary':
 import dew.nn.mixers.attention as blocks
 raw_rotate=blocks.rotate
 def rotate(x,cos,sin,**kwargs):
  with jax.enable_x64():
   return raw_rotate(x.astype(jnp.float64),cos.astype(jnp.float64),sin.astype(jnp.float64),**kwargs).astype(x.dtype)
 blocks.rotate=rotate
if mode=='softmax':
 import jax._src.nn.functions as nf
 raw_softmax=nf.softmax
 def softmax(x,*args,**kwargs):
  with jax.enable_x64():return raw_softmax(x.astype(jnp.float64),*args,**kwargs).astype(x.dtype)
 nf.softmax=softmax
 jax.nn.softmax=softmax
jax.clear_caches()
errs=[];refs=[]
with jax.default_matmul_precision(FP32_SUMS):
 for seed in range(40):
  truth=st['llama-cpp.truth.logits'][seed];ref=st['llama-cpp.reference.logits'][seed]
  result=np.asarray(model.apply(vs,jnp.asarray(draw(seed)['ids'])))
  errs.append(np.mean((result.astype('float64')-truth)**2));refs.append(np.mean((ref.astype('float64')-truth)**2))
print('K result',mode,'rms',float(np.sqrt(np.mean(errs))),'ref',float(np.sqrt(np.mean(refs))),'ratio',float(np.sqrt(np.mean(errs)/np.mean(refs))),flush=True)
for k,v in counts.items():print('K count',mode,k,v,flush=True)
