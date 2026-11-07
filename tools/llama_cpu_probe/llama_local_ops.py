import os,sys,json,collections
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),str(Path.cwd()/'tests'),str(Path.cwd()/'tools')]
import numpy as np
import jax,jax.numpy as jnp
import torch
torch.set_num_threads(1)
from flax import linen as nn
from test_llama_cpp_export import draw,exported
from dew.nn.precision import FP32_SUMS
from dew.nn.dense import Dense,DenseGeneral
model,vs=exported()
os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
stats=collections.defaultdict(lambda:np.zeros(5))
shapes={}
seed=0
def record(key,x,y,k):
 global seed
 x=np.asarray(x);y=np.asarray(y);k=np.asarray(k.value if hasattr(k,'value') else k)
 if k.ndim!=2:
  print('K unsupported',key,x.shape,y.shape,k.shape,flush=True);return
 truth=x.astype('float64')@k.astype('float64')
 reference=(torch.from_numpy(x.copy()).float()@torch.from_numpy(k.copy()).float()).numpy()
 numpy_ref=x.astype('float32')@k.astype('float32')
 stats[key]+=np.array([np.sum((y-truth)**2),np.sum((reference-truth)**2),np.sum((numpy_ref-truth)**2),np.sum(truth**2),truth.size])
 shapes[key]=str((x.shape,k.shape))
def interceptor(next_fun,args,kwargs,context):
 result=next_fun(*args,**kwargs)
 if context.method_name=='__call__' and isinstance(context.module,(Dense,DenseGeneral)):
  k=context.module.get_variable('params','kernel')
  if k is not None:record(context.module.scope.path_text,args[0],result,k)
 return result
with jax.default_matmul_precision(FP32_SUMS),nn.intercept_methods(interceptor):
 for seed in range(40):
  result=np.asarray(model.apply(vs,jnp.asarray(draw(seed)['ids'])))
for key,v in stats.items():
 print('K local',key,'shape',shapes[key],'xla_rms',np.sqrt(v[0]/v[4]),'torch_rms',np.sqrt(v[1]/v[4]),'numpy_rms',np.sqrt(v[2]/v[4]),'ratio_torch',np.sqrt(v[0]/v[1]),'relative',np.sqrt(v[0]/v[3]),flush=True)
