import os,sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),str(Path.cwd()/'tests'),str(Path.cwd()/'tools')]
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from test_llama_cpp_export import draw,exported
from dew.nn.precision import FP32_SUMS
from dew.nn.dense import Dense,DenseGeneral
model,vs=exported()
os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
def interceptor(next_fun,args,kwargs,context):
 y=next_fun(*args,**kwargs)
 if context.method_name=='__call__' and isinstance(context.module,(Dense,DenseGeneral)):
  x=np.asarray(args[0]);k=context.module.get_variable('params','kernel');k=np.asarray(k.value if hasattr(k,'value') else k)
  yn=np.asarray(y);acc=np.zeros(y.shape,np.float32);nonfused=acc.copy()
  for i in range(k.shape[0]):
   acc=(acc.astype('float64')+x[...,i:i+1].astype('float64')*k[i].astype('float64')).astype('float32')
   nonfused=nonfused+x[...,i:i+1]*k[i]
  products=x.astype('float64')[...,None]*k.astype('float64')
  terms=products.astype('float32')
  while terms.shape[-2]>1:
   terms=terms[...,::2,:]+terms[...,1::2,:]
  balanced=terms[...,0,:]
  truth=x.astype('float64')@k.astype('float64')
  print('K order',context.module.scope.path_text,'sequential_fma_equal',np.mean(yn==acc),'sequential_nonfused_equal',np.mean(yn==nonfused),'balanced_equal',np.mean(yn==balanced),'rms_xla',np.sqrt(np.mean((yn-truth)**2)),'rms_fma',np.sqrt(np.mean((acc-truth)**2)),'rms_balanced',np.sqrt(np.mean((balanced-truth)**2)),flush=True)
 return y
with jax.default_matmul_precision(FP32_SUMS),nn.intercept_methods(interceptor):
 model.apply(vs,jnp.asarray(draw(0)['ids']))
