import sys; sys.path[:0]=["src","tests","."]
import jax
import jax.numpy as j
import numpy as np
from dew.algebra import scan, linear_scan
from dew.nn.kernels import recurrence_chunk as rc
for line in str(scan._form(linear_scan.Extents(16,8,8,1),True)).splitlines():
 print("K FORM",line,flush=True)
keys=jax.random.split(jax.random.key(1861),8)
scale=2.**62
q,k,v=(jax.random.normal(key,(1,37,2,8))*scale for key in keys[:3])
g=-jax.nn.softplus(jax.random.normal(keys[3],(1,37,2,1)))
b=jax.nn.sigmoid(jax.random.normal(keys[4],(1,37,2)))
c=jax.random.normal(keys[7],q.shape)
orig=scan._paired
def paired(left,right,between,issue=None,role="scores",strict=False):
    result=orig(left,right,between,issue,role,strict=strict)
    if role=="scores":
      jax.debug.print("K pair leftmax {l} rightmax {r} finite {f}",l=j.max(j.abs(left)),r=j.max(j.abs(right)),f=j.sum(j.isfinite(result)))
    return result
scan._paired=paired
def rule(q,k,v,g,b):
    return rc.xla_mainloop(rc.Tiles(16),q,k,v,g[...,0],b,None,None,scale=8**-.5,delta=True,dtype=j.float32)[0]
out=jax.jit(rule)(q,k,v,g,b)
grad=jax.jit(jax.grad(lambda q:j.sum(rule(q,k,v,g,b)*c)))(q)
for h in (0,1):
 print("K OUT",h,np.sum(np.isfinite(out[0,:,h]),axis=1).tolist(),flush=True)
 print("K DQ",h,np.sum(np.isfinite(grad[0,:,h]),axis=1).tolist(),flush=True)
