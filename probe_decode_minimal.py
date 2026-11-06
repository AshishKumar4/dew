import os
import sys
sys.path[:0] = [os.path.join(os.getcwd(), p) for p in ("src", "tests", "tools")]
import jax
import jax.numpy as jnp
import numpy as np
from test_contracts_recurrence import delta_rule_draw

p,d,c = delta_rule_draw(2669,2.0**63,key_heads=1,value_heads=2,width=128,length=1,per_key=False)
q,k,v,g,b,s = p
dq,dk,dv,dg,db,ds = d
q,k,dk = [jnp.repeat(x[:,0],2,axis=1) for x in (q,k,dk)]
b,db = b[:,0,None].swapaxes(1,2),db[:,0,None].swapaxes(1,2)
co = c[0][:,0]
cs = c[1]

def terms(q,k,dk,b,db,co,cs):
    keyed_direction = dk*b+k*db
    updated_cotangent = q[..., :,None]*co[...,None,:]+cs
    return updated_cotangent*keyed_direction[..., :,None]

t = terms(q,k,dk,b,db,co,cs)
with jax.enable_x64():
    tw = terms(*(x.astype(jnp.float64) for x in (q,k,dk,b,db,co,cs)))
    wide = jnp.sum(tw,axis=-2)
red = jnp.sum(t,axis=-2)
green = jnp.sum(t/128,axis=-2)*128
reordered = jnp.sum(jnp.swapaxes(t,-1,-2),axis=-1)
for col in (51,62):
    print("K REDGREEN",col,"terms_max32",np.max(np.abs(np.asarray(t)[0,1,:,col])),
          "sum32",np.asarray(red)[0,1,col],"sum64",np.asarray(wide)[0,1,col],
          "scaled_sum32",np.asarray(green)[0,1,col],
          "reordered_sum32",np.asarray(reordered)[0,1,col],
          "prefix_max64",np.max(np.abs(np.cumsum(np.asarray(tw)[0,1,:,col]))))
