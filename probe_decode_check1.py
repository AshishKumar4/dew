import os
import sys
sys.path[:0] = [os.path.join(os.getcwd(),p) for p in ("src","tests","tools")]
import jax
import jax.numpy as jnp
import numpy as np
from contract_support import normal_draw
from test_contracts_recurrence import delta_rule_draw
from dew.nn.kernels import recurrence
from reference_error import assert_as_exact, with_gradients

mode = sys.argv[1]
def changed(state,query,key,value,decay,write,*,delta):
    corrected = value-jnp.sum((key*decay)[...,:,None]*state,axis=-2) if delta else value
    if mode == "transpose":
        written = jnp.swapaxes(corrected[..., :,None]*(key*write)[...,None,:],-1,-2)
    else:
        written = ((key*write)/2)[...,:,None]*(corrected*2)[...,None,:]
    updated = state*decay[..., :,None]+written
    return jnp.sum(query[..., :,None]*updated,axis=-2),updated

def fn(update):
    def run(q,k,v,g,b,s,skip):
        q,k = [jnp.repeat(x[:,0],2,axis=1) for x in (q,k)]
        out,held = update(s,q,k,v[:,0],jnp.broadcast_to(jnp.exp(g[:,0,:,None]),q.shape),
                          b[:,0,:,None],delta=True)
        return jnp.concatenate([(out+skip[:,None]*v[:,0]).ravel(),held.ravel()])
    return with_gradients(run)

def draw(seed):
    p,d,c = delta_rule_draw(2669+seed,1.0,key_heads=1,value_heads=2,width=128,length=1,per_key=False)
    (skip,) = normal_draw(2672+seed,(2,))
    cot = jnp.concatenate([c[0][:,0].ravel(),c[1].ravel()])
    return (*p,skip,cot)

try:
    assert_as_exact(fn(changed),fn(recurrence.update),draw,"decode write "+mode,seeds=16)
    print("K CHECK1 PASS",mode,"16 seeds")
except Exception as e:
    print("K CHECK1 FAIL",mode,type(e).__name__,str(e))
