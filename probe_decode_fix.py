import os
import sys
sys.path[:0] = [os.path.join(os.getcwd(), p) for p in ("src", "tests", "tools")]
import jax
import jax.numpy as jnp
import numpy as np
from contract_support import normal_draw, assert_contract
from test_contracts_recurrence import delta_rule_draw, flattened, plain_delta_rule, INTERPRETED_REMAT
from dew.nn.kernels import recurrence

mode = sys.argv[1]
factor = float(sys.argv[2]) if len(sys.argv) > 2 else 128.0
reference_only = len(sys.argv) > 3 and sys.argv[3] == "reference"
if len(sys.argv) > 3 and sys.argv[3] == "reference-forward":
    recurrence._launched = lambda hooks, operands, *, dtype: recurrence.step_reference(hooks, operands, dtype=dtype)
original = recurrence.update

def candidate(state, query, key, value, decay, write, *, delta):
    if mode == "delta":
        corrected = value - jnp.sum(((key * decay)/factor)[..., :, None] * state, axis=-2)*factor if delta else value
    else:
        corrected = value - jnp.sum((key * decay)[..., :, None] * state, axis=-2) if delta else value
    if mode == "write":
        updated = state * decay[..., :, None] + ((key * write)/factor)[..., :, None] * (corrected*factor)[..., None, :]
    elif mode == "write-first":
        updated = state * decay[..., :, None] + key[..., :, None] * (write * corrected)[..., None, :]
    elif mode == "einsum":
        updated = state * decay[..., :, None] + jnp.einsum("...k,...v->...kv",key*write,corrected)
    elif mode == "transpose":
        written = corrected[..., :, None] * (key*write)[..., None, :]
        updated = state * decay[..., :, None] + jnp.swapaxes(written,-1,-2)
    else:
        updated = state * decay[..., :, None] + (key * write)[..., :, None] * corrected[..., None, :]
    if mode == "output":
        out = jnp.sum((query/factor)[..., :, None] * updated, axis=-2)*factor
    else:
        out = jnp.sum(query[..., :, None] * updated, axis=-2)
    return out,updated

if mode != "original":
    recurrence.update = candidate
hooks = recurrence.Hooks(decay="scalar",skip=True)

def fn(*x):
    q,k,v,g,b,s,d = x
    operands = recurrence.Step(s,q[:,0],k[:,0],v[:,0],log_decay=g[:,0],beta=b[:,0],skip=d)
    out,_,state = (recurrence.step_reference if reference_only else recurrence.step)(hooks,operands)
    return out,state

def plain(*x):
    q,k,v,g,b,s,d = x
    out,held = plain_delta_rule(q,k,v,g,b,s,1.0)
    return out[:,0]+d[:,None]*v[:,0],held

for edge,scale in (("largest",2.0**63),("normal",1.0),("least",2.0**-63)):
    p,d,c = delta_rule_draw(2669,scale,key_heads=1,value_heads=2,width=128,length=1,per_key=False)
    (skip,) = normal_draw(2672,(2,))
    (sd,) = normal_draw(2673,(2,))
    cot = jnp.concatenate([c[0][:,0].ravel(),c[1].ravel()])
    try:
        assert_contract(flattened(fn),(*p,skip),(*d,sd),cot,f"{mode}/{factor}/{edge}",
                        truth_fn=flattened(plain),plain_fn=flattened(plain),without=INTERPRETED_REMAT)
        print("K PASS",mode,factor,edge)
    except AssertionError as e:
        print("K FAIL",str(e))
