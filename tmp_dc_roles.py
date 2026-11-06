import sys
sys.path[:0] = ["src","tests"]
import jax
import jax.numpy as j
import numpy as np
import test_algebra_scan as t
from reference_error import without_ynn
from dew.algebra import scan
from dew.nn.precision import float32_matmul
jax.config.update("jax_enable_x64",True)
mode=sys.argv[1]
target=int(sys.argv[2]) if len(sys.argv)>2 else 16
original=scan._issued
def product(left,right):
    if mode=="inner":
        return j.sum(left[:,None,:]*right.T[None,:,:],axis=-1)
    if mode=="rows":
        return jax.lax.map(lambda row: float32_matmul(row,right),left)
    if mode=="cols":
        return jax.lax.map(lambda col: float32_matmul(left,col),right.T).T
    if mode=="swap":
        return float32_matmul(right.T,left.T).T
    if mode=="einsum":
        return j.sum(left[:,:,None]*right[None,:,:],axis=1)
    if mode=="wide":
        return (left.astype(j.float64)@right.astype(j.float64)).astype(left.dtype)
    if mode in ("asc","desc"):
        indices=j.arange(left.shape[-1])
        if mode=="desc": indices=indices[::-1]
        def step(acc,k):
            return acc+left[:,k,None]*right[k,None,:],None
        return jax.lax.scan(step,j.zeros((left.shape[0],right.shape[1]),left.dtype),indices)[0]
    if mode=="flip": return float32_matmul(left[:,::-1],right[::-1])
    return float32_matmul(left,right)
@jax.custom_vjp
def score(left,right):
    return original("readout_local",left,right)
def fwd(left,right):
    return original("readout_local",left,right),(left,right)
def bwd(saved,d):
    left,right=saved
    return product(d,right.T),float32_matmul(left[::-1].T,d[::-1])
score.defvjp(fwd,bwd)
def issue(name,left,right):
    if name=="readout_local" and (target==0 or left.shape[-1]==target): return score(left,right)
    return original(name,left,right)
if mode!="base": scan._issued=issue
forms=scan.derive_scan(t.DECLARED,saturation=t.SATURATION)
def chunked(v,i):
    return t.chunk_of({**v,"rate":j.float32(t.RATE),"skip":j.float32(t.SKIP)},i)
adjoint=t.adjoint(forms,chunked,4)
def dc(x,dt,B,C,dy,dh):
    return adjoint(dict(x=x,dt=dt,B=B,C=C),(dy,dh))["C"]
def tc(x,dt,B,C,dy,dh):
    return jax.vjp(t.tokenwise,x,dt,B,C)[1]((dy,dh))[3]
dc,tc=without_ynn(dc),without_ynn(tc)
errs=np.zeros(2)
for seed in range(16):
    vals=(*t.deep_draw(seed),*t.cotangents_of(seed,(128,32),(16,32)))
    truth=np.asarray(tc(*(x.astype(j.float64) for x in vals)))
    for i,fn in enumerate((dc,tc)):
        errs[i]+=np.sum((np.asarray(fn(*vals),np.float64)-truth)**2)
print("K mode",mode,"rms",np.sqrt(errs/(16*128*16)).tolist(),"ratio",np.sqrt(errs[0]/errs[1]))
