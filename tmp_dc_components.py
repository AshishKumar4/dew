import sys
sys.path[:0]=["src","tests"]
import jax
import jax.numpy as j
import numpy as np
import test_algebra_scan as t
from reference_error import without_ynn
from dew.algebra import scan
from dew.nn.precision import float32_matmul
jax.config.update("jax_enable_x64",True)
mode=sys.argv[1]
original=scan._issued
def cut(name,left,right):
    @jax.custom_vjp
    def f(a,b):return original(name,a,b)
    def fwd(a,b):return original(name,a,b),(a,b)
    def bwd(saved,d):
        a,b=saved
        return j.zeros_like(a),float32_matmul(a[::-1].T,d[::-1])
    f.defvjp(fwd,bwd)
    return f(left,right)
def issue(name,left,right):
    if (mode=="local" and name=="readout_memory") or (mode=="memory" and name=="readout_local" and left.shape[-1]==16):
        return cut(name,left,right)
    return original(name,left,right)
scan._issued=issue
forms=scan.derive_scan(t.DECLARED,saturation=t.SATURATION)
def chunked(v,i):return t.chunk_of({**v,"rate":j.float32(t.RATE),"skip":j.float32(t.SKIP)},i)
adjoint=t.adjoint(forms,chunked,4)
def dc(x,dt,B,C,dy,dh):return adjoint(dict(x=x,dt=dt,B=B,C=C),(dy,dh))["C"]
def tc(x,dt,B,C,dy,dh):return jax.vjp(t.tokenwise,x,dt,B,C)[1]((dy,dh))[3]
dc,tc=without_ynn(dc),without_ynn(tc)
sums=np.zeros(4)
for seed in range(80):
    vals=(*t.deep_draw(seed),*t.cotangents_of(seed,(128,32),(16,32)))
    wide=tuple(v.astype(j.float64) for v in vals)
    xx,dt,bb,cc,yy,dh=map(lambda v:np.asarray(v,np.float64),vals)
    state=np.zeros((16,32))
    local=np.zeros((16,32))
    locals_,memories=[],[]
    for i in range(128):
        if i%32==0:local[:]=0
        write=bb[i,:,None]*(xx[i]*dt[i])[None,:]
        state=np.exp(dt[i]*t.RATE)*state+write
        local=np.exp(dt[i]*t.RATE)*local+write
        locals_.append(local@yy[i])
        memories.append((state-local)@yy[i])
    exact={"local":np.asarray(locals_),"memory":np.asarray(memories),
           "base":np.asarray(locals_)+np.asarray(memories)}[mode]
    a=np.asarray(dc(*vals),np.float64)
    tok=np.asarray(tc(*vals),np.float64);truth=np.asarray(tc(*wide))
    sums+=np.array([np.sum((a-exact)**2),np.sum((tok-truth)**2),np.sum(a*a),np.sum(exact*exact)])
print("K component",mode,"RMS",np.sqrt(sums/(80*128*16)).tolist(),"ratio",np.sqrt(sums[0]/sums[1]))
