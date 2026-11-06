import sys
sys.path[:0] = ["src", "tests"]
import jax
import jax.numpy as j
import numpy as np
import test_algebra_scan as t
from reference_error import without_ynn
from dew.algebra import scan
from dew.nn.precision import float32_matmul

jax.config.update("jax_enable_x64", True)
forms = scan.derive_scan(t.DECLARED, saturation=t.SATURATION)
def chunked(v, i):
    return t.chunk_of({**v, "rate": j.float32(t.RATE), "skip": j.float32(t.SKIP)}, i)
derived = t.adjoint(forms, chunked, 4)
def derived_c(x, dt, B, C, dy, dh):
    return derived(dict(x=x, dt=dt, B=B, C=C), (dy, dh))["C"]
def token_c(x, dt, B, C, dy, dh):
    return jax.vjp(t.tokenwise, x, dt, B, C)[1]((dy,dh))[3]
derived_c, token_c = without_ynn(derived_c), without_ynn(token_c)
def parts(x, dt, B, C, dy, dh):
    z=dt*j.asarray(t.RATE,dt.dtype)
    def step(h, v):
        zz,bb,uu=v
        h=j.exp(zz)*h+bb[:,None]*uu[None,:]
        return h,h
    states=jax.lax.scan(step,j.zeros((16,32),x.dtype),(z,B,x*dt[:,None]))[1]
    hs=j.concatenate([j.zeros_like(states[:1]), states[:-1]])[::32]
    zs=z.reshape(4,32)
    def one(zz, bb, uu, yy, h):
        dec=scan._decays(zz[:,None],False)
        ds=float32_matmul(yy,uu.T)
        terms=ds[:,:,None]*dec.between*bb[None,:,:]
        local=float32_matmul(ds*dec.between[:,:,0],bb)
        memory=dec.entering*float32_matmul(yy,h.T)
        asc=jax.lax.scan(lambda acc,v:(acc+v,None),j.zeros_like(local),j.moveaxis(terms,1,0))[0]
        desc=jax.lax.scan(lambda acc,v:(acc+v,None),j.zeros_like(local),j.moveaxis(terms[:,::-1],1,0))[0]
        return local,memory,asc,desc,terms,dec.entering,ds
    return jax.vmap(one)(zs,B.reshape(4,32,16),(x*dt[:,None]).reshape(4,32,32),
                          dy.reshape(4,32,32),hs)
parts=without_ynn(parts)
sq={}
def err(name,a,b):
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    sq[name]=sq.get(name,0)+np.sum(d*d)
for seed in range(16):
    vals=(*t.deep_draw(seed),*t.cotangents_of(seed,(128,32),(16,32)))
    vals64=tuple(v.astype(j.float64) for v in vals)
    actual=derived_c(*vals)
    tok=token_c(*vals)
    truth=token_c(*vals64)
    pp=parts(*vals); p64=parts(*vals64)
    local,mem,asc,desc,terms,enter,ds=pp
    l64,m64,*_=p64
    err("token",tok,truth); err("derived",actual,truth)
    err("local",local,l64);err("memory",mem,m64)
    err("asc",asc+mem,truth.reshape(4,32,16))
    err("desc",desc+mem,truth.reshape(4,32,16))
    err("reconstruct",local+mem,actual.reshape(4,32,16))
    err("local_only64",l64+mem,truth.reshape(4,32,16))
    err("memory_only64",local+m64,truth.reshape(4,32,16))
    if seed==0:
        for lag in (0,1,2,4,8,16,31):
            tt=np.asarray(terms)
            mag=np.concatenate([tt[:,i,i-lag] for i in range(lag,32)])
            print("K lag",lag,"rms",np.sqrt(np.mean(mag*mag)))
        print("K local_rms",float(j.sqrt(j.mean(local*local))),"memory_rms",float(j.sqrt(j.mean(mem*mem))))
for name,value in sq.items():
    print("K error",name,"rms",np.sqrt(value/(16*128*16)),"ratio",np.sqrt(value/sq["token"]))
