import sys
sys.path[:0]=["src","tests"]
import jax
import jax.numpy as j
import numpy as np
from reference_error import without_ynn
from dew.nn.precision import float32_matmul
q=j.ones((16,),j.float32)
h=j.ones((16,32),j.float32)
dy=j.ones((32,),j.float32)
def token(h,dy):
    return jax.vjp(lambda q:q@h,q)[1](dy)[0]
print("K token",str(jax.make_jaxpr(token)(h,dy)).replace("\n"," "))
d=j.ones((32,32),j.float32)
u=j.ones((32,32),j.float32)
def matrix(d,u):return float32_matmul(d,u.T)
def columns(d,u):return jax.lax.map(lambda row:float32_matmul(d,row),u).T
for name,fn in (("matrix",matrix),("columns",columns),("token",token)):
    args=(h,dy) if name=="token" else (d,u)
    txt=without_ynn(fn).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    for line in txt.splitlines():
        if "dot(" in line or "reduce(" in line:
            print("K hlo",name,line)
    txt=without_ynn(fn).lower(*args).compile().as_text()
    for line in txt.splitlines():
        if "dot(" in line or "reduce(" in line:
            print("K optimized",name,line)
rng=np.random.default_rng(0)
aa=j.asarray(rng.standard_normal((32,32)),j.float32)
bb=j.asarray(rng.standard_normal((32,32)),j.float32)
def reductions(a,b):
    direct=a@b.T
    gemv=columns(a,b)
    seq=jax.lax.scan(lambda acc,k:(acc+a[:,k,None]*b[None,:,k],None),j.zeros_like(a),j.arange(32))[0]
    return direct,gemv,seq
direct,gemv,seq=map(np.asarray,without_ynn(reductions)(aa,bb))
exact=np.asarray(aa,np.float64)@np.asarray(bb,np.float64).T
for name,value in (("GEMM",direct),("GEMV",gemv),("serial",seq)):
    print("K rounding",name,"rms",np.sqrt(np.mean((value-exact)**2)),
          "different_from_GEMM",np.count_nonzero(value!=direct))
