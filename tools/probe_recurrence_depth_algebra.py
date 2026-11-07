import sys,inspect,numpy as np
sys.path[:0]=['src','tests','tools']
import jax,jax.numpy as jnp
from dew.nn.kernels import recurrence_algebra as alg
original=alg.key_decayed_pairs_back
source=inspect.getsource(original).replace('d_pair.rows(place, sub).T, down_rows.rows(place, sub)', 'd_pair.T, lambda down_rows=down_rows: jnp.where(within, down_rows.value(), 0.0)')
exec(source,alg.__dict__)
modified=alg.key_decayed_pairs_back
class Tile:
 def __init__(self,x):self.x=x
 @property
 def T(self):return Tile(self.x.T)
 def rows(self,start,count):return Tile(jax.lax.dynamic_slice_in_dim(self.x,start,count,axis=0))
 def value(self):return self.x
rng=np.random.default_rng(8)
def rand(shape):return jnp.asarray(rng.normal(size=shape),jnp.float32)
size,width=64,32
rows=[rand((size,width)),rand((size,width))];key=rand((size,width))
gate=-jnp.cumsum(jnp.abs(rand((size,width)))*0.02,axis=0)
pairs=[Tile(jnp.tril(rand((size,size)))) for _ in range(2)]
def summed(products,shape=None):
 def array(x):
  x=x() if callable(x) else x
  return x.value() if isinstance(x,Tile) else x
 return sum(array(l)@array(r) for _,l,r in products)
def run(fn):
 out={}
 fn(pairs,[lambda c,x=x:x for x in rows],lambda c:key,1,
    lambda c:(gate,jnp.zeros_like(gate)),lambda s,c:(gate[s*16],jnp.zeros(width)),summed,Tile,
    chunk=size,emit=lambda kind,p,c,v:out.__setitem__((kind,p,c),v))
 return out
before,after=run(original),run(modified)
for k in before:print('K ALGEBRA',k,'max_abs',float(jnp.max(jnp.abs(before[k]-after[k]))),flush=True)
