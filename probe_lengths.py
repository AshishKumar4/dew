import sys; sys.path[:0]=["src","tests","."]
import jax
import jax.numpy as j
import numpy as np
from dew.nn.kernels import recurrence_chunk as rc
for length in (1,2,3,5,8,16,17,32,37):
    x=j.ones((1,length,1,1),j.float32)*2.**62
    g=j.ones((1,length,1),j.float32)*-.5
    b=j.ones((1,length,1),j.float32)*.5
    def fn(v):
        return rc.xla_mainloop(rc.Tiles(16),x,x,v,g,b,scale=1.,delta=True,dtype=j.float32)[0].sum()
    value=jax.jit(jax.grad(fn))(x)
    print("K length",length,"gradv",np.asarray(value).ravel().tolist(),flush=True)
