import os,sys
sys.path[:0]=[os.getcwd(),os.path.join(os.getcwd(),'tests'),os.path.join(os.getcwd(),'tools')]
from lane_environment import configure_lane
configure_lane(os.environ)
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
import flaxdiff_family_worker as w
import test_flaxdiff_family_source as t
from flax.traverse_util import unflatten_dict
original=nn.dot_product_attention
calls=[]
def watched(query,*args,**kwargs):
    calls.append((str(query.dtype),kwargs.get('force_fp32_for_softmax',False)))
    return original(query,*args,**kwargs)
nn.dot_product_attention=watched
for case in list(t.CASES)[int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 0:]:
    width=t.CASES[case][2]['emb_features']
    w.fourier_table(width[0] if isinstance(width,tuple) else width,16)
    inputs=t.drawing(case)(0)
    calls.clear()
    stem,cls,config,_,_=t.CASES[case]
    with jax.enable_x64():
        model=getattr(w.flaxdiff(True)[stem],cls)(**config)
        params=unflatten_dict({tuple(k.removeprefix('param.').split('/')):jnp.asarray(v,jnp.float64) for k,v in inputs.items() if k.startswith('param.')})
        out=model.apply({'params':params},*[jnp.asarray(inputs[k],jnp.float64) for k in ('image','times','text')])
        jax.block_until_ready(out)
    print('K census',case,'attentions',len(calls),'forced',sum(flag for _,flag in calls),'dtypes',sorted(set(dtype for dtype,_ in calls)),flush=True)
