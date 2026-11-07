"""Trace float64 UNet3D boundaries; scratch probe, armada only."""
import os,sys
sys.path[:0]=[os.getcwd(),os.path.join(os.getcwd(),'tests'),os.path.join(os.getcwd(),'tools')]
from lane_environment import configure_lane
configure_lane(os.environ)
os.environ['XLA_FLAGS'] += ' --xla_cpu_use_xnnpack=false --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import dataclasses
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import test_flaxdiff_family_source as t
import flaxdiff_family_worker as w
from arm_arithmetic import unrewritten,summed_as_run
from reference_error import assert_computes_the_oracle,chain_roundings
case='unet_3d/'+sys.argv[1]
mode=sys.argv[2] if len(sys.argv)>2 else 'trace'
w.fourier_table(t.CASES[case][2]['emb_features'],16)
modules=w.flaxdiff(True)
if mode in ('ref-softmax','both'):
    original=modules['attention'].scaled_dot_product_attention
    def no_pin(*args,**kwargs):
        kwargs['force_fp32_for_softmax']=False
        return original(*args,**kwargs)
    modules['attention'].scaled_dot_product_attention=no_pin
    modules['vit_common'].scaled_dot_product_attention=no_pin
if mode in ('dew-gate','both'):
    import dew.nn.activations as acts
    def widened(activate):
        return lambda gate,up: (activate(gate.astype(jnp.promote_types(gate.dtype,jnp.float32))) * up.astype(jnp.promote_types(up.dtype,jnp.float32))).astype(jnp.result_type(gate,up))
    acts._rounded=widened
_,cls,config,_,_=t.CASES[case]
reference=getattr(modules['unet_3d'],cls)(**config)
dew=t.dew_model(cls,config)
inputs=t.drawing(case)(int(sys.argv[3]) if len(sys.argv)>3 else 0)
names=sorted(t.LAYOUT[case]['params'])

def compute(which):
    def run(inputs):
        params=unflatten_dict({tuple(p.split('/')):jnp.asarray(inputs['param.'+p],jnp.float64) for p in names})
        image,text,times,probe=(jnp.asarray(inputs[k],jnp.float64) for k in ('image','text','times','probe'))
        constants={}
        if which=='dew':
            params=t.nest({t.dew_path(cls,p):t.to_dew(cls,p,params_leaf,config.get('patch_size')) for p,params_leaf in [('/'.join(p),v) for p,v in flatten_dict(params).items()]})
            constants=t.nest({(*t.fourier_path(cls),'frequencies'):jnp.asarray(t.LAYOUT[case]['fourier'],jnp.float64)})
        def forward(params,image,text):
            capture=lambda module,method: method=='__call__' and ('/'.join(module.scope.path) in ('FourierEmbedding_0','TimeProjection_0','down_0_residual_0','down_0_attention_0','down_0_attention_1','down_0_temporal','down_0_temporal/RMSNorm_0','down_0_temporal/temporal_attention','down_0_temporal/temporal_out') or len(module.scope.path)==0)
            if which=='dew':
                return dew.apply({'params':params,'constants':constants},image,times,t.TextContext(text,jnp.ones(text.shape[:2],jnp.float64)),capture_intermediates=capture,mutable=['intermediates'])
            return reference.apply({'params':params},image,times,text,capture_intermediates=capture,mutable=['intermediates'])
        output,pb,aux=jax.vjp(forward,params,image,text,has_aux=True)
        gp,gi,gt=pb(probe)
        flat=flatten_dict(gp)
        gradients=jnp.concatenate([gi.ravel(),gt.ravel(),*[((t.from_dew(cls,p,flat[t.dew_path(cls,p)],t.LAYOUT[case]['params'][p])) if which=='dew' else flat[tuple(p.split('/'))]).ravel() for p in names]])
        captured={'/'.join(p[:-1]):v[0] for p,v in flatten_dict(aux['intermediates']).items()}
        return output,gradients,captured
    return unrewritten(jax.jit(run))

def rms(v):return float(np.sqrt(np.mean(np.square(np.asarray(v,np.float64)))))
with jax.enable_x64():
    da,rb=compute('dew'),compute('ref')
    a=jax.device_get(da(inputs));b=jax.device_get(rb(inputs))
    roundings=chain_roundings(jax.make_jaxpr(da)(inputs))+chain_roundings(jax.make_jaxpr(rb)(inputs))
print('K result',case,mode,'output',rms(a[0]-b[0]),'gradients',rms(a[1]-b[1]),flush=True)
for p,bv in b[2].items():
    dp='/'.join(t.dew_path(cls,p+'/dummy')[:-1]) if p else ''
    if dp in a[2] and bv.shape==a[2][dp].shape:
        av=a[2][dp]
        print('K layer',case,mode,p,'dtype',av.dtype,bv.dtype,'rms',rms(av-bv),'size',rms(bv),flush=True)
from dew.nn.precision import float32_dot_general
with jax.enable_x64():
    value=jax.jit(lambda x:float32_dot_general(x,jnp.ones((1,),jnp.float64),(((0,),(0,)),((),()))))(jnp.array([1+2**-35],jnp.float64))
    print('K dot64',value.dtype,float(value-1),flush=True)
print('K oracle-roundings',case,mode,roundings,flush=True)
for index,name in enumerate(('output','gradients')):
    assert_computes_the_oracle(b[index],a[index],case+' '+name,roundings=roundings)
print('K FLOAT64-PASS',case,mode,flush=True)
