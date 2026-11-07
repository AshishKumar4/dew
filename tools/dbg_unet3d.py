"""Scratch diagnostics for row 4763; all numerical execution on armada."""
import os
import sys
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'tests'), os.path.join(os.getcwd(), 'tools')]
from lane_environment import configure_lane
configure_lane(os.environ)
os.environ['XLA_FLAGS'] += ' --xla_cpu_use_xnnpack=false --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import dataclasses
import json
import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import summed_as_run, unrewritten
import test_flaxdiff_family_source as t
from flaxdiff_family_worker import applying

case = 'unet_3d/variant'
mode = sys.argv[1] if len(sys.argv)>1 else 'split'
start = int(sys.argv[2]) if len(sys.argv)>2 else 0
count = int(sys.argv[3]) if len(sys.argv)>3 else 8

def rms(x):
    return np.sqrt(np.mean(np.square(np.asarray(x,np.float64))))

def groups(inputs):
    entries = [('image', inputs['image'].shape), ('text',inputs['text'].shape)]
    entries += [(p,np.shape(inputs['param.'+p])) for p in sorted(t.LAYOUT[case]['params'])]
    offset=0
    result={}
    for path,shape in entries:
        n=int(np.prod(shape)); group=path.split('/')[0]
        result.setdefault(group,[]).append(slice(offset,offset+n)); offset+=n
    return result

original_model = t.dew_model

def wide_model(cls, config):
    m = original_model(cls,config)
    return m.clone(attention_configs=tuple(None if s is None else dataclasses.replace(s,dtype=jnp.float64) for s in m.attention_configs))

old=t.computing(case)
t.dew_model=wide_model
wide=t.computing(case)
t.dew_model=original_model
r32ctx=applying(case,'float32'); ref32=r32ctx.__enter__()
r64ctx=applying(case,'float64'); ref64=r64ctx.__enter__()

def ref(apply,inputs):
    return apply({k.removeprefix('param.'):v for k,v in inputs.items() if k.startswith('param.')},inputs)

narrow=jax.jit(lambda x: old(jnp.float32,x))
old64=unrewritten(jax.jit(lambda x: old(jnp.float64,x)))
new64=unrewritten(jax.jit(lambda x: wide(jnp.float64,x)))
rn=unrewritten(jax.jit(lambda x:ref(ref32,x)))
rw=unrewritten(jax.jit(lambda x:ref(ref64,x)))
errors={}
for seed in range(start,start+count):
    inputs=t.drawing(case)(seed)
    with jax.enable_x64():
        a64=jax.device_get(old64(inputs)); b64=jax.device_get(new64(inputs)); truth=jax.device_get(rw(inputs))
    with summed_as_run():
        a=jax.device_get(narrow(inputs)); b=jax.device_get(rn(inputs))
    gs=groups(inputs)
    leaves={'output': (a['output'],b['output'],truth['output'],a64['output'],b64['output'])}
    leaves['gradients']=(a['gradients'],b['gradients'],truth['gradients'],a64['gradients'],b64['gradients'])
    for g,spans in gs.items():
        leaves[g]=tuple(np.concatenate([np.ravel(v['gradients'])[s] for s in spans]) for v in (a,b,truth,a64,b64))
    for name,(av,bv,tr,ov,nv) in leaves.items():
        err=np.array([rms(av-tr),rms(bv-tr),rms(ov-tr),rms(nv-tr),rms(av-ov),rms(av-nv)])
        errors.setdefault(name,[]).append(err)
    print('K seed',seed,'output',errors['output'][-1].tolist(),'gradients',errors['gradients'][-1].tolist(),flush=True)
for name,errs in errors.items():
    er=np.sqrt(np.mean(np.square(errs),axis=0))
    print('K group',name,json.dumps(dict(zip(['dew_reftruth','ref_reftruth','old64_reftruth','new64_reftruth','dew_oldtruth','dew_newtruth'],er.tolist()))),'ratios',er[[0,4,5]]/er[1],flush=True)
