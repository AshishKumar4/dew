"""Run live reference check 1 and sizing for UNet3D; scratch, armada only."""
import os,sys
sys.path[:0]=[os.getcwd(),os.path.join(os.getcwd(),'tests'),os.path.join(os.getcwd(),'tools')]
from lane_environment import configure_lane
configure_lane(os.environ)
os.environ['XLA_FLAGS']+=' --xla_cpu_use_xnnpack=false --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import jax,jax.numpy as jnp,numpy as np
import flaxdiff_family_worker as w
import test_flaxdiff_family_source as t
from reference_error import assert_as_exact, as_given, seeds_for
import reference_error as gate
from arm_arithmetic import unrewritten
case='unet_3d/'+sys.argv[1]
mode=sys.argv[2]
seeds=int(sys.argv[3]) if len(sys.argv)>3 else 8
w.fourier_table(t.CASES[case][2]['emb_features'],16)
ctx=w.applying(case,'float32');r32=ctx.__enter__()
ctxw=w.applying(case,'float64');r64=ctxw.__enter__()
if mode in ('fixed','sizing','eager-fixed'):
    modules={p:sys.modules['flaxdiff.models.'+p] for p in ('attention','vit_common')}
    orig=modules['attention'].scaled_dot_product_attention
    def no_pin(*args,**kwargs):
        kwargs['force_fp32_for_softmax']=False
        return orig(*args,**kwargs)
    for m in modules.values():m.scaled_dot_product_attention=no_pin

def ref(apply,inputs,dtype):
    return apply({k.removeprefix('param.'):jnp.asarray(v,dtype) for k,v in inputs.items() if k.startswith('param.')},inputs)
# Compare both the recorder's eager arithmetic and whole-program compilation.
eager=mode.startswith('eager') or mode=='sizing'
r=unrewritten(lambda x:ref(r32,x,jnp.float32))
truth=unrewritten(lambda x:ref(r64,x,jnp.float64))
if not eager:
    r=unrewritten(jax.jit(r));truth=unrewritten(jax.jit(truth))
dew=t.computing(case)
compiled=jax.jit(lambda x:dew(jnp.float32,x))
candidate=as_given(lambda x:compiled(x))
judgement=gate._judgement
def logged(draws,label,*args,**kwargs):
    for index,name in enumerate(draws.names):
        print('K ratios',case,mode,name,len(draws.mine),gate._judged(draws,index),flush=True)
    return judgement(draws,label,*args,**kwargs)
gate._judgement=logged
group_errors={}
shape_image,shape_text=t.CASES[case][3:]
entries=[('image',(2,*shape_image)),('text',(2,*shape_text)),*[(p,s) for p,s in sorted(t.LAYOUT[case]['params'].items())]]
spans={};offset=0
for path,shape in entries:
    size=int(np.prod(shape))
    spans.setdefault(path.split('/')[0],[]).append(slice(offset,offset+size));offset+=size
evidence=gate._evidence
def split(prepared,reference,their_truth,**kwargs):
    if not isinstance(reference['gradients'],gate.RecordedLeaf):
        a=np.asarray(prepared.leaves[0],np.float64)-np.asarray(prepared.truth['gradients'],np.float64)
        b=np.asarray(reference['gradients'],np.float64)-np.asarray(their_truth['gradients'],np.float64)
        for group,ranges in spans.items():
            ae=np.concatenate([a[s] for s in ranges]);be=np.concatenate([b[s] for s in ranges])
            sums=group_errors.setdefault(group,np.zeros(3))
            sums+=np.array([np.square(ae).sum(),np.square(be).sum(),ae.size])
    return evidence(prepared,reference,their_truth,**kwargs)
gate._evidence=split
try:
    if mode=='sizing':
        n=seeds_for(as_given(lambda x:r(x)),lambda seed:(t.drawing(case)(seed),),f'UNet3D {case}',dew_fn=candidate,truth_fn=lambda x:dew(jnp.float64,x),start=seeds)
        print('K sizing',case,n,flush=True)
    else:
        assert_as_exact(candidate,as_given(lambda x:r(x)),lambda seed:(t.drawing(case)(seed),),f'UNet3D {case} {mode}',seeds=seeds,truth_fn=lambda x:truth(x))
        print('K PASS',case,mode,seeds,flush=True)
except Exception as error:
    print('K FAIL',case,mode,type(error).__name__,str(error),flush=True)
    raise
for group,(ours,theirs,size) in group_errors.items():
    print('K group',case,mode,group,'rms',np.sqrt(ours/size),np.sqrt(theirs/size),'ratio',np.sqrt(ours/theirs) if theirs else 0,flush=True)
