import os, sys, re
variant = sys.argv.pop(1)
for bit in variant.split('+'):
    if bit != 'base': os.environ['DBG_' + bit.upper()] = '1'
print('K VARIANT',variant,flush=True)
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path[:0] = ['src', 'tests']
import jax
import jax.numpy as jnp
if os.environ.get('DBG_LOAD4'):
    import inspect
    from jax.experimental.mosaic.gpu import tcgen05 as tc
    source=inspect.getsource(tc._transfer_32xcols)
    assert '32 // regs_per_instr' in source
    exec(source.replace('32 // regs_per_instr','4 // regs_per_instr'),tc.__dict__)
import kernel_compile as kc
import test_recurrence_mosaic as t
from fla_gdn_support import KernelCase
from dew.nn.kernels import generation
from dew.nn.kernels.templates.recurrence import template
target, rows, length = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
name, dtype = (sys.argv[4] if len(sys.argv) > 4 else 'nemotron-h-4b-ssd'), jnp.bfloat16
facts = generation.gpu_target(target)
family = template.bundle(facts)
args = t.resource_operands(name, dtype, rows, length)
bound = template.MOSAIC.bind(*args, None, max_segments=None, bound=None if name in t.RATED else 1.0, **t._form(name, dtype), **t._skipped(name, dtype))
with generation.assumed(facts):
    tiles = template.MOSAIC.derived(bound, facts)
    launches = template.MOSAIC.launches(bound, tiles, facts, backward=True)
    models = t._models(launches, family, facts)
print('K TILES',variant,name,length,tiles,flush=True)
for launch in launches:
    print('K PLAN',variant,name,length,launch.name,'programs',launch.programs,'stages',launch.stages,'model',launch.registers,flush=True)
call = lambda: t._gate_call(name, dtype, [*args, jax.ShapeDtypeStruct(args[2].shape, dtype),
                                          jax.ShapeDtypeStruct(args[5].shape, jnp.float32)], None, tiles, family,
                            training=not bool(os.environ.get('DBG_FORWARD'))) if not os.environ.get('DBG_FORWARD') else t._gate_call(
                                name,dtype,args,None,tiles,family,training=False)
if os.environ.get('DBG_DIRECT'):
    from dew.nn.kernels.templates.recurrence import mainloop_backward as back
    original = back._scores_back
    captured=[]
    def capture(*xs,**kwargs):
        captured.append((xs,kwargs))
        return original(*xs,**kwargs)
    back._scores_back=capture
    fn, supplied=call()
    with generation.assumed(facts):
        jax.eval_shape(fn,*supplied)
    back._scores_back=original
    xs, kwargs = captured[-1]
    static=xs[-1]
    inputs=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype) if hasattr(x,'shape') else x,xs[:-1])
    call=lambda:(lambda *xs:original(*xs,static,**kwargs),inputs)
for record in kc.compile_mosaic_kernels([KernelCase(f'{name} {length}', call, models)], target, least=True):
    for kernel in record.get('kernels', ()):
        short = kernel['kernel'].split('__')[0]
        model = models.get(short)
        print('K RESULT',variant,target,name[:6],length,short,'model',getattr(model,'registers',None),getattr(model,'budget',None),
              'least',kernel.get('least'),'registers',kernel.get('registers'),'spilled',kernel.get('spilled'),flush=True)
        if short == 'recurrence_scores_back':
            ptx=kc.PTX.get(kernel.get('module'),'')
            consumer=ptx.split('setmaxnreg.inc',1)[-1]
            before=consumer.split('tcgen05.mma',1)[0]
            print('K PREFIXPOS',variant,'shared_loads_before_first_MMA',len(re.findall(r'ld.shared.b(?:32|64)',before)),flush=True)
            if os.environ.get('DBG_ASSERT_MODEL'):
                limit=(model.registers+7)//8*8
                needed=kernel.get('least')
                fits=needed is not None and needed<=limit
                print('K CHECK',variant,'GREEN' if fits else 'RED','need',needed,'model_cap',limit,flush=True)
                assert fits,(needed,limit)
    if record.get('error'): print('K ERROR', str(record['error'])[:300],flush=True)
