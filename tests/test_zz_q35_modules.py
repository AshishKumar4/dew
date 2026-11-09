"""Scratch residual and norm sensitivity."""
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import test_multimodal_model as T
from dew.registry import ModelConfig
from arm_arithmetic import summed_as_run


def test_modules():
    events=[]
    def tap(x,label):
        @jax.custom_vjp
        def identity(x):return x
        def fwd(x):return x,x
        def bwd(x,dy):
            jax.debug.callback(lambda a,b:events.append((label,np.asarray(a),np.asarray(b))),x,dy)
            return (dy,)
        identity.defvjp(fwd,bwd)
        return identity(x)
    def interceptor(next_fun,args,kwargs,context):
        chosen=type(context.module).__name__ in ('RMSNorm','RMSNormGated','GatedDeltaNet','MultiHeadDotProductAttention') and context.method_name=='__call__'
        label='/'.join(context.module.scope.path)
        if chosen:args=(tap(args[0],label+':in'),*args[1:])
        result=next_fun(*args,**kwargs)
        return tap(result,label+':out') if chosen and isinstance(result,jax.Array) else result
    loaded,_,_=T._family('qwen35');inputs=T.drawing_text('qwen35')(10)
    sample=loaded.processor.from_hf(T.text_rows(inputs['ids']))
    def run(dtype):
        model=loaded.model if dtype==jnp.float32 else ModelConfig.from_model(loaded.model).with_dtype('float64').build()
        variables=jax.tree.map(lambda a:jnp.asarray(a,dtype) if jnp.issubdtype(jnp.asarray(a).dtype,jnp.floating) else a,loaded.variables)
        with nn.intercept_methods(interceptor):
            return jax.jit(lambda v:T._stepped(model,v,sample)[1])(variables)
    with summed_as_run():jax.block_until_ready(run(jnp.float32))
    a=dict((label,(x,dx)) for label,x,dx in events);events.clear()
    with jax.enable_x64(True):jax.block_until_ready(run(jnp.float64))
    b=dict((label,(x,dx)) for label,x,dx in events)
    def rel(a,b):return np.linalg.norm((a.astype(np.float64)-b).ravel())/max(np.linalg.norm(b.ravel()),1e-300)
    lines=[]
    for label in sorted(a):
        x,dx=a[label];y,dy=b[label];t=(3,0)
        lines.append('%s x %.3g dx %.3g target x %.3g dx %.3g rms %.3g grad %.3g'%(label,rel(x,y),rel(dx,dy),rel(x[t],y[t]),rel(dx[t],dy[t]),np.sqrt(np.mean(y[t]**2)),np.linalg.norm(dy[t])))
    raise AssertionError('PROBE\n'+'\n'.join(lines))
