"""Scratch: exact backward at the float32 program's rounded module boundaries."""
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import test_multimodal_model as T
from dew.registry import ModelConfig
from arm_arithmetic import summed_as_run


def test_shadow():
    stored={};mode=['capture']
    def transform(value,key):
        leaves,tree=jax.tree.flatten(value)
        found=[]
        for i,x in enumerate(leaves):
            if isinstance(x,jax.Array) and jnp.issubdtype(x.dtype,jnp.floating):
                tag=key+':'+str(i)
                if mode[0]=='capture':
                    jax.debug.callback(lambda a,t=tag:stored.__setitem__(t,np.asarray(a)),x)
                elif tag in stored:
                    x=x+jax.lax.stop_gradient(jnp.asarray(stored[tag],x.dtype)-x)
            found.append(x)
        return jax.tree.unflatten(tree,found)
    def interceptor(next_fun,args,kwargs,ctx):
        if ctx.method_name not in ('__call__','seam'):return next_fun(*args,**kwargs)
        tag='/'.join(ctx.module.scope.path)+'.'+ctx.method_name
        args=transform(args,tag+':in')
        return transform(next_fun(*args,**kwargs),tag+':out')
    loaded,_,_=T._family('qwen35');inputs=T.drawing_text('qwen35')(10)
    sample=loaded.processor.from_hf(T.text_rows(inputs['ids']))
    def run(dtype,intercept):
        model=loaded.model if dtype==jnp.float32 else ModelConfig.from_model(loaded.model).with_dtype('float64').build()
        variables=jax.tree.map(lambda a:jnp.asarray(a,dtype) if jnp.issubdtype(jnp.asarray(a).dtype,jnp.floating) else a,loaded.variables)
        def compute(v):return T._stepped(model,v,sample)[1]
        if intercept:
            with nn.intercept_methods(interceptor):return jax.jit(compute)(variables)
        return jax.jit(compute)(variables)
    with summed_as_run():g32=jax.block_until_ready(run(jnp.float32,True))
    count=len(stored)
    with jax.enable_x64(True):
        g64=jax.block_until_ready(run(jnp.float64,False))
        mode[0]='replay'
        shadow=jax.block_until_ready(run(jnp.float64,True))
    rows=[]
    for (path,a),b,c in zip(jax.tree_util.tree_leaves_with_path(g32),jax.tree.leaves(g64),jax.tree.leaves(shadow)):
        a,b,c=map(np.asarray,(a,b,c));norm=max(np.linalg.norm(b.ravel()),1e-300)
        err=np.linalg.norm((a-b).ravel())/norm
        residual=np.linalg.norm((a-c).ravel())/norm
        rows.append((err,jax.tree_util.keystr(path),residual,np.linalg.norm((c-b).ravel())/norm))
    rows.sort(reverse=True)
    lines=['rounded boundaries %d'%count]
    for err,path,residual,predicted in rows[:6]:lines.append('%s original %.5g residual %.5g shadow-shift %.5g'%(path,err,residual,predicted))
    raise AssertionError('PROBE\n'+'\n'.join(lines))
