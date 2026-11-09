"""Scratch forward/backward sensitivity at one real token."""
import jax
import jax.numpy as jnp
import numpy as np
import test_multimodal_model as T
from dew.nn import linear
from dew.registry import ModelConfig


def test_global(monkeypatch):
    events=[]; counter=[0]
    def instrument(name,fn,count):
        def wrapped(*args,**kwargs):
            xs,rest=args[:count],args[count:]
            tag=(counter[0],name); counter[0]+=1
            def pure(*ys): return fn(*ys,*rest,**kwargs)
            @jax.custom_vjp
            def call(*ys): return pure(*ys)
            def fwd(*ys): return pure(*ys),ys
            def bwd(ys,dy):
                y,pull=jax.vjp(pure,*ys); dx=pull(dy)
                jax.debug.callback(lambda a,b,c,d: events.append((tag,a,b,c,d)),ys,y,dy,dx)
                return dx
            call.defvjp(fwd,bwd)
            return call(*xs)
        return wrapped
    for name,count in [('l2norm',1),('chunk_gated_delta_rule',5),('gated_rms_normalized',3)]:
        monkeypatch.setattr(linear,name,instrument(name,getattr(linear,name),count))
    loaded,_,_=T._family('qwen35'); inputs=T.drawing_text('qwen35')(10)
    sample=loaded.processor.from_hf(T.text_rows(inputs['ids']))
    def run(dtype):
        model=loaded.model if dtype==jnp.float32 else ModelConfig.from_model(loaded.model).with_dtype('float64').build()
        variables=jax.tree.map(lambda a:jnp.asarray(a,dtype) if jnp.issubdtype(jnp.asarray(a).dtype,jnp.floating) else a,loaded.variables)
        return jax.jit(lambda v:T._stepped(model,v,sample)[1])(variables)
    jax.block_until_ready(run(jnp.float32)); a=sorted(events,key=lambda e:e[0][0]);events.clear()
    with jax.enable_x64(True): jax.block_until_ready(run(jnp.float64))
    b=sorted(events,key=lambda e:e[0][0])
    lines=['events %d/%d'%(len(a),len(b))]
    def rel(a,b): return np.linalg.norm((np.asarray(a,np.float64)-b).ravel())/max(np.linalg.norm(np.asarray(b).ravel()),1e-300)
    for (ta,xa,ya,dya,dxa),(tb,xb,yb,dyb,dxb) in zip(a,b):
        xa,ya,dya,dxa,xb,yb,dyb,dxb=jax.tree.map(np.asarray,(xa,ya,dya,dxa,xb,yb,dyb,dxb))
        name=ta[1]
        if name=='chunk_gated_delta_rule':ya,yb=ya[0],yb[0];dya,dyb=dya[0],dyb[0]
        target=(3,0)
        lines.append('%s: input %.3g output %.3g dy %.3g dx %.3g target x %.3g y %.3g dy %.3g dx %.3g' %(name,rel(xa[0],xb[0]),rel(ya,yb),rel(dya,dyb),rel(dxa[0],dxb[0]),rel(xa[0][target],xb[0][target]),rel(ya[target],yb[target]),rel(dya[target],dyb[target]),rel(dxa[0][target],dxb[0][target])))
        if name=='chunk_gated_delta_rule':
            q,k,v,g,beta=xa
            dot=np.sum(q[3,0].astype(np.float64)*k[3,0],-1)
            lines.append(' qk dot %s q norm %s k norm %s out rms %s' %(dot.tolist(),np.linalg.norm(q[3,0],axis=-1).tolist(),np.linalg.norm(k[3,0],axis=-1).tolist(),np.sqrt(np.mean(yb[3,0]**2,-1)).tolist()))
    raise AssertionError('PROBE\n'+'\n'.join(lines))
