"""Scratch stage backward diagnosis."""
import functools
import jax
import jax.numpy as jnp
import numpy as np
import test_multimodal_model as T
from dew.nn import linear
from dew.registry import ModelConfig


def test_stages(monkeypatch):
    events = []
    functions = {}
    counter = [0]

    def instrument(name, fn, count):
        def wrapped(*args, **kwargs):
            xs, rest = args[:count], args[count:]
            tag = name + ':' + str(counter[0]); counter[0] += 1
            def pure(*ys):
                return fn(*ys, *rest, **kwargs)
            functions[tag] = pure
            @jax.custom_vjp
            def call(*ys):
                return pure(*ys)
            def fwd(*ys):
                return pure(*ys), ys
            def bwd(ys, dy):
                y, pull = jax.vjp(pure, *ys)
                dx = pull(dy)
                jax.debug.callback(lambda a,b,c,d: events.append((tag,a,b,c,d)), ys, y, dy, dx)
                return dx
            call.defvjp(fwd,bwd)
            return call(*xs)
        return wrapped

    for name, count in [('l2norm',1),('chunk_gated_delta_rule',5),('gated_rms_normalized',3)]:
        monkeypatch.setattr(linear,name,instrument(name,getattr(linear,name),count))
    loaded, _, _ = T._family('qwen35')
    sample = loaded.processor.from_hf(T.text_rows(np.full((4,32), T.TEXT_IDS[0])))
    tree = jax.tree.structure(sample)
    def grads(inputs):
        encoded = jax.tree.unflatten(tree,[inputs[f'dew.{i}'] for i in range(tree.num_leaves)])
        return T._stepped(loaded.model,loaded.variables,encoded)[1]
    run = jax.jit(grads)
    lines=[]
    inputs=T.drawing_text('qwen35')(10)
    jax.block_until_ready(run(inputs))
    lines.append('events %d' % len(events))
    for tag, xs, y, dy, dx in events:
        fn=functions[tag]
        # Match the output dtype argument of the output norm when widening its oracle.
        if tag.startswith('gated_rms'):
            fn=lambda x,g,w: linear.gated_rms_normalized.__wrapped__(x,g,w)
            # Use the same norm formula, retaining its epsilon and silu gate.
            def fn(x,g,w):
                return x * jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+1e-6) * w * jax.nn.silu(g)
        with jax.enable_x64(True):
            x64=jax.tree.map(lambda a:jnp.asarray(a,jnp.float64),xs)
            dy64=jax.tree.map(lambda a:jnp.asarray(a,jnp.float64),dy)
            y64,pull=jax.vjp(fn,*x64)
            dx64=pull(dy64)
        yerr=np.sqrt(sum(np.sum((np.asarray(a)-np.asarray(b))**2) for a,b in zip(jax.tree.leaves(y),jax.tree.leaves(y64))))
        ynorm=np.sqrt(sum(np.sum(np.asarray(b)**2) for b in jax.tree.leaves(y64)))
        lines.append('%s forward rel %.5g' % (tag,yerr/max(ynorm,1e-300)))
        for i,(a,b) in enumerate(zip(dx,dx64)):
            a,b=np.asarray(a),np.asarray(b)
            e=np.abs(a-b); k=np.unravel_index(e.argmax(),e.shape)
            rel=np.linalg.norm(e.ravel())/max(np.linalg.norm(b.ravel()),1e-300)
            lines.append(' arg%d rel %.5g max %.5g at %s true %.5g' %(i,rel,e[k],k,b[k]))
        if tag.startswith('l2norm'):
            sq=np.sum(np.asarray(xs[0],np.float64)**2,-1)
            lines.append(' squared norm min %.5g max %.5g' %(sq.min(),sq.max()))
    raise AssertionError('PROBE\n'+'\n'.join(lines))
