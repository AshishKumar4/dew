import sys
import ctypes
import struct
import platform

variant, packed = sys.argv[1], sys.argv[2] == "1"
if variant == "ftz":
    libm = ctypes.CDLL("libm.so.6")
    env = ctypes.create_string_buffer(32)
    libm.fegetenv(env)
    prior = struct.unpack_from("I", env, 28)[0]
    struct.pack_into("I", env, 28, prior | 0x8040)
    libm.fesetenv(env)
    print("K fp", platform.machine(), hex(prior), hex(struct.unpack_from("I", env, 28)[0]), flush=True)

import statistics
import time
sys.path[:0] = ['src', 'tests', 'tools']
import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.algebra import scan

if variant == "lateftz":
    from dew.nn.precision import vanishing
    # Populate numpy.finfo while gradual underflow is still enabled.
    print("K vanish",vanishing(jnp.float32,False),flush=True)
    libm=ctypes.CDLL("libm.so.6")
    env=ctypes.create_string_buffer(32)
    libm.fegetenv(env)
    prior=struct.unpack_from("I",env,28)[0]
    struct.pack_into("I",env,28,prior|0x8040)
    libm.fesetenv(env)
    print("K latefp",platform.machine(),hex(prior),flush=True)

if variant == "ordinary":
    scan._decays_of = scan._decays_value
if variant == "stopbetween":
    original = scan._decays_of
    def changed(x, base2):
        d = original(x, base2)
        return d._replace(between=jax.lax.stop_gradient(d.between))
    scan._decays_of = changed
if variant in ("zerodecay", "simpledecay", "barrier", "barriereach", "barrierbetween", "leftonly", "spanonly"):
    original_backward = scan._decays_backward
    def changed_backward(base2, decays, cotangents):
        if variant == "zerodecay":
            return (jnp.zeros_like(decays.entering),)
        if variant in ("barrier", "barriereach", "barrierbetween", "leftonly", "spanonly"):
            weight = np.log(2.) if base2 else 1.
            def summed(x, reverse=False):
                result=scan._scanned(x,reverse=reverse)
                return jax.lax.optimization_barrier(result) if variant=="barriereach" else result
            entering = cotangents.entering * decays.entering
            to_last = cotangents.to_last * decays.to_last
            between = cotangents.between * decays.between
            if variant=="barrierbetween":
                between=jax.lax.optimization_barrier(between)
            left=jnp.swapaxes(scan._earlier(summed(jnp.swapaxes(between,0,1))),0,1)
            if variant=="barrier":
                left=jax.lax.optimization_barrier(left)
            spanning=jnp.diagonal(summed(left,True),axis1=0,axis2=1).T
            if variant=="leftonly":
                spanning=jnp.sum(left,axis=0)
            if variant=="spanonly":
                spanning=jnp.diagonal(summed(between,True),axis1=0,axis2=1).T
            return ((summed(entering,True)+scan._earlier(summed(to_last))+spanning+
                     (cotangents.whole*decays.whole)[None])*weight,)
        weight = np.log(2.) if base2 else 1.
        def summed(x, reverse=False):
            return jnp.flip(jnp.cumsum(jnp.flip(x,0),axis=0),0) if reverse else jnp.cumsum(x,axis=0)
        entering = cotangents.entering * decays.entering
        to_last = cotangents.to_last * decays.to_last
        between = cotangents.between * decays.between
        left = jnp.swapaxes(scan._earlier(summed(jnp.swapaxes(between,0,1))),0,1)
        spanning = jnp.diagonal(summed(left,True),axis1=0,axis2=1).T
        return ((summed(entering,True)+scan._earlier(summed(to_last))+spanning+
                 (cotangents.whole*decays.whole)[None])*weight,)
    scan._decays_backward = changed_backward

rng = np.random.default_rng(0)
rows, length, heads, width = 2, 1024, 4, 128
def unit(x):
    return x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True))
q, k = (jnp.asarray(unit(rng.standard_normal((rows,length,heads,width))), jnp.float32) for _ in range(2))
v = jnp.asarray(rng.standard_normal((rows,length,heads,width)), jnp.float32)
g = jnp.asarray(-np.logaddexp(0,rng.standard_normal((rows,length,heads,width))+1)*0.1,jnp.float32)
if variant == "mild":
    g = g * 0.1
if variant == "strong":
    g = g * 4
beta = jnp.asarray(1/(1+np.exp(-rng.standard_normal((rows,length,heads)))),jnp.float32)
starts = None
if packed:
    starts=np.zeros((rows,length),bool)
    starts[:,[0,300,700]]=True
    starts=jnp.asarray(starts)
hooks=recurrence.Hooks(decay="vector",scale=width**-0.5,write_bound=1.)
def forward(q,k,v,g,beta):
    with generation.chosen(gated_delta_rule="derived"):
        return chunk(hooks,Chunk(q,k,v,log_decay=g,beta=beta,starts=starts))
def loss(*xs):
    out,final=forward(*xs)
    return jnp.sum(out*out)+jnp.sum(final)
xs=(q,k,v,g,beta)
if variant == "proof":
    import inspect
    original=scan._decays_backward
    source=inspect.getsource(original)
    before="between = cotangents.between * decays.between"
    after="between = lax.optimization_barrier(cotangents.between * decays.between)"
    assert source.count(before)==1
    compiled=jax.jit(jax.grad(loss,argnums=(0,1,2,3,4))).lower(*xs).compile()
    start=time.perf_counter()
    red=jax.block_until_ready(compiled(*xs))
    red_ms=(time.perf_counter()-start)*1000
    print("K proof red",int(packed),round(red_ms,1),"under5s",red_ms<5000,flush=True)
    namespace=dict(scan.__dict__)
    exec(source.replace(before,after),namespace)
    scan._decays_backward=namespace["_decays_backward"]
    jax.clear_caches()
    compiled2=jax.jit(jax.grad(loss,argnums=(0,1,2,3,4))).lower(*xs).compile()
    start=time.perf_counter()
    green=jax.block_until_ready(compiled2(*xs))
    green_ms=(time.perf_counter()-start)*1000
    print("K proof green",int(packed),round(green_ms,1),"under5s",green_ms<5000,flush=True)
    for name,a,b in zip(("q","k","v","g","beta"),red,green):
        a,b=np.asarray(a),np.asarray(b)
        print("K proof equality",int(packed),name,np.array_equal(a,b),"maxdiff",float(np.max(np.abs(a-b))),flush=True)
        if name=="g":
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=3e-7)
        else:
            np.testing.assert_array_equal(a,b)
    print("K proof costs",int(packed),"red",{k:compiled.cost_analysis()[k] for k in ("flops","bytes accessed")},
          "green",{k:compiled2.cost_analysis()[k] for k in ("flops","bytes accessed")},flush=True)
    assert red_ms>5000 and green_ms<5000
    print("K proof PASS",int(packed),flush=True)
    sys.exit()
argnums=(0,1,2,4) if variant == "nog" else (3,) if variant == "onlyg" else (0,1,2,3,4)
for name,fn in (("fwd",jax.jit(forward)),("bwd",jax.jit(jax.grad(loss,argnums=argnums)))):
    if variant == "profile" or len(sys.argv)>3:
        import re
        executable=fn.lower(*xs).compile()
        text=executable.as_text()
        cost=executable.cost_analysis()
        print("K cost",variant,int(packed),name,{k:cost[k] for k in ("flops","bytes accessed","transcendentals") if k in cost},flush=True)
        sizes={}
        for ty,shape in re.findall(r'\b(f32|f64|s32|pred)\[([0-9,]+)\]',text):
            count=int(np.prod([int(d) for d in shape.split(",")]))*(8 if ty=="f64" else 1 if ty=="pred" else 4)
            sizes[(ty,shape)]=count
        print("K shapes",int(packed),name,sorted(sizes.items(),key=lambda p:p[1],reverse=True)[:10],flush=True)
        print("K hlo",variant,int(packed),name,"chars",len(text),"ops",[(op,text.count(" "+op+"(")) for op in ("while","dot","scatter","fusion","copy","reduce")],flush=True)
        biggest=sorted((line for line in text.splitlines() if " fusion(" in line),key=len,reverse=True)[:3]
        for line in biggest:
            operands=line.split(" fusion(",1)[1].split("),",1)[0]
            print("K fusion",variant,int(packed),name,"operands",operands.count("%"),line.split('metadata=',1)[-1][-300:],flush=True)
    for _ in range(2):
        jax.block_until_ready(fn(*xs))
    times=[]
    for _ in range(3):
        start=time.perf_counter()
        jax.block_until_ready(fn(*xs))
        times.append((time.perf_counter()-start)*1000)
    print("K",variant,int(packed),name,round(statistics.median(times),1),flush=True)
