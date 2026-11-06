import sys
import os
sys.path[:0] = [os.path.join(os.getcwd(), p) for p in ("src", "tests", "tools")]
import jax
import jax.numpy as jnp
import numpy as np
from contract_support import EDGES, normal_draw, normal_float32, orders
from test_contracts_recurrence import delta_rule_draw, flattened, plain_delta_rule
from dew.nn.kernels import recurrence

mode = sys.argv[1] if len(sys.argv) > 1 else "trace"
primals, direction, cotangent = delta_rule_draw(
    2669, 2.0**63, key_heads=1, value_heads=2, width=128, length=1, per_key=False)
(skip,) = normal_draw(2672, (2,))
(skip_direction,) = normal_draw(2673, (2,))
primals = (*primals, skip)
direction = (*direction, skip_direction)
cotangent = jnp.concatenate([cotangent[0][:, 0].ravel(), cotangent[1].ravel()])
hooks = recurrence.Hooks(decay="scalar", skip=True)

def fn(*x):
    query, key, value, g, beta, state, skip = x
    operands = recurrence.Step(state, query[:, 0], key[:, 0], value[:, 0],
                               log_decay=g[:, 0], beta=beta[:, 0], skip=skip)
    out, _, updated = recurrence.step_reference(hooks, operands)
    return jnp.concatenate([out.ravel(), updated.ravel()])

def grad(*x):
    return jax.grad(lambda *p: jnp.sum(fn(*p) * cotangent),
                    argnums=tuple(range(7)))(*x)

def gg(*x):
    return jax.grad(lambda *p: sum(jnp.sum(a*b) for a,b in zip(direction, grad(*p))),
                    argnums=tuple(range(7)))(*x)

def wide(x):
    return x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x

def plain(*x):
    q,k,v,g,b,s,d = x
    out,held = plain_delta_rule(q,k,v,g,b,s,1.0)
    return jnp.concatenate([(out[:,0]+d[:,None]*v[:,0]).ravel(),held.ravel()])

if mode == "trace":
    plain_gg = orders(plain,primals,direction,cotangent)["grad of grad"]()
    closed = jax.make_jaxpr(gg)(*primals)
    env32, env64 = {}, {}
    intermediates = {}
    with jax.enable_x64():
        direction, cotangent = tuple(wide(x) for x in direction), wide(cotangent)
        closed64 = jax.make_jaxpr(gg)(*(wide(x) for x in primals))
        print("K SIZES",len(closed.jaxpr.eqns),len(closed64.jaxpr.eqns))
        for var, x in zip(closed.jaxpr.invars, primals):
            env32[var] = x
        for var, x in zip(closed64.jaxpr.invars, primals):
            env64[var] = wide(x)
        for var, x in zip(closed.jaxpr.constvars, closed.consts):
            env32[var] = x
        for var, x in zip(closed64.jaxpr.constvars, closed64.consts):
            env64[var] = x
        def read(env, v, iswide=False):
            if isinstance(v, jax.extend.core.Literal):
                x = jnp.asarray(v.val)
                return wide(x) if iswide else x
            return env[v]
        for i, (eq, eq64) in enumerate(zip(closed.jaxpr.eqns, closed64.jaxpr.eqns)):
            args32 = [read(env32,v) for v in eq.invars]
            args64 = [read(env64,v) for v in eq64.invars]
            out32 = eq.primitive.bind(*args32, **eq.params)
            out64 = eq64.primitive.bind(*args64, **eq64.params)
            if i in (209,210,232,233,239):
                intermediates[i] = (args32,args64,out32,out64)
            if not eq.primitive.multiple_results:
                out32, out64 = [out32], [out64]
            for v,v64,a,b in zip(eq.outvars,eq64.outvars,out32,out64):
                env32[v], env64[v64] = a,b
                a,b = np.asarray(a),np.asarray(b)
                if np.issubdtype(a.dtype,np.floating):
                    bad = normal_float32(b) & ~np.isfinite(a)
                    if bad.any() and False:
                        print("K BAD",i,str(eq), "count",bad.sum(),"wide",b[bad][:4],
                              "input_max", [float(np.max(np.abs(np.asarray(t)))) for t in args64])
        print("K OUT", [(int(np.sum(normal_float32(np.asarray(env64[v64])) & ~np.isfinite(np.asarray(env32[v]))))) for v,v64 in zip(closed.jaxpr.outvars,closed64.jaxpr.outvars)])
        a,b = env32[closed.jaxpr.outvars[2]],env64[closed64.jaxpr.outvars[2]]
        lost = normal_float32(b) & ~np.isfinite(a) & np.isfinite(plain_gg[2])
        print("K CONTRACT_LOST",np.argwhere(lost).tolist(),"wide",np.asarray(b)[lost],"plain",np.asarray(plain_gg[2])[lost])
        for pos in np.argwhere(lost):
            h,v = pos[-2:]
            for idx in (209,210,232,233,239):
                x,y,a,b = intermediates[idx]
                if idx in (209,232):
                    print("K INTER",idx,"head/value",h,v,"max_fp32",np.max(np.abs(np.asarray(a)[0,h,:,v])),
                          "max_fp64",np.max(np.abs(np.asarray(b)[0,h,:,v])),
                          "sum_fp64",np.sum(np.asarray(b)[0,h,:,v]))
                else:
                    print("K INTER",idx,"head/value",h,v,"fp32",np.asarray(a).reshape(2,128)[h,v],"fp64",np.asarray(b).reshape(2,128)[h,v])
            t32 = np.asarray(intermediates[209][2])[0,h,:,v]
            t64 = np.asarray(intermediates[209][3])[0,h,:,v]
            all_terms = intermediates[209][2]
            print("K SUM_PROOF",h,v,"fp32",jnp.sum(all_terms,axis=-2)[0,h,v],
                  "fp64",np.sum(t64),"scaled_fp32",(jnp.sum(all_terms/128,axis=-2)*128)[0,h,v],
                  "prefix_max_fp64",np.max(np.abs(np.cumsum(t64))),
                  "block8_max_fp64",np.max(np.abs(np.sum(t64.reshape(-1,8),axis=1))))
            q,k,v0,g,b,s,sk = [np.asarray(x) for x in primals]
            dq,dk,dv,dg,db,ds,dsk = [np.asarray(x) for x in direction]
            co = np.asarray(cotangent[:256]).reshape(1,2,128)
            cs = np.asarray(cotangent[256:]).reshape(1,2,128,128)
            r = cs[0,h,:,v] + q[0,0,0,:].astype(np.float64)*co[0,h,v]
            p = dk[0,0,0,:]*b[0,0,h]+k[0,0,0,:]*db[0,0,h]
            print("K MATCH_209",np.max(np.abs(r*p-t64))/np.max(np.abs(t64)),
                  "db",db[0,0,h],"beta",b[0,0,h])
        for i,eq in enumerate(closed.jaxpr.eqns):
            if False:
                print("K EQ",i,str(eq).replace("\n", " "), "IN",[(str(v),v.aval.shape) for v in eq.invars],
                      "OUT",[str(v) for v in eq.outvars])
else:
    a = gg(*primals)
    with jax.enable_x64():
        b = orders(fn,tuple(wide(x) for x in primals),tuple(wide(x) for x in direction),wide(cotangent))["grad of grad"]()
    print("K RESULT",[(i,int(np.sum(normal_float32(y) & ~np.isfinite(x)))) for i,(x,y) in enumerate(zip(a,b))])
