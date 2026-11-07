import sys; sys.path[:0] = ["tools", "src", "tests", "."]
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from flax.traverse_util import flatten_dict, unflatten_dict
import flaxdiff_family_worker as w
import test_flaxdiff_family_source as t

case = sys.argv[1].replace("-", "/", 1)
mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
_, cls, config, _, _ = t.CASES[case]
inputs = t.drawing(case)(0)
params = {k.removeprefix("param."): jnp.asarray(v, jnp.float64)
          for k,v in inputs.items() if k.startswith("param.")}
image, times, text = (jnp.asarray(inputs[k], jnp.float64) for k in ("image", "times", "text"))
mods = w.flaxdiff(True)
refmodel = getattr(mods[t.CASES[case][0]], cls)(**config)
reftree = unflatten_dict({tuple(k.split("/")):v for k,v in params.items()})
dewmodel = t.dew_model(cls, config)
if mode == "clone":
    dewmodel = dewmodel.clone(dtype=jnp.float64)
dewtree = t.nest({t.dew_path(cls,k):t.to_dew(cls,k,v,config.get("patch_size")) for k,v in params.items()})
constants = t.nest({(*t.fourier_path(cls),"frequencies"):jnp.asarray(t.LAYOUT[case]["fourier"],jnp.float64)})
def rf(tree, image, text):
    return refmodel.apply({"params": tree},image,times,text)
def df(tree, image, text):
    return dewmodel.apply({"params":tree,"constants":constants},image,times,t.TextContext(text,jnp.ones(text.shape[:2],jnp.float64)))
ra, rpb = jax.vjp(rf,reftree,image,text)
da, dpb = jax.vjp(df,dewtree,image,text)
rg,ri,rt = rpb(jnp.asarray(inputs["probe"],jnp.float64))
dg,di,dt = dpb(jnp.asarray(inputs["probe"],jnp.float64))
def report(label, a, b):
    a,b=np.asarray(a),np.asarray(b)
    delta=a-b
    print("K",mode,label,"RMS",float(np.sqrt(np.mean(delta**2))),"REL",float(np.linalg.norm(delta)/(np.linalg.norm(a) or 1)))
report("output",ra,da)
report("image",ri,di)
report("text",rt,dt)
names=sorted(params)
rflat=flatten_dict(rg)
rr=[ri,rt]
dd=[di,dt]
for k in names:
    a=rflat[tuple(k.split("/"))]
    b=t.from_dew(cls,k,t.node(dg,t.dew_path(cls,k)),t.LAYOUT[case]["params"][k])
    rr.append(a); dd.append(b)
    rms=float(np.sqrt(np.mean(np.asarray(a-b)**2)))
    if rms>1e-12:
        report("PARAM "+k,a,b)
report("gradients",jnp.concatenate([a.ravel() for a in rr]),jnp.concatenate([a.ravel() for a in dd]))
refout, refstate=refmodel.apply({"params":reftree},image,times,text,capture_intermediates=True,mutable=["intermediates"])
dewout, dewstate=dewmodel.apply({"params":dewtree,"constants":constants},image,times,t.TextContext(text,jnp.ones(text.shape[:2],jnp.float64)),capture_intermediates=True,mutable=["intermediates"])
def arrays(state):
    result={}
    for k,v in flatten_dict(state["intermediates"]).items():
        leaves=[a for a in jax.tree.leaves(v) if hasattr(a,"dtype") and jnp.issubdtype(a.dtype,jnp.inexact)]
        if leaves: result["/".join(k[:-1])]=leaves[0]
    return result
rc,dc=arrays(refstate),arrays(dewstate)
for k in rc:
    if k in dc and rc[k].shape==dc[k].shape and (k.count("/")<1 or "norm" in k.lower()):
        report("INTER "+k,rc[k],dc[k])
