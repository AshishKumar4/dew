import sys; sys.path[:0] = ["tools", "src", "tests", "."]
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from flax.traverse_util import flatten_dict, unflatten_dict
import flaxdiff_family_worker as w
import test_flaxdiff_family_source as t

mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
if mode == "fixed":
    w.FOURIER = "jnp.asarray(freqs, dtype=jnp.float64) * self.scale"
print("K MODE", mode)
case = sys.argv[1].replace("-", "/", 1)
_, cls, config, _, _ = t.CASES[case]
inputs = t.drawing(case)(0)
params = {k.removeprefix("param."): jnp.asarray(v, jnp.float64) for k,v in inputs.items() if k.startswith("param.")}
image, times, text = (jnp.asarray(inputs[k], jnp.float64) for k in ("image", "times", "text"))
mods = w.flaxdiff(True)
freqs = np.asarray(mods["common"].FourierEmbedding(features=config["emb_features"]).bind({}).freqs)
table = np.asarray(t.LAYOUT[case]["fourier"], np.float64)
print("K FOURIER_TABLE",float(np.sqrt(np.mean((freqs-table)**2))),float(np.linalg.norm(freqs-table)/np.linalg.norm(table)))
refmodel = getattr(mods[t.CASES[case][0]], cls)(**config)
reftree = unflatten_dict({tuple(k.split("/")):v for k,v in params.items()})
refout, refstate = refmodel.apply({"params":reftree}, image,times,text, capture_intermediates=True, mutable=["intermediates"])
dewmodel = t.dew_model(cls, config).clone(dtype=jnp.float64)
dewtree = t.nest({t.dew_path(cls,k):t.to_dew(cls,k,v,config.get("patch_size")) for k,v in params.items()})
constants = t.nest({(*t.fourier_path(cls),"frequencies"):jnp.asarray(t.LAYOUT[case]["fourier"],jnp.float64)})
dewout, dewstate = dewmodel.apply({"params":dewtree,"constants":constants}, image,times,t.TextContext(text,jnp.ones(text.shape[:2],jnp.float64)), capture_intermediates=True,mutable=["intermediates"])
def arrays(state):
    result={}
    for k,v in flatten_dict(state["intermediates"]).items():
        leaves = jax.tree.leaves(v)
        leaves = [a for a in leaves if hasattr(a,"dtype") and jnp.issubdtype(a.dtype,jnp.inexact)]
        if leaves: result["/".join(k[:-1])] = leaves[0]
    return result
ra,da = arrays(refstate),arrays(dewstate)
for k in ra:
    if k not in ("", "embed", "embed/patch_embed", "conditioning", "conditioning/time_embed",
                 "conditioning/time_embed/layers_0", "conditioning/time_embed/layers_1",
                 "conditioning/text_context_proj", "dit_block_0"):
        continue
    if k in da and ra[k].shape == da[k].shape:
        a,b = np.asarray(ra[k]),np.asarray(da[k])
        print("K INTER",k,a.dtype,b.dtype,"rms",float(np.sqrt(np.mean((a-b)**2))),"rel",float(np.linalg.norm(a-b)/np.linalg.norm(a)))
    elif k == "conditioning/text_context_proj":
        a,b=np.mean(np.asarray(ra[k]),axis=1),np.asarray(da[k])
        print("K TEXT_POOLED",float(np.sqrt(np.mean((a-b)**2))),float(np.linalg.norm(a-b)/np.linalg.norm(a)))
    else: print("K UNMATCHED",k,ra[k].shape)
with w.applying(case,"float64") as apply:
    ref = apply(params,{k:v for k,v in inputs.items() if not k.startswith("param.")})
ours = t.computing(case)(jnp.float64,inputs)
for k in ours:
    a,b=np.asarray(ref[k]),np.asarray(ours[k])
    print("K FINAL",k,"rms",float(np.sqrt(np.mean((a-b)**2))),"rel",float(np.linalg.norm(a-b)/np.linalg.norm(a)))
    if mode == "fixed":
        assert np.linalg.norm(a-b)/np.linalg.norm(a) < 1e-14, k
