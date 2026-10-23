"""Measurement script (never lands): where the fp16 huge-state family goes non-finite in the Triton path."""
import jax
import jax.numpy as jnp
import numpy as np
import test_attention_stability as suite
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention as template
from dew.nn.kernels.templates import attention_triton

family = suite.FAMILIES["fp16 values near their largest beside a state near fp32's largest"]
q, k, v, state, dout = suite._cast(family, family.draw)(0)
spans = suite._spans(family, q, k)
hooks = suite._hooks(q, state)
with tuning.pinned(suite.PINS):
    fwd = attention_triton.forward(q, k, v, spans, hooks, residuals=True, natural=True, interpret=True)
    for name, leaf in zip(["out", "m", "total", "anchors", "others", "own", "others_total", "shift"],
                          jax.tree.leaves(fwd)):
        arr = np.asarray(leaf, np.float64)
        print("forward", name, arr.dtype, "finite", bool(np.isfinite(arr).all()), "max", float(np.nanmax(np.abs(arr))))
    for split in (True, False):
        grads = attention_triton.backward(q, k, v, spans, hooks, dout, natural=True, split=split, interpret=True)
        for name, leaf in zip(["dq", "dk", "dv", "m0", "total0", "acc0"], jax.tree.leaves(grads)):
            arr = np.asarray(leaf, np.float64)
            print("split", split, name, arr.shape, np.asarray(leaf).dtype, "max", float(np.nanmax(np.abs(arr))))
print("state dtypes", [np.asarray(x).dtype for x in state], "q", np.asarray(q).dtype, "dout", np.asarray(dout).dtype)
from reference_error import with_gradients
ref = with_gradients(suite._reference(family, anchored=False))(q, k, v, state, dout)
for i, leaf in enumerate(jax.tree.leaves(ref)):
    arr = np.asarray(leaf, np.float64)
    print("reference", i, np.asarray(leaf).dtype, "finite", bool(np.isfinite(arr).all()), "max", float(np.nanmax(np.abs(arr))))
mine = with_gradients(suite._mainloops(family, True)["triton"])(q, k, v, state, dout)
for i, leaf in enumerate(jax.tree.leaves(mine)):
    arr = np.asarray(leaf, np.float64)
    print("mine", i, np.asarray(leaf).dtype, "finite", bool(np.isfinite(arr).all()), "max", float(np.nanmax(np.abs(arr))))
