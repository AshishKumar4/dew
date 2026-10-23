"""Single-draw aligned-key precision localization; CPU only, not a product test."""
import functools
import dataclasses
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import test_recurrence_stability as st
from reference_error import widened, with_gradients
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import recurrence_algebra as a, recurrence_chunk as c
from dew.nn.kernels.recurrence_chunk import chunk


def wide(fn):
    def run(*args, **kwargs):
        dtype = next(x.dtype for x in jax.tree.leaves(args)
                     if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating))
        with jax.enable_x64(True):
            args = jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, "dtype")
                                and jnp.issubdtype(x.dtype, jnp.floating) else x, args)
            out = fn(*args, **kwargs)
            return jax.tree.map(lambda x: x.astype(dtype) if hasattr(x, "dtype")
                                and jnp.issubdtype(x.dtype, jnp.floating) else x, out)
    return run


def rms(x):
    return np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))


def run(mode, name, seed):
    start = time.monotonic()
    if "DEW_PROBE_WIDTH" in os.environ:
        import test_recurrence_chunk as token
        token.KEY_WIDTH = int(os.environ["DEW_PROBE_WIDTH"])
        st.SITES["scalar delta"] = dataclasses.replace(st.SITES["scalar delta"],
                                                       scale=token.KEY_WIDTH**-.5)
    if "DEW_PROBE_CHUNK" in os.environ:
        st.CHUNK = int(os.environ["DEW_PROBE_CHUNK"])
        c.held_chunk = lambda size, width: size
    with jax.enable_x64(False):
        drawn = st.cotangents(st.SITES[name], "aligned keys", seed)
    jax.config.update("jax_enable_x64", True)
    hooks = st.SITES[name]
    ref = jax.jit(with_gradients(functools.partial(token_by_token, hooks)))
    ref32 = ref(*drawn)
    with jax.enable_x64(True):
        truth = ref(*widened(drawn))
    if mode == "chunk-adjoint":
        from probe_aligned_chunk_adjoint_nosysquid import chunk_adjoint
        c.chunk_delta_rule = chunk_adjoint
    elif mode.startswith("comp:"):
        import probe_aligned_dots_nosysquid as dots
        from probe_aligned_compensated_nosysquid import dot
        dots.wide_dot = dot
        dots.install(mode[5:])
    elif mode == "compall":
        from probe_aligned_compensated_nosysquid import dot
        a.float32_matmul = dot
    elif mode.startswith("wide:"):
        from probe_aligned_dots_nosysquid import install
        install(mode[5:])
    elif mode == "chronological":
        from probe_aligned_jvp_nosysquid import stable_jvp
        c.chunk_delta_rule = stable_jvp
    elif mode == "regroup":
        from probe_aligned_regroup_nosysquid import scalar_regroup
        c.chunk_delta_rule = scalar_regroup
    elif mode == "factor_decay":
        original_factor, original_intra = a.factor, a.intra
        captured = []

        def intra(key, keyed, pairs, issue=a.portable):
            captured.append(pairs)
            return original_intra(key, keyed, jnp.ones_like(pairs), issue)

        def factor(matrix, issue=a.portable):
            pairs = captured.pop()
            return original_factor(matrix, issue) * pairs[..., 0]

        a.intra, a.factor = intra, factor
    elif mode.startswith("dot:"):
        selected = mode[4:].split(",")
        original = a.portable
        widened_dot = wide(original)

        def issue(name, left, right):
            return (widened_dot if name in selected else original)(name, left, right)

        for piece_name in ("factor", "paired", "intra", "apply", "carry", "readout"):
            fn = getattr(a, piece_name)
            fn.__defaults__ = (*fn.__defaults__[:-1], issue)
    else:
        for piece in mode.split("+"):
            if piece != "base":
                target = c if piece == "l2norm" else c.recurrence if piece == "gated" else a
                setattr(target, piece, wide(getattr(target, piece)))
    mine = jax.jit(with_gradients(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=False)))(*drawn)
    leaves = jax.tree_util.tree_flatten_with_path(mine)[0]
    refs, exacts = jax.tree.leaves(ref32), jax.tree.leaves(truth)
    for (path, ours), refleaf, exact in zip(leaves, refs, exacts, strict=True):
        err = rms(np.asarray(ours, np.float64) - np.asarray(exact))
        referr = rms(np.asarray(refleaf, np.float64) - np.asarray(exact))
        print(f"{mode}|{name}|seed={seed}|{jax.tree_util.keystr(path)}"
              f"|ours={err:.10g}|ref={referr:.10g}|ratio={err/referr:.8g}", flush=True)
    print(f"seconds={time.monotonic()-start:.2f}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "scalar delta",
        int(sys.argv[3]) if len(sys.argv) > 3 else 0)
