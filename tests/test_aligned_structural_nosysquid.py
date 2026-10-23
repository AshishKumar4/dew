"""Fixed-draw diagnostic grid; no production bound or product code is changed."""
import dataclasses
import functools
import time

import jax
import numpy as np
import pytest
import test_recurrence_chunk as token
import test_recurrence_stability as st
from reference_error import widened, with_gradients

from dew.nn.kernels import recurrence_chunk as c

GEOMETRIES = [pytest.param(width, size, id=f"w{width}-c{size}")
              for width in (16, 64, 128) for size in (16, 32, 64)]


@pytest.mark.parametrize("width,size", GEOMETRIES)
@pytest.mark.parametrize("name,seeds", [pytest.param("scalar delta", 40, id="scalar"),
                                       pytest.param("gated", 320, id="gated")])
def test_structural(width, size, name, seeds, monkeypatch):
    monkeypatch.setattr(token, "KEY_WIDTH", width)
    monkeypatch.setattr(c, "held_chunk", lambda size, width: size)
    hooks = st.SITES[name]
    if name == "scalar delta":
        hooks = dataclasses.replace(hooks, scale=width**-.5)
    ours = jax.jit(with_gradients(functools.partial(c.chunk, hooks, chunk_size=size, ported=False)))
    reference = jax.jit(with_gradients(functools.partial(token.token_by_token, hooks)))
    mine, theirs = {}, {}
    start = time.monotonic()
    with jax.enable_x64(True):
        for seed in range(seeds):
            with jax.enable_x64(False):
                drawn = st.cotangents(hooks, "aligned keys", seed)
            found, ref = ours(*drawn), reference(*drawn)
            exact = reference(*widened(drawn))
            pairs = jax.tree_util.tree_flatten_with_path(found)[0]
            for (path, leaf), refleaf, truth in zip(pairs, jax.tree.leaves(ref),
                                                   jax.tree.leaves(exact), strict=True):
                key = jax.tree_util.keystr(path)
                truth = np.asarray(truth, np.float64)
                mine.setdefault(key, []).append(np.mean((np.asarray(leaf, np.float64) - truth)**2))
                theirs.setdefault(key, []).append(np.mean((np.asarray(refleaf, np.float64) - truth)**2))
            if seed == 0:
                first = drawn
        for key in mine:
            own, ref = np.sqrt(np.mean(mine[key])), np.sqrt(np.mean(theirs[key]))
            print(f"GRID|{name}|width={width}|chunk={size}|seeds={seeds}|{key}"
                  f"|ours={own:.12g}|reference={ref:.12g}|ratio={own/ref:.9g}", flush=True)
            assert np.isfinite(own)
        times = []
        for _ in range(12):
            then = time.monotonic()
            jax.block_until_ready(ours(*first))
            times.append(time.monotonic()-then)
    print(f"TIME|{name}|width={width}|chunk={size}|median={np.median(times):.9g}"
          f"|total={time.monotonic()-start:.3f}", flush=True)
