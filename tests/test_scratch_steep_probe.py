import functools

import jax
import numpy as np
import pytest
import test_recurrence_stability as st
from reference_error import widened
from test_recurrence_chunk import token_by_token

from dew.nn.kernels.recurrence_chunk import chunk


@pytest.mark.parametrize("name", ["scalar delta", "gated"])
def test_probe(name):
    hooks = st.SITES[name]
    reference = jax.jit(st.pulled(functools.partial(token_by_token, hooks)))
    arms = {arm: jax.jit(st.pulled(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=arm == "port")))
            for arm in ("portable", "port")}
    lines = []
    for seed in range(4):
        (operands,) = st.stressed(hooks, "steep then mild", seed)
        with jax.enable_x64(True):
            truth = reference(*widened((operands,)))
        theirs = reference(operands)
        mine = {arm: fn(operands) for arm, fn in arms.items()}
        for field in truth:
            t = np.asarray(truth[field], np.float64)
            def rms(x):
                return float(np.sqrt(np.mean((np.asarray(x, np.float64) - t) ** 2)))
            lines.append(f"{seed} {field}: ref {rms(theirs[field]):.3e} portable {rms(mine['portable'][field]):.3e} "
                         f"port {rms(mine['port'][field]):.3e} scale {float(np.sqrt(np.mean(t ** 2))):.3e}")
    raise AssertionError("PROBE|" + name + "\n" + "\n".join(lines))
