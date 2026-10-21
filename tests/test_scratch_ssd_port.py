import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_mamba_ssd as t

from dew.nn.mixers.mamba2 import chunk_ssd


@pytest.mark.parametrize("first", [True, False], ids=["row starts a document", "row continues"])
@pytest.mark.parametrize("width, state", [(32, 32), (24, 40)])
def test_port_against_portable(first, width, state):
    operands, starts = t.scan_operands(2, 48, 4, width, 2, state, documents=True, seed=3)
    starts = starts.at[0, 0].set(first)
    found = {}
    for port in (True, False):
        with t.row_names_port(port=port):
            out, final = jax.jit(lambda *a: chunk_ssd(*a, chunk_size=16, starts=starts))(*operands)
        found[port] = (np.asarray(out), np.asarray(final))
    lines = [f"{name} max {np.max(np.abs(a - b)):.2e} per row "
             + " ".join(f"{np.max(np.abs(a[r] - b[r])):.1e}" for r in range(2))
             for name, a, b in zip(("out", "final"), found[True], found[False], strict=True)]
    raise AssertionError(f"PORT|{width},{state},{first}|" + "; ".join(lines))
