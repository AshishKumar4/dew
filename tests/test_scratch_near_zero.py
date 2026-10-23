import functools

import jax
import jax.numpy as jnp
import numpy as np
import test_recurrence_stability as st
from reference_error import widened, with_gradients
from test_recurrence_chunk import token_by_token

from dew.nn.kernels.recurrence_chunk import chunk


def test_where():
    hooks = st.SITES["scalar delta"]
    regime = "decays near zero"
    port = jax.jit(with_gradients(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=True)))
    portable = jax.jit(with_gradients(functools.partial(chunk, hooks, chunk_size=st.CHUNK, ported=False)))
    reference = jax.jit(with_gradients(functools.partial(token_by_token, hooks)))
    lines = []
    sums = {}
    for seed in range(16):
        drawn = st.cotangents(hooks, regime, seed)
        with jax.enable_x64(True):
            truth = reference(*widened(drawn))
        arms = {"port": port(*drawn), "portable": portable(*drawn), "reference": reference(*drawn)}
        for leaf in ("log_decay", "state"):
            t = np.asarray(getattr(truth[1], leaf), np.float64)
            for arm, out in arms.items():
                e = np.asarray(getattr(out[1], leaf), np.float64) - t
                sums.setdefault((leaf, arm), []).append(e)
            sums.setdefault((leaf, "truth"), []).append(t)
    for leaf in ("log_decay",):
        t = np.stack(sums[leaf, "truth"])
        for arm in ("port", "portable", "reference"):
            e = np.stack(sums[leaf, arm])
            # per position within its chunk, RMS over seeds/rows/heads
            per = np.sqrt(np.mean(e ** 2, axis=(0, 1, 3)))
            lines.append(f"{arm} log_decay rms by token: " + " ".join(f"{x:.1e}" for x in per[:40]))
        lines.append("truth log_decay rms by token: " + " ".join(f"{x:.1e}" for x in np.sqrt(np.mean(t ** 2, axis=(0, 1, 3)))[:40]))
    t = np.stack(sums["state", "truth"])
    for arm in ("port", "portable", "reference"):
        e = np.stack(sums["state", arm])
        lines.append(f"{arm} state rms {np.sqrt(np.mean(e**2)):.3e} (truth {np.sqrt(np.mean(t**2)):.3e}); worst entry {np.max(np.abs(e)):.3e}")
    raise AssertionError("NEAR|\n" + "\n".join(lines))
