"""Probe, not for main: the three nn-kernels findings on a GPU, at the default matmul precision."""

import json
import os
import subprocess
import sys

import pytest

SCRIPT = r'''
import json
import jax, jax.numpy as jnp, numpy as np
from dew.nn.attention import scaled_dot_product_attention, with_key_lengths
report = {"device": jax.devices()[0].device_kind, "precision": str(jax.config.jax_default_matmul_precision)}

# A: fp32 values 1 + 2**-12, zero queries and keys, one visible key.
q = jnp.zeros((1, 1, 1, 16), jnp.float32)
k = jnp.zeros((1, 32, 1, 16), jnp.float32)
v = jnp.full((1, 32, 1, 16), 1 + 2.0**-12, jnp.float32)
lengths = jnp.asarray([1], jnp.int32)
kernel = scaled_dot_product_attention(q, k, v, implementation="xla", key_value_seq_lengths=lengths)
folded = scaled_dot_product_attention(q, k, v, implementation="xla",
                                      mask=with_key_lengths(None, lengths, 32))
report["A kernel - 1"] = float(np.asarray(kernel).ravel()[0]) - 1
report["A folded - 1"] = float(np.asarray(folded).ravel()[0]) - 1
plain = jnp.einsum("ij,jk->ik", jnp.full((1, 32), 1 / 32, jnp.float32), v[0, :, 0, :])
report["xla einsum - 1"] = float(np.asarray(plain).ravel()[0]) - 1

# C: the gradient of a lengths call the kernel takes.
rng = np.random.default_rng(0)
q2 = jnp.asarray(rng.normal(size=(1, 1, 2, 16)), jnp.float32)
k2 = jnp.asarray(rng.normal(size=(1, 32, 2, 16)), jnp.float32)
v2 = jnp.asarray(rng.normal(size=(1, 32, 2, 16)), jnp.float32)
try:
    grads = jax.grad(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation="xla", key_value_seq_lengths=jnp.asarray([20], jnp.int32)).sum(),
        argnums=(0, 1, 2))(q2, k2, v2)
    report["C grad"] = "ok"
except Exception as error:
    report["C grad"] = (type(error).__name__ + ": " + str(error).splitlines()[0])[:200]

# B: FlashAttention given keys of another batch, and keys of another width.
try:
    import dew_flash_attn
    report["B version"] = dew_flash_attn.__version__ if hasattr(dew_flash_attn, "__version__") else "?"
    for name, (qs, ks) in {"batch": ((2, 32, 1, 64), (1, 32, 1, 64)), "width": ((1, 32, 1, 64), (1, 32, 1, 32))}.items():
        try:
            out = dew_flash_attn.flash_mha(jnp.ones(qs, jnp.bfloat16), jnp.ones(ks, jnp.bfloat16), jnp.ones(ks, jnp.bfloat16))
            report[f"B {name}"] = f"ran, out {out.shape}, finite {bool(jnp.isfinite(out.astype(jnp.float32)).all())}"
        except Exception as error:
            report[f"B {name}"] = (type(error).__name__ + ": " + str(error).splitlines()[0])[:200]
except Exception as error:
    report["B"] = repr(error)[:200]
print("PROBE " + json.dumps(report))
'''


@pytest.mark.gpu(reason="probe")
def test_probe():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DEW_XLA_CACHE") and key != "JAX_DEFAULT_MATMUL_PRECISION"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, ["src", env.get("PYTHONPATH")]))
    lines = []
    for flags in (env.get("XLA_FLAGS", ""), ""):
        done = subprocess.run([sys.executable, "-c", SCRIPT], env={**env, "XLA_FLAGS": flags}, capture_output=True,
                              text=True, timeout=900)
        found = [line for line in done.stdout.splitlines() if line.startswith("PROBE ")]
        lines.append(f"## [{flags or 'no flags'}]\n" + (json.dumps(json.loads(found[0][6:]), indent=1) if found
                                                       else f"exit {done.returncode}\n{done.stderr[-2000:]}"))
    pytest.fail("\n".join(lines), pytrace=False)
