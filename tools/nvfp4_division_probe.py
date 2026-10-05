#!/usr/bin/env python3
"""Dense proof of the checkpoint QDQ divider against IEEE fp32 division.

The default x64-disabled process draws 2.5 million ratios across normed
activations, MLP products and inverse global scales: denominators in
[2**-24, 2**24], quotients up to 128. Half of those draws sit at or beside
E2M1's seven bucket boundaries. NumPy's float32 divide supplies the IEEE
result, bit for bit. Run on CPU and on the 4080 through the lane runners:

    PYTHONPATH=src python tools/nvfp4_division_probe.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from dew.training.quantization import _nvfp4_divide


def main() -> None:
    if jax.config.jax_enable_x64:
        raise ValueError("this probe must run with the default x64-disabled configuration")
    rng = np.random.default_rng(2046)
    count = 1_000_000
    denominators = np.exp2(rng.uniform(-24, 24, count)).astype(np.float32)
    quotients = (rng.uniform(-8, 8, count) * np.exp2(rng.uniform(-24, 4, count))).astype(np.float32)
    numerators = (denominators.astype(np.float64) * quotients.astype(np.float64)).astype(np.float32)
    boundaries = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    q = rng.choice(boundaries, count // 2)
    b = denominators[:count // 2]
    a = (q.astype(np.float64) * b.astype(np.float64)).astype(np.float32)
    numerators = np.concatenate([numerators, a, np.nextafter(a, np.inf), np.nextafter(a, -np.inf)])
    denominators = np.concatenate([denominators, b, b, b])
    expected = numerators / denominators
    actual = np.asarray(jax.jit(_nvfp4_divide)(jnp.asarray(numerators), jnp.asarray(denominators)))
    bad = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    print(jax.default_backend(), "x64", jax.config.jax_enable_x64, "draws", expected.size,
          "mismatches", bad.size)
    if bad.size:
        examples = [(float(numerators[i]), float(denominators[i]), float(actual[i]), float(expected[i]))
                    for i in bad[:12]]
        raise AssertionError((bad.size, examples))


if __name__ == "__main__":
    main()
