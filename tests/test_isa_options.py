"""Probe (not for landing): what the optimized HLO of a dot shows with and without the dot fusions."""
import re

import jax
import jax.numpy as jnp
import numpy as np


def test_probe():
    left, right = np.ones((16, 48), np.float32), np.ones((48, 4), np.float32)
    for options in ({}, {"xla_cpu_experimental_ynn_fusion_type": "", "xla_cpu_experimental_xnn_fusion_type": ""}):
        text = jax.jit(jnp.dot, compiler_options=options).lower(left, right).compile().as_text()
        kinds = sorted(set(re.findall(r'kind=(\w+)', text)) | set(re.findall(r'(\w*(?:ynn|xnn|YNN|XNN)\w*)', text)))
        print("PROBE options", bool(options), kinds, flush=True)
