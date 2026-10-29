"""Pilot of the two-mode mixture's learning gate at fewer training steps (ledger 2159): the units
`learning_gate.units_for` declares over 192 runs of `steps` each, and the seconds they take."""

import json
import sys
import time

sys.path.insert(0, "tests")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import test_flow_matching as flow  # noqa: E402
from learning_gate import units_for  # noqa: E402

steps = int(sys.argv[1])
source = open("tests/test_flow_matching.py").read()
assert "jax.lax.fori_loop(0, 1500," in source
flow_code = compile(source.replace("jax.lax.fori_loop(0, 1500,", f"jax.lax.fori_loop(0, {steps},"),
                    "tests/test_flow_matching.py", "exec")
exec(flow_code, flow.__dict__)
began = time.perf_counter()
contrasts = flow.mixture_contrasts(np.arange(192))
seconds = time.perf_counter() - began
print("RESULT", json.dumps({"steps": steps, "units": units_for(contrasts, "two-mode mixture"),
                            "seconds_192": seconds,
                            "weakest": min((float(np.mean(v) / np.std(v, ddof=1)), k)
                                           for k, v in contrasts.items())}), flush=True)
