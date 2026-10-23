"""Measurement script (never lands): the bf16 small-weight family's worst dk rows over seeds."""
import json
import sys

import jax
import numpy as np
import test_attention_stability as suite
from reference_error import with_gradients

name = "a small weight on a large value, bfloat16"
family = suite.FAMILIES[name]
draw = suite._cast(family, family.draw)
form = sys.argv[1]
first, count = int(sys.argv[2]), int(sys.argv[3])
mainloops = suite._mainloops(family, suite.FORMS[form])
reference = jax.jit(with_gradients(suite._reference(family, anchored=False)))
truth = with_gradients(suite._reference(family, anchored=True))
for seed in range(first, first + count):
    args = draw(seed)
    with jax.enable_x64(True):
        wide = [None if a is None else np.asarray(a, np.float64) for a in args]
        exact = [np.asarray(x, np.float64) for x in truth(*wide)]
    ref = [np.asarray(x, np.float64) for x in reference(*args)]
    for loop, fn in mainloops.items():
        ours = [np.asarray(x, np.float64) for x in jax.jit(with_gradients(fn))(*args)]
        dk_ours, dk_ref, dk_true = ours[2], ref[2], exact[2]
        size = np.abs(dk_true).max(axis=(1, 2, 3)) + 1e-300
        err = np.abs(dk_ours - dk_true).max(axis=(1, 2, 3)) / size
        ref_err = np.abs(dk_ref - dk_true).max(axis=(1, 2, 3)) / size
        worst = int(np.argmax(err))
        if err[worst] > 1e-2:
            row = {"seed": seed, "loop": loop, "row": worst, "err": float(err[worst]), "ref": float(ref_err[worst]),
                   "key_logit": float(np.asarray(args[1], np.float32)[worst, 1, 0, 0]),
                   "value": float(np.asarray(args[2], np.float32)[worst, 1, 0, 0]),
                   "dk_true": np.asarray(dk_true[worst, :2, 0, :2]).tolist(),
                   "dk_ours": np.asarray(dk_ours[worst, :2, 0, :2]).tolist()}
            print(json.dumps(row), flush=True)
