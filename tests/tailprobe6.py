"""A pytest plugin replacing check 1's assertion with a summary of both arms' tails over seeds 0 to
TAILPROBE_SEEDS - 1: per leaf, the pooled RMS ratio, each arm's four largest seeds' share of its pooled
squared error and those seeds; then the run stops."""
import json
import os

import jax
import numpy as np
import reference_error

SEEDS = int(os.environ.get("TAILPROBE_SEEDS", "320"))


def probe(dew_fn, reference_fn, draw, label, *, seeds=None, truth_fn=None, **_):
    errors = {}
    for seed in range(int(os.environ.get("TAILPROBE_FROM", "0")), int(os.environ.get("TAILPROBE_FROM", "0")) + SEEDS):
        args = draw(seed)
        mine = jax.tree_util.tree_flatten_with_path(jax.device_get(dew_fn(*args)))[0]
        ref = jax.tree.leaves(jax.device_get(reference_fn(*args)))
        with jax.enable_x64(new_val=True):
            truth = jax.tree.leaves(jax.device_get((truth_fn or reference_fn)(*reference_error.widened(args))))
        for (path, a), b, true in zip(mine, ref, truth, strict=True):
            a, b, true = (np.asarray(v, np.float64) for v in (a, b, true))
            if not np.issubdtype(true.dtype, np.floating) or true.size == 0:
                continue
            errors.setdefault(jax.tree_util.keystr(path), []).append(
                (float(np.square(a - true).sum()), float(np.square(b - true).sum())))
    for name, pairs in errors.items():
        e = np.asarray(pairs)
        if e[:, 1].sum() == 0:
            continue
        top = {arm: np.argsort(-e[:, at])[:4] for at, arm in enumerate(("dew", "ref"))}
        print("TAILSUM " + json.dumps({
            "leaf": name, "ratio": float(np.sqrt(e[:, 0].sum() / e[:, 1].sum())),
            **{f"{arm}_share": float(e[seeds, at].sum() / e[:, at].sum()) for at, (arm, seeds) in
               enumerate(top.items())},
            **{f"{arm}_seeds": seeds.tolist() for arm, seeds in top.items()}}), flush=True)
    raise SystemExit("tail probe done")


def pytest_configure(config):
    reference_error.assert_as_exact = probe
