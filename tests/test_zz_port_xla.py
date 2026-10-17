"""Scratch, not for landing (560): the port on [joint]'s and [marker]'s packed windows against the real
portable rule chosen through the kernel-choice seam (gated_delta_rule -> xla) on the same windows, with
IEEE products and with one-TF32 products (the whole reference call under 'tensorfloat32'); each leaf's
distance from float64 rows alone, RMS and worst element over its size."""
import copy
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import float64_terms

import test_decision_packing as t
from dew.nn.kernels import generation


@pytest.mark.gpu(reason="the port runs on a GPU")
def test_port_xla():
    lines = []
    for arm in ("joint", "marker"):
        objective = t.objective_for(arm)
        batch = t.windows(objective, 2)
        variables = objective.init(jax.random.key(0))
        rows = t.apart(batch)
        runs = {}
        for label, kernel, precision in (("port", None, None), ("xla ieee", "xla", None),
                                         ("xla tf32", "xla", "tensorfloat32")):
            begun = time.time()
            with generation.chosen(**({"gated_delta_rule": kernel} if kernel else {})), \
                    jax.default_matmul_precision(precision or jax.config.jax_default_matmul_precision):
                runs[label] = jax.tree.map(np.asarray, jax.jit(lambda v: t.gradient(objective, v, batch))(variables))
            lines.append(f"[{arm}] {label}: {time.time() - begun:.0f} s")
        with jax.enable_x64(new_val=True):
            wide = copy.copy(objective)
            wide.substitute((objective.model.backbone.clone(dtype=jnp.float64),
                             objective.model.head.clone(dtype=jnp.float64)))
            widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
            truth, terms = float64_terms(lambda held: t.gradient(wide, held, rows), widened)
        names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(truth)]
        truth_l, terms_l = (jax.tree.leaves(x) for x in (truth, terms))
        flat = {label: jax.tree.leaves(g) for label, g in runs.items()}
        stats = []
        for index, name in enumerate(names):
            true = np.asarray(truth_l[index], np.float64)
            size = np.maximum(np.abs(true), np.asarray(terms_l[index], np.float64))
            ok = size > 0
            if not ok.any():
                continue
            scale = max(np.sqrt(np.mean(true ** 2)), 1e-300)
            stats.append((name, {label: (float(np.sqrt(np.mean((np.asarray(leaves[index], np.float64) - true) ** 2))
                                               / scale),
                                         float(np.max(np.abs(np.asarray(leaves[index], np.float64) - true)[ok]
                                                      / size[ok])))
                                 for label, leaves in flat.items()}))
        for level, which in (("rms", 0), ("worst element", 1)):
            for mine, ref in (("port", "xla ieee"), ("port", "xla tf32"), ("xla tf32", "xla ieee")):
                ratio = np.array([r[mine][which] / r[ref][which] for _, r in stats if r[ref][which] > 0])
                lines.append(f"[{arm}] {level}: {mine}/{ref} over {len(ratio)} leaves: median {np.median(ratio):.2f}, "
                             f"p90 {np.percentile(ratio, 90):.2f}, max {ratio.max():.2f}, over 2: "
                             f"{int(np.sum(ratio > 2))}, over 4: {int(np.sum(ratio > 4))}")
        for name, r in sorted(stats, key=lambda s: -s[1]["port"][1] / max(s[1]["xla tf32"][1], 1e-30))[:6]:
            lines.append(f"    {name}: " + "  ".join(f"{label} el {r[label][1]:.2e}" for label in flat))
    pytest.fail("\n".join(lines))
