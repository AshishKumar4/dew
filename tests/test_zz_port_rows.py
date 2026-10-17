"""Scratch, not for landing: whether the port's excess over its formula is packing's. For [joint] and
[marker], the head and backbone leaves' worst element from float64 rows alone over its size (max of
|truth| and its float64 terms), the port and the formula, each on the packed windows and on the rows
alone (no packing)."""
import copy
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import port_formula
from reference_error import float64_terms

import test_decision_packing as t
from dew.nn import linear
from dew.nn.kernels import generation


@pytest.mark.gpu(reason="the port runs on a GPU")
def test_port_rows(monkeypatch):
    lines = []
    for arm in ("joint", "marker"):
        objective = t.objective_for(arm)
        batch = t.windows(objective, 2)
        variables = objective.init(jax.random.key(0))
        rows = t.apart(batch)
        runs = {}
        for rule_name in ("port", "interpreted", "formula"):
            for layout, laid in (("packed", batch), ("rows", rows)):
                begun = time.time()
                with monkeypatch.context() as patched:
                    if rule_name == "formula":
                        patched.setattr(linear, "fla_rule", port_formula)
                    if rule_name == "interpreted":
                        # The same kernels, fla's algorithm and Dew's short conv, run as XLA ops.
                        patched.setattr(generation, "triton_runs", lambda: False)
                    runs[f"{rule_name} {layout}"] = jax.tree.map(np.asarray, jax.jit(
                        lambda v, laid=laid: t.gradient(objective, v, laid))(variables))
                lines.append(f"[{arm}] {rule_name} {layout}: {time.time() - begun:.0f} s")
        with jax.enable_x64(new_val=True):
            wide = copy.copy(objective)
            wide.substitute((objective.model.backbone.clone(dtype=jnp.float64),
                             objective.model.head.clone(dtype=jnp.float64)))
            widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
            truth, terms = float64_terms(lambda held: t.gradient(wide, held, rows), widened)
        names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(truth)]
        truth_l, terms_l = (jax.tree.leaves(x) for x in (truth, terms))
        flat = {label: jax.tree.leaves(g) for label, g in runs.items()}
        labels = list(flat)
        stats = []
        for index, name in enumerate(names):
            true = np.asarray(truth_l[index], np.float64)
            size = np.maximum(np.abs(true), np.asarray(terms_l[index], np.float64))
            ok = size > 0
            if not ok.any():
                continue
            scale = max(np.sqrt(np.mean(true ** 2)), 1e-300)
            row = {}
            for label in labels:
                err = np.asarray(flat[label][index], np.float64) - true
                row[label] = (float(np.sqrt(np.mean(err ** 2)) / scale), float(np.max(np.abs(err)[ok] / size[ok])))
            stats.append((name, row))
        for level, which in (("rms", 0), ("worst element", 1)):
            for layout, against in (("packed", "formula"), ("rows", "formula"), ("packed", "interpreted"),
                                    ("rows", "interpreted")):
                ratio = np.array([r[f"port {layout}"][which] / r[f"{against} {layout}"][which]
                                  for _, r in stats if r[f"{against} {layout}"][which] > 0])
                lines.append(f"[{arm}] {level}, {layout}: port/{against} over {len(ratio)} leaves: median "
                             f"{np.median(ratio):.2f}, p90 {np.percentile(ratio, 90):.2f}, max {ratio.max():.2f}, "
                             f"over 2: {int(np.sum(ratio > 2))}, over 4: {int(np.sum(ratio > 4))}")
            ratio = np.array([r["port rows"][which] / max(r["port packed"][which], 1e-30) for _, r in stats])
            lines.append(f"[{arm}] {level}: port rows/port packed median {np.median(ratio):.2f}")
        worst = sorted(stats, key=lambda s: -s[1]["port packed"][1] / max(s[1]["interpreted packed"][1], 1e-30))[:6]
        for name, r in worst:
            lines.append(f"    {name}: " + "  ".join(f"{label} rms {r[label][0]:.2e} el {r[label][1]:.2e}"
                                                    for label in labels))
    pytest.fail("\n".join(lines))
