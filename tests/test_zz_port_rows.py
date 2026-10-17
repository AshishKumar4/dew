"""Scratch, not for landing: whether the port's excess over its formula is packing's. For [joint] and
[marker], the head and backbone leaves' worst element from float64 rows alone over its size (max of
|truth| and its float64 terms), the port and the formula, each on the packed windows and on the rows
alone (no packing)."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import port_formula
from reference_error import float64_terms

import test_decision_packing as t
from dew.nn import linear


@pytest.mark.gpu(reason="the port runs on a GPU")
def test_port_rows(monkeypatch):
    lines = []
    for arm in ("joint", "marker"):
        objective = t.objective_for(arm)
        batch = t.windows(objective, 2)
        variables = objective.init(jax.random.key(0))
        rows = t.apart(batch)
        runs = {}
        for rule_name, rule in (("port", None), ("formula", port_formula)):
            for layout, laid in (("packed", batch), ("rows", rows)):
                with monkeypatch.context() as patched:
                    if rule is not None:
                        patched.setattr(linear, "fla_rule", rule)
                    runs[f"{rule_name} {layout}"] = jax.tree.map(np.asarray, jax.jit(
                        lambda v, laid=laid: t.gradient(objective, v, laid))(variables))
        with jax.enable_x64(new_val=True):
            wide = copy.copy(objective)
            wide.substitute((objective.model.backbone.clone(dtype=jnp.float64),
                             objective.model.head.clone(dtype=jnp.float64)))
            widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
            truth, terms = float64_terms(lambda held: t.gradient(wide, held, rows), widened)
        names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(truth)]
        truth_l, terms_l = (jax.tree.leaves(x) for x in (truth, terms))
        flat = {label: jax.tree.leaves(g) for label, g in runs.items()}
        ratios = {"packed": [], "rows": []}
        for index, name in enumerate(names):
            true = np.asarray(truth_l[index], np.float64)
            size = np.maximum(np.abs(true), np.asarray(terms_l[index], np.float64))
            ok = size > 0
            if not ok.any():
                continue
            worst = {label: float(np.max(np.abs(np.asarray(leaves[index], np.float64) - true)[ok] / size[ok]))
                     for label, leaves in flat.items()}
            for layout in ("packed", "rows"):
                ratios[layout].append((worst[f"port {layout}"] / max(worst[f"formula {layout}"], 1e-30), name,
                                       worst[f"port {layout}"], worst[f"formula {layout}"]))
        for layout in ("packed", "rows"):
            values = np.array([r[0] for r in ratios[layout]])
            lines.append(f"[{arm}] {layout}: port/formula worst-element ratio over {len(values)} leaves: "
                         f"median {np.median(values):.2f}, p90 {np.percentile(values, 90):.2f}, max {values.max():.2f}, "
                         f"leaves over 4: {int(np.sum(values > 4))}")
            for ratio, name, port, formula in sorted(ratios[layout], reverse=True)[:5]:
                lines.append(f"    {ratio:5.2f}  {name}  port {port:.2e} formula {formula:.2e}")
    pytest.fail("\n".join(lines))
