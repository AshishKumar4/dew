"""562 probe, not for main: the bf16 joint head's attention gradients from float64, packed and alone,
on flash and on xla, on this GPU."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import distance
from test_decision_packing import SPECIALS, TOKENIZER, apart, attending, decided, gradient, rows_alone, windows

from dew.decision import DecisionObjective, JointLayout, LogLoss
from dew.decision.head import JointSchemaHead

LEAVES = ("['multihead_attn']['to_k']['kernel']", "['multihead_attn']['to_q']['kernel']",
          "['multihead_attn']['to_q']['bias']", "['self_attn']['to_k']['bias']", "['self_attn']['to_q']['kernel']")


@pytest.mark.gpu(reason="probe")
def test_probe_head(without_deterministic_ops):
    layout = JointLayout(max_len=1536)

    def built(implementation):
        return DecisionObjective(
            attending(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
            head=JointSchemaHead(32, width=128, routing_layers=1, layers=1, heads=2, feedforward=64,
                                 dtype=jnp.bfloat16, attention_impl=implementation),
            loss=LogLoss(), shuffle_options=False)

    flash, xla = built("flash"), built("xla")
    batch = windows(flash, 2)
    variables = flash.init(jax.random.key(0))
    found = {}
    for name, objective in (("flash", flash), ("xla", xla)):
        found[f"packed {name}"] = decided(objective, variables, batch, objective.bounds.segments)[1]
        found[f"alone {name}"] = rows_alone(objective, variables, batch)[1]
    rows = apart(batch)
    with jax.enable_x64(new_val=True):
        wide = copy.copy(xla)
        wide.substitute((xla.model.backbone.clone(dtype=jnp.float64), xla.model.head.clone(dtype=jnp.float64)))
        truth = gradient(wide, jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables), rows)

    def leaf(tree, tail):
        [(path, value)] = [(jax.tree_util.keystr(p), v) for p, v in jax.tree_util.tree_leaves_with_path(tree)
                           if jax.tree_util.keystr(p).endswith(tail) and "['head']" in jax.tree_util.keystr(p)]
        return np.asarray(value, np.float64)

    lines = [f"device {jax.devices()[0].device_kind}"]
    for tail in LEAVES:
        want = leaf(truth, tail)
        lines.append(f"{tail}: " + ", ".join(f"{name} {float(distance(leaf(grads, tail), want)):.3e}"
                                              for name, grads in found.items()))
    pytest.fail("\n".join(lines), pytrace=False)
