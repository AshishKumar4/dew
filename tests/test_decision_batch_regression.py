"""Diagnostic draws of a decision row alone and beside mixed rows."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.decision import Choice, Decide, DecisionInputs


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("kind", ["laya", "joint"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_mixed_rows_keep_each_probability(tmp_path, kind, dtype):
    source = FIXTURES / "laya" / "tiny"
    if kind == "joint":
        source = tmp_path / "release"
        source.mkdir()
        backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
        joint = FIXTURES / "joint_checkpoint" / "tiny"
        for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                     joint / "joint_head.safetensors"]:
            (source / item.name).symlink_to(item.resolve())
    task = Decide.from_pretrained(source, attention_impl="xla", dtype=dtype)
    task = replace(task, layout=replace(task.layout, max_len=512))
    short = ("refund", {"q": Choice("team", ["pay", "tech", "sales"])})
    long = ("invoice " * 20, {"q": Choice("which department", ["billing", "technical", "sales",
                                                             "support", "account", "other"])})
    requests = [short, long]
    rows = [own[0] for own in task.rows(requests)]
    print("DIAGNOSTIC", kind, dtype, "lengths", [len(row.tokens) for row in rows],
          "model", task.model, flush=True)
    for enabled in (False, True):
        with algebra.passes(enabled):
            together = task(requests)
            apart = [task(state, questions) for state, questions in requests]
            print("PUBLIC", kind, dtype, enabled,
                  [(np.asarray(batch["q"].probabilities).tolist(),
                    np.asarray(lone["q"].probabilities).tolist())
                   for batch, lone in zip(together, apart, strict=True)], flush=True)
            batched = np.asarray(task._forward(
                task.variables, DecisionInputs.collate(rows, task.specials.pad)))
            for index, row in enumerate(rows):
                own = np.asarray(task._forward(
                    task.variables, DecisionInputs.collate([row], task.specials.pad)))
                width = len(row.questions[0].options)
                print("COLLATED", kind, dtype, enabled, index,
                      batched[index, 0, :width].tolist(), own[0, 0, :width].tolist(), flush=True)
    print("ENV", jax.config.jax_default_matmul_precision, jnp.dtype(dtype), str(task.layout), flush=True)
