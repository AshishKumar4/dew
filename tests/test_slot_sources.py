"""Released slot decisions through the native composite loader and task export."""

import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.decision import Choice, Decide
from dew.interop import Pretrained

BACKBONE = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"


@pytest.mark.parametrize("readout", ["slot_lm", "head", "fresh_head"])
def test_a_slot_release_loads_its_readout_and_exports_the_same_task(tmp_path, readout):
    """The stored head keeps all its rows and native q/k orientation. A fresh head draws only
    the available label rows; the byte tokenizer has all letters but no single-token pairs.
    All three become decision tasks and survive a native task export with their parameters."""
    import torch

    source = tmp_path / "release"
    shutil.copytree(BACKBONE, source)
    config = {"readout": "slot_lm" if readout == "slot_lm" else "head",
              "head_dim": 8, "n_slots": 32}
    (source / "readout.json").write_text(json.dumps(config))
    stored = {"slot": np.arange(32 * 32, dtype=np.float32).reshape(32, 32) / 1024,
              "q.weight": np.arange(8 * 32, dtype=np.float32).reshape(8, 32) / 256,
              "k.weight": np.arange(8 * 32, dtype=np.float32).reshape(8, 32) / 512}
    if readout == "head":
        torch.save({name: torch.from_numpy(value) for name, value in stored.items()},
                   source / "readout_head.pt")
    bundle = Pretrained.load(source, dtype="float32", attention_impl="xla")
    task = Decide.of(bundle)
    [row] = task.layout.rows(task.tokenizer, task.specials, "Refund requested.",
                            {"team": Choice("Which team?", ["billing", "technical"])})
    task.model.head.validate_rows([row])
    assert row.questions[0].label_ids == tuple((task.tokenizer.encode(label)[0],) for label in "AB")
    if readout == "head":
        for name, native in (("slot", "slots"), ("q.weight", "q"), ("k.weight", "k")):
            np.testing.assert_array_equal(bundle.head_variables["params"][native], stored[name])
    elif readout == "fresh_head":
        labels = [task.tokenizer.encode(label)[0] for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        table = task.model.table(task.variables)
        np.testing.assert_array_equal(bundle.head_variables["params"]["slots"], table[labels])
        assert np.count_nonzero(bundle.head_variables["params"]["q"]) > 0
        assert np.count_nonzero(bundle.head_variables["params"]["k"]) == 0
    else:
        assert not jax.tree.leaves(bundle.head_variables)
    exported = tmp_path / "exported"
    bundle.save(exported)
    restored = Decide.from_pretrained(exported, dtype="float32", attention_impl="xla")
    assert restored.layout.rows(restored.tokenizer, restored.specials, "Refund requested.",
                                {"team": Choice("Which team?", ["billing", "technical"])}) == [row]
    for actual, expected in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(task.variables),
                               strict=True):
        np.testing.assert_array_equal(actual, expected)
