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
    shapes = Pretrained.load(source, dtype="float32", attention_impl="xla", weights=False)
    host = Pretrained.load(source, dtype="float32", attention_impl="xla", on_host=True)
    for abstract, held, original in zip(jax.tree.leaves((shapes.variables, shapes.head_variables)),
                                       jax.tree.leaves((host.variables, host.head_variables)),
                                       jax.tree.leaves((bundle.variables, bundle.head_variables)), strict=True):
        assert isinstance(abstract, jax.ShapeDtypeStruct)
        assert isinstance(held, np.ndarray)
        assert abstract.shape == held.shape == original.shape
        assert abstract.dtype == held.dtype == original.dtype
        np.testing.assert_array_equal(held, original)
    exported = tmp_path / "exported"
    task.save_pretrained(exported)
    restored = Decide.from_pretrained(exported, dtype="float32", attention_impl="xla")
    assert restored.layout.rows(restored.tokenizer, restored.specials, "Refund requested.",
                                {"team": Choice("Which team?", ["billing", "technical"])}) == [row]
    for actual, expected in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(task.variables),
                               strict=True):
        np.testing.assert_array_equal(actual, expected)


class HostRows:
    """A host array with a transfer budget: a native gather may cross JAX's array boundary,
    the complete vocabulary may not. Both paths read the same real NumPy values."""

    def __init__(self, values):
        self.values = values
        self.dtype = values.dtype

    def __jax_array__(self, dtype=None):
        raise AssertionError("initializing a few slots must not transfer the entire host vocabulary")

    def take(self, indices, axis=None):
        return self.values.take(indices, axis=axis)


def test_a_slot_draw_transfers_only_selected_host_rows_and_matches_a_device_draw():
    import jax.numpy as jnp
    from test_decision_readout import ids, rows_of

    from dew.decision import DecisionInputs
    from dew.decision.head import SlotReadout

    table = np.arange(256 * 32, dtype=np.float32).reshape(256, 32) / 8192
    head = SlotReadout(32, tuple(row[0] for row in ids()), head_dim=8, slot_rows=8)
    inputs = DecisionInputs.collate(rows_of(0), 0)
    states = jnp.zeros((*inputs.tokens.shape, 32), jnp.float32)
    key = jax.random.key(3)
    host = head.init(key, states, inputs, table)
    limited = head.init(key, states, inputs, HostRows(table))
    device = head.init(key, states, inputs, jnp.asarray(table))
    for actual, constrained, expected in zip(jax.tree.leaves(host), jax.tree.leaves(limited),
                                            jax.tree.leaves(device), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(constrained, actual)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(host["params"]["slots"][:3], table[[65, 66, 67]])
