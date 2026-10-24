"""What a layout's validation reports before a run's first step, read against the state the run then places.

`Trainer.validate` reads the state `fit` would place without placing anything: where each
leaf splits and the part each device holds, the bytes each device keeps between steps, the axes
a batch's rows split over, and the logical axes no rule names. Each claim is held to the arrays
`fit` then places on the eight simulated CPU devices, and a capacity those bytes overflow is
refused, as a CPU mesh standing in for GPUs states it.
"""

import jax
import optax
import pytest
from model_support import TINY_DECODER
from sharded import assert_sharded
from token_support import repeated_token_batch, token_rows

from dew.data.dataset import Loader
from dew.nn.sharding import LayoutRefused, mesh_axes
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import DEFAULT_RULES, Layout, MeshSpec, Trainer

pytestmark = pytest.mark.mesh

# A vocabulary no other dimension of the tiny decoder shares, so a shard that holds it whole says so.
VOCAB = 96
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256

CASES = {
    # Between them the meshes place a state over fsdp alone and beside tensor, with the optimizer
    # state and the EMA in pinned host memory, and with the gradient accumulator of two steps.
    "fsdp": (MeshSpec(fsdp=8), Layout(min_shard=TINY_SHARD), 1),
    "fsdp_tensor_accumulating": (MeshSpec(fsdp=4, tensor=2), Layout(min_shard=TINY_SHARD), 2),
    "fsdp_host_accumulating": (MeshSpec(fsdp=8), Layout(min_shard=TINY_SHARD, host=("opt_state", "ema")), 2),
}


def fine_tune(mesh, layout, accumulation=1):
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN, ema_decay=0.999), optax.adam(1e-3), key=jax.random.key(0),
                   mesh=mesh, layout=layout, accumulation=accumulation, split=1)


def step_batch():
    return token_rows(VOCAB, BATCH, SEQ_LEN + 1)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("case", list(CASES))
def test_a_validated_fine_tune_is_placed_and_held_as_its_report_said(case):
    """The report read before launch names every leaf of the state the run then holds, each placed
    as reported; each device holds, byte for byte, what the report said it would of the leaves in
    its own memory; and the step reads its batch's rows split over the reported axes."""
    mesh, layout, accumulation = CASES[case]
    run = fine_tune(mesh, layout, accumulation)
    report = run.validate(step_batch())
    state = run.fit(Loader(train=lambda partition: repeated_token_batch(VOCAB, BATCH, SEQ_LEN + 1),
                           val=None, records=None, batch=BATCH), steps=2 * accumulation)

    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    paths = [jax.tree_util.keystr(path, simple=True, separator="/") for path, _ in leaves]
    assert list(report.leaves) == paths
    held = dict.fromkeys(run.device_mesh.device_ids.ravel().tolist(), 0)
    for (_, leaf), placed in zip(leaves, report.leaves.values(), strict=True):
        assert (placed.spec, placed.shard_shape, placed.memory_kind) == (
            leaf.sharding.spec, leaf.sharding.shard_shape(leaf.shape), leaf.sharding.memory_kind)
        if leaf.sharding.memory_kind == "device":
            for shard in leaf.addressable_shards:
                held[shard.device.id] += shard.data.nbytes
    assert report.device_bytes == held
    assert_sharded(state.variables["params"], run.device_mesh)

    arguments, _ = run.executable.input_shardings
    rows = {mesh_axes(sharding.spec[0]) for sharding in jax.tree.leaves(arguments[1]) if sharding.spec}
    assert rows == {report.batch_axes}


@pytest.mark.mesh(devices=8)
def test_a_fine_tune_over_a_devices_capacity_is_refused_before_anything_is_placed():
    """A capacity one byte below what a device would hold of the state, its accumulator included, is
    refused, and exactly that much passes; neither reading leaves an array behind."""
    mesh, layout, accumulation = CASES["fsdp_tensor_accumulating"]
    run = fine_tune(mesh, layout, accumulation)
    before = jax.live_arrays()
    held = max(run.validate(step_batch()).device_bytes.values())
    run.validate(step_batch(), capacity=held)
    with pytest.raises(LayoutRefused):
        run.validate(step_batch(), capacity=held - 1)
    assert {id(array) for array in jax.live_arrays()} <= {id(array) for array in before}


@pytest.mark.mesh(devices=8)
def test_rules_naming_an_absent_axis_are_refused_and_a_logical_axis_no_rule_names_is_reported():
    """A rule naming an axis the mesh lacks fails validation, where the same rule naming one it has
    passes. Dropping the vocabulary's rule unbinds that axis alone, and every leaf the report lists
    under it holds the vocabulary whole on each device, where the rule split it."""
    mesh = MeshSpec(fsdp=4, tensor=2)
    fine_tune(mesh, Layout(rules=(("embed", "fsdp"), *DEFAULT_RULES), min_shard=TINY_SHARD)).validate()
    with pytest.raises(ValueError):
        fine_tune(mesh, Layout(rules=(("embed", "context"), *DEFAULT_RULES), min_shard=TINY_SHARD)).validate()

    bound = fine_tune(mesh, Layout(min_shard=TINY_SHARD)).validate()
    unbound = fine_tune(mesh, Layout(rules=tuple(rule for rule in DEFAULT_RULES if rule[0] != "vocab"),
                                     min_shard=TINY_SHARD)).validate()
    assert set(unbound.unbound) - set(bound.unbound) == {"vocab"}
    listed = unbound.unbound["vocab"]
    assert all(VOCAB in unbound.leaves[path].shard_shape for path in listed)
    assert any(VOCAB not in bound.leaves[path].shard_shape for path in listed)
