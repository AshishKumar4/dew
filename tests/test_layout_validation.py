"""What a layout's validation reports before a run's first step, read against the state the run then places.

`Trainer.validate` reads the state `fit` would place without placing anything: where each
leaf splits and the part each device holds, the bytes each device keeps between steps, the axes
a batch's rows split over, and the logical axes no rule names. Each claim is held to the arrays
`fit` then places on the eight simulated CPU devices, as is fit's own validation, and a capacity
those bytes overflow is refused, as a CPU mesh standing in for GPUs states it.
"""

import jax
import numpy as np
import optax
import pytest
from jax.sharding import Mesh
from model_support import TINY_DECODER
from sharded import collective_axes
from test_parallelism import BoxedReadout
from token_support import repeated_token_batch, token_rows

from dew.config import OptimConfig
from dew.data.dataset import Loader
from dew.nn.sharding import LayoutRefused, mesh_axes
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import DEFAULT_RULES, Layout, MeshSpec, Trainer
from dew.training.distributed import STEP_BATCH, placed_shapes

pytestmark = pytest.mark.mesh

# A vocabulary no other dimension of the tiny decoder shares, so a shard that holds it whole says so.
VOCAB = 96
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256


def native():
    """A mesh of its own axes: a replica axis beside two fsdp levels, all three splitting the rows."""
    return Mesh(np.asarray(jax.devices()[:8]).reshape(2, 2, 2), ("replica", "zero", "zero2"))


NATIVE = Layout(rules={"embed": ("zero", "zero2"), "mlp": "zero", "heads": "zero", "kv": "zero",
                       "vocab": "zero", "linear": "zero"},
                data_axes=("replica", "zero", "zero2"), min_shard=TINY_SHARD)

CASES = {
    # Between them the meshes place a state over fsdp alone, beside tensor, beside sequence (which
    # holds the state with fsdp) and beside pipeline stages, over a native mesh's own axes, with the
    # optimizer state and the EMA in pinned host memory, and with the gradient accumulator of two steps.
    "fsdp": (lambda: MeshSpec(fsdp=8), Layout(min_shard=TINY_SHARD), 1),
    "fsdp_tensor_accumulating": (lambda: MeshSpec(fsdp=4, tensor=2), Layout(min_shard=TINY_SHARD), 2),
    "fsdp_host_accumulating": (lambda: MeshSpec(fsdp=8),
                               Layout(min_shard=TINY_SHARD, host=("opt_state", "ema")), 2),
    "fsdp_sequence": (lambda: MeshSpec(fsdp=4, sequence=2), Layout(min_shard=TINY_SHARD), 1),
    "fsdp_stage": (lambda: MeshSpec(fsdp=2, stage=2), Layout(min_shard=TINY_SHARD), 1),
    "native_accumulating": (native, NATIVE, 2),
}


def fine_tune(mesh, layout, accumulation=1, optimizer=None):
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN, ema_decay=0.999), optimizer or optax.adam(1e-3),
                   key=jax.random.key(0), mesh=mesh, layout=layout, accumulation=accumulation, split=1)


def step_batch():
    return token_rows(VOCAB, BATCH, SEQ_LEN + 1)


def leaf_path(path) -> str:
    return jax.tree_util.keystr(path, simple=True, separator="/")


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("case", list(CASES))
def test_a_validated_fine_tune_is_placed_and_held_as_its_report_said(case):
    """The report read before launch, and fit's own once it compiled its step, name every leaf of
    the state the run then holds, each placed as reported; each device holds, byte for byte, what
    they said it would of the leaves in its own memory; the step reads its batch's rows split over
    the reported axes; and it runs every collective the report says each axis implies, over it."""
    mesh, layout, accumulation = CASES[case]
    run = fine_tune(mesh(), layout, accumulation)
    report = run.validate(step_batch())
    state = run.fit(Loader(train=lambda partition: repeated_token_batch(VOCAB, BATCH, SEQ_LEN + 1),
                           val=None, records=None, batch=BATCH), steps=2 * accumulation)

    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    held = dict.fromkeys(run.device_mesh.device_ids.ravel().tolist(), 0)
    whole = 0
    for _, leaf in leaves:
        if leaf.sharding.memory_kind == "device":
            whole += leaf.nbytes
            for shard in leaf.addressable_shards:
                held[shard.device.id] += shard.data.nbytes
    # The mesh splits the state: no device holds all of it.
    assert max(held.values()) < whole
    for validation in (report, run.validation):
        assert list(validation.leaves) == [leaf_path(path) for path, _ in leaves]
        for (_, leaf), placed in zip(leaves, validation.leaves.values(), strict=True):
            assert (placed.spec, placed.shard_shape, placed.memory_kind) == (
                leaf.sharding.spec, leaf.sharding.shard_shape(leaf.shape), leaf.sharding.memory_kind)
        assert validation.device_bytes == held

    arguments, _ = run.executable.input_shardings
    rows = {mesh_axes(sharding.spec[0]) for sharding in jax.tree.leaves(arguments[STEP_BATCH])
            if sharding.spec}
    assert rows == {report.batch_axes}
    # Every collective the report says an axis implies, the compiled step runs over that axis.
    ran = collective_axes(run.executable.as_text(), run.device_mesh)
    assert set(report.collectives) == {axis for axis, size in run.device_mesh.shape.items() if size > 1}
    for axis, ops in report.collectives.items():
        for op in ops:
            assert any(kind == op and axis in axes for kind, axes in ran), (axis, op)


@pytest.mark.mesh(devices=8)
def test_a_fine_tune_over_a_devices_capacity_is_refused_before_anything_is_placed():
    """A configured optimizer's run, read as `fit` builds it over its steps. Between what a device
    holds of the state alone and of the state with the accumulator its steps pool, the state alone
    passes and the whole is refused, and exactly the whole passes."""
    mesh, layout, accumulation = CASES["fsdp_tensor_accumulating"]
    run = fine_tune(mesh(), layout, accumulation, OptimConfig(optimizer="adam", learning_rate=1e-3))
    alone = max(run.validate(steps=8).device_bytes.values())
    whole = max(run.validate(step_batch(), steps=8).device_bytes.values())
    assert alone < whole
    run.validate(steps=8, capacity=whole - 1)
    run.validate(step_batch(), steps=8, capacity=whole)
    with pytest.raises(LayoutRefused):
        run.validate(step_batch(), steps=8, capacity=whole - 1)


@pytest.mark.mesh(devices=8)
def test_rules_naming_an_absent_axis_are_refused_and_a_logical_axis_no_rule_names_is_reported():
    """A rule naming an axis the mesh lacks fails validation, the trainer's and the layout's own
    over a placed state, where the same rule naming an axis it has passes. Dropping the
    vocabulary's rule unbinds that axis alone, and every leaf the report lists under it holds the
    vocabulary whole on each device, where the rule split it."""
    mesh = MeshSpec(fsdp=4, tensor=2)
    present = Layout(rules=(("embed", "fsdp"), *DEFAULT_RULES), min_shard=TINY_SHARD)
    absent = Layout(rules=(("embed", "context"), *DEFAULT_RULES), min_shard=TINY_SHARD)
    run = fine_tune(mesh, present)
    before = jax.live_arrays()
    run.validate(step_batch())
    # Validating places nothing: no array outlives it.
    assert {id(array) for array in jax.live_arrays()} <= {id(array) for array in before}
    with pytest.raises(ValueError):
        fine_tune(mesh, absent).validate()
    state = jax.eval_shape(run.initial_state)
    placed = placed_shapes(state, run.shardings(state))
    present.validate(run.device_mesh, placed)
    with pytest.raises(ValueError):
        absent.validate(run.device_mesh, placed)

    bound = fine_tune(mesh, Layout(min_shard=TINY_SHARD)).validate()
    unbound = fine_tune(mesh, Layout(rules=tuple(rule for rule in DEFAULT_RULES if rule[0] != "vocab"),
                                     min_shard=TINY_SHARD)).validate()
    assert set(unbound.unbound) - set(bound.unbound) == {"vocab"}
    listed = unbound.unbound["vocab"]
    assert all(VOCAB in unbound.leaves[path].shard_shape for path in listed)
    assert any(VOCAB not in bound.leaves[path].shard_shape for path in listed)


@pytest.mark.mesh(devices=8)
def test_a_parameter_boxed_with_names_no_rule_places_reports_every_one_of_them():
    """A Flax module's kernel boxed with two names the rules lack takes the shape heuristic,
    and the report names both; a rule for one of them names the other alone."""
    def validated(rules):
        return Trainer(BoxedReadout(), optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(fsdp=2),
                       layout=Layout(min_shard=1, rules=rules)).validate()

    loose = validated((("vocab", "fsdp"),))
    named = validated((("vocab", "fsdp"), ("foreign_a", "fsdp")))
    assert set(loose.unbound) - set(named.unbound) == {"foreign_a"}
    assert loose.unbound["foreign_a"] == loose.unbound["foreign_b"] == named.unbound["foreign_b"]
