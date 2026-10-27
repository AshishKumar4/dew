"""What a layout's validation reports before a run's first step, read against the state the run then places.

`Trainer.validate` reads the state `fit` would place without placing anything: where each
leaf splits and the part each device holds, the bytes each device keeps between steps, the axes
a batch's rows split over, and the logical axes no rule names. Each claim is held to the arrays
`fit` then places on the eight simulated CPU devices, as is fit's own validation, and a capacity
those bytes overflow is refused, as a CPU mesh standing in for GPUs states it. The collectives it
says a step runs over each axis are the GPU's: each is held to a step compiled by XLA's CUDA backend
for compile-only GPUs on this CPU (tests/kernel_compile.py), whose partitioner sums a gradient into
shares as a GPU does, where XLA:CPU sums it whole.
"""


from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kernel_compile import compile_only_gpus
from model_support import TINY_DECODER
from sharded import collective_axes
from test_parallelism import BoxedReadout
from token_support import repeated_token_batch, token_rows

from dew.data.dataset import Loader
from dew.distributed.placement import placed_shapes
from dew.mesh import DEFAULT_RULES, LayoutRefused, mesh_axes
from dew.nn.kernels.generation import assumed
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import STEP_BATCH
from dew.training.memory import step_compiler_options

pytestmark = pytest.mark.mesh

# A vocabulary no other dimension of the tiny decoder shares, so a shard that holds it whole says so.
VOCAB = 96
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256


def native(devices: Sequence | None = None) -> Mesh:
    """A mesh of its own axes over eight `devices`, this process's first by default: a replica axis
    beside two fsdp levels, all three splitting the rows."""
    return Mesh(np.asarray(jax.devices()[:8] if devices is None else devices).reshape(2, 2, 2),
                ("replica", "zero", "zero2"))


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
    the reported axes; and it names a collective for every axis above one."""
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
    assert set(report.collectives) == {axis for axis, size in run.device_mesh.shape.items() if size > 1}


GPU_BYTES = 40 * 2**30
"""A compile-only GPU's capacity, which reports none: the smallest the generations here come in, an
A100's 40 GB."""

PLACEMENT = ("all-gather", "all-reduce", "reduce-scatter")
"""The collectives a placement implies (`Layout.validate`'s), beside those a model chooses itself (an
attention's or an expert dispatch's exchange) and a pipeline's hand-offs."""


GPU_STEPS = {
    # The 8-GPU wave's rows (ledger 1306): eight A100s over fsdp alone and beside two tensor shards,
    # four H100s over fsdp alone and two beside two tensor shards. Then the sequence axis holding
    # the state in shares (ledger 156), alone on two H100s and beside fsdp, the stage axis's
    # hand-offs beside fsdp, and a native mesh's own axes. Each builds its mesh over that many
    # compile-only GPUs of the generation (`compile_only_gpus`).
    "a100_fsdp8": ("sm80", 8, MeshSpec(fsdp=8).build, Layout(min_shard=TINY_SHARD)),
    "a100_fsdp4_tensor2": ("sm80", 8, MeshSpec(fsdp=4, tensor=2).build, Layout(min_shard=TINY_SHARD)),
    "h100_fsdp4": ("sm90", 4, MeshSpec(fsdp=4).build, Layout(min_shard=TINY_SHARD)),
    "h100_fsdp2_tensor2": ("sm90", 4, MeshSpec(fsdp=2, tensor=2).build, Layout(min_shard=TINY_SHARD)),
    "h100_sequence2": ("sm90", 2, MeshSpec(sequence=2).build, Layout(min_shard=TINY_SHARD)),
    "h100_fsdp2_sequence2": ("sm90", 4, MeshSpec(fsdp=2, sequence=2).build, Layout(min_shard=TINY_SHARD)),
    "h100_fsdp2_stage2": ("sm90", 4, MeshSpec(fsdp=2, stage=2).build, Layout(min_shard=TINY_SHARD)),
    "a100_native": ("sm80", 8, native, NATIVE),
}


def step_loss(objective, variables, batch):
    """`objective`'s loss over `batch` at the first step, a function of the parameters, the rest of
    `variables` held."""
    def loss(params):
        at = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)
        return objective.scalar_loss({**variables, "params": params}, batch, at)[0]
    return loss


@pytest.mark.parametrize("case", list(GPU_STEPS))
def test_a_gpu_step_runs_the_collectives_its_layout_plans(case):
    """A fine-tune's step (its forward, backward and Adam update over the placed state, under the
    step's XLA options for those GPUs) compiled for compile-only GPUs runs every collective its
    layout's report says an axis implies, over that axis; and over each axis whose devices compute
    their own share of a batch and hold the state in shares (fsdp's rows, sequence's positions, a
    native mesh's fsdp levels), the placement's collectives are the report's: it gathers each
    weight whole and sums each gradient into its shares, which a sequence axis's report left out."""
    generation, count, build, layout = GPU_STEPS[case]
    mesh = build(compile_only_gpus(generation, count))
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
    objective, optimizer = LMObjective(model, SEQ_LEN), optax.adam(1e-3)
    variables = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32)))
    shardings = layout.shardings(mesh, variables)
    placed = placed_shapes(variables, shardings)
    report = layout.validate(mesh, placed, capacity=GPU_BYTES)
    rows = NamedSharding(mesh, P(report.batch_axes))
    batch = {"text": jax.ShapeDtypeStruct((BATCH, SEQ_LEN + 1), jnp.int32, sharding=rows)}

    def step(variables, batch):
        params = variables["params"]
        grads = jax.grad(step_loss(objective, variables, batch))(params)
        updates, _ = optimizer.update(grads, optimizer.init(params), params)
        return optax.apply_updates(params, updates)

    with assumed(generation), jax.set_mesh(mesh), nn.logical_axis_rules(layout.axis_rules):
        options = step_compiler_options(objective, BATCH * SEQ_LEN, frozen=False, devices=mesh.size)
        compiled = jax.jit(step, out_shardings=shardings["params"]).trace(placed, batch).lower().compile(
            compiler_options=options)
    ran = collective_axes(compiled.as_text(), mesh)
    for axis, ops in report.collectives.items():
        for op in ops:
            assert any(kind == op and axis in axes for kind, axes in ran), (axis, op)
    split = {axis for leaf in report.leaves.values() for entry in leaf.spec for axis in mesh_axes(entry)}
    for axis in split & {*report.batch_axes, "sequence"}:
        assert {kind for kind, axes in ran if axis in axes and kind in PLACEMENT} == set(
            report.collectives[axis]), axis


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


INITIALIZED: list[bool] = []
"""One entry each time a `Watched` objective's initializer runs on the devices."""


class Watched(LMObjective):
    """An LM objective that notes each time its initializer runs on the devices (`INITIALIZED`), which
    tracing it for shapes never does."""

    def init(self, key, variables=None):
        jax.debug.callback(lambda: INITIALIZED.append(True))
        return super().init(key, variables)


def capped(limit: int) -> type[Trainer]:
    """A trainer class on devices of `limit` bytes each, as a GPU's allocator states its limit."""

    class Capped(Trainer):
        def _validation(self, placed, capacity=None):
            return super()._validation(placed, limit)

    return Capped


@pytest.mark.mesh(devices=8)
def test_a_fit_whose_whole_state_overflows_its_devices_is_refused_before_it_initializes_any():
    """`fit` reads its first batch before it places the state, and checks against it the whole state
    its steps keep (the accumulator its window pools beside the parameters, optimizer state and EMA):
    on devices that hold the state alone but not the whole, it is refused before the objective's
    initializer runs, not after the base is placed."""
    mesh, layout, accumulation = CASES["fsdp_tensor_accumulating"]
    whole = max(fine_tune(mesh(), layout, accumulation).validate(step_batch()).device_bytes.values())
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
    run = capped(whole - 1)(Watched(model, SEQ_LEN, ema_decay=0.999), optax.adam(1e-3),
                            key=jax.random.key(0), mesh=mesh(), layout=layout, accumulation=accumulation,
                            split=1)
    INITIALIZED.clear()
    with pytest.raises(LayoutRefused):
        run.fit(Loader(train=lambda partition: repeated_token_batch(VOCAB, BATCH, SEQ_LEN + 1), val=None,
                       records=None, batch=BATCH), steps=2 * accumulation)
    jax.effects_barrier()
    assert INITIALIZED == []


@pytest.mark.mesh(devices=8)
def test_the_check_before_placing_reads_a_run_in_the_splits_its_plan_may_take():
    """The whole state a run is checked for before anything is placed is read in the splits its plan
    may take: a run left to the plan in its first that fits, here the batch whole, which pools no
    accumulator; a run fixed to two microbatches in those two, whose window pools one. On devices that
    hold the whole batch's state, the first passes and the second is refused (`Trainer.validate`, as
    `fit` checks)."""
    mesh, layout, _ = CASES["fsdp_tensor_accumulating"]

    def peak(split: int | None, capacity: int | None = None) -> int:
        model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
        trainer = Trainer(LMObjective(model, SEQ_LEN, ema_decay=0.999), optax.adam(1e-3),
                          key=jax.random.key(0), mesh=mesh(), layout=layout, split=split)
        return max(trainer.validate(step_batch(), capacity=capacity).device_bytes.values())

    whole = peak(1)
    assert peak(2) > whole
    assert peak(None, capacity=whole) == whole
    with pytest.raises(LayoutRefused):
        peak(2, capacity=whole)


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
