"""The derived peak of a decoder step with a whole vocabulary head is at most what XLA compiles it to (row
2440): on Qwen3.5-4B at two rows of 4,096 tokens the derivation held 56.8 GB where XLA's compile held 29.8,
and 29.7 against 15.0 at one row twice. The walk reads the jaxpr and XLA compiles the program JAX lowers it
to, and five terms the walk counted are not in that program. Each test states one, with the rule that
removes it, on a program small enough to read."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from affine_run import Features
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from model_support import TINY_DECODER
from token_support import token_rows

from dew._records import listed, record
from dew.data.dataset import Loader, replayed
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, staging
from dew.training.ahead import compile_ahead, compile_only_gpus
from dew.training.compiled_program import compiled
from dew.training.footprint import Footprint, Workspace, gradient_peak, peak_bytes, traced_loss
from dew.training.inspection import Inspection
from dew.training.transaction import consumed_where_made, leaf_local

TOKENS, VOCAB, WIDTH = 256, 4096, 64
PRODUCT = TOKENS * VOCAB * 4
"""One float32 value over every token and vocabulary column, 4 MiB."""


def program_peak(function, *specs) -> int:
    """The most bytes the program JAX lowers `function` to for CUDA holds at once, as the derivation reads
    it, with no cuBLAS workspace."""
    program = jax.make_jaxpr(function)(*specs).jaxpr
    return peak_bytes(compiled(program, "cuda"), "gpu", Workspace(0, 0))


def compiled_temporaries(function, *specs) -> int:
    """The temporaries XLA's compile of `function` for a compile-only H100 states, its Triton GEMMs off as
    the planner compiles a step, so each product is one cuBLAS call."""
    (gpu,) = compile_only_gpus("sm90", 1, 1 << 30)
    sharding = NamedSharding(Mesh(np.asarray([gpu]), ("data",)), P())
    placed = jax.tree.map(lambda spec: jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding), specs)
    stats = jax.jit(function).lower(*placed).compile(
        compiler_options={"xla_gpu_enable_triton_gemm": False}).memory_analysis()
    assert stats is not None, "the compile states no memory"
    return stats.temp_size_in_bytes


HIDDEN = jax.ShapeDtypeStruct((TOKENS, WIDTH), jnp.float32)
TABLE = jax.ShapeDtypeStruct((VOCAB, WIDTH), jnp.float32)


def product(hidden, table):
    """Each token's logits over the vocabulary, contracting the width of both, as the head's product does."""
    return jax.lax.dot_general(hidden, table, (((1,), (1,)), ((), ())))


def test_a_variadic_reduction_reads_the_values_it_fuses_from_nowhere():
    """XLA fuses the elementwise values a reduction reads into the reduction's own kernel, a `kInput` fusion
    whatever the number of operands (`PriorityFusion::ChooseKind`, `CanFuseWithNativeEmitter`,
    xla/backends/gpu/transforms/priority_fusion.cc), so the variadic `jax.lax.reduce` that takes a loss's row
    terms holds its product and its two outputs and nothing else. The walk made a buffer of each operand's
    chain for a reduction whose primitive is not `reduce_sum` or `reduce_max`: four float32 values over the
    tokens and the vocabulary at the head, 32.5 GB of Qwen3.5-4B's 56.8."""
    def combine(left, right):
        return jnp.maximum(left[0], right[0]), left[1] + right[1]

    def row_terms(hidden, table):
        logits = product(hidden, table)
        peak, total = jax.lax.reduce((logits, jnp.exp(logits)), (-jnp.inf, 0.0), combine, (1,))
        return jnp.sum(peak + total)

    held = program_peak(row_terms, HIDDEN, TABLE)
    assert held == PRODUCT + 2 * TOKENS * 4
    assert held <= compiled_temporaries(row_terms, HIDDEN, TABLE)


def test_an_optimization_barrier_holds_the_storage_of_what_it_is_given():
    """An optimization barrier defines no value: the values of its result are its operand's
    (`HloDataflowAnalysis::UpdateOptimizationBarrierValueSet`, xla/hlo/analysis/hlo_dataflow_analysis.cc).
    A logits product rounded to bfloat16 behind one (`rounded_operand`) holds the product and the one
    rounded copy, not a copy for the barrier to read beside the copy it returns."""
    def rounded(hidden, table):
        kept = jax.lax.optimization_barrier(product(hidden, table).astype(jnp.bfloat16))
        return jnp.sum(jnp.square(kept.astype(jnp.float32)))

    held = program_peak(rounded, HIDDEN, TABLE)
    assert held == PRODUCT + PRODUCT // 2
    assert held <= compiled_temporaries(rounded, HIDDEN, TABLE)


def test_a_barrier_over_storage_holds_that_storage_alone():
    """A barrier over a view of a product, and a barrier over that barrier, hold the product's storage and
    no copy of it (`HloDataflowAnalysis::UpdateOptimizationBarrierValueSet`)."""
    def nested(hidden, table):
        kept = jax.lax.optimization_barrier(jax.lax.optimization_barrier(product(hidden, table).reshape(-1)))
        return jnp.sum(jnp.square(kept))

    held = program_peak(nested, HIDDEN, TABLE)
    assert PRODUCT <= held < PRODUCT + PRODUCT // 2
    assert held <= compiled_temporaries(nested, HIDDEN, TABLE)


def test_each_result_of_a_barrier_holds_its_own_operand_to_its_own_last_read():
    """Each result of a barrier is its own operand's storage, read where that result is read: the second
    product, read last by its sum, is gone by the time a third product is made beside the first."""
    def apart(hidden, table):
        first, second = jax.lax.optimization_barrier((product(hidden, table), product(2 * hidden, table)))
        early = jnp.sum(second)
        third = product(3 * hidden, table) + first
        return early + jnp.sum(third)

    held = program_peak(apart, HIDDEN, TABLE)
    assert 2 * PRODUCT <= held < 3 * PRODUCT
    assert held <= compiled_temporaries(apart, HIDDEN, TABLE)


def test_a_cotangent_sum_is_an_elementwise_value_like_any_other():
    """JAX lowers the sum of two cotangents, `add_any`, to one `add` (`add_jaxvals_lowering`,
    jax/_src/interpreters/mlir.py), which XLA's fusion recomputes in its consumers as it does any elementwise
    value. A value read twice, here a product an exponential and a sine both read, takes the gradient's sum of
    their cotangents into the one matrix product that reads it, held once for that read."""
    weights = jax.ShapeDtypeStruct((WIDTH, VOCAB), jnp.float32)

    def twice(params, rows):
        wide = rows @ params
        return jnp.sum(jnp.exp(wide) + jnp.sin(wide)), ()

    peak = gradient_peak(twice, weights, (HIDDEN,), platform="gpu", workspace=Workspace(0, 0)).bytes
    # The product the exponential and the sine read, the cotangent sum made for the product that reads it,
    # and the weights' gradient that product makes.
    assert peak == 2 * PRODUCT + WIDTH * VOCAB * 4


def test_a_gradient_the_step_updates_where_it_is_made_is_held_while_it_is_made():
    """The step reads each gradient where it is made, into the buffers the state donates: the optimizer's
    update writes the parameter and its moments, or the accumulator the sum, in place (the donated state's
    outputs alias its arguments, `HloInputOutputAliasConfig`), and XLA's memory-minimizing schedule runs a
    consumer that frees its operand as soon as it is ready (`ListScheduler::BytesFreedIfScheduled`,
    xla/hlo/transforms/simplifiers/hlo_memory_scheduler.cc; `ScheduleGpuModule`,
    xla/service/gpu/gpu_hlo_schedule.cc). Eight layers' gradients, 4 MiB each, are not held at once, as
    XLA's compile of the step shows: it holds what the backward holds and one gradient at a time."""
    layers = [jax.ShapeDtypeStruct((512, 512), jnp.float32)] * 8
    rows = jax.ShapeDtypeStruct((64, 512), jnp.float32)

    def loss(layers, rows):
        hidden = rows
        for weights in layers:
            hidden = jnp.sin(hidden @ weights)
        return jnp.sum(hidden), ()

    def step(layers, rows):
        gradient = jax.grad(lambda layers: loss(layers, rows)[0])(layers)
        return [weights - 0.1 * each for weights, each in zip(layers, gradient, strict=True)]

    peak = gradient_peak(loss, layers, (rows,), platform="gpu", workspace=Workspace(0, 0),
                         consumed=True).bytes
    assert peak < 8 * 512 * 512 * 4 // 2
    assert peak <= compiled_temporaries(step, layers, rows)


def test_a_gradient_is_consumed_where_it_is_made_only_by_an_update_that_reads_it_alone():
    """Adam reads each gradient into its own parameter's update and moments alone, so the step consumes
    each where it is made; clipping by the global norm reads every gradient into each update, so the step
    holds them all until it has read the last (`leaf_local`)."""
    params = {"first": jax.ShapeDtypeStruct((16, 64), jnp.float32),
              "second": jax.ShapeDtypeStruct((64, 16), jnp.float32)}
    adam, clipped = optax.adam(1e-3), optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    assert leaf_local(adam, params, jax.eval_shape(adam.init, params), progressed=False)
    assert not leaf_local(clipped, params, jax.eval_shape(clipped.init, params), progressed=False)

    handed = []

    def scaled(updates, state, params=None, *, progress=None, **extra):
        handed.append(progress is not None)
        return jax.tree.map(lambda update: update if progress is None else update * progress, updates), state

    # An update on the run's progress is handed it where the run has one, as the step hands it
    # (`Transaction.commit`, `optimizer_update`), and none where the run is bounded otherwise.
    clocked = optax.GradientTransformationExtraArgs(lambda params: optax.EmptyState(), scaled)
    assert leaf_local(clocked, params, optax.EmptyState(), progressed=True)
    assert leaf_local(clocked, params, optax.EmptyState(), progressed=False)
    assert handed == [True, False]


def test_a_step_consumes_a_gradient_where_it_is_made_only_into_device_resident_state():
    """A step whose optimizer state and EMA are in device memory updates each gradient where it is made;
    one whose EMA is fetched from the host fetches it behind one barrier over every gradient
    (`Transaction.past`), and one whose loss is dynamically scaled commits only once every gradient is
    known finite, either holding them all (`consumed_where_made`)."""
    device, host = Layout(min_shard=1, tolerance=1.0), Layout(min_shard=1, tolerance=1.0, host=("ema",))
    for layout, scaled, consumed in ((device, False, True), (host, False, False), (device, True, False)):
        trainer = Trainer(Features(), optax.adam(0.1), key=jax.random.key(0), layout=layout,
                          dynamic_scale=scaled)
        state, _, _ = trainer.place()
        assert consumed_where_made(trainer.objective, trainer.optimizer, state, 1, shared=True,
                                   progressed=False) is consumed


def test_a_diloco_groups_step_is_planned_by_its_own_optimizer_state():
    """A DiLoCo run's state holds the outer optimizer's state, Nesterov's trace, and each group its own
    (`TrainState.inner`), which the group's step commits by the run's optimizer (`diloco.replica`). Planned
    for a GPU at a stated capacity, compiled ahead, a run of one group derives its step's peak, reading
    whether that step consumes each gradient where it is made from the group's own fields (`group_of`),
    Adam over its moments; the run's optimizer traced over the outer state failed, and the group's view
    made an array on a GPU that is only compiled for."""
    from dew.training import DiLoCo
    from dew.training.diloco import group_of

    trainer = Trainer(Features(), optax.adam(0.1), key=jax.random.key(0), diloco=DiLoCo(every=2),
                      layout=Layout(min_shard=1, tolerance=1.0),
                      mesh=MeshSpec(diloco=1).build(compile_only_gpus("sm90", 1)))
    rows = {"x": np.ones((8, 3), np.float32), "y": np.zeros((8, 2), np.float32)}
    ahead = compile_ahead(trainer, rows, capacity=1 << 30)
    assert [probe["derived"]["peak"] is not None for probe in ahead.plan["probes"]] == [True]
    # The answer the peak read, over the group's view of a run of one group placed on this host: Adam over
    # the group's in-device moments consumes each gradient where it is made (row 3806).
    placed = Trainer(Features(), optax.adam(0.1), key=jax.random.key(0), diloco=DiLoCo(every=2),
                     layout=Layout(min_shard=1, tolerance=1.0), mesh=MeshSpec(diloco=1))
    state, _, _ = placed.place()
    with group_of(state, rows) as (view, _):
        assert consumed_where_made(placed.objective, placed.optimizer, view, 1, shared=True, progressed=False)


def test_a_step_whose_head_takes_its_gradient_in_the_backward_holds_every_gradient():
    """A decoder's head that takes its gradient in the backward a tile at a time holds every gradient to the
    step's end, as XLA schedules its weight gradients and updates after the whole backward (row 3653); its
    whole logits, and its tile taking its gradient in the forward, consume each where it is made, as does
    a head past a transform, which no matrix alone is and so keeps its whole logits whatever its tile
    (`chunked.head_cross_entropy`), the step's loss traced as the derivation traces it (`traced_loss`)."""
    batch = {key: jnp.asarray(rows) for key, rows in token_rows(VOCAB, 2, 17).items()}
    for transform, cases in ((None, (((16, 128), False, False), ((16, 128), True, True), (None, True, True))),
                             ("gelu", (((16, 128), False, True),))):
        model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, head_transform=transform)
        objective = LMObjective(model, 16, head_tile=(16, 128))
        trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                          layout=Layout(min_shard=1, tolerance=1.0))
        state, _, _ = trainer.place()
        for tile, fused, consumed in cases:
            objective.restore_head(tile, fused=fused)
            loss, arguments = traced_loss(objective, state, 1, 1)
            jax.eval_shape(loss, arguments[0], batch, *arguments[1:])
            assert consumed_where_made(objective, trainer.optimizer, state, 1, shared=True,
                                       progressed=False) is consumed


class Layers(Features):
    """`Features` over four like layers, each a 32 x 32 kernel and its bias."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential([nn.Dense(32) for _ in range(3)] + [nn.Dense(2)])

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, 32)))


def staged_of(trainer: Trainer) -> int:
    """The bytes of its host-resident commit fields `trainer`'s step holds at once, as its own staged cut
    measures them wherever its fit traces it (`staging.measured`), the sampled step among them; the plan's
    footprint (`Footprint.staged`) holds the step's own."""
    rows = {"x": np.ones((16, 32), np.float32), "y": np.zeros((16, 2), np.float32)}
    data = Loader(replayed(lambda partition: itertools.repeat(rows)), None, batch=16, records=None)
    with staging.measured() as cuts:
        trainer.fit(data, steps=1, log_every=1)
    measured = max(cuts, default=0)
    assert [probe["derived"]["staged"] for probe in trainer.plan["probes"]] == [measured], trainer.plan
    return measured


def test_a_commit_over_host_resident_state_holds_two_of_its_chunks():
    """A step whose Adam moments and EMA rest in pinned host memory stages them a parameter at a time,
    each parameter's moments and average one chunk (`dew.training.staging`): its commit holds the two
    largest, two 32 x 32 kernels' three copies, beside its gradient and the moments' count, which it
    fetches first, as its own traced cut measures it (`staging.measured`) and its plan's footprint reads
    it, as does one that samples its update's norms (`Trainer(inspection=...)`, row 3659); a window of
    two, committing under a conditional, stages it whole; one whose state the device holds stages none."""
    device = Layout(min_shard=1, tolerance=1.0)
    host = Layout(min_shard=1, tolerance=1.0, host=("opt_state", "ema"))
    trainer = Trainer(Layers(), optax.adam(0.1), key=jax.random.key(0), layout=host)
    state, _, _ = trainer.place()
    leaves = jax.tree.leaves((state.opt_state, state.ema))
    whole, count = sum(leaf.nbytes for leaf in leaves), sum(leaf.nbytes for leaf in leaves if leaf.ndim == 0)
    assert count == 4
    assert staged_of(trainer) == count + 2 * 3 * 32 * 32 * 4
    sampled = Trainer(Layers(), optax.adam(0.1), key=jax.random.key(0), layout=host,
                      inspection=Inspection(every=1))
    assert staged_of(sampled) == count + 2 * 3 * 32 * 32 * 4
    windowed = Trainer(Layers(), optax.adam(0.1), key=jax.random.key(0), layout=host, accumulation=2)
    assert staged_of(windowed) == whole
    resident = Trainer(Layers(), optax.adam(0.1), key=jax.random.key(0), layout=device)
    assert staged_of(resident) == 0


def test_a_device_resident_steps_temporaries_are_its_gradient_programs_peak():
    """A step whose state the device holds consumes each gradient where it is made, its derived peak below
    its whole gradient, and stages nothing: its temporaries are that peak. One that stages host-resident
    fields holds its whole gradient and them as it commits."""
    streamed = Footprint(residuals=0, gradients=100, held=0, peak=60)
    assert streamed.temporaries == 60
    assert dataclasses.replace(streamed, staged=10).temporaries == 110


def test_a_differentiated_checkpoint_keeps_its_operands_behind_its_barrier():
    """A differentiated checkpoint that prevents CSE lowers its operands through an optimization barrier
    (`_remat_lowering`, jax/_src/ad_checkpoint.py), which keeps them in a buffer the backward reads: the
    logits a checkpointed square reads are held, where without the barrier they are recomputed."""
    weights = jax.ShapeDtypeStruct((WIDTH, VOCAB), jnp.float32)

    def squared(prevent_cse: bool):
        def loss(weights, rows):
            square = jax.checkpoint(lambda logits: jnp.sum(logits * logits), prevent_cse=prevent_cse)
            return square(jnp.sin(rows @ weights)), ()
        return loss

    kept, free = (gradient_peak(squared(prevent), weights, (HIDDEN,), platform="gpu",
                                workspace=Workspace(0, 0)).bytes for prevent in (True, False))
    assert kept >= free + PRODUCT


def test_a_whole_head_step_is_derived_at_most_what_xla_compiles_it_to(tmp_path, monkeypatch):
    """A decoder computing in bfloat16 over a vocabulary of 32,768 columns, its head whole, compiled ahead
    for an H100: the derived peak of its first candidate is at most the temporaries XLA's compile of it
    holds, and not under nine tenths of them. The miss at Qwen3.5-4B's widths ran 1.9 to 2.0 times the
    compile's; here the same terms ran 3.2 times."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    vocab, seq_len = 32768, 128
    model = models.build("causal_transformer", vocab_size=vocab, **{**TINY_DECODER, "max_seq_len": seq_len},
                         dtype=jnp.bfloat16, num_kv_heads=2)
    trainer = Trainer(LMObjective(model, seq_len), optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec().build(compile_only_gpus("sm90", 1)), layout=Layout(min_shard=256))
    # The suite's lane multiplies at its highest precision, which keeps a head's logits in float32.
    with jax.default_matmul_precision("default"):
        ahead = compile_ahead(trainer, token_rows(vocab, 2, seq_len + 1), capacity=80 << 30)
    (probe,) = listed(record(ahead.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert 0.9 * probe["temp"] <= probe["derived"]["peak"] <= probe["temp"]
