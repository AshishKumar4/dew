"""The derived peak of a decoder step with a whole vocabulary head is at most what XLA compiles it to (row
2440): on Qwen3.5-4B at two rows of 4,096 tokens the derivation held 56.8 GB where XLA's compile held 29.8,
and 29.7 against 15.0 at one row twice. The walk reads the jaxpr and XLA compiles the program JAX lowers it
to, and five terms the walk counted are not in that program. Each test states one, with the rule that
removes it, on a program small enough to read."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from model_support import TINY_DECODER
from token_support import token_rows

from dew._records import listed, record
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead, compile_only_gpus
from dew.training.compiled_program import compiled
from dew.training.footprint import Workspace, gradient_peak, peak_bytes

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


def test_a_call_returns_the_buffers_of_its_program_as_the_program_holds_them():
    """XLA sees no call: a `jit` is a call its `CallInliner` inlines (`_pjit_lowering`, jax/_src/pjit.py;
    `GpuCompiler::RunPreSPMDPartitionerPasses`, xla/service/gpu/gpu_compiler.cc), and a custom derivative's
    primal lowers to its program in the caller (`_custom_jvp_vjp_call_lowering`,
    jax/_src/custom_derivatives.py). A result the program returns as its own parameter is the operand's
    buffer, the embedding table a lookup returns beside its rows (2.5 GB of Qwen3.5-4B's step, held from the
    lookup to the end), and one the program computes of another result by a cheap conversion is recomputed
    where it is read, the float32 logits its head returns beside the bfloat16 ones (8.1 GB at two rows)."""
    ids = jax.ShapeDtypeStruct((TOKENS,), jnp.int32)

    def lookup(table, ids):
        rows = table[ids]
        return rows, table

    def passed(table, ids):
        rows, same = jax.jit(lookup)(table, ids)
        return jnp.sum(rows) + jnp.sum(jnp.square(same))

    def alone(table, ids):
        rows, same = lookup(table, ids)
        return jnp.sum(rows) + jnp.sum(jnp.square(same))

    assert program_peak(passed, TABLE, ids) == program_peak(alone, TABLE, ids)

    @jax.custom_jvp
    def logits(hidden, table):
        narrow = product(hidden, table).astype(jnp.bfloat16)
        return narrow, narrow.astype(jnp.float32)

    @logits.defjvp
    def logits_jvp(primals, tangents):
        raise AssertionError("the walk differentiates nothing here")

    def both(hidden, table):
        narrow, wide = logits(hidden, table)
        return jnp.sum(jnp.square(wide)) + jnp.sum(narrow)

    def alone_logits(hidden, table):
        narrow = product(hidden, table).astype(jnp.bfloat16)
        return jnp.sum(jnp.square(narrow.astype(jnp.float32))) + jnp.sum(narrow)

    # The product and the two sums, each a scalar: both results are conversions of the product, recomputed
    # where the sums read them, as without the call.
    assert program_peak(both, HIDDEN, TABLE) == program_peak(alone_logits, HIDDEN, TABLE) == PRODUCT + 2 * 4


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

    peak = gradient_peak(loss, layers, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes
    assert peak < 8 * 512 * 512 * 4 // 2
    assert peak <= compiled_temporaries(step, layers, rows)


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
