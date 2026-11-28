"""A hybrid model's decode step compiled for a GPU with its cache donated, as serving donates it: each gated
delta net's conv taps read in their checkpoint's order by either decode kernel, no launch repacking its
weight (row 14174)."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from generation_support import COMPILES_ON_CPU
from jax._src import config as jax_config
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.sharding import NamedSharding, PartitionSpec
from kernel_compile import _device, assumed, gpu_target

from dew.nn.kernels import generation
from dew.nn.kernels.templates.recurrence.step import donated_state
from dew.nn.linear import decode_gated_delta_rule
from dew.registry import models
from dew.telemetry import hlo
from tools.benchmark_cases import LM_HYBRID


def _decode_module(target: str, delta_decode: str):
    """`LM_HYBRID`'s (three gated delta nets to each attention layer) B8 decode step, its cache donated,
    compiled for a compile-only `target` at the default matmul precision, its gated delta nets' tokens on
    `delta_decode`'s kernel (`generation.chosen`): its optimized module."""
    model = models.build('causal_transformer', {**LM_HYBRID, 'vocab_size': 1024, 'dtype': 'bfloat16'})
    ids = jnp.zeros((8, 1), jnp.int32)
    mesh = _device(target)
    sharding = NamedSharding(mesh, PartitionSpec())
    with (generation.chosen(layer_scan='plain', delta_decode=delta_decode),
          jax.default_matmul_precision('default')):
        variables = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids)))
        cache = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids, decode=True)))['cache']
        placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
                              (variables, cache))

        def step(variables, cache):
            return model.apply({**variables, 'cache': cache}, ids, decode=True, mutable=['cache'])

        with (assumed(target), jax.set_mesh(mesh),
              mgpu_core.artificial_shared_memory_limit(gpu_target(target).shared_limit)):
            compiled = jax.jit(step, donate_argnums=(1,)).trace(*placed).lower().compile()
    (module,) = compiled.runtime_executable().hlo_modules()
    return module


@COMPILES_ON_CPU
@pytest.mark.parametrize('delta_decode', ['step', 'fused'])
def test_a_donated_decode_reads_each_conv_weight_as_stored(delta_decode):
    """On an H100, each gated delta net's conv weight reaches the kernel that reads it (the short conv's
    before the step alone, the fused step's taking the conv) through no-copy views alone (`bitcast`): the
    kernels read their taps in the checkpoint's `[C, 1, K]` order (`conv.read_taps`), where read down
    `[K, C]` the step transposed each layer's weight first, weight-only work a layer a step (row 14174: 12 of
    an LM_HYBRID B8 step's), and a repack of it, a copy or a transpose of its own, would read it twice."""
    every = list(hlo.instructions(_decode_module('sm90', delta_decode)))
    users: dict[tuple[str, str], list] = {(op.computation, op.name): [] for op in every}
    for op in every:
        for operand in op.operands:
            users[op.computation, operand].append(op)
    weights = [op.name for op in every
               if op.entry and op.opcode == 'parameter' and 'conv1d____weight' in op.name]

    def parameter(computation: str, index: int) -> str:
        return next(op.name for op in every if op.computation == computation and op.opcode == 'parameter'
                    and op.proto.parameter_number == index)

    def reached(computation: str, name: str) -> tuple[set, set]:
        """The kernels `name` reaches through no-copy views (`bitcast`), into a fusion through its parameter
        (XLA's dynamic-slice fusion wrapping the fused step's launch), and the other instructions it meets."""
        kernels, others, frontier = set(), set(), [(computation, name)]
        while frontier:
            where, value = frontier.pop()
            for op in users[where, value]:
                if op.opcode == 'bitcast':
                    frontier.append((where, op.name))
                elif op.opcode == 'custom-call':
                    kernels.add(op.name)
                elif op.opcode == 'fusion':
                    (called,) = op.called
                    frontier.extend((called, parameter(called, index))
                                    for index, operand in enumerate(op.operands) if operand == value)
                else:
                    others.add(f'{op.opcode} {op.name}')
        return kernels, others

    entry = next(op.computation for op in every if op.entry)
    found = {weight: reached(entry, weight) for weight in weights}
    assert weights and all(kernels and not others for kernels, others in found.values()), found


@COMPILES_ON_CPU
@pytest.mark.parametrize('shards,checked', [(1, False), (2, False), (2, True)],
                         ids=['one device', 'split', 'in a checked map'])
def test_a_donated_state_steps_in_place_unless_a_checked_map_types_it(shards, checked):
    """A gated delta net's decode token (`decode_gated_delta_rule`) on H100s, its state donated by a program
    that owns it (`donated_state`): on one device and split over two (the step's own map, `manual_map`), its
    launch reads and writes the state in place (one operand aliased to its output), where an output of its
    own had the donated cache copy it back, 4 MiB a layer a step (row 14174); called inside a checked map
    (`jax.shard_map`, `check_vma`), whose state keeps its manual axes' type, which no launch takes as a ref,
    it takes the state's own output and compiles (head review of row 14174's in-place state). The same jitted
    function traced outside the scope, first, keeps the state's own output: the scope keys its trace."""
    mesh = _device('sm90', (('data', shards),))
    sharding = NamedSharding(mesh, PartitionSpec('data'))

    def spec(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)

    rows, heads, width = 8, 16, 128
    operands = (*(spec((rows, 1, heads, width), jnp.bfloat16) for _ in range(3)),
                *(spec((rows, 1, heads), jnp.float32) for _ in range(2)),
                spec((rows, heads, width, width), jnp.bfloat16))
    decode = decode_gated_delta_rule
    if checked:
        rows_spec = PartitionSpec('data')
        decode = jax.shard_map(decode_gated_delta_rule, in_specs=(rows_spec,) * 6, out_specs=rows_spec,
                               check_vma=True)
    stepped = jax.jit(decode, donate_argnums=(5,))

    def aliased() -> list:
        """The custom calls of `stepped`'s program whose output aliases an operand."""
        with (assumed('sm90'), jax.set_mesh(mesh),
              mgpu_core.artificial_shared_memory_limit(gpu_target('sm90').shared_limit)):
            (module,) = stepped.trace(*operands).lower().compile().runtime_executable().hlo_modules()
        return [op.name for op in hlo.instructions(module) if op.opcode == 'custom-call'
                and op.proto.output_operand_aliasing]

    # The same jitted function on either side of the scope: its trace keyed on it (`_cache.DONATED_STATE`).
    kept = aliased()
    with donated_state():
        donated = aliased()
    assert not kept and bool(donated) == (not checked), (kept, donated)


@COMPILES_ON_CPU
def test_a_donated_states_step_in_a_checked_map_types_its_state_as_its_operands():
    """A decode token (`decode_gated_delta_rule`) traced in a program that donates its state (`donated_state`)
    inside a checked map over two H100s (`jax.shard_map`, `check_vma`): its initial state replicated, one
    zero row, beside a split `active`, the next state varies over the map's axis as its rows do, so a
    `psum` of it traces with no implicit cast (`jax_auto_pcast` off). Frozen from an in-place ref, the next
    state kept the initial state's replicated type, and the `psum` refused it (train 1511's K-F1)."""
    mesh = _device('sm90', (('data', 2),))
    rows, heads, width = 1, 16, 128
    split, whole = PartitionSpec('data'), PartitionSpec()

    def step(query, key, value, g, beta, state, active):
        _, next_state = decode_gated_delta_rule(query, key, value, g, beta, state, active)
        return jax.lax.psum(next_state.astype(jnp.float32), 'data')

    operands = (*(jax.ShapeDtypeStruct((rows, 1, heads, width), jnp.bfloat16) for _ in range(3)),
                *(jax.ShapeDtypeStruct((rows, 1, heads), jnp.float32) for _ in range(2)),
                jax.ShapeDtypeStruct((rows, heads, width, width), jnp.bfloat16),
                jax.ShapeDtypeStruct((2,), jnp.bool_))
    mapped = jax.shard_map(step, in_specs=(whole,) * 6 + (split,), out_specs=whole, check_vma=True)
    with (assumed('sm90'), jax.set_mesh(mesh), donated_state(), jax_config.auto_pcast(new_val=False),
          mgpu_core.artificial_shared_memory_limit(gpu_target('sm90').shared_limit)):
        jax.jit(mapped).trace(*operands).lower()
