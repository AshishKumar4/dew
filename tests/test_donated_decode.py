"""A hybrid model's decode step compiled for a GPU with its cache donated, as serving donates it: each gated
delta net's recurrent state stepped in place and its conv taps read in their checkpoint's order (row
14174)."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.sharding import NamedSharding, PartitionSpec
from kernel_compile import _device, assumed, gpu_target

from dew.nn.kernels import generation
from dew.registry import models
from dew.telemetry import hlo
from tools.benchmark_cases import LM_HYBRID


def _decode_module(target: str):
    """`LM_HYBRID`'s (three gated delta nets to each attention layer) B8 decode step, its cache donated,
    compiled for a compile-only `target` at the default matmul precision: its optimized module."""
    model = models.build('causal_transformer', {**LM_HYBRID, 'vocab_size': 1024, 'dtype': 'bfloat16'})
    ids = jnp.zeros((8, 1), jnp.int32)
    mesh = _device(target)
    sharding = NamedSharding(mesh, PartitionSpec())
    with generation.chosen(layer_scan='plain'), jax.default_matmul_precision('default'):
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


def test_a_donated_decode_steps_each_recurrent_state_in_place_reading_its_taps_as_stored():
    """On an H100, each gated delta net's step reads its donated recurrent state and writes its successor in
    one operand (its custom call's output aliased to it), so no copy of a state's shape follows it: returned
    as an output of its own, the call wrote the successor apart and the cache copied it back, 4 MiB a layer
    a step; and no fusion reads a conv weight alone, the short conv taking its taps in the checkpoint's
    `[C, 1, K]` order: read down `[K, C]`, the step transposed each layer's weight first (row 14174: 12 of
    each a step)."""
    instructions = list(hlo.instructions(_decode_module('sm90')))
    entry = {op.name: op for op in instructions if op.entry}
    users = {name: [] for name in entry}
    for op in instructions:
        if op.entry:
            for operand in op.operands:
                users[operand].append(op)

    def parameters(named: str) -> list:
        return [op for op in entry.values() if op.opcode == 'parameter' and named in op.name]

    def shaped(op) -> tuple:
        return tuple(op.proto.shape.dimensions), op.proto.shape.element_type

    states, weights = parameters('recurrent_state'), parameters('conv1d____weight')
    copies = [op for op in entry.values() if op.opcode == 'copy' and shaped(op) in set(map(shaped, states))]
    alone = [op for weight in weights for op in users[weight.name] if op.opcode == 'fusion']
    aliased = [state for state in states for op in users[state.name]
               if any(op.operands[alias.operand_index] == state.name
                      for alias in op.proto.output_operand_aliasing)]
    assert states and len(states) == len(weights)
    assert (len(copies), len(alone), len(aliased)) == (0, 0, len(states))
