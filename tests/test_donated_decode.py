"""A hybrid model's decode step compiled for a GPU with its cache donated, as serving donates it: each gated
delta net's conv taps read in their checkpoint's order, no launch transposing its weight (row 14174)."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.sharding import NamedSharding, PartitionSpec
from generation_support import COMPILES_ON_CPU
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


@COMPILES_ON_CPU
def test_a_donated_decode_reads_each_conv_weight_as_stored():
    """On an H100, no fusion of the step reads a gated delta net's conv weight alone: the short conv takes
    its taps in the checkpoint's `[C, 1, K]` order, where read down `[K, C]` the step transposed each layer's
    weight first, weight-only work a layer a step (row 14174: 12 of an LM_HYBRID B8 step's)."""
    instructions = list(hlo.instructions(_decode_module('sm90')))
    entry = {op.name: op for op in instructions if op.entry}
    weights = {op.name for op in entry.values() if op.opcode == 'parameter' and 'conv1d____weight' in op.name}
    alone = [op.name for op in entry.values() if op.opcode == 'fusion' and op.operands
             and set(op.operands) <= weights]
    assert weights and not alone, alone
