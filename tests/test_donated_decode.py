"""A hybrid model's decode step compiled for a GPU with its cache donated, as serving donates it: each gated
delta net's conv taps read in their checkpoint's order by either decode kernel, no launch repacking its
weight (row 14174)."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from generation_support import COMPILES_ON_CPU
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.sharding import NamedSharding, PartitionSpec
from kernel_compile import _device, assumed, gpu_target

from dew.nn.kernels import generation
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
    instructions = [op for op in hlo.instructions(_decode_module('sm90', delta_decode)) if op.entry]
    users: dict[str, list] = {op.name: [] for op in instructions}
    for op in instructions:
        for operand in op.operands:
            users[operand].append(op)
    weights = [op.name for op in instructions if op.opcode == 'parameter' and 'conv1d____weight' in op.name]

    def reached(name: str) -> tuple[set, set]:
        """The kernels `name` reaches through no-copy views, and the other instructions it meets."""
        kernels, others, frontier = set(), set(), [name]
        while frontier:
            for op in users[frontier.pop()]:
                if op.opcode == 'bitcast':
                    frontier.append(op.name)
                elif op.opcode == 'custom-call':
                    kernels.add(op.custom_call_target)
                else:
                    others.add(f'{op.opcode} {op.name}')
        return kernels, others

    found = {weight: reached(weight) for weight in weights}
    assert weights and all(kernels and not others for kernels, others in found.values()), found
