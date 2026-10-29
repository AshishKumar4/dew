"""Diagnostic only: independent-process H100 forward variants."""
import os
import sys

if "nograph" in sys.argv[1]:
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
if "fullcore" in sys.argv[1]:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention as template
from dew.nn.kernels.templates import attention_mosaic as a
from dew.nn.kernels.templates import mosaic, mosaic_sm90
from dew.nn.kernels.templates.attention_mosaic_program import Tiles

variant = sys.argv[1]
width = int(variant[1:]) if variant.startswith("d") else 128
consumers = 2 if variant in ("consumers2", "consumers2forced") else 1
stages = int(variant[-1]) if variant.startswith("stages") else 2
residuals = variant != "noresiduals"
causal = variant == "causal"

if "coordzero" in variant:
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import llvm
    original_asm = llvm.inline_asm
    def is_zero(value):
        op = value.owner
        if not isinstance(op, ir.Operation):
            return False
        if op.name in ("arith.index_cast", "arith.index_castui", "llvm.trunc", "llvm.sext"):
            return is_zero(op.operands[0])
        if op.name in ("arith.constant", "llvm.mlir.constant"):
            return ir.IntegerAttr(op.attributes["value"]).value == 0
        return False
    def coordinate_asm(result_type, operands, asm, constraints, **kwargs):
        if "cp.async.bulk.tensor.5d" in asm and "shared::cta.global" in asm:
            operands = list(operands)
            for index in range(4, len(operands)):
                if is_zero(operands[index]):
                    operands[index] = original_asm(
                        operands[index].type, [],
                        "mov.u32 $0, %tid.x; and.b32 $0, $0, 127;", "=r",
                        has_side_effects=True, asm_dialect=0)
        return original_asm(result_type, operands, asm, constraints, **kwargs)
    llvm.inline_asm = coordinate_asm

if variant == "consumers2forced":
    a.MosaicForward.register_budget = lambda self, config: 256
if "swizzle64" in variant:
    previous_staged = mosaic.staged
    mosaic.staged = lambda shape, dtype, swizzle=None: previous_staged(
        shape, dtype, 64 if swizzle is None else swizzle)

if "nosetmax" in variant:
    def roles(wg, consumers, consumer, producer):
        pl.when(wg < consumers)(lambda: consumer(wg))
        pl.when(wg == consumers)(producer)
    mosaic.roles = roles
elif variant == "serialmma":
    original = mosaic_sm90.Wgmma.register_products
    def serial(self, acc, parts, rhs, interpret):
        self.settled()
        result = original(self, acc, parts, rhs, interpret)
        self.settled()
        return result
    mosaic_sm90.Wgmma.register_products = serial

if variant in ("initfence", "initfence_nosetmax"):
    from jax.experimental.pallas import mosaic_gpu as plgpu
    previous_roles = mosaic.roles
    def fenced_roles(wg, consumers, consumer, producer):
        plgpu.commit_smem()
        previous_roles(wg, consumers, consumer, producer)
    mosaic.roles = fenced_roles

rng = np.random.default_rng(0)
q, k, v = (
    jnp.asarray(rng.standard_normal((2, 4096, heads, width), dtype=np.float32) * scale,
                jnp.bfloat16)
    for heads, scale in ((32, 1), (8, 0.3), (8, 1))
)
spans = template.spans(4096, 4096, 2, causal=causal)
tiles = Tiles(64, 64, stages, consumers, "wgmma")
print("VARIANT", variant, "TILES", tiles, "DEVICE", jax.devices(), flush=True)
with tuning.pinned({a.MosaicForward: tiles}):
    run = jax.jit(lambda q, k, v: a.forward(q, k, v, spans, residuals=residuals))
    for i in range(2):
        out = jax.block_until_ready(run(q, k, v))
        print("ITERATION", i, flush=True)
print("OK", variant,
      all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(out)
          if x.dtype != jnp.int32), flush=True)
