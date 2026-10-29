"""Diagnostic only: independent-process H100 forward variants."""
import sys

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
consumers = 2 if variant == "consumers2" else 1
stages = int(variant[-1]) if variant.startswith("stages") else 2
residuals = variant != "noresiduals"
causal = variant == "causal"

if variant == "nosetmax":
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
