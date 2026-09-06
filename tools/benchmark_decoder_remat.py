#!/usr/bin/env python3
"""Measure a decoder's block recomputation in one fresh process per mode.

Run from the repository root with PYTHONPATH=src and JAX_PLATFORMS=cpu or
cuda. GPU runs set XLA_PYTHON_CLIENT_PREALLOCATE=false. For example:

    python tools/benchmark_decoder_remat.py --width 256 --depth 8 --length 256
    python tools/benchmark_decoder_remat.py --width 256 --depth 8 --length 256 --remat

The timed operation is a donated AdamW update on next-token cross entropy.
Compiler temporary bytes include activation and gradient workspaces. Device
peak bytes are the allocator high-water mark for this fresh process, including
initialization; they are not a pure activation-memory counter. Autodiff's
saved activation bytes exclude input and constant references and count the
logical residual shapes, before XLA scheduling and buffer reuse.

RTX 4080 16 GiB, driver 595.84, JAX/jaxlib 0.11.1, Flax 0.12.9, Optax
0.2.8: default dimensions with --dtype bfloat16 (FP32 master weights),
three warmups and twenty timed steps. Each row was a separate process;
columns compare --remat absent/present. MB means 1,000,000 bytes.

case         activation MB   compiler temp MB   device peak MB   ms/step       compile s
dense        326.24/13.48     138.49/31.88        326.68/177.21    2.555/3.026   7.52/8.18
dense --scan 300.28/32.90     321.33/72.27        379.58/192.46    4.737/3.520   6.86/7.90
shared       316.73/14.00     142.35/31.71        323.53/177.21    2.455/2.866   8.22/8.44
sparse       368.77/9.81      270.42/75.68        443.02/270.53    5.539/6.857   7.95/8.73

Initialization reached 177.21 MB before each first step, which bounds the
device peak from below. Recompute costs 17-24% in the unrolled cases; in
the scanned case the smaller loop residual buffers also reduce step time.
The final bf16 losses differ by at most 5.3e-4 after 23 updates. FP32
behavior and checkpoint tests run separately; these short runs do not
establish long-run convergence or TPU/multi-host memory behavior.
"""

import argparse
import json
import time

import flax
import jax
from jax.core import ShapedArray
import jax.numpy as jnp
import numpy as np
import optax
# This is the shape analysis behind jax.ad_checkpoint.print_saved_residuals,
# read directly so the measurement need not parse printed shapes or HLO text.
from jax._src.ad_checkpoint import saved_residuals

from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.objectives import scalar_loss
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--case", choices=("dense", "sparse", "shared"), default="dense")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--remat", action="store_true")
    args = parser.parse_args()
    model = CausalTransformer(
        vocab_size=256, emb_features=args.width, num_layers=args.depth,
        num_heads=4, num_kv_heads=2, mlp_features=2 * args.width,
        max_seq_len=args.length, dtype=jnp.dtype(args.dtype), attention_impl="xla",
        scan_layers=args.scan, remat=args.remat,
        mixture=Mixture(experts=4, top_k=2) if args.case == "sparse" else None,
        num_kv_shared_layers=args.depth // 2 if args.case == "shared" else 0)
    objective = LMObjective(model, args.length)
    tokens = {"text": jnp.asarray(np.random.default_rng(0).integers(
        1, 256, (args.batch, args.length + 1)), jnp.int32)}
    info = Step(jnp.zeros((), jnp.int32), jax.random.key(3), None)
    shapes = jax.eval_shape(objective.init, jax.random.key(0))["params"]
    optimizer = optax.adamw(1e-4)

    def loss(params, tokens):
        return scalar_loss(objective, {"params": params}, tokens, info)[0]

    residuals = saved_residuals(loss, shapes, jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tokens))
    activation_bytes = 0
    for aval, origin in residuals:
        if origin.startswith("output of"):
            if not isinstance(aval, ShapedArray):
                raise TypeError(f"Expected an array residual, got {aval}")
            activation_bytes += int(np.prod(aval.shape)) * np.dtype(aval.dtype).itemsize

    def step(params, opt_state, tokens):
        value, grads = jax.value_and_grad(loss)(params, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    jitted = jax.jit(step, donate_argnums=(0, 1))
    opt_shapes = jax.eval_shape(optimizer.init, shapes)
    started = time.perf_counter()
    compiled = jitted.lower(shapes, opt_shapes, tokens).compile()
    compile_seconds = time.perf_counter() - started
    memory = compiled.memory_analysis()
    params = jax.jit(objective.init)(jax.random.key(0))["params"]
    opt_state = jax.jit(optimizer.init)(params)
    jax.block_until_ready((params, opt_state, tokens))
    device = jax.devices()[0]
    allocated_before = device.memory_stats()
    params, opt_state, value = jitted(params, opt_state, tokens)
    for _ in range(2):
        params, opt_state, value = jitted(params, opt_state, tokens)
    jax.block_until_ready((params, opt_state, value))
    started = time.perf_counter()
    for _ in range(args.steps):
        params, opt_state, value = jitted(params, opt_state, tokens)
    jax.block_until_ready((params, opt_state, value))
    milliseconds = 1000 * (time.perf_counter() - started) / args.steps
    allocated_after = device.memory_stats()
    print(json.dumps({
        **vars(args), "device": device.device_kind, "backend": jax.default_backend(),
        "jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__,
        "loss": float(value), "compile_seconds": compile_seconds, "ms_per_step": milliseconds,
        "activation_residual_bytes": activation_bytes,
        "compiler_temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        "compiler_argument_bytes": None if memory is None else memory.argument_size_in_bytes,
        "compiler_output_bytes": None if memory is None else memory.output_size_in_bytes,
        "compiler_alias_bytes": None if memory is None else memory.alias_size_in_bytes,
        "device_live_bytes_before": None if allocated_before is None else allocated_before["bytes_in_use"],
        "device_peak_bytes_before": None if allocated_before is None else allocated_before["peak_bytes_in_use"],
        "device_peak_bytes_after": None if allocated_after is None else allocated_after["peak_bytes_in_use"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
