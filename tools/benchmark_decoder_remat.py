#!/usr/bin/env python3
"""Measure a decoder's block recomputation in one fresh process per mode.

Run from the repository root with PYTHONPATH=src and JAX_PLATFORMS=cpu or
cuda. GPU runs set XLA_PYTHON_CLIENT_PREALLOCATE=false. For example:

    python tools/benchmark_decoder_remat.py --width 256 --depth 8 --length 256
    python tools/benchmark_decoder_remat.py --width 256 --depth 8 --length 256 --remat full

`--remat` takes a name from `decoder_block.REMAT_POLICIES`; absent, the
blocks keep every residual. `--mixer mamba2` builds Mamba-2 layers in place
of attention, with heads of 64 at twice the width, a state of 128, one group
and a conv of 4 taps.

The timed operation is a donated AdamW update on next-token cross entropy.
Compiler temporary bytes include activation and gradient workspaces. Device
peak bytes are the allocator high-water mark for this fresh process, including
initialization; they are not a pure activation-memory counter. The activation
bytes are the residuals the loss's pullback keeps, each stored once, inputs and
constants left out (`dew.training.footprint.residual_bytes`, what a run's fit
search derives), before XLA scheduling and buffer reuse; the table's column
counted each residual slot of `print_saved_residuals` on its own.

RTX 4080 16 GiB, driver 595.84, JAX/jaxlib 0.11.1, Flax 0.12.9, Optax
0.2.8: default dimensions with --dtype bfloat16 (FP32 master weights),
three warmups and twenty timed steps. Each row was a separate process;
columns compare --remat absent/`full`. MB means 1,000,000 bytes.

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
import jax.numpy as jnp
import numpy as np
import optax

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES, Mixture
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import mixers
from dew.telemetry.benchmark import carried
from dew.training.footprint import residual_bytes


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
    parser.add_argument("--remat", choices=sorted(REMAT_POLICIES), default=None)
    parser.add_argument("--mixer", choices=("attention", "mamba2"), default="attention")
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--mlp", type=int, default=None, help="the MLP's width, twice the model's if absent")
    parser.add_argument("--attention-impl", default="xla")
    args = parser.parse_args()
    mixer = {} if args.mixer == "attention" else {"mixer": mixers.build("mamba2", {
        "num_heads": 2 * args.width // 64, "head_dim": 64, "state_size": 128, "n_groups": 1,
        "conv_kernel": 4})}
    model = CausalTransformer(
        vocab_size=args.vocab, emb_features=args.width, num_layers=args.depth,
        num_heads=args.heads, num_kv_heads=args.kv_heads,
        mlp_features=2 * args.width if args.mlp is None else args.mlp,
        max_seq_len=args.length, dtype=jnp.dtype(args.dtype), attention_impl=args.attention_impl,
        scan_layers=args.scan, remat=args.remat, **mixer,
        mixture=Mixture(experts=4, top_k=2) if args.case == "sparse" else None,
        kv_shared_layers=tuple(range(args.depth // 2, args.depth)) if args.case == "shared" else None)
    objective = LMObjective(model, args.length)
    tokens = {"text": jnp.asarray(np.random.default_rng(0).integers(
        1, args.vocab, (args.batch, args.length + 1)), jnp.int32)}
    info = Step(jnp.zeros((), jnp.int32), jax.random.key(3), None)
    shapes = jax.eval_shape(objective.init, jax.random.key(0))["params"]
    optimizer = optax.adamw(1e-4)

    def loss(variables, tokens):
        return objective.scalar_loss({"params": variables}, tokens, info)

    activation_bytes = residual_bytes(loss, shapes, (jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tokens),))

    def step(params, opt_state, tokens):
        (value, _), grads = jax.value_and_grad(loss, has_aux=True)(params, tokens)
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
    def stepped(carry):
        return jitted(carry[0], carry[1], tokens)

    warm, _ = carried(stepped, (params, opt_state, None), 3)
    (params, opt_state, value), (seconds,) = carried(stepped, warm, args.steps)
    milliseconds = 1000 * seconds / args.steps
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
        "device_peak_bytes_before":
            None if allocated_before is None else allocated_before["peak_bytes_in_use"],
        "device_peak_bytes_after": None if allocated_after is None else allocated_after["peak_bytes_in_use"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
