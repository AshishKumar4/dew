# GPU performance

What was measured on one RTX 4080, what was adopted, and what was rejected
with the number that rejected it. `docs/benchmarks.md` is the step table per
architecture; this file is the evidence behind the kernel and flag defaults.

Every number here comes from one of three harnesses:

```
python tools/benchmark_attention.py --json-out attention.json
python tools/benchmark_step.py --preset small --architectures <arch> \
    --attention-impl <kernel> --warmup 3 --steps 10
XLA_FLAGS=<flags> python tools/benchmark_step.py --preset small \
    --architectures <arch> --warmup 3 --steps 10
```

Conditions: jax 0.11.1 / jaxlib 0.11.1 (CUDA 13), driver 595.84, RTX 4080
16 GiB, single device, bf16 compute, adam, 3 warmup and 10 measured steps, one
architecture per process. The card was idle before each measurement
(`nvidia-smi --query-compute-apps=pid` showing only the desktop's pid), at
210 MHz and 30 W at rest and 2760 MHz and 120-220 W under load. An XLA flag
is read once when a backend opens, so every flag configuration ran in a fresh
process.

## The attention kernels

`tools/benchmark_attention.py`, bf16, batch chosen so query tokens times
heads is 524288 in every row. Forward only, and forward with the gradient
wrt q, k and v, in milliseconds.

| S | D | causal | reference fwd | xla fwd | cudnn fwd | reference bwd | xla bwd | cudnn bwd |
|------|-----|-------|------|------|------|------|------|------|
| 256 | 64 | no | 4.36 | 3.99 | 0.62 | 8.52 | 9.50 | 3.95 |
| 256 | 64 | yes | 4.64 | 4.11 | 0.63 | 9.44 | 10.24 | 3.97 |
| 256 | 128 | no | 6.77 | 10.43 | 1.22 | 18.31 | 11.95 | 10.33 |
| 256 | 128 | yes | 7.49 | 7.43 | 2.13 | 20.47 | 15.09 | 13.62 |
| 1024 | 64 | no | 19.65 | 21.67 | 1.63 | 37.91 | 35.24 | 10.35 |
| 1024 | 64 | yes | 16.36 | 19.57 | 1.15 | 34.62 | 31.06 | 5.89 |
| 1024 | 128 | no | 13.88 | 13.27 | 3.24 | 28.64 | 32.56 | 15.42 |
| 1024 | 128 | yes | 13.90 | 13.43 | 2.22 | 29.05 | 33.51 | 10.57 |
| 4096 | 64 | no | oom | oom | 6.53 | oom | oom | 24.31 |
| 4096 | 64 | yes | oom | oom | 3.93 | oom | oom | 14.87 |
| 4096 | 128 | no | oom | oom | 12.12 | oom | oom | 46.89 |
| 4096 | 128 | yes | oom | oom | 6.96 | oom | oom | 26.38 |

The reference and xla paths materialize the S x S logits, so they run out of
16 GiB at S=4096 and are 3 to 12 times slower than the fused kernel
everywhere they fit. cudnn is the kernel for a GPU run, forward and
backward, which is why `'auto'` reaches for it wherever it can.

## Which kernel 'auto' picks

cudnn's fused kernel has no backward pass for an odd query or key length. The
forward pass takes any length, so this only appears at the first training
step, as `NotImplementedError: Unsupported sequence length Q 333, KV 333`
out of jax. 77 CLIP text tokens are odd, and so is 256+77 concatenated.

Before this wave `attention_impl='auto'` meant plain cudnn on a GPU, so six
of the twelve architectures in the small preset could not train on this card
at their default settings. `'auto'` now picks cudnn only for the shapes
cudnn can train and xla for the rest (`dew.nn.attention.cudnn_supports`),
per call rather than per run, since one model holds both kinds of shape.
Explicit `'cudnn'` still raises: a run that names a kernel gets that kernel.

Where 'auto' lands, at the small preset's shapes:

| architecture | cudnn | xla |
|---|---|---|
| simple_dit, simple_udit, hybrid_dit | q256/kv256 | |
| uvit | q334/kv334 | |
| video_dit | q8/kv8, q256/kv256 | |
| causal_transformer | q512/kv512 | |
| unet, unet_3d | | q256/kv77, q1024/kv77 |
| simple_mmdit | | q333/kv333 |
| hierarchical_mmdit | | q141, q333, q1101 |
| jepa_encoder | q130/kv130 | q193/kv193 |
| jepa_video_encoder | q8/kv8, q130/kv130 | q193/kv193 |

The rerouting costs the affected models the fused kernel's speed and memory,
which is the price of training at all. It changes no numbers: a fixed-seed
20-step loss trajectory under `'auto'` is bitwise identical to the same run
under the kernel 'auto' selected, both for a rerouted model (simple_mmdit,
max delta 0.0) and for one that stays on cudnn (simple_dit, max delta 0.0).

## XLA flags

`TrainerConfig.xla_flags` appends to `XLA_FLAGS`, applied by
`prepare_process` before JAX opens a backend. The default is None. This is
the sweep behind that default: three architectures, one fresh process per
configuration, median of the runs with the range and count where a
configuration was repeated.

| configuration | simple_dit | causal_transformer | unet |
|---|---|---|---|
| baseline | 7.01 [6.96-7.53] n=5 | 75.70 | 17.38 [17.10-17.50] n=4 |
| `--xla_gpu_triton_gemm_any=true` | 7.43 | 75.64 | 17.05 [16.78-17.43] n=4 |
| `--xla_gpu_autotune_level=4` | 7.02 [6.95-7.48] n=5 | 75.58 | 17.36 [16.88-17.58] n=4 |
| `--xla_gpu_enable_latency_hiding_scheduler=true` | 7.33 | 75.60 | 17.08 |
| `--xla_gpu_enable_command_buffer=` (off) | 7.49 | 76.14 | 17.90 [17.45-18.15] n=4 |
| `--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,WHILE` | 7.03 [6.95-7.42] n=5 | 75.78 | 16.94 |
| `--xla_gpu_enable_while_loop_double_buffering=true` | 6.95 [6.93-7.09] n=5 | 75.73 | 17.30 |
| the two above with any signal, together | 7.00 [6.99-7.02] n=2 | 75.75 [75.67-75.84] n=2 | 17.09 [16.93-17.26] n=2 |

Nothing was adopted. The noise band decides it: four repeats of the *same*
configuration on simple_dit spread from 6.97 to 7.53 ms, 8%, because a fresh
process autotunes afresh. Against that, every simple_dit column is one
distribution. The causal_transformer is the quiet measurement, spread 0.7%,
and no flag moves it by more than 0.2%. The unet is the only architecture
where a flag shows: `--xla_gpu_triton_gemm_any=true` takes the median from
17.38 to 17.05 ms, 1.9%, over four runs each.

So one architecture gains 2%, one is unchanged, one cannot tell. The rule was
faster on all three and within noise on none, so the default stays None. A
run that wants the unet flag can pass `--trainer.xla-flags`.

Two flags are worth knowing about for a different reason:

- `--xla_gpu_autotune_level=4` changes nothing anywhere, which is how you
  learn it is already the default in this build.
- `--xla_gpu_enable_command_buffer=` (command buffers off) is the only
  configuration that is reliably *slower*: 17.90 against 17.38 on the unet
  over four runs, and slower on the other two as well. Command buffers are on
  by default and worth 3% on the launch-heavy architecture. Passing a longer
  type list than the default does not add to that.

Numerics were not part of any candidate: only kernel selection and
scheduling flags were measured. No flag that relaxes precision was tested and
none would be adopted, because an adopted change has to keep a fixed-seed
20-step loss trajectory within 1e-5.

## Ceilings

Measured, not adopted. These say where the remaining room is.

### fp8 against bf16

`/tmp/Kernels/ceilings.py matmul`, square matmuls with fp32 accumulation
through `preferred_element_type`.

| dtype | n | ms | TFLOP/s |
|---|---|---|---|
| bfloat16 | 4096 | 1.52 | 90.3 |
| float8_e4m3fn | 4096 | 0.80 | 171.0 |
| bfloat16 | 8192 | 10.84 | 101.4 |
| float8_e4m3fn | 8192 | 5.64 | 194.9 |

fp8 is 1.9 times bf16 on this card and needs no code beyond the cast, which
makes it the largest single ceiling here. It is also the one that cannot be
adopted under the no-tradeoff rule without a scaling scheme and a numerics
argument, since e4m3 carries 3 mantissa bits. The bf16 number is also the
honest ceiling for `train/mfu`: 101.4 TFLOP/s measured against the 97.5 the
utilisation column divides by, so a row reading 100% would be right.

### RMSNorm and SwiGLU

The question was whether XLA leaves them as separate kernels, in which case
the pallas `rms_norm` op would be worth trying. It does not. A 4096-token
1024-wide block with a 2816-wide gated MLP compiles to four kernels:

```
%fusion.11              rms_norm: mean of squares, rsqrt, scale, convert
%gemm_fusion_dot        gate and up projections, merged into one 5632-wide gemm
%loop_convert_fusion    silu(gate) * up
%gemm_fusion_dot_general.5   down projection
```

Both constructs are already one kernel each, and XLA merged the two MLP
projections into a single gemm. The block runs at 69.9 TFLOP/s of the 101.4
ceiling, and the norm is 0.040 ms of its 1.113 ms, so a perfect norm kernel
could take back at most 3.6% of an MLP block. No comparison against
`ops/gpu/rms_norm.py` was run: there is nothing unfused for it to win.

### The unet's 1.7%

| run | batch | ms/step | GFLOP/step | util |
|---|---|---|---|---|
| unet | 16 | 17.37 | 28.6 | 1.69% |
| unet | 64 | 57.59 | 102.9 | 1.83% |
| unet, command buffers extended | 16 | 17.12 | 28.6 | 1.71% |
| unet, command buffers extended | 64 | 57.94 | 102.9 | 1.82% |

Four times the batch costs 3.3 times the step, so about 4 ms of the 17.4 ms
step (23%) does not scale with the batch and 0.84 ms per sample does.
Utilisation moves from 1.69% to 1.83%: the batch is not what holds this model
at 2% of peak. A convolutional stack at 64 channels moves far more activation
bytes per FLOP than a transformer does, and that ratio does not improve with
batch. Command buffers are worth 1.4% at batch 16 and nothing at batch 64.

### Compile time and scan over layers

A 28-layer 1024-wide `causal_transformer` at 1024 tokens, batch 2, which is
140 ms per step at 38% utilisation:

| compile | seconds |
|---|---|
| no persistent cache | 11.7 |
| cold cache (writing) | 12.3 |
| warm cache | 2.2 |

Twelve seconds for 28 layers, and 2.2 with the cache the trainer enables by
default. Scan over layers would trade a lower unrolled compile time for a
different step; there is no compile-time argument for it at this depth.

### fp32 matmuls and TF32

XLA:GPU answers an fp32 dot at DEFAULT precision with TF32 tensor cores: 10
mantissa bits, not 23. `jax_default_matmul_precision` is None in this repo, so
that is what the fp32 matmuls in a bf16 step get. `dot_general` operand
dtypes in the lowered training step:

| architecture | bf16 x bf16 | f32 x f32 | f32 x bf16 |
|---|---|---|---|
| simple_dit | 131 | 8 | 0 |
| causal_transformer | 63 | 3 | 0 |
| unet | 115 | 5 | 10 |
| jepa_encoder | 217 | 0 | 6 |

The f32 pairs are the projections a model keeps in fp32 on purpose: the
language model's `lm_head` (`dtype=jnp.float32` at
`causal_transformer.py:337`, and the tied-embedding path which casts both
operands), and the DiT's conditioning and output projections. The mixed pairs
are the fp32 attention logits on the xla path. Every one of them carries
`precision=DEFAULT`, so every one runs at TF32.

What the precision costs, at 4096 x 4096 x 4096 and at the real shapes,
against a float64 reference:

| dot | default ms | highest ms | cost | max abs error, default | highest |
|---|---|---|---|---|---|
| 4096 cube | 3.06 | 4.11 | 1.34x | 1.10e-1 | 9.2e-4 |
| lm_head, 8192 x 768 x 50304 | 13.06 | 19.75 | 1.51x | 5.4e-2 | 2.3e-4 |
| DiT adaLN, 16 x 384 x 1536 | 0.058 | 0.054 | 0.94x | 2.3e-2 | 7.1e-5 |
| DiT output, 4096 x 384 x 48 | 0.059 | 0.067 | 1.13x | 2.8e-2 | 7.1e-5 |

`precision='high'` measures the same error as DEFAULT, which confirms both
are TF32.

The finding: where this repo asks for fp32 it gets TF32. For the small
projections the difference is free (0.054 against 0.058 ms), so passing
`precision='highest'` there costs nothing and honors what the code asks for.
For the language model's head it is not free: the forward dot alone goes from
13.06 to 19.75 ms, 9% of a 75.7 ms step, and the step holds three of these
dots. Nothing was changed here: `lm_head` and the DiT heads are not this
wave's files, and the choice between 11 mantissa bits and 9% of the step
belongs with the objective that wants fp32 logits.
