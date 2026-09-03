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

Conditions: jax 0.11.1 / jaxlib 0.11.1 / jax_cuda12_plugin 0.11.1, driver
595.84, RTX 4080 16 GiB, single device, bf16 compute, adam, 3 warmup and 10
measured steps, one architecture per process. The card was idle before each
measurement (`nvidia-smi --query-compute-apps=process_name` showing only
gnome-remote-desktop-daemon, the desktop itself), at 210 MHz and 30 W at rest
and 2760 MHz and 120-220 W under load. An XLA flag is read once when a
backend opens, so every flag configuration ran in a fresh process.

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
cudnn supports and xla for the rest (`dew.nn.attention.cudnn_supports`),
per call rather than per run, since one model holds both kinds of shape.
Explicit `'cudnn'` still raises. A run that names a kernel gets that kernel.

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

The rule reads shapes, so it applies to a forward-only call as well. Decode
asks for one query position at a time, so sampling from a language model
runs on xla where it used to run on cudnn. Both kernels accumulate the
logits and run the softmax in fp32, so the samples are the same; the speed
of that path was not measured.

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

Nothing was adopted. The noise band decides it: four repeats of the same
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
  configuration that is reliably slower, 17.90 against 17.38 on the unet
  over four runs, and slower on the other two as well. Command buffers are on
  by default and worth 3% on the launch-heavy architecture. Passing a longer
  type list than the default does not add to that.

No candidate flag changes numerics. The sweep covered kernel selection and
scheduling only. No flag that relaxes precision was tested, and none would be
adopted, because an adopted change has to keep a fixed-seed 20-step loss
trajectory within 1e-5.

## What batch size buys the unet

These numbers were measured and nothing was adopted from them. They show
where the remaining room is on the architecture whose step is least sensitive
to batch.

```
python tools/benchmark_step.py --preset small --architectures unet \
    --batch-size 16 --warmup 3 --steps 10
```

once per batch size, and again with
`--xla-flags=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,WHILE`
for the extended rows.

| run | batch | ms/step |
|---|---|---|
| unet | 16 | 17.37 |
| unet | 64 | 57.59 |
| unet, command buffers extended | 16 | 17.12 |
| unet, command buffers extended | 64 | 57.94 |

Four times the batch costs 3.3 times the step, so about 4 ms of the 17.4 ms
step (23%) does not scale with the batch and 0.84 ms per sample does. Command
buffers are worth 1.4% at batch 16 and nothing at batch 64.

These rows carried a utilisation column when they were measured, reading 1.7%,
and that number was the counter rather than the card: XLA's `cost_analysis()`
cannot see inside the cuDNN convolution calls the backend emits, and it
undercounted this model 22.5 times. Counted off the optimized HLO the unet
runs at 40.5% of peak, which `docs/benchmarks.md` reports.
