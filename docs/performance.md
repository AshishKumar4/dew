# GPU performance

What was measured on one RTX 4080, what was adopted, and what was rejected
with the number that rejected it. `docs/benchmarks.md` is the step table per
architecture; this file is the evidence behind the kernel and flag defaults.

Every number here comes from one of four harnesses:

```
python tools/benchmark_attention.py --json-out attention.json
python tools/benchmark_step.py --preset small --architectures <arch> \
    --attention-impl <kernel> --warmup 3 --steps 10
XLA_FLAGS=<flags> python tools/benchmark_step.py --preset small \
    --architectures <arch> --warmup 3 --steps 10
python tools/optimizer_curve.py --dataset <tokens> --optimizer <name> \
    --learning-rate <lr> --out <json>
```

Conditions: jax 0.11.1 / jaxlib 0.11.1 / jax_cuda12_plugin 0.11.1, driver
595.84, RTX 4080 16 GiB, single device, bf16 compute, adam, 3 warmup and 10
measured steps, one architecture per process. The card was idle before each
measurement (`nvidia-smi --query-compute-apps=process_name` showing only
gnome-remote-desktop-daemon, the desktop itself), at 210 MHz and 30 W at rest
and 2760 MHz and 120-220 W under load. An XLA flag is read once when a
backend opens, so every flag configuration ran in a fresh process.

## Where a step's time goes, 2026-09-05

```
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    python tools/benchmark_step.py --preset small --architectures <arch> \
    --warmup 3 --steps 30 --profile-dir /tmp/dew-trace --profile-steps 5
```

The traced window is read back by `tools/benchmark_step.py` itself: busy
is the union of every kernel interval on the device's streams, kernels are
counted per step, and kernel time is summed per category from the kernel
names. dew at `9886c20`, the tree before the cudnn padding below.

| architecture | ms/step | device busy | kernels/step | gemm | elementwise | reduce | convert | attention | copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_dit | 7.0 | 100% in steady state | 532 | 3.29 | 0.69 | 1.19 | 0.79 | 0.69 | 0.16 |
| causal_transformer | 88.8 | 100% | 282 | 63.3 | 13.3 | 7.6 | 0.8 | 1.8 | 0.7 |

The trace reports 81.6% busy for the DiT over its five steps because the
profiler's start costs the first two steps 3 ms gaps each; the step-to-step
interval settles at 6.9 ms, which is the kernel time, so the device is not
idle between steps once the loop is running. The DiT's reductions are the
bias gradients of every Dense (the q, k and v projections' 6-by-64 biases
cost three full passes over the activation gradient per layer, 0.25 ms a
step) and the norm statistics; its converts are XLA's own split-K partial
sums in fp32 and the per-use casts of the fp32 parameters to bf16 (0.15 ms
of the 0.79). The decoder's gemm time is the fp32 (TF32) vocabulary head:
two cutlass `s1688gemm` kernels at 12.9 and 12.5 ms and four Triton tiles of
3.2 ms for the third product, against a 12.8 ms floor per product at the
49.5 TFLOP/s TF32 ceiling measured in `docs/research/benchmark-parity.md`.

### The host's budget per step

What the host spends per step on simple_dit, from the trace's host plane
and from timing the dispatch loop with the device deliberately left
behind. The device step is 6.9 ms.

| host work per step | ms | how measured |
|---|---:|---|
| XLA thunk execution inside PjRt Execute | 3.3 | `GpuExecutable::ExecuteThunks` on the host plane; 2.4 of it is three CUDA-graph launches |
| Python in `jax.stages.Compiled.__call__` before Execute | 1.8 | `$stages.py __call__` 6.7 ms against `PjRtCApiLoadedExecutable::Execute` 5.0 ms |
| placing a fresh batch (`shard_batch`) | 0.25 | 200 calls timed in isolation, image plus tokens |
| the loop with a fixed device batch | 5.0 | dispatch loop time, 100 steps, device 27 steps behind |
| the loop with a fresh batch per step | 6.5 | same, device 7 steps behind |
| the loop with XLA command buffers off | 7.3 | `--xla_gpu_enable_command_buffer=`; wall 7.46 ms/step, the host is now the step |

So on the smallest step the host is at 94% of the device with a fresh
batch every step, and 106% without command buffers. Two things follow.
The `Compiled.__call__` Python (4.5 us a leaf, 396 leaves here, more on a
mesh) is why `Trainer.compile` returns the jitted step since `de6b22c`:
wall time on this card does not move, host time drops by 1.8 ms a step.
And the 1.5 ms a fresh batch costs over a fixed one is the command buffer
being updated for the new buffer addresses; it caps how far the loop runs
ahead (7 steps against 27) and would be the wall clock on a faster card or
a smaller model. It does not come from the placement itself (0.25 ms) nor
from freeing the consumed batch (keeping every batch alive changes
nothing under the default preallocation), and prefetch depth 2, 8 and 32
measure the same. No fix is adopted: the addresses are the runtime's.

Waiting on the device every step, the other way to lose this, costs 45%:
the same simple_dit loop with `block_until_ready` after each step runs at
10.3 ms against 7.1. The trainer's loop never waits between logging ticks,
and the peak allocation does not grow with how far the loop runs ahead
(0.823 GiB at 27 steps ahead, 0.819 in lockstep; 3.499 against 3.495 GiB
for hierarchical_mmdit).

### Antipatterns audited

Read across `src/dew` and measured on the small preset. The classes are
the owner's nine; a row names the cost it found and what was done.

| class | site | what was measured | verdict |
|---|---|---|---|
| 1 sync in hot paths | `training/trainer.py` `fit`, per-step `loss.astype`, `interval_loss + loss`, `jnp.where(finite, ...)`, `bad_run + 1`, `jnp.maximum` | `jax_log_compiles` over a 50-step fit: five one-op executables compiled and dispatched eagerly every step, no host sync; wall cost not measured | left, reported to the trainer's owner as a candidate |
| 1 sync in hot paths | `jax.stages.Compiled.__call__` in `Trainer.compile` | 1.8 ms/step of Python at 396 leaves (table above) | fixed on main in `de6b22c` (jit dispatch) |
| 2 recompilation | `Trainer.fit` with evaluation every 25 of 50 steps, diffusion and LM objectives | one `jit(step)`, one `jit(initial_state)`, one evaluation executable (`_sample_impl`, `scored`); no per-step or per-eval retrace | none found |
| 3 baked constants | the compiled step's optimized HLO | simple_dit: 20 constants, 0.19 MiB, the largest the 2D sincos table bf16[256, 384]; causal_transformer: none | none found; the encoder's table moved into the state before this pass |
| 4 dtype churn | HLO dots by output dtype and the trace's convert kernels | simple_dit: 65 bf16 dots, 40 fp32 outputs that are XLA split-K partials and the fp32 `final_proj`; parameter casts 0.15 ms/step; decoder: the fp32 head by design | none found in dew's code; XLA's split-K choice is the card's |
| 5 redundant work | `nn/attention.py` odd-length routing to the xla kernel | hierarchical_mmdit 33.9 to 20.9 ms, simple_mmdit 12.9 to 11.0, peaks 3.50 to 1.85 and 1.43 to 1.08 GiB | fixed, `3b67135` |
| 5 redundant work | `objectives/lm` head chunking at its default of 4 | 1.9 ms/step (2.2%) against one chunk, for 1.2 GiB | reported to the LM lane with the sweep in `docs/benchmarks.md` |
| 5 redundant work | `objectives/diffusion/objective.py:141`, `null = self.encode(...)` every step | a frozen encoder's forward on the unconditional tokens, once per step, inside the step; free with the table encoder used here, a text tower's forward at batch 1 with CLIP; not measured with CLIP | reported to the diffusion lane |
| 6 data path | `DevicePrefetchIterator` depth, `shard_batch` cost, main-thread placement | 0.15 to 0.25 ms/step waiting for a batch at depth 2, 8 and 32; placement 0.25 ms; the loop is bounded by dispatch, not the transfer | none found |
| 7 sharding | the compiled step's `input_output_alias` | every state leaf aliased (396 of 396 on simple_dit, 143 of 143 on the decoder): donation happens | none found; collectives on a mesh not measured this pass |
| 8 compile time | `Trainer.compile` | one compile per fit (class 2 row); the FLOP count reads the same executable | none found; the persistent cache was not timed this pass |
| 9 memory | peak against the state, run-ahead against lockstep | simple_dit 0.82 GiB peak on a 303 MiB state, unchanged by run-ahead; hierarchical_mmdit 3.50 GiB on 847 MiB, 1.65 GiB of it the xla attention's fp32 logits | fixed by the class-5 row |

Not run this pass, so nothing is claimed about them: the
`jax_default_matmul_precision` settings (`bfloat16` would change the fp32
head's numerics and is refused by the precision rule regardless), remat
on a step that fits in memory, XLA flags beyond command buffers, and the
cost of the class-1 eager scalars.

### Against PyTorch today

`tools/benchmark_torch.py` rerun in a fresh venv, torch 2.14.0+cu130 with
cuDNN 9.24 (last week was 2.11.0+cu128 with 9.19), `--mode compile
--warmup 20 --steps 100`, the small presets, one process per row, against
the dew rows of `docs/benchmarks.md` and the table above:

| case | dew ms/step | torch compile, reference attention | torch compile, SDPA cudnn | dew against the best torch row |
|---|---:|---:|---:|---:|
| simple_dit | 7.02 | 9.28 | 8.39 | 1.19x faster |
| causal_transformer | 88.78 | 81.50 | 72.46 | 0.82x, torch faster by 18% |

Last week's 0.95x set the parity harness's fixed-batch decoder row (75.70)
against torch's 72.18. Today's dew row is the benchmark's own prefetching
loop at 88.78. Of the 13 ms between them, 1.9 are the chunked head, 3.8
are the decoder's own changes since `6b0f119`, and the remaining 7.6 are
the distance between that harness's fixed-batch row and this tool's loop
at the same commit (`6b0f119` reruns at 83.26 here today). The DiT gained:
1.16x last week, 1.19x now, with torch's SDPA row 0.4 ms slower than a
week ago on the newer torch.

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

## Odd sequence lengths on cudnn

cudnn's fused kernel has no backward pass for an odd query or key length. The
forward pass takes any length, so this only appeared at the first training
step, as `NotImplementedError: Unsupported sequence length Q 333, KV 333`
out of jax. 77 CLIP text tokens are odd, and so is 256+77 concatenated.

Until 2026-09-05 `'auto'` sent those shapes to the xla kernel, which
materializes the [B, H, Q, K] logits and their probabilities in fp32 and
keeps them for the backward pass. `cudnn_attention` now pads an odd length
to an even one instead: one zero row on the query, sliced off the output,
and one zero key hidden by the kernel's own padding mask
(`key_value_seq_lengths`), so every real query attends to exactly the keys
it had. `'auto'` is cudnn on every shape on a GPU, and an explicit
`'cudnn'` takes any length too. The check is
`tests/test_kernels.py::test_cudnn_trains_odd_lengths_and_agrees_with_xla`:
at q1024/kv77, q9/kv7 and q333/kv333 causal, outputs and the three input
gradients agree with the xla kernel to within two bf16 ulps of their scale,
which is also how far the two kernels sit apart at an even length (q256:
1.6e-2 at scale 2.9 on the output, 7.8e-2 at scale 15.6 on the gradients,
both one ulp). Leaving the pad key unmasked moves the q9/kv7 output by
0.26 at scale 2.4 and fails the test; leaving the pad query row in changes
the shape and fails it.

What it is worth, `--warmup 3 --steps 50` on the small preset, `'xla'`
(the kernel 'auto' used to pick for these shapes) against `'auto'`:

| architecture | shapes | xla ms/step | cudnn ms/step | xla peak GiB | cudnn peak GiB | loss at the end, xla / cudnn |
|---|---|---:|---:|---:|---:|---|
| hierarchical_mmdit | q141, q333, q1101 | 33.86 | 20.86 | 3.50 | 1.85 | 0.551035 / 0.551038 |
| simple_mmdit | q333/kv333 | 12.86 | 11.01 | 1.43 | 1.08 | 0.584398 / 0.584407 |
| unet | q256/kv77, q1024/kv77 | 16.30 | 16.13 | 0.78 | 0.71 | 0.597518 / 0.597516 |

The 1101-token stage's xla attention kept its fp32 logits and probabilities
for the backward pass; that is where the 1.65 GiB and the 13 ms went. The
unet's attention is a small part of its step, so it gains little. The
losses are after 103 steps on one fixed batch and differ in the sixth
digit, which is the two kernels' bf16 rounding compounded by Adam. Decode
asks for one query position at a time, an odd length, and now runs on
cudnn with the cache mask as an additive bias; its speed was not measured.

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

## Muon against AdamW at equal tokens

The rows below are the only CPU rows in this file. A loss curve at equal
tokens asks nothing of the card's kernels, and the run is small enough that
one workstation CPU does nine of them in under an hour.

```
python tools/tokenize_text.py --input data/shakespeare.txt \
    --out data/shakespeare-byte --tokenizer byte --val-fraction 0.02
JAX_PLATFORMS=cpu taskset -c 0-5 python tools/optimizer_curve.py \
    --dataset data/shakespeare-byte --optimizer muon --learning-rate 3e-3 \
    --steps 2000 --emb-features 128 --num-layers 2 --num-heads 2 --seed 0 \
    --out /tmp/muon-3e-3.json
```

once per arm, learning rate and seed. Conditions: `causal_transformer`, 128
wide, 2 layers, 2 heads, tied head, byte vocabulary of 256, sequence length
128, batch 16, 557,952 parameters, bf16 compute, weight decay 0.1 on both
groups, no schedule, no clipping. 2000 steps is 4,096,000 tokens, which is
3.75 passes over the 1,093,086 training tokens of the Shakespeare corpus.
12th Gen i9-12900K, jax 0.11.1, `JAX_PLATFORMS=cpu`, six cores pinned per
run, three runs at a time on disjoint cores. Every arm sees the same batches
in the same order at the same seed, so a difference between two arms is the
solver.

Three arms: `adamw`, `muon` as this branch builds it, and `muon-unsplit`,
which is `optax.contrib.muon` with its own ndim == 2 rule, the shape the
'muon' entry had before the parameter groups. Final loss is the mean over
the last 50 steps.

| arm | lr 1e-3 | lr 3e-3 | lr 1e-2 |
|---|---|---|---|
| adamw | 1.4723 | 1.4842 | 1.5885 |
| muon | 1.5229 | 1.4438 | 1.4713 |
| muon-unsplit | 1.5762 | 1.4598 | 1.4916 |

Each arm at its own best learning rate, averaged over seeds 0, 1 and 2, as
the loss at five token counts:

| arm | 0.51M | 1.02M | 2.05M | 3.07M | 4.10M |
|---|---|---|---|---|---|
| adamw, lr 1e-3 | 2.0136 | 1.7376 | 1.5737 | 1.5015 | 1.4764 |
| muon, lr 3e-3 | 1.9885 | 1.6744 | 1.5179 | 1.4572 | 1.4386 |
| muon-unsplit, lr 3e-3 | 2.2454 | 1.7646 | 1.5559 | 1.4812 | 1.4561 |

Muon with the groups reaches 1.4386 where AdamW reaches 1.4764, 0.038 nats
lower at the same tokens. The three seeds of an arm spread 0.007 to 0.013,
so the gap to AdamW is three times that noise. The gap to unsplit Muon is
0.018, one and a half times it, and the split is ahead on each of the three
seeds by 0.016, 0.020 and 0.017. Muon also holds its loss at a learning
rate ten times its best, losing 0.028 against AdamW's 0.116, which is the
tolerance the labs report (`docs/research/frontier-training.md:183`).

These numbers say nothing about 0.4B parameters, which is the run section
4.9 of `docs/design/plan.md` asks for and which needs a v5e-16. Wall-clock
is not comparable either, because the runs shared a machine.
