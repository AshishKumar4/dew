# Performance measurements

This page records experiments on one RTX 4080, at the revisions and settings stated in each section. For architecture comparisons, see [step benchmarks](benchmarks.md). For how distributed training is configured today, see [distributed training](concepts/distributed.md). A result at one shape and one revision does not settle a default for every case, and it says nothing about TPUs.

The timeline busy percentages below were taken before `e5ee70d`, which fixed the measurement window for nested kernel intervals. Before you reuse those percentages, replay the original traces. The synchronized wall-clock step times are separate measurements, and that arithmetic bug does not affect them.

These are command templates. Replace the angle-bracket fields with the architecture, kernel and data of your experiment:

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
measurement: `nvidia-smi --query-compute-apps=process_name` showed only
gnome-remote-desktop-daemon, which is the desktop itself. The card ran at 210
MHz and 30 W at rest and at 2760 MHz and 120-220 W under load. XLA reads a
flag once, when a backend opens, so every flag configuration ran in a fresh
process.

## Where a step's time goes, 2026-09-05

```
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    python tools/benchmark_step.py --preset small --architectures <arch> \
    --warmup 3 --steps 30 --profile-dir /tmp/dew-trace --profile-steps 5
```

`tools/benchmark_step.py` reads the traced window back itself. Busy time is
the union of every kernel interval on the device's streams. Kernels are
counted per step, and kernel time is summed per category, where the category
comes from the kernel name. Dew was at `9886c20`, the tree before the cudnn
padding described below.

| architecture | ms/step | device busy | kernels/step | gemm | elementwise | reduce | convert | attention | copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_dit | 7.0 | 100% in steady state | 532 | 3.29 | 0.69 | 1.19 | 0.79 | 0.69 | 0.16 |
| causal_transformer | 88.8 | 100% | 282 | 63.3 | 13.3 | 7.6 | 0.8 | 1.8 | 0.7 |

The trace reports 81.6% busy for the DiT over its five steps. The profiler's
start puts a 3 ms gap into each of the first two steps. After that, the
interval from one step to the next settles at 6.9 ms, which equals the kernel
time. Once the loop is running, the device does not sit idle between steps.

The DiT's reductions are the bias gradients of every Dense layer and the norm
statistics. The 6-by-64 biases of the q, k and v projections cost three full
passes over the activation gradient per layer, 0.25 ms a step. Its converts
are XLA's own split-K partial sums in fp32 and the per-use casts of the fp32
parameters to bf16 (0.15 ms of the 0.79). The decoder's gemm time is the fp32
(TF32) vocabulary head. It runs as two cutlass `s1688gemm` kernels at 12.9 and
12.5 ms and four Triton tiles of 3.2 ms for the third product. The floor per
product is 12.8 ms at the 49.5 TFLOP/s TF32 ceiling measured in
`docs/research/benchmark-parity.md`.

### The host's budget per step

This table shows what the host spends per step on simple_dit. The numbers
come from the trace's host plane and from timing the dispatch loop while the
device was deliberately left behind. The device step is 6.9 ms.

| host work per step | ms | how measured |
|---|---:|---|
| XLA thunk execution inside PjRt Execute | 3.3 | `GpuExecutable::ExecuteThunks` on the host plane; 2.4 of it is three CUDA-graph launches |
| Python in `jax.stages.Compiled.__call__` before Execute | 1.8 | `$stages.py __call__` 6.7 ms against `PjRtCApiLoadedExecutable::Execute` 5.0 ms |
| placing a fresh batch (`shard_batch`) | 0.25 | 200 calls timed in isolation, image plus tokens |
| the loop with a fixed device batch | 5.0 | dispatch loop time, 100 steps, device 27 steps behind |
| the loop with a fresh batch per step | 6.5 | same, device 7 steps behind |
| the loop with XLA command buffers off | 7.3 | `--xla_gpu_enable_command_buffer=`; wall 7.46 ms/step, the host is now the step |

On the smallest step, the host takes 94% of the device's time with a fresh
batch every step, and 106% without command buffers. Two conclusions follow.

First, the Python in `Compiled.__call__` costs 4.5 us per leaf, and this
state has 396 leaves (more on a mesh). That is why `Trainer.compile` returns
the jitted step, starting with `de6b22c`. Since the sequence axis landed, the jitted step is wrapped in the
mesh context, and a dispatch costs 32 us on the i9-12900K with or without
that wrapper. On this card the wall time stays the same, and host time drops
by 1.8 ms a step.

Second, a fresh batch costs 1.5 ms more than a fixed one, because the command
buffer has to be updated for the new buffer addresses. That cost limits how
far the loop runs ahead (7 steps against 27). On a faster card or a smaller
model it would set the wall clock. The placement itself (0.25 ms) is not the
cause. Freeing the consumed batch is not the cause either: keeping every
batch alive changes nothing under the default preallocation. Prefetch depths
2, 8 and 32 measure the same. I adopted no fix, because the runtime owns the
addresses.

The other way to lose this time is to wait on the device every step, and that
costs 45%. The same simple_dit loop with `block_until_ready` after each step
runs at 10.3 ms against 7.1. The trainer's loop does not wait between logging
ticks. The peak allocation does not grow with how far the loop runs ahead
(0.823 GiB at 27 steps ahead, 0.819 in lockstep; 3.499 against 3.495 GiB for
hierarchical_mmdit).

### Antipatterns audited

I read through `src/dew` and measured on the small preset. The nine classes
are the ones the owner listed. Each row names the cost it found and what was
done about it.

| class | site | what was measured | verdict |
|---|---|---|---|
| 1 sync in hot paths | `training/trainer.py` `fit`, per-step `loss.astype`, `interval_loss + loss`, `jnp.where(finite, ...)`, `bad_run + 1`, `jnp.maximum` | `jax_log_compiles` over a 50-step fit: five one-op executables compiled and dispatched eagerly every step, no host sync; 176 us a step on the CPU backend of the i9-12900K | fixed on `systems/parallelism`: one jitted `bookkeep`, 37 us a step, a Regression fit from 756 to 702 us a step |
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

This pass did not run the following, so I claim nothing about them: the
`jax_default_matmul_precision` settings, remat on a step that fits in memory,
XLA flags other than command buffers, and the cost of the class-1 eager
scalars. (`bfloat16` matmul precision would change the numerics of the fp32
head, and the precision rule refuses it in any case.)

Correction, 2026-09-22, checked against current main. The jitted `bookkeep`
from the first class-1 row is on main (`src/dew/training/trainer.py:150`,
called at `:770`). The class-5 row about `objective.py:141` no longer matches
the code. The diffusion objective encodes the unconditional prompt once, when
it is built, and each step only casts that stored encoding to the batch's
dtypes (`blank_conditions`, `src/dew/objectives/diffusion/objective.py:142-149`,
used at `:204-207`).

### Against PyTorch

I reran `tools/benchmark_torch.py` in a fresh venv with torch 2.14.0+cu130 and
cuDNN 9.24. The run the week before used 2.11.0+cu128 with cuDNN 9.19. The
flags were `--mode compile --warmup 20 --steps 100`, with the small presets
and one process per row. The dew columns are the rows of
`docs/benchmarks.md` and the table above:

| case | dew ms/step | torch compile, reference attention | torch compile, SDPA cudnn | dew against the best torch row |
|---|---:|---:|---:|---:|
| simple_dit | 7.02 | 9.28 | 8.39 | 1.19x faster |
| causal_transformer | 88.78 | 81.50 | 72.46 | 0.82x, torch faster by 18% |

The week before, the decoder read 0.95x. That figure compared the parity
benchmark's fixed-batch decoder row (75.70) with torch's 72.18. The dew row
here is the benchmark's own prefetching loop, at 88.78. The two dew numbers
are 13 ms apart. Of that, 1.9 ms is the chunked head and 3.8 ms is the
decoder's own changes since `6b0f119`. The remaining 7.6 ms is the gap
between the fixed-batch row and this tool's loop at the same commit
(`6b0f119` reruns at 83.26 here on the same day). The DiT gained, from 1.16x
the week before to 1.19x. On the newer torch, torch's SDPA row is 0.4 ms
slower than the week before.

## The attention kernels

`tools/benchmark_attention.py`, bf16. The batch is chosen so that query tokens
times heads is 524288 in every row. The table shows the forward pass alone,
and the forward pass with the gradient with respect to q, k and v, in
milliseconds.

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

The reference and xla paths materialize the S x S logits. They run out of 16
GiB at S=4096, and wherever they fit they are 3 to 12 times slower than the
fused kernel. cudnn is the kernel to use for a GPU run, forward and backward,
and `'auto'` picks it wherever it can.

### Head dimension 256 through tokamax's Triton flash attention

Before Hopper, cudnn refuses head dimensions above 128. So a Gemma 3 4B or 12B
shape (heads of 256) trains through the xla path on every Ampere and Ada card,
and that path materializes the S x S logits. tokamax 0.0.13 ships a
Pallas-Triton flash attention for compute capability 8.0 and up. It has a
forward and a backward, and it takes grouped query heads, causal masks,
windows and any power-of-two head dimension. JAX 0.11.1 deprecates its own
`jax.experimental.pallas.ops.gpu.attention` in favour of it.

Measured on the RTX 4080 (compute capability 8.9, 99 KiB of shared memory per
block), driver 595.84, jax/jaxlib 0.11.1, tokamax 0.0.13. Inputs were bf16,
with 8 query and 4 key/value heads, causal, 3 warmup and 20 timed calls. The
error is against the fp32 reference einsum on the same values. The command
was `tools/benchmark_attention.py` with `--implementations xla triton
--head-dims 256 --kv-groups 2 --reference-error --triton-device-kind "NVIDIA
GeForce RTX 4090"`. The device kind is needed because JAX's Pallas-Triton
backend compiles for a table of named cards, and that table lists the 4090
but not the 4080.

| S | window | xla fwd | triton fwd | xla temp | triton temp | xla fwd+bwd | triton fwd+bwd |
|---|---|---:|---:|---:|---:|---:|---:|
| 1024 (B=2) | none | 0.468 | 0.210 | 96 MiB | 0 | 1.47 | fails, shared memory |
| 2048 (B=1) | none | 1.19 | 0.380 | 192 MiB | 0 | 2.66 | fails, shared memory |
| 4096 (B=1) | none | 3.97 | 1.03 | 768 MiB | 0 | 10.5 | fails, shared memory |
| 4096 (B=1) | 1024 | 4.10 | 0.540 | 768 MiB | 0 | 10.7 | fails, shared memory |

The Triton forward is 2.2 to 7.6 times faster than xla, uses no temporary
memory, and has a smaller error (0.0081 against xla's 0.0112, on outputs of
size 3.4). The backward does not run. tokamax's Triton VJP uses one fixed
tiling for every card (`pallas_triton_vjp.py` carries a `TODO: Implement
heuristics`). At head dimension 256 that tiling asks for 102784 bytes of
shared memory, and the card has 101376, so it fails with `RESOURCE_EXHAUSTED:
Shared memory size limit exceeded`.

I probed other tilings through tokamax's private classes. A 32x32 tiling with
one stage fits and is correct (gradient error 0.031, the same as xla). It
runs forward and backward in 1.80 ms against xla's 2.66 at S=2048. Two
16-row tilings compile and run at the same speed, but they return wrong
gradients (error 6.6 on gradients of size 6.3). tokamax's autotuner picks a
tiling by its time on random inputs and never compares numerics, so I cannot
trust autotuning to find the correct one. At head dimension 128 the Triton
kernel ties cudnn (0.235 against 0.236 ms forward, 0.75 against 0.78 forward
and backward at S=2048), so it gains nothing where cudnn already runs.

Two other features are still missing. The first is Gemma 2's logit softcap.
The Triton forward takes it (0.35 against xla's 1.01 ms at S=2048, head
dimension 256), but the VJP raises `NotImplementedError: logits_soft_cap
unsupported`. tokamax also applies the cap after adding the bias, while Gemma
applies it before (1.4e-2 apart on CPU with a bias, identical without one).
The second is attention sinks: no tokamax implementation takes them.

I added no Dew route for this kernel. A forward-only kernel cannot serve
training, and the only backward tiling that works is reachable through
private tokamax classes. The route needs an upstream tokamax release whose
VJP picks a tiling that fits the card, or a public tiling setting, with a
correctness check next to it. Installing tokamax 0.0.13 next to Dew also
pins `typeguard==2.13.3`, while tyro 1.0.16 requires `typeguard>=4.0.0`. That
breaks the command line of every recipe, so I ran the tool through its `main`
function in a separate environment.

## Odd sequence lengths on cudnn

cudnn's fused kernel has no backward pass for an odd query or key length. The
forward pass takes any length, so the problem only appeared at the first
training step, as `NotImplementedError: Unsupported sequence length Q 333, KV
333` from jax. 77 CLIP text tokens are an odd length, and so is 256+77
concatenated.

Until 2026-09-05, `'auto'` sent those shapes to the xla kernel. That kernel
materializes the [B, H, Q, K] logits and their probabilities in fp32 and
keeps them for the backward pass. `cudnn_attention` pads an odd length to an
even one instead. It adds one zero row to the query and slices it off the
output. It adds one zero key and hides it with the kernel's own padding mask
(`key_value_seq_lengths`), so every real query attends to exactly the keys it
had. On a GPU, `'auto'` picks cudnn at any sequence length, and an explicit
`'cudnn'` also takes any length.

`tests/test_kernels.py::test_cudnn_trains_odd_lengths_and_agrees_with_xla`
checks this at q1024/kv77, q9/kv7 and q333/kv333 causal. The outputs and the
three input gradients agree with the xla kernel to within two bf16 ulps of
their scale. The two kernels sit the same distance apart at an even length
(q256: 1.6e-2 at scale 2.9 on the output, 7.8e-2 at scale 15.6 on the
gradients, both one ulp). If the pad key is left unmasked, the q9/kv7 output
moves by 0.26 at scale 2.4 and the test fails. If the pad query row is left
in, the shape changes and the test fails.

To see what the padding is worth, I ran `--warmup 3 --steps 50` on the small
preset with `'xla'` (the kernel these shapes ran on before the padding)
against `'auto'`:

| architecture | shapes | xla ms/step | cudnn ms/step | xla peak GiB | cudnn peak GiB | loss at the end, xla / cudnn |
|---|---|---:|---:|---:|---:|---|
| hierarchical_mmdit | q141, q333, q1101 | 33.86 | 20.86 | 3.50 | 1.85 | 0.551035 / 0.551038 |
| simple_mmdit | q333/kv333 | 12.86 | 11.01 | 1.43 | 1.08 | 0.584398 / 0.584407 |
| unet | q256/kv77, q1024/kv77 | 16.30 | 16.13 | 0.78 | 0.71 | 0.597518 / 0.597516 |

The xla attention on the 1101-token stage kept its fp32 logits and
probabilities for the backward pass. That is where the 1.65 GiB and the 13 ms
went. Attention is a small part of the unet's step, so the unet gains little.
The losses are after 103 steps on one fixed batch and differ in the sixth
digit. That difference is the two kernels' bf16 rounding, compounded by Adam.
Decoding asks for one query position at a time, which is an odd length. It
runs on cudnn with the cache mask as an additive bias. I did not measure its
speed.

## Attention metadata and the masked conv, 2026-09-07

Before `14622ba`, any `AttentionMetadata` cost the fused kernel, whatever the
metadata said. The mixer built its `[B, 1, S, S]` mask and forced the xla
path as soon as any metadata arrived. A batch that only spelled out rotary
positions, or one whose validity marked every slot as real, paid for a mask
that excluded nothing. At `14622ba` the mixer checks what the metadata
restricts: key validity, or image groups on a bidirectional-image layer. A
validity array is opaque at trace time, so an all-true array still builds the
mask. The host producers that used to emit one leave it out when they know
the rows are whole: `pad_token_rows`, the processor's `from_hf`, generation's
input validation, the rollout collector and episode cohorts, the PPO critic
without lengths, and every MTP depth.

The Gated DeltaNet short conv had the same kind of problem inside it.
`_masked_conv1d` convolved one token per scan step to keep a paused row's
history still. At `14622ba` it compacts each row's real tokens by
`cumsum(valid) - 1` and calls the same fp32 `causal_conv1d` once.

Conditions: one RTX 4080, bf16 compute with fp32 master parameters, one fresh
process per case, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, no XLA flags, 5
warmups then 3 windows of 50 calls. The numbers are medians of the time from
dispatch to `block_until_ready`, at `14622ba` against `83f08e5`:

| case | fwd before | fwd after | fwd+bwd before | fwd+bwd after | peak MiB before / after | kernels a call before / after |
|---|---:|---:|---:|---:|---|---|
| attention, no metadata | 0.378 | 0.380 | 1.198 | 1.200 | 169 / 169 | 43 / 43 |
| attention, opaque all-true validity | 1.030 | 1.048 | 2.960 | 2.960 | 496 / 496 | 50 / 50 |
| attention, canonical metadata | 1.017 | 0.378 | 2.959 | 1.175 | 496 / 169 | 50 / 43 |
| attention, packed segments | 1.040 | 1.046 | 2.974 | 2.956 | 496 / 496 | 50 / 50 |
| GDN, no mask | 4.382 | 4.404 | 16.14 | 16.12 | 1460 / 1460 | 1882 / 1882 |
| GDN, all-true dynamic mask | 16.15 | 4.915 | 54.79 | 18.56 | 1923 / 1544 | 36700 / 1884 |
| GDN, lengths 2048 and 1537 | 16.09 | 4.876 | 54.70 | 18.46 | 1923 / 1544 | 36700 / 1884 |

The attention cases use batch 1, 2048 tokens, and 8 query and 4 key heads of
128. The GDN cases use batch 2, 2048 tokens, 8 key and 16 value heads, and
conv kernel 4. Peaks and kernel counts are the forward+backward figures.

The canonical row is the same batch as the opaque row with the redundant
validity left out, so it has the shape of a real unpadded request. Its before
column is that same call measured at `83f08e5`. The compiled HLO holds a
`__cudnn$fmhaSoftmax` custom call after the change and none before, which
shows that the route changed and the gain is not clock noise. The opaque and
packed rows are unchanged by design, and their spread across windows covers
the difference. The peaks the process allocator reports move by up to 20 MiB
between identical runs. The packed forward gave 496.02 and 476.02 MiB on two
repeats of the same executable, whose own `memory_analysis` is
byte-identical. Read the peak column at that resolution.

Case by case: canonical metadata runs the plain call exactly. Its outputs are
bitwise equal to the no-metadata forward, and its parameter gradients are
within 2.4e-06 of it. The opaque all-true mask stays on the xla kernel at its
old cost, because the shape of a validity array does not say that its
contents are all true. The GDN rows time the whole mixer (projections, gates,
rule and norm), and the masked conv is the only part that changed. With a
mask, the mixer is 3.3 times faster forward and 3.0 times faster with the
gradient. Its kernel launches drop 22.8 times forward (10707 to 470 a call)
and 19.5 times with the gradient (36700 to 1884). The scan's `while` loop is
gone from the HLO, and the `__cudnn$convForward` of the unmasked path takes
its place. The result is exact where it has to be. On lengths 2048 and 1537
the outputs agree with row-by-row evaluation to 2.4e-04 (the layer's bound is
5e-4). The padded row's input gradients and outputs are exactly zero. Against
the token scan on CPU at fp32, the largest difference over left, right,
interior and paused padding at kernels 2, 4 and 8 is 4.8e-07.

Leaving the field out changes the batch's pytree, so every process in a pool
has to agree on it. Whether a process's own rows needed padding is known only
to that process. If one process leaves the field out while another carries
it, the same step gets two different pytrees. A generation request first
agrees on the signature that ignores validity, then on one fixed-size
presence vector. Every process runs the same collectives in the same order,
whatever it holds, and materializes the field wherever any process carries
it. Where no process carries it, the field stays out and the call keeps the
fused kernel. `shard_batch` cannot run that agreement. Placement runs on the
worker thread of `DevicePrefetchIterator`, while the step's collectives run
on the caller's thread. So in a pool, every `ModelInputs` of a training batch
that lacks the field gets it materialized. Single-process runs, which is what
the table measures, are untouched. So are batches of plain token arrays,
which carry no validity anywhere.

I did not rerun the head-chunk and head-dimension-256 cases, because nothing
in this change reaches them. To reproduce, run `run_batch.sh` in
`.cache/dew/mask-routing-83f08e5`. Its `kernel_cases.py` holds the case
definitions, the allocator and HLO capture, and the correctness groups.

## XLA flags

`TrainerConfig.xla_flags` appends to `XLA_FLAGS`. `prepare_process` applies
it before JAX opens a backend. The default is None, and this sweep is the
reason. It covers three architectures, with one fresh process per
configuration. Each cell is the median of the runs, with the range and count
where I repeated a configuration.

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

I adopted no flag, and the noise band is the reason. Four repeats of the same
configuration on simple_dit spread from 6.97 to 7.53 ms, or 8%, because each
fresh process autotunes again. Against that spread, every simple_dit number
in the table comes from one distribution. The causal_transformer is the quiet
measurement, with a spread of 0.7%, and no flag moves it by more than 0.2%.
The unet is the only architecture where a flag shows an effect:
`--xla_gpu_triton_gemm_any=true` takes the median from 17.38 to 17.05 ms, or
1.9%, over four runs each.

So the unet gains 2%, the decoder is unchanged, and simple_dit cannot tell
the difference. The adoption rule asks for a flag to be faster on all three
architectures and outside the noise on each, so the default stays None. A
run that wants the unet flag can pass `--trainer.xla-flags`.

Two flags stand out for other reasons:

- `--xla_gpu_autotune_level=4` changes nothing on any architecture, because
  it is already the default in this build.
- `--xla_gpu_enable_command_buffer=` (command buffers off) is the only
  configuration that is reliably slower: 17.90 against 17.38 on the unet over
  four runs, and slower on the other two as well. Command buffers are on by
  default and save 3% on the launch-heavy architecture. Passing a longer type
  list than the default adds nothing to that.

None of the candidate flags changes numerics. The sweep covered only kernel
selection and scheduling. I tested no flag that relaxes precision, and none
would be adopted, because an adopted change has to keep a fixed-seed 20-step
loss trajectory within 1e-5.

## What batch size buys the unet

I measured these numbers and adopted nothing from them. They show where the
remaining room is on the architecture whose step is least sensitive to
batch.

```
python tools/benchmark_step.py --preset small --architectures unet \
    --batch-size 16 --warmup 3 --steps 10
```

I ran this once per batch size, and again with
`--xla-flags=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,WHILE`
for the extended rows.

| run | batch | ms/step |
|---|---|---|
| unet | 16 | 17.37 |
| unet | 64 | 57.59 |
| unet, command buffers extended | 16 | 17.12 |
| unet, command buffers extended | 64 | 57.94 |

Four times the batch costs 3.3 times the step. So about 4 ms of the 17.4 ms
step (23%) does not scale with the batch, and 0.84 ms per sample does.
Command buffers save 1.4% at batch 16 and nothing at batch 64.

When I measured these rows they had a utilisation column that read 1.7%. That
number was wrong because of the counter. XLA's `cost_analysis()` cannot see
inside the cuDNN convolution calls the backend emits, and it undercounted
this model 22.5 times. Counted off the optimized HLO, the unet runs at 40.5%
of peak, as `docs/benchmarks.md` reports.

## Muon against AdamW at equal tokens

These are the only CPU rows in this file. A loss curve at equal tokens does
not depend on the card's kernels, and the run is small enough that one
workstation CPU does nine of them in under an hour.

```
python tools/tokenize_text.py --input data/shakespeare.txt \
    --out data/shakespeare-byte --tokenizer byte --val-fraction 0.02
JAX_PLATFORMS=cpu taskset -c 0-5 python tools/optimizer_curve.py \
    --dataset data/shakespeare-byte --optimizer muon --learning-rate 3e-3 \
    --steps 2000 --emb-features 128 --num-layers 2 --num-heads 2 --seed 0 \
    --out /tmp/muon-3e-3.json
```

I ran the second command once per arm, learning rate and seed.
`data/shakespeare.txt` is not in the repository; point `--input` at your own
copy of the corpus.

Conditions: `causal_transformer`, 128 wide, 2 layers, 2 heads, tied head, byte
vocabulary of 256, sequence length 128, batch 16, 557,952 parameters, bf16
compute, weight decay 0.1 on both groups, no schedule, no clipping. 2000
steps is 4,096,000 tokens, which is 3.75 passes over the 1,093,086 training
tokens of the Shakespeare corpus. 12th Gen i9-12900K, jax 0.11.1,
`JAX_PLATFORMS=cpu`, six cores pinned per run, three runs at a time on
disjoint cores. Every arm sees the same batches in the same order at the same
seed, so a difference between two arms comes from the solver.

There are three arms. `adamw` is AdamW. `muon` is Muon as this branch builds
it. `muon-unsplit` is `optax.contrib.muon` with its own ndim == 2 rule, which
is how the 'muon' entry worked before the parameter groups. Final loss is the
mean over the last 50 steps.

| arm | lr 1e-3 | lr 3e-3 | lr 1e-2 |
|---|---|---|---|
| adamw | 1.4723 | 1.4842 | 1.5885 |
| muon | 1.5229 | 1.4438 | 1.4713 |
| muon-unsplit | 1.5762 | 1.4598 | 1.4916 |

This table shows each arm at its own best learning rate, averaged over seeds
0, 1 and 2, as the loss at five token counts:

| arm | 0.51M | 1.02M | 2.05M | 3.07M | 4.10M |
|---|---|---|---|---|---|
| adamw, lr 1e-3 | 2.0136 | 1.7376 | 1.5737 | 1.5015 | 1.4764 |
| muon, lr 3e-3 | 1.9885 | 1.6744 | 1.5179 | 1.4572 | 1.4386 |
| muon-unsplit, lr 3e-3 | 2.2454 | 1.7646 | 1.5559 | 1.4812 | 1.4561 |

Muon with the parameter groups reaches 1.4386 where AdamW reaches 1.4764,
0.038 nats lower at the same tokens. The three seeds of an arm spread 0.007
to 0.013, so the gap to AdamW is three times that noise. The gap to unsplit
Muon is 0.018, one and a half times the noise, and the split version is ahead
on each of the three seeds, by 0.016, 0.020 and 0.017. Muon also holds its
loss at ten times its best learning rate: it loses 0.028, where AdamW loses
0.116. That matches the tolerance the labs report
(`docs/research/frontier-training.md:183`).

These numbers say nothing about 0.4B parameters. That is the run section 4.9
of `docs/design/plan.md` asks for, and it needs a v5e-16. The wall-clock
times are not comparable either, because the runs shared a machine.

## Quantized training on the RTX 4080

The fp8 trunk compiles and runs on the card, but the step does not get
faster. Conditions: RTX 4080 16 GiB, driver 595.84, jax/jaxlib 0.11.1, Qwix
0.1.8, `JAX_PLATFORMS=cuda`, one process, one device, bf16 compute with the
`xla` attention kernel, `Quantization(dtype="fp8")` over the whole trunk,
adamw, 3 warmup and 10 measured steps. I ran two sizes, each in its own
process: 8 layers of width 256 (mlp 512) and 8 layers of width 1024 (mlp
2048), both with 8 heads, vocabulary 512, sequence 64 and batch 8.

| width | bf16 compile | bf16 ms/step | fp8 compile | fp8 ms/step |
|---|---:|---:|---:|---:|
| 256 | 7.95 s | 2.01 | 5.58 s | 2.18 |
| 1024 | 8.36 s | 9.59 | 6.29 s | 9.75 |

The compiled fp8 step holds `f8e4m3fn` converts (146 mentions in the HLO at
width 256, against 12 GPU gemm calls), so the quantization reaches the
device. At these sizes the converts cost more than the gemms save, and
nothing raises an error. The losses go down (2.44 bf16 against 2.68 fp8 at
width 256, 0.009 against 0.011 at width 1024, each after 14 steps from the
same init). On this card, at these sizes, fp8 gives no speedup to adopt.

## Kernel choices per backend, 2026-09-22

Each choice below is made in one place per kernel and keyed by hardware generation (`dew.nn.moe.device_generation`: `sm89`, `v6e`); a generation without a measurement here runs the XLA path. The measurements are one process per row, jax 0.11.1, bf16 compute: a Colab NVIDIA L4 (the RTX 4080's architecture, sm_89), a Colab TPU v6e-1, and the local RTX 4080 for the kernel-level rows. Step rows are `tools/benchmark_step.py` cases, 30 timed steps after 5 warmup; lm-moe is 321.8M parameters, 8 experts top-2, lm-dense 359.8M, both at sequence 1024. Batch is 4 (moe) and 1 (dense) on the L4, 8 and 8 on the v6e. "before" is main at c1f7e2dd.

### The MoE grouped matmul: `GROUPED_MATMUL_BY_GENERATION`

| device | path | ms/step | p50 ms | peak GiB |
|---|---|---|---|---|
| L4 | lm-moe before (xla) | 601.55 | 610.84 | 12.46 |
| L4 | lm-moe after, `auto` = pallas | 213.14 | 216.45 | 8.15 |
| L4 | lm-moe after, xla | 598.54 | 609.07 | 12.46 |
| v6e | lm-moe before (xla) | 76.04 | 76.43 | 5.05 |
| v6e | lm-moe after, `auto` = xla | 74.68 | 75.25 | 5.05 |
| v6e | lm-moe after, tokamax (`mosaic_tpu_v2`) | 75.40 | 76.04 | 5.05 |

`expert_projection` alone, 8192 rows, 768 to 2048, 8 experts, forward plus backward: XLA 26.21 ms and Pallas 3.38 ms on the L4; XLA 14.84 ms and Pallas 1.86 ms on the RTX 4080. Errors against a float64 oracle of the rounded operands are the same or lower for Pallas (kernel gradient 4.2e-6 against 6.8e-6 relative). The L4 step is 2.82x faster; JAX's stock Pallas lowering with an out-sharding fix measured 1.97x on the same step, because its tangents run in fp32 and Dew's backward multiplies the bf16 cotangent.

Rejected: a pure-JAX loop of dense per-tile products. On the RTX 4080 it was 2.2x faster than XLA for the projection alone (6.67 ms), but it doubled the step's temporaries (4.22 GiB against 2.17 at batch 1), and on the v6e it was 2.2x slower than XLA (1.71 ms against 0.79). The Pallas kernels are JAX's own `gmm` and `tgmm` from the jax-v0.11.2 source tree, vendored because no wheel ships them, and called through a custom VJP. jax 0.11.2 deprecates the Pallas Triton backend they run on and warns at every lowering. They stay the sm80 to sm89 path: JAX's Mosaic GPU grouped matmul (`pallas/ops/gpu/ragged_dot_mgpu.py`) uses wgmma and fails to compile on the RTX 4080, and tokamax's sm80 Mosaic config exceeds Ada's shared memory. Dew filters that one message once the kernels are used. A Mosaic GPU grouped matmul for sm90 and later waits for Hopper hardware to measure it on. Under a mesh the kernels run inside `shard_map` on each device's share of the sorted rows; that path is checked for parity on an 8-device CPU mesh and not measured on multiple GPUs.

On TPU, tokamax's `mosaic_tpu_v2` is within 1% of XLA on the step; tokamax's default dispatch picks its v1 kernel there, 13x slower, so Dew names the kernel.

### bf16 Adam state: `OptimConfig.state_dtype`

| device | measurement | fp32 state | bf16 state, hash rounding | bf16 state, threefry rounding |
|---|---|---|---|---|
| L4 | one AdamW update, lm-dense tree | 49.10 ms | 37.20 ms | 51.54 ms |
| v6e | one AdamW update, lm-dense tree | 11.45 ms | 8.89 ms | 20.07 ms |
| L4 | lm-dense step | 132.43 ms, 7.31 GiB | 120.09 ms, 5.86 GiB | |
| L4 | lm-moe step | 213.14 ms, 8.15 GiB | 208.52 ms, 6.92 GiB | |
| v6e | lm-dense step | 123.85 ms, 5.77 GiB | 124.94 ms, 4.50 GiB | |
| v6e | lm-moe step | 74.68 ms, 5.05 GiB | 71.96 ms, 3.87 GiB | |

The rounding noise is a counter hash of the step, the leaf and the element index. threefry noise (`jax.random.bits`) makes the update slower than fp32 state on both devices. The saving is memory everywhere; on the v6e lm-dense step it costs 0.9% instead of saving time, so the option stays off by default.

### The vocabulary head: `head_logits`

Forward plus backward of the chunked head alone, 8 x 1024 tokens, 1024 features, vocabulary 50304:

| device | before (fp32 operands) | bf16 operands, with argmax | bf16 operands, no argmax | fused linear cross entropy (Pallas port of Liger) |
|---|---|---|---|---|
| L4 | 206.16 ms | 134.02 ms | 133.91 ms | 142.63 ms |
| v6e | 8.04 ms | 8.03 ms | 7.26 ms | not run |

On the v6e the fp32 operands already multiplied in one bf16 pass, so only skipping the argmax moves the head. On the L4 the argmax fuses into the head's own kernels. The lm-dense step on the L4 went from 142.21 ms to 132.43 ms with the head change. Rejected: the fused Pallas kernel, 6% slower than the chunked head on the L4, and tokamax's `mosaic_tpu` head, 2.24x slower on the v6e (kernel catalog, 2026-09-22).
