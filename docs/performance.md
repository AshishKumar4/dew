# Performance measurements

This page records performance experiments: where a training step's time goes, which kernel each hardware generation runs and why, which XLA flags and optimizer settings were tried, and how expert parallelism and rematerialization behave. Each section states its hardware (an RTX 4080 unless it says otherwise; some sections use an L4, an A100, 4x RTX 3090 or a TPU v6e), revision and settings. A result at one shape and one revision does not settle a default for every case. [Step benchmarks](benchmarks.md) compares architectures, and [Distributed training](concepts/distributed.md) describes how distributed training is configured.

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

## Training scoreboard, 2026-10-02

Dew against `torch.compile` on the same models: the same batch, bf16
compute over fp32 master weights, the same optimizer constants, the same
step (forward, backward, update, and the EMA where both keep one), warm,
one process per row. A ratio is Dew's throughput over the best reference
row: above 1, Dew is faster. The torch rows come from
`tools/reference_runs/torch_lm.py` (transformers models) and
`tools/benchmark_torch.py` (line-by-line ports of Dew's modules), the Dew
rows from `tools/reference_runs/dew_lm.py` and `tools/benchmark_step.py`;
`tools/reference_runs/scoreboard.py` builds the reference-run rows into one
table.

RTX 4080 16 GiB, Dew at `42292e99` (jax 0.11.2.post3), torch 2.13.0+cu130,
transformers 5.17.0, SDPA attention. Each Dew row is two processes, given
as their range; each torch row is one process, or two where a range is
given:

| model | step | Dew | best torch.compile | Dew / torch |
|---|---|---:|---:|---:|
| Qwen3-0.6B, pretrained | 1 x 1024 tokens, AdamW | 100.4-101.9 ms, MFU 44.1-44.8% | 112.1-112.4 ms, 40.0% | 1.10-1.12 |
| Qwen3-0.6B, pretrained | 2 x 1024 tokens, AdamW | 154.0-154.2 ms, MFU 58.3-58.4% | 168.6-169.1 ms, 53.3% | 1.09-1.10 |
| 99M Qwen3-MoE shape, 8 experts, top 2 | 8 x 1024 tokens, AdamW | 79.2 ms | 112.5 ms | 1.42 |
| decoder, GPT-2 small widths, 3 layers | 16 x 512, Adam, EMA | 52.2 ms | 49.4 ms (flash), 50.0 (cuDNN) | 0.95 |
| SimpleDiT, width 384, 6 layers, 64 px | batch 16, Adam, EMA | 7.43-7.52 ms | 8.07 ms | 1.07-1.09 |
| SimpleDiT, width 768, 12 layers, 64 px | batch 32, Adam, EMA | 76.4-76.5 ms, MFU 59.2-59.3% | 76.4 ms (flash) | 1.00 |
| 176M hybrid DiT (published config) | batch 16, Adam, EMA | 69.7-72.7 ms, MFU 35.9-37.4% | no torch port | |
| 176M hybrid DiT (published config) | batch 32 | 116.6-122.3 ms, MFU 42.6-44.8% | no torch port | |

The Dew decoder and SimpleDiT rows take a fresh host batch every step, and
match the fixed-batch rows ("Comparison with PyTorch" below has both ways
for both frameworks, and the commands). On Qwen3-0.6B at 1 x 1024 torch
waits 25.1 ms a step on its host, which its 2 x 1024 step hides; on the
MoE it waits 20.9 ms, and on device time alone Dew is 1.24x faster (79.0
against 98.3 ms busy).

Since `42ddfc14`: the hybrid DiT's dilated depthwise convolutions run as
undilated ones over their interleaved grids (75.4 to 70.2 ms at batch 16);
the vocabulary head's logsumexp, maximum and argmax come from one pass, and
its gradient products read one bf16 copy of the logits' gradient, so the
MoE's whole logits fit (119.6 to 92.7 ms); pretrained weights are placed
without the hole that made Qwen3-0.6B at 2 x 1024 recompute (185.7 to
161.2 ms); and at the default precision the head rounds its logits and
their gradient to bf16 once, as torch autocast and MaxText do ("The
vocabulary head" below has the quality comparison): the 3-layer decoder
59.9 to 52.2 ms, the MoE 92.7 to 79.2, Qwen3-0.6B at 2 x 1024 161.2 to
154.0.

Where Dew wins: the optimizer update. XLA fuses Adam (or AdamW), the EMA
and the finiteness guard into one bandwidth-bound pass over the state,
6.9 ms on the 768-wide SimpleDiT against torch's fused Adam and foreach EMA
at 16.2, and 27.6 ms on Qwen3-0.6B against torch's fused AdamW and gradient
clipping at 39.5. GEMMs run at par or better (44.5 against 47.2 ms on the
768-wide SimpleDiT). Dew's host cost is at most 2.6 ms a step on these
rows, where torch.compile's reaches 25 ms; it is 13.0 ms on Qwen3-0.6B at
2 x 1024, hidden behind the device.

Where Dew loses:

- Attention: cuDNN's fused kernels take 5.6 ms forward and backward on the
  768-wide SimpleDiT (head dimension 64, 256 tokens) against
  FlashAttention-2's 4.4 in torch, and 8.7 against 7.2 on Qwen3-0.6B (head
  dimension 128, causal, 1024 tokens). With tokamax installed
  (`dewml[kernels]`), 'auto' runs its Pallas-Triton kernel for heads up to
  64 wide: 4.3 ms on the SimpleDiT, and its step 73.1 to 71.5 ms ("tokamax's
  attention" below).
- Converts and reductions: 11.0 and 5.6 ms of Qwen3-0.6B's step at 1 x
  1024, 15.9 and 9.8 of the MoE's, where torch.compile casts inside its
  GEMMs and elementwise kernels. Not yet attributed by scope.
- The 3-layer decoder, whose vocabulary head (50304 columns, 8192 tokens)
  is most of the step: 2.8 ms behind, after the head's rounding took 7.7
  ms off.

A100 40 GB (Colab), the latest records:

| model | step | Dew | reference | Dew / reference | Dew commit |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B, pretrained | 4 x 1024 tokens, AdamW | 141.0 ms | torch.compile 138.3 ms | 0.98 | `8c391009` |
| Qwen3-0.6B, pretrained | 4 x 1024 tokens | 161.9 ms | MaxText 0.2.4 GPU recipe 164.9 ms | 1.02 | `157bc21a` |
| 99M Qwen3-MoE shape, 8 experts, top 2 | 8 x 1024 tokens | 74.4 ms | torch.compile 110.5 ms | 1.48 | `157bc21a` |
| mamba2-130m | 4 x 1024 tokens | 127.9 ms | torch with mamba_ssm kernels, eager, 172.5 ms | 1.35 | `9490c9e6` |
| SimpleDiT, width 384, 8 layers, 64 px | batch 64, EMA | 21.9 ms | flaxdiff 21.4 ms | 0.98 | `9490c9e6` |

These rows predate every change listed above. On Qwen3-0.6B torch idles
18.3 ms a step on the host, so its device does 120 ms of work against
Dew's 138: Dew's attention is 21.4 against 16.6 ms (cuDNN's sm80 backward
against FlashAttention-2), its converts 17.8 ms and its reductions 10.9
against 4.2 (the norms and the fp32 head), while its GEMMs are 64.9 against
70.9 and its update, inside 22.7 ms of copies, beats torch's 19.2 ms of
copies plus 17.8 of optimizer. The MoE and Mamba-2 rows win only because
torch idles on the host (77 to 177 ms a step); on device time Dew is 1.7
and 2.0 times slower (its expert GEMMs and the XLA path of the SSD scan).
The MaxText row ran on another VM, before the whole-logits head took Dew's
step from 161.9 to 141 ms.

## The hybrid DiT's SSM blocks, 2026-10-01

The published 176M hybrid DiT (16 blocks, 12 of them S5 blocks with the
2D fusion convolution, 32x32x4 latents, patch 2) at batch 16, bf16, RTX
4080, `tools/benchmark_step.py` with `--cases` of its config. XProf with
command buffers off names each kernel's HLO instruction, and the optimized
HLO gives its JAX scope; per step, at `42ddfc14`:

| scope | ms |
|---|---:|
| MLPs (GEMMs at about 88 TFLOP/s) | 21.0 |
| S5 layers: complex projections, `associative_scan` | 15.9 |
| optimizer update (Adam and the EMA, fp32 state) | 10.0 |
| 2D fusion: dilated depthwise convolutions and the scan-order gather | 8.8 |
| SSM output projections | 4.3 |
| attention projections | 2.9 |
| norms, modulation, attention kernels, embeddings, the S5 reversals | 6.4 |
| unattributed (converts and reductions outside a scope) | 6.5 |

Each dilated depthwise convolution (dilations 2 and 3) ran as nine shifted
products in fp32, because cuDNN's dilated grouped kernels are slow; its
weight gradient read the input and the output's cotangent once per tap. A
pixel's dilation-d taps are its neighbours in the grid of pixels sharing
its row and column residues mod d, so the convolution now runs as cuDNN's
dilation-1 kernel over the d^2 interleaved grids: forward and VJP 0.097 ms
(dilation 2) and 0.089 ms (dilation 3) against 0.31 and 0.33, and the step
75.30 to 70.17 ms. In fp32, where cuDNN's depthwise kernels are slower,
the step goes from 109.70 to 113.68 ms; the published config trains in
bf16. At HIGHEST precision the fp32 output and input gradient equal lax's
dilated convolution's exactly.

Unless a section says otherwise, the sections below were measured on jax
0.11.1 / jaxlib 0.11.1 / jax_cuda12_plugin 0.11.1, driver 595.84, RTX 4080
16 GiB, single device, bf16 compute, adam, 3 warmup and 10 measured steps,
one architecture per process. The card was idle before each
measurement: `nvidia-smi --query-compute-apps=process_name` showed only
gnome-remote-desktop-daemon, which is the desktop itself. The card ran at 210
MHz and 30 W at rest and at 2760 MHz and 120-220 W under load. XLA reads a
flag once, when a backend opens, so every flag configuration ran in a fresh
process.

## Step time breakdown, 2026-09-05

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

### Host time per step

This table shows what the host spends per step on simple_dit. The numbers
come from the trace's host plane and from timing the dispatch loop while the
device was deliberately left behind. The device step is 6.9 ms.

| host work per step | ms | how measured |
|---|---:|---|
| XLA thunk execution inside PjRt Execute | 3.3 | `GpuExecutable::ExecuteThunks` on the host plane; 2.4 of it is three CUDA-graph launches |
| Python in `jax.stages.Compiled.__call__` before Execute | 1.8 | `$stages.py __call__` 6.7 ms against `PjRtCApiLoadedExecutable::Execute` 5.0 ms. The `$` events come from JAX's Python tracer, which adds time to every Python and C call, so 1.8 is an upper bound on the untraced cost |
| placing a fresh batch (`shard_batch`) | 0.25 | 200 calls timed in isolation, image plus tokens |
| the loop with a fixed device batch | 5.0 | dispatch loop time, 100 steps, device 27 steps behind |
| the loop with a fresh batch per step | 6.5 | same, device 7 steps behind |
| the loop with XLA command buffers off | 7.3 | `--xla_gpu_enable_command_buffer=`; wall 7.46 ms/step, the host is now the step |

On the smallest step, the host takes 94% of the device's time with a fresh
batch every step, and 106% without command buffers. Two conclusions follow.

First, the Python in `Compiled.__call__` costs at most 4.5 us per leaf, and this
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
2, 8 and 32 measure the same. No fix was adopted, because the runtime owns the
addresses.

The other way to lose this time is to wait on the device every step, and that
costs 45%. The same simple_dit loop with `block_until_ready` after each step
runs at 10.3 ms against 7.1. The trainer's loop does not wait between logging
ticks. The peak allocation does not grow with how far the loop runs ahead
(0.823 GiB at 27 steps ahead, 0.819 in lockstep; 3.499 against 3.495 GiB for
hierarchical_mmdit).

Correction, 2026-10-01, at `42ddfc14` (jax 0.11.2.post3, same card). The
three "loop" rows above time a dispatch loop that runs into the runtime's
limit on executions in flight, so once the device is a few dozen steps
behind each dispatch waits for a step to finish: they measure the device,
not the host. Timed right after a synchronization, eight dispatches stay
under that limit, and they cost 0.95 ms each on simple_dit with a fixed
device batch, 1.38 ms when the main thread also places a fresh batch, and
0.40 and 0.63 ms on the small decoder. With a fresh batch from
`DevicePrefetchIterator`, as `Trainer.fit` reads it, the placement runs on
the worker thread. Every loop then runs at the device's pace:

| loop, 3 repeats of 100 steps (40 on the decoder) | simple_dit ms/step | decoder ms/step |
|---|---:|---:|
| one device batch reused | 7.41-7.46 | 63.54-63.64 |
| a fresh placement each step, `DevicePrefetchIterator` | 7.43-7.50 | 63.55-63.64 |
| `Trainer.fit` over the same host batch, logging every 100 (40) steps | 7.456-7.460 | 63.69-63.73 |

The fit row includes its logging, which waits on the device once an
interval. Where the host is the step, on the cpu-smoke decoder (0.3 ms of
device work), `Trainer.fit` costs 0.51-0.62 ms a step against 0.31-0.37 for
the bare loop and 0.32-0.44 with the prefetch iterator. Its pieces, timed one
by one there: the compiled step's dispatch 234 us, the prefetch iterator's
`next` 85 us, the jitted `bookkeep` 22 us, and the batch's shapes, its row
count and the profiler regions under 1 us each. So `fit` adds about 0.2 ms
of host work a step, which no step that keeps the device busy for longer
than about 0.6 ms can see.

### Antipattern audit

An audit of `src/dew` for nine classes of performance antipattern, measured on
the small preset. Each row names the cost found and what was done about it.

| class | site | what was measured | verdict |
|---|---|---|---|
| 1 sync in hot paths | `training/trainer.py` `fit`, per-step `loss.astype`, `interval_loss + loss`, `jnp.where(finite, ...)`, `bad_run + 1`, `jnp.maximum` | `jax_log_compiles` over a 50-step fit: five one-op executables compiled and dispatched eagerly every step, no host sync; 176 us a step on the CPU backend of the i9-12900K | fixed on `systems/parallelism`: one jitted `bookkeep`, 37 us a step, a Regression fit from 756 to 702 us a step |
| 1 sync in hot paths | `jax.stages.Compiled.__call__` in `Trainer.compile` | 1.8 ms/step of Python at 396 leaves (table above) | fixed on main in `de6b22c` (jit dispatch) |
| 2 recompilation | `Trainer.fit` with evaluation every 25 of 50 steps, diffusion and LM objectives | one `jit(step)`, one `jit(initial_state)`, one evaluation executable (`_sample_impl`, `scored`); no per-step or per-eval retrace | none found |
| 3 baked constants | the compiled step's optimized HLO | simple_dit: 20 constants, 0.19 MiB, the largest the 2D sincos table bf16[256, 384]; causal_transformer: none | none found; the encoder's table moved into the state before this pass |
| 4 dtype churn | HLO dots by output dtype and the trace's convert kernels | simple_dit: 65 bf16 dots, 40 fp32 outputs that are XLA split-K partials and the fp32 `final_proj`; parameter casts 0.15 ms/step; decoder: the fp32 head by design | none found in dew's code; XLA's split-K choice is the card's |
| 5 redundant work | `nn/attention.py` odd-length routing to the xla kernel | hierarchical_mmdit 33.9 to 20.9 ms, simple_mmdit 12.9 to 11.0, peaks 3.50 to 1.85 and 1.43 to 1.08 GiB | fixed, `3b67135` |
| 5 redundant work | `objectives/lm` head chunking at its default of 4 | 1.9 ms/step (2.2%) against one chunk, for 1.2 GiB | kept as the default; the sweep is in [Step benchmarks](benchmarks.md) |
| 5 redundant work | `objectives/diffusion/objective.py:141`, `null = self.encode(...)` every step | a frozen encoder's forward on the unconditional tokens, once per step, inside the step; free with the table encoder used here, a text tower's forward at batch 1 with CLIP; not measured with CLIP | since fixed (see the correction below) |
| 6 data path | `DevicePrefetchIterator` depth, `shard_batch` cost, main-thread placement | 0.15 to 0.25 ms/step waiting for a batch at depth 2, 8 and 32; placement 0.25 ms; the loop is bounded by dispatch, not the transfer | none found |
| 7 sharding | the compiled step's `input_output_alias` | every state leaf aliased (396 of 396 on simple_dit, 143 of 143 on the decoder): donation happens | none found; collectives on a mesh not measured this pass |
| 8 compile time | `Trainer.compile` | one compile per fit (class 2 row); the FLOP count reads the same executable | none found; the persistent cache was not timed this pass |
| 9 memory | peak against the state, run-ahead against lockstep | simple_dit 0.82 GiB peak on a 303 MiB state, unchanged by run-ahead; hierarchical_mmdit 3.50 GiB on 847 MiB, 1.65 GiB of it the xla attention's fp32 logits | fixed by the class-5 row |

The audit did not measure the `jax_default_matmul_precision` settings, remat on
a step that fits in memory, XLA flags other than command buffers, or the cost
of the class-1 eager scalars, and claims nothing about them. (`bfloat16` matmul
precision would change the numerics of the fp32 head, and the precision rule
refuses it in any case.)

Correction, 2026-09-22, checked against current main. The jitted `bookkeep`
from the first class-1 row is on main (`src/dew/training/trainer.py`, called
from `Trainer.fit`). The class-5 row about `objective.py:141` no longer matches
the code: the diffusion objective encodes the unconditional prompt once, when
it is built, and each step only casts that stored encoding to the batch's
dtypes (`DiffusionObjective.blank_conditions` in
`src/dew/objectives/diffusion/objective.py`).

### Comparison with PyTorch

`tools/benchmark_torch.py` ran in a fresh venv with torch 2.14.0+cu130 and
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

Correction, 2026-10-01. The 0.82x row set Dew's loop, which places a fresh
batch every step, against torch's run without `--h2d`, which keeps one batch
on the device, and the 7.6 ms was never shown to be loop overhead. At
`42ddfc14` the two tools compare like with like: `tools/benchmark_step.py
--fixed-batch` against `tools/benchmark_torch.py` without `--h2d`, and the
default fresh-batch loop against `--h2d` (pinned host memory, copied every
step). torch 2.13.0+cu130, transformers 5.17.0, `--mode compile --attention
sdpa`, `--warmup 20 --steps 100`, one process per row:

| case | Dew, fixed | Dew, fresh | torch.compile, fixed | torch.compile, `--h2d` | Dew against torch, fixed / fresh |
|---|---:|---:|---:|---:|---:|
| causal_transformer, small preset | 63.68 | 63.63 | 49.44 (flash), 50.00 (cudnn) | 49.99 (cudnn) | 0.78x / 0.79x |
| simple_dit, small preset | 7.46 | 7.43 | 8.53 (cudnn) | 8.07 (cudnn) | 1.14x / 1.09x |
| simple_dit, width 768, 12 layers, batch 32 (`--size large`) | 76.09 | 76.11 | 76.40 (flash), 78.18 (cudnn) | 76.51 (flash), 78.38 (cudnn) | 1.00x / 1.01x |

The torch decoder takes its head's product in bf16 (`--head-dtype bfloat16`,
now the twin's default), as Dew's LM objective does; with the fp32 head the
twin carried before, torch ran 72.38 ms. A fresh batch costs Dew nothing
measurable on these rows, and `Trainer.fit` costs nothing on top (the
host-time correction above).

At `42ddfc14` the decoder's 14 ms was its vocabulary head (8192 tokens,
vocabulary 50304, three layers, so the head is most of the step). Dew kept
the logits in fp32 and carried their fp32 cotangent into the state product
as a bf16 high half and its rest, two products; torch rounds both to bf16.
Three changes since took the decoder to 52.2 ms against torch's 49.4 (the
scoreboard above): one pass for the log-sum-exp and argmax, one bf16 copy
of the cotangent for both products, and torch's rounding of the logits and
their cotangent at the default precision. XProf with
command buffers off against torch.profiler, ms per step: the forward's
log-sum-exp and argmax read the fp32 logits in two passes, 5.1, against
torch's fused log-softmax at 1.3; the backward wrote the logits' cotangent
three times in bf16 (the high half, the rest and the plain rounding for the
head's own gradient), 6.7, against 2.7; and the state product ran twice,
which was about 3 of Dew's 39.4 ms of GEMMs against torch's 33.9. Attention
is 1.7 against 1.3.

On the large DiT the two frameworks spend the step
differently (XProf with command buffers off against torch.profiler, ms per
step): GEMMs 44.5 against 47.2, the optimizer update 6.9 (XLA fuses Adam and
the EMA into one pass over the state) against 16.2, attention 5.3 (cuDNN's
forward 0.9, backward 2.9 and its two pre-Hopper backward helpers 1.5)
against 4.4 (FlashAttention-2), and Dew's reductions and converts 16.5 (the
bias gradients with the GELU backward 5.4, the norm statistics 1.9) against
torch's elementwise and norm kernels 5.8 and copies 8.9.

## Attention kernels

### tokamax's attention, 2026-10-02

tokamax's Pallas-Triton flash attention (openxla/tokamax main at `47d3d663`)
against cuDNN on an RTX 4080, bf16, forward plus backward, medians of 7
rounds of 10 calls, each checked against fp32 XLA at HIGHEST
(`max|err| / max|ref|` for dq and dk):

| shape | cuDNN | tokamax, its heuristic config | tokamax, best of a config grid | JAX's Pallas `mha`, best blocks |
|---|---:|---:|---:|---:|
| 32 x 256, 12 heads of 64 | 0.726 ms | 0.506 | 0.452 | 0.374 |
| 16 x 512 causal, 12 heads of 64 | 0.793 | 0.515 | 0.522 | 0.465 |
| 4 x 1024 causal, 16 heads of 64 | 0.833 | 0.592 | 0.576 | 0.523 |
| 4 x 1024 causal, 16 query heads over 4, of 64 | 0.784 | 0.608 | | 0.574 (keys repeated) |
| 4 x 1024 causal, 16 over 8, of 128 (Qwen3-0.6B) | 1.541 | 1.288 | 1.242 | 1.260 |
| 4 x 1024 causal, window of 256 | 0.579 | 0.587 | 0.535 | no window |

tokamax's errors are cuDNN's at every shape (dq and dk 4.8e-3 to 6.6e-3);
JAX's `mha` reaches 8.5e-3 because its backward forms `rowsum(o * do)` as a
bf16 product, and an fp32 one gives cuDNN's errors at the same speed. In the
training step the gain holds at 64-wide heads and not at 128: SimpleDiT-B
at batch 32, attention 5.62 to 4.33 ms and the step 73.1 to 71.5; the
3-layer decoder, 1.82 to 1.24 ms and 50.8 to 50.2; Qwen3-0.6B's widths at
1 x 1024, 9.13 to 9.24 ms and 97.6 to 98.4. So with tokamax installed
'auto' takes it for heads up to 64 wide and calls with no window, mask or
bias (`dew.nn.attention.triton_runs`), at tokamax's heuristic config.

tokamax trails JAX's `mha` by 10-18% at 64-wide heads in its backward
(0.35 against 0.29 ms of the SimpleDiT-B call; the forwards are 0.093 and
0.087), and no block size, warp count or stage count of its own grid closes
that. Its VJP computes wrong gradients for a causal call when
`block_m1 > block_n1` (dk and dv off by 10^2) or `block_n2 > block_m2` (dq
off by 0.7), configs its autotuning grid includes; its heuristic config is
not one of them, which is why Dew runs that and checks each shape it routes
(`tests/test_kernels.py`). tokamax's heuristic at 256-wide heads asks for
more shared memory than sm89 has (102784 of 101376 bytes) and fails; cuDNN
takes no 256-wide head either, so those calls run on XLA.

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

Other tilings, probed through tokamax's private classes: a 32x32 tiling with
one stage fits and is correct (gradient error 0.031, the same as xla). It
runs forward and backward in 1.80 ms against xla's 2.66 at S=2048. Two
16-row tilings compile and run at the same speed, but they return wrong
gradients (error 6.6 on gradients of size 6.3). tokamax's autotuner picks a
tiling by its time on random inputs and never compares numerics, so
autotuning cannot be trusted to find the correct one. At head dimension 128 the Triton
kernel ties cudnn (0.235 against 0.236 ms forward, 0.75 against 0.78 forward
and backward at S=2048), so it gains nothing where cudnn already runs.

Two other features are still missing. The first is Gemma 2's logit softcap.
The Triton forward takes it (0.35 against xla's 1.01 ms at S=2048, head
dimension 256), but the VJP raises `NotImplementedError: logits_soft_cap
unsupported`. tokamax also applies the cap after adding the bias, while Gemma
applies it before (1.4e-2 apart on CPU with a bias, identical without one).
The second is attention sinks, which no tokamax implementation takes.

Dew has no route for this kernel. A forward-only kernel cannot serve
training, and the only backward tiling that works is reachable through
private tokamax classes. The route needs an upstream tokamax release whose
VJP picks a tiling that fits the card, or a public tiling setting, with a
correctness check next to it. Installing tokamax 0.0.13 next to Dew also
pins `typeguard==2.13.3`, while tyro 1.0.16 requires `typeguard>=4.0.0`. That
breaks the command line of every recipe, so these measurements ran the tool
through its `main` function in a separate environment.

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

The padding's value, from `--warmup 3 --steps 50` on the small preset with
`'xla'` (the kernel these shapes ran on before the padding) against `'auto'`:

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
runs on cudnn with the cache mask as an additive bias; its speed was not
measured.

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

The head-chunk and head-dimension-256 cases were not rerun, because nothing
in this change reaches them.

## XLA flags

`TrainerConfig.xla_flags` appends to `XLA_FLAGS`. `prepare_process` applies
it before JAX opens a backend, and also sets `--xla_allow_excess_precision=false`
unless the run named that flag (`dew.training.runtime.keep_roundings`): with
XLA's default a fusion may skip a bf16 rounding the program states, and which
it skips depends on the layout, so one device and four computed different
bf16 forwards of one model. The recipes and the CLI call `prepare_process`;
a script or notebook that builds a `Trainer` itself calls it first, or sets
`XLA_FLAGS=--xla_allow_excess_precision=false` before importing jax. On the
RTX 4080 the flag was faster: the 176M hybrid DiT at batch 16 69.60 to 66.62
ms, SimpleDiT-B at batch 32 76.00 to 73.03, Qwen3-0.6B's widths at 1 x 1024
110.52 to 109.62. The default `xla_flags` is None, and this sweep is the
reason. It covers three architectures, with one fresh process per
configuration. Each cell is the median of the runs, with the range and count
where a configuration was repeated.

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

No flag was adopted, and the noise band is the reason. Four repeats of the same
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
  it is already the default in this build. Level 0 turns autotuning off,
  which removes XLA's compile-time kernel choice as a source of run-to-run
  differences (see [checkpoints](guides/checkpoints.md)). On the 4080 it
  slowed the 176M hybrid DiT's step from 139 to 151 ms and a 67M decoder's
  from 79.8 to 81.5 ms, two fresh processes each.
- `--xla_gpu_enable_command_buffer=` (command buffers off) is the only
  configuration that is reliably slower: 17.90 against 17.38 on the unet over
  four runs, and slower on the other two as well. Command buffers are on by
  default and save 3% on the launch-heavy architecture. Passing a longer type
  list than the default adds nothing to that.

None of the candidate flags changes numerics. The sweep covered only kernel
selection and scheduling. No flag that relaxes precision was tested, and none
would be adopted, because an adopted change has to keep a fixed-seed 20-step
loss trajectory within 1e-5.

## UNet batch scaling

These numbers show where the remaining room is on the architecture whose step
is least sensitive to batch; nothing was adopted from them.

```
python tools/benchmark_step.py --preset small --architectures unet \
    --batch-size 16 --warmup 3 --steps 10
```

It ran once per batch size, and again with
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

When first recorded, these rows had a utilisation column that read 1.7%. That
number was wrong because of the counter. XLA's `cost_analysis()` cannot see
inside the cuDNN convolution calls the backend emits, and it undercounted
this model 22.5 times. Counted off the optimized HLO, the unet runs at 40.5%
of peak, as `docs/benchmarks.md` reports.

## Muon against AdamW at equal tokens

These are the only CPU rows in this file. They compare optimizers at equal
token budgets, not accelerator speed; the run is small enough that one
workstation CPU does nine of them in under an hour.

```
curl -o data/shakespeare.txt --create-dirs \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
dew tokenize --input data/shakespeare.txt \
    --out data/shakespeare-byte --tokenizer byte --val-fraction 0.02
JAX_PLATFORMS=cpu taskset -c 0-5 python tools/optimizer_curve.py \
    --dataset data/shakespeare-byte --optimizer muon --learning-rate 3e-3 \
    --steps 2000 --emb-features 128 --num-layers 2 --num-heads 2 --seed 0 \
    --out /tmp/muon-3e-3.json
```

The first command downloads the corpus, which is not in the repository. The
last command ran once per arm, learning rate and seed.

Conditions: `causal_transformer`, 128 wide, 2 layers, 2 heads, tied head, byte
vocabulary of 256, sequence length 128, batch 16, 557,952 parameters, bf16
compute, weight decay 0.1 on both groups, no schedule, no clipping. 2000
steps is 4,096,000 tokens, which is 3.75 passes over the 1,093,086 training
tokens of the Shakespeare corpus. 12th Gen i9-12900K, jax 0.11.1,
`JAX_PLATFORMS=cpu`, six cores pinned per run, three runs at a time on
disjoint cores. Every arm sees the same batches in the same order at the same
seed, so a difference between two arms comes from the solver.

There are three arms. `adamw` is AdamW. `muon` is Dew's Muon, with its
parameter groups. `muon-unsplit` is `optax.contrib.muon` with its own ndim == 2 rule, which
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
on each of the three seeds, by 0.016, 0.020 and 0.017. Raising the learning
rate from each arm's best to 1e-2 costs Muon 0.028 (3.3 times its best rate)
and AdamW 0.116 (10 times its best rate).

These numbers say nothing about 0.4B parameters. That is the run section 4.9
of `docs/design/plan.md` asks for, and it needs a v5e-16. The wall-clock
times are not comparable either, because the runs shared a machine.

## Quantized training on the RTX 4080

The fp8 trunk compiles and runs on the card, but the step does not get
faster. Conditions: RTX 4080 16 GiB, driver 595.84, jax/jaxlib 0.11.1, Qwix
0.1.8, `JAX_PLATFORMS=cuda`, one process, one device, bf16 compute with the
`xla` attention kernel, `Quantization(dtype="fp8")` over the whole trunk,
adamw, 3 warmup and 10 measured steps. Two sizes ran, each in its own
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

## Quantized serving of the 176M text-to-image model, 2026-09-28

`TextToImage.quantized` serves the denoiser with its kernels stored as int8
or fp8 values and their scales, through Qwix's post-training quantization
(`dew.training.quantization.quantize_for_serving`). `int8` and `fp8`
quantize weights and activations, so a matmul of two quantized operands runs
in the quantized dtype; `int8w` and `fp8w` quantize weights only and
dequantize them into the compute dtype. The model is dewml/hybrid-dit-176m.
Each row is one process of `tools/benchmark_quantized_serving.py`: forward
is the warm guided denoiser call over 12 prompts (batch 24, median of 5),
sample is the warm wall time of 12 images with 20 DPM-Solver++(2M) steps at
guidance 5 including text encoding and decoding, CLIP is the mean ViT-L/14
cosine over 12 prompts at seeds 0 and 1, and memory is the denoiser's weight
bytes and the compiled forward's temporaries, in MiB.

RTX 4080 16 GiB, jax 0.11.2, Qwix 0.1.8:

| compute | precision | forward ms | sample s | CLIP | weights MiB | temporaries MiB |
|---|---|---:|---:|---:|---:|---:|
| fp32 | none | 53.3 | 1.24 | 0.2474 | 670 | 182 |
| fp32 | int8w | 52.0 | 1.26 | 0.2489 | 183 | 183 |
| fp32 | fp8w | 53.8 | 1.27 | 0.2462 | 183 | 183 |
| fp32 | int8, fusion in fp32 | 34.7 | 0.95 | 0.2470 | 183 | 158 |
| fp32 | fp8, fusion in fp32 | 32.5 | 0.87 | 0.2471 | 183 | 173 |
| bf16 | none | 36.0 | 0.82 | 0.2481 | 670 | 128 |
| bf16 | int8w | 34.8 | 0.81 | 0.2483 | 183 | 110 |
| bf16 | fp8w | 34.8 | 0.87 | 0.2472 | 183 | 110 |
| bf16 | int8, fusion in bf16 | 28.3 | 0.72 | 0.2490 | 183 | 108 |
| bf16 | fp8, fusion in bf16 | 28.3 | 0.78 | 0.2464 | 183 | 106 |

Weight-only quantization saves memory, not time: the kernels take 27% of
their fp32 bytes and the forward runs as fast as the unquantized one in the
same compute dtype. Weights and activations in int8 or fp8 take 21% off the
bf16 forward's time (28.3 ms against 36.0) and 35% to 39% off the fp32
one's (34.7 and 32.5 ms against 53.3). Every
quantized row keeps CLIP within 0.002 of fp32.

The RTX 4080's bf16 rows with quantized activations were measured before
serving scaled the product of two quantized operands in float32 (below). On
that card the change moved the bf16 forward's distance from unquantized, at
noise level 0.5 over the same batch, from 1.90% to 1.88% in int8 and from
4.23% to 4.26% in fp8.

A100-SXM4-40GB on Colab, jax 0.11.2, Qwix 0.1.8. The bf16 rows with int8 or
fp8 activations are from 2026-09-30, the rest from 2026-09-28:

| compute | precision | forward ms | sample s | CLIP | weights MiB | temporaries MiB |
|---|---|---:|---:|---:|---:|---:|
| fp32 | none | 27.5 | 1.31 | 0.2471 | 670 | 182 |
| fp32 | int8w | 33.2 | 1.70 | 0.2485 | 183 | 164 |
| fp32 | fp8w | 33.2 | 1.70 | 0.2468 | 183 | 164 |
| fp32 | int8, fusion in fp32 | 30.3 | 1.75 | 0.2496 | 183 | 168 |
| fp32 | fp8, fusion in fp32 | 34.8 | 1.42 | 0.2461 | 183 | 186 |
| bf16 | none | 25.0 | 1.82 | 0.2485 | 670 | 128 |
| bf16 | int8w | 24.9 | 1.82 | 0.2481 | 183 | 110 |
| bf16 | fp8w | 25.3 | 1.84 | 0.2470 | 183 | 110 |
| bf16 | int8, fusion in bf16 | 28.4 | 1.90 | 0.2484 | 183 | 118 |
| bf16 | fp8, fusion in bf16 | 31.7 | 2.05 | 0.2467 | 183 | 171 |

On the A100 quantizing saves memory and no time. With weights and
activations quantized the forward is slower than unquantized in the same
compute dtype: 28.4 ms in bf16 int8 against 25.0, 30.3 ms in fp32 int8
against 27.5, and slower again in fp8, which the A100 has no units for.
Every quantized row keeps CLIP within 0.0025 of fp32.

TPU v6e, one chip on Colab, jax 0.11.2, libtpu 0.0.48, Qwix 0.1.8. The bf16
rows without a weight-only precision are from 2026-09-30, the rest from
2026-09-28:

| compute | precision | forward ms | sample s | CLIP | weights MiB | temporaries MiB |
|---|---|---:|---:|---:|---:|---:|
| fp32 | none | 13.7 | 28.36 | 0.2492 | 670 | 72 |
| fp32 | int8w | 13.5 | 27.73 | 0.2488 | 183 | 74 |
| fp32 | fp8w | 13.6 | 27.75 | 0.2466 | 183 | 74 |
| fp32 | int8 | 7.1 | 28.50 | 0.2476 | 182 | 79 |
| fp32 | fp8 | 13.3 | 28.08 | 0.2472 | 182 | 79 |
| fp32 | int8, fusion in fp32 | 14.9 | 28.02 | 0.2469 | 183 | 61 |
| fp32 | fp8, fusion in fp32 | 20.4 | 28.82 | 0.2452 | 183 | 61 |
| bf16 | none | 5.7 | 29.56 | 0.2493 | 670 | 126 |
| bf16 | int8w | 5.2 | 29.63 | 0.2497 | 183 | 46 |
| bf16 | fp8w | 5.6 | 30.58 | 0.2478 | 183 | 46 |
| bf16 | int8 | 6.5 | 28.45 | 0.2499 | 182 | 52 |
| bf16 | fp8 | 9.8 | 28.19 | 0.2462 | 182 | 52 |
| bf16 | int8, fusion in bf16 | 6.3 | 28.40 | 0.2521 | 183 | 50 |
| bf16 | fp8, fusion in bf16 | 9.3 | 28.56 | 0.2443 | 183 | 50 |

On the v6e int8 weights and activations halve the fp32 forward (7.1 ms
against 13.7) with the depthwise convolutions quantized too; with them in
fp32 the forward takes 14.9 ms. In bf16 the weight-only rows run in the
unquantized forward's time (5.2 and 5.6 ms against 5.7), and with activations
quantized the forward is slower (6.3 to 9.8 ms). Sampling takes 28
to 31 s in every row, whatever the forward's time, so on this machine
something other than the denoiser's 20 steps sets it; this section does not
break it down. Every quantized row keeps CLIP within 0.005 of fp32.

Before serving scaled the product of two quantized operands in float32,
every bf16 row with int8 or fp8 activations sampled NaN images on the v6e
(CLIP 0.1481, with the depthwise convolutions quantized or not). Qwix 0.1.8
scales that product in the scales' dtype, bf16 in a bf16 model. In plain
JAX on the v6e, an int8 depthwise convolution whose int32 product is scaled
in bf16 came out NaN in all but a few outputs, while the model's dense and
attention forms scaled the same way stayed finite. In the served model the
NaN began in the depthwise convolutions in int8 and in an attention block in
fp8. Scaled in float32, the bf16 model's 8-bit operations have the result
types of the fp32 model's, and it samples as above.

XLA:CPU, 12 threads of a Colab L4 host, jax 0.11.2, Qwix 0.1.8, latents
decoded two at a time (`--decode-batch 2`). The bf16 int8 row is from
2026-09-30, the rest from 2026-09-28:

| compute | precision | forward ms | sample s | CLIP | weights MiB | temporaries MiB |
|---|---|---:|---:|---:|---:|---:|
| fp32 | none | 4912 | 111.72 | 0.2472 | 670 | 563 |
| fp32 | int8w | 4953 | 112.50 | 0.2486 | 183 | 844 |
| fp32 | int8 | 9804 | 208.77 | 0.2465 | 182 | 192 |
| bf16 | none | 5472 | 122.54 | 0.2495 | 670 | 657 |
| bf16 | int8w | 5550 | 123.50 | 0.2494 | 183 | 584 |
| bf16 | int8 | 10550 | 224.61 | 0.2497 | 182 | 152 |

On XLA:CPU int8 weights and activations double the forward's time (9.8 s
against 4.9 in fp32) and weight-only int8 leaves it as it was. Every
quantized row keeps CLIP within 0.0025 of fp32.

On a GPU, Dew refuses to quantize the activations of a grouped
convolution, so the GPU rows with quantized activations keep the spatial
fusion's depthwise convolutions in float (`--float spatial_fusion`), and
`TextToImage.quantized` raises without it. XLA:GPU (jax 0.11.2) computes
those convolutions wrongly or not at all. On the RTX 4080, an int8
convolution with one or two input channels per group returns wrong values
without an error: before the refusal, the whole-model int8 row ran in 42.3
ms and scored CLIP 0.1391. In fp8 the same convolutions fail to compile
there (`Failed to get configs for: 36 out of 126 instructions`, one per
depthwise convolution). On the A100 the whole-model int8 row scored CLIP
0.1373 in fp32 and failed to compile in bf16 (`UNIMPLEMENTED`). fp8
computed on the A100 (CLIP 0.2443 in fp32), but Dew refuses it there too,
with the rest of the GPUs.

## Kernel choices per generation, 2026-09-22

Each choice below is made in one place per kernel and keyed by hardware generation (`dew.nn.kernels.device_generation`: `sm80`, `sm86`, `sm89`, `v5e`, `v6e`, ...); a generation without a measurement here runs the XLA path. `tools/benchmark_kernels.py` and `tools/benchmark_lm_head.py` reproduce the rows. The measurements are one process per row, jax 0.11.1, bf16 compute: a Colab NVIDIA L4 (the RTX 4080's architecture, sm_89), a Colab TPU v6e-1, and the local RTX 4080 for the kernel-level rows. Step rows are `tools/benchmark_kernels.py step` (built on `tools/benchmark_step.py`'s trainer), 30 timed steps after 5 warmup; lm-moe is 321.8M parameters, 8 experts top-2, lm-dense 359.8M, both at sequence 1024. Batch is 4 (moe) and 1 (dense) on the L4, 8 and 8 on the v6e. "before" is main at c1f7e2dd.

### The MoE grouped matmul: `GROUPED_MATMUL_BY_GENERATION`

| device | path | ms/step | p50 ms | peak GiB |
|---|---|---|---|---|
| L4 | lm-moe before (xla) | 601.55 | 610.84 | 12.46 |
| L4 | lm-moe after, `auto` = pallas | 213.14 | 216.45 | 8.15 |
| L4 | lm-moe after, xla | 598.54 | 609.07 | 12.46 |
| v6e | lm-moe before (xla) | 76.04 | 76.43 | 5.05 |
| v6e | lm-moe after, `auto` = xla | 74.68 | 75.25 | 5.05 |
| v6e | lm-moe after, tokamax (`mosaic_tpu_v2`) | 75.40 | 76.04 | 5.05 |

Rerun at jax 0.11.2, one Colab L4 session (2026-09-22 19:55 to 20:13 CDT), `tools/benchmark_kernels.py step --path lm-moe --batch 4`: `auto` (pallas) 224.90 ms, 8.15 GiB peak; `--implementation xla` 602.36 ms, 12.46 GiB. `projection`: Pallas 3.37 ms, XLA 25.19 ms forward plus backward.

On a mesh, 2x RTX 3090 (jax 0.11.2, one bf16 ExpertMLP layer forward plus backward, XLA against Pallas): fsdp 1 expert 1 141.9 against 64.4 ms; fsdp 2 83.3 against 98.1 ms, where the Pallas path all-gathers the fsdp-sharded expert kernel (128 MiB of temporaries against XLA's 3120); expert 2 global 174.0 against 102.7; expert 2 exchange 93.0 against 14.9. Whole lm-moe steps: data 2 1024.0 against 578.3 ms, expert 2 exchange 780.9 against 537.1, same losses. That made `auto` take XLA where fsdp alone sharded the experts, until the dispatch moved every routed layer inside its row map, where both kernels see gathered experts, and there the Pallas kernels win on the RTX 3090 ([Expert parallelism on 4x RTX 3090](#expert-parallelism-on-4x-rtx-3090-2026-09-23)).

Kernel-matrix rows, forward plus backward, jax 0.11.2, checked against float64: at lm-moe's up projection the Pallas kernels take 1.21 ms against XLA's 6.25 on an A100, 3.15 against 25.9 on an L4 and 1.59 against 15.7 on the RTX 4080; at 128 experts 0.43 against 15.7 (A100), 1.38 against 77.9 (L4) and 0.55 against 33.5 (RTX 4080). On a TPU v5e and v6e XLA wins at 128 experts (v6e 0.346 ms against `mosaic_tpu_v2`'s 0.408). On an RTX 3090 (sm86, jax 0.11.2): 2.54 ms against 17.40 up and 2.37 against 16.97 down, same forward error. An sm75 card (T4) cannot compile the Triton kernels, and no sm90 or sm120 card was available, so those run XLA.

`expert_projection` alone, 8192 rows, 768 to 2048, 8 experts, forward plus backward: XLA 26.21 ms and Pallas 3.38 ms on the L4; XLA 14.84 ms and Pallas 1.86 ms on the RTX 4080. Errors against a float64 oracle of the rounded operands are the same or lower for Pallas (kernel gradient 4.2e-6 against 6.8e-6 relative). The L4 step is 2.82x faster; JAX's stock Pallas lowering with an out-sharding fix measured 1.97x on the same step, because its tangents run in fp32 and Dew's backward multiplies the bf16 cotangent.

Rejected: a pure-JAX loop of dense per-tile products. On the RTX 4080 it was 2.2x faster than XLA for the projection alone (6.67 ms), but it doubled the step's temporaries (4.22 GiB against 2.17 at batch 1), and on the v6e it was 2.2x slower than XLA (1.71 ms against 0.79). The Pallas kernels are JAX's own `gmm` and `tgmm` from the jax-v0.11.2 source tree, vendored because no wheel ships them, and called through a custom VJP. jax 0.11.2 deprecates the Pallas Triton backend they run on and warns at every lowering. They stay the sm80 to sm89 path: JAX's Mosaic GPU grouped matmul (`pallas/ops/gpu/ragged_dot_mgpu.py`) uses wgmma and fails to compile on the RTX 4080, and tokamax's sm80 Mosaic config exceeds Ada's shared memory. Dew does not silence the warning: it is the user's to filter, and moving this path to Mosaic GPU on sm90 and later is an open item that waits for Hopper hardware to measure on. Under a mesh the kernels run inside `shard_map` on each device's share of the sorted rows; that path is checked for parity on an 8-device CPU mesh and not measured on multiple GPUs.

On TPU, tokamax's `mosaic_tpu_v2` is within 1% of XLA on the step; tokamax's default dispatch picks its v1 kernel there, 13x slower, so Dew names the kernel.

### bf16 Adam state: `OptimConfig.state_dtype`

| device | measurement | fp32 state | bf16 state, hash rounding | bf16 state, threefry rounding |
|---|---|---|---|---|
| L4 | one AdamW update, lm-dense tree | 49.10 ms | 37.20 ms | 51.54 ms |
| v6e | one AdamW update, lm-dense tree | 11.45 ms | 8.89 ms | 20.07 ms |
| L4 | lm-dense step | 138.72 ms, 7.34 GiB | 126.45 ms, 5.89 GiB | |
| L4 | lm-moe step | 224.90 ms, 8.15 GiB | 218.15 ms, 6.92 GiB | |
| v6e | lm-dense step | 123.85 ms, 5.77 GiB | 124.94 ms, 4.50 GiB | |
| v6e | lm-moe step | 74.68 ms, 5.05 GiB | 71.96 ms, 3.87 GiB | |

The two L4 step rows are jax 0.11.2, from one Colab session (2026-09-22, 19:55 to 20:13 CDT); the update rows and the v6e rows are jax 0.11.1. The rounding noise is a counter hash of the step, the leaf and the element index. threefry noise (`jax.random.bits`) makes the update slower than fp32 state on both devices. The saving is memory everywhere; on the v6e lm-dense step it costs 0.9% instead of saving time, so the option stays off by default.

### The vocabulary head: the compute dtype's product

The head's product follows the compute dtype, as torch autocast and MaxText (`logits_dot_in_fp32=False`) run it: under bf16 compute both operands multiply as bf16 with fp32 accumulation, in the model's head and the chunked loss alike, and the softmax and the loss stay fp32; an fp32 model keeps its fp32 head. Forward plus backward of the chunked head alone, 8 x 1024 tokens, 1024 features, vocabulary 50304:

| device | fp32 operands (before) | bf16 operands, with argmax | bf16 operands, no argmax | fused linear cross entropy (Pallas port of Liger) |
|---|---|---|---|---|
| L4 | 206.16 ms | 134.02 ms | 133.91 ms | 142.63 ms |
| v6e | 8.04 ms | 8.03 ms | 7.26 ms | not run |

On the v6e the fp32 operands already multiplied in one bf16 pass, so only skipping the argmax (`token_accuracy=False`) moves the head. On the L4 the argmax fuses into the head's own kernels. The lm-dense step on the L4 went from 138.72 ms to 129.36 ms with the bf16 product; on the RTX 4080 (jax 0.11.2) the head at 4 x 1024 tokens went from 45.25 ms to 28.00 ms.

The bf16 product changes the loss by less than its own rerun spread. `tools/lm_step_parity.py`, 100 steps of the 39M-parameter decoder on the RTX 4080, twice each way: two fp32-head runs differ by at most 2.3e-4 relative at any step, two bf16-head runs by 7.7e-4, and a bf16-head run differs from an fp32-head run by 3.4e-4 and 7.2e-4, within the bf16 head's own rerun spread. Final losses 0.0078378 and 0.0078376 (fp32 head), 0.0078368 and 0.0078387 (bf16 head). Rejected: the fused Pallas kernel, 6% slower than the chunked head on the L4, and tokamax's `mosaic_tpu` head, 2.24x slower on the v6e (kernel catalog, 2026-09-22).

On an A100, the reference runs measured the fp32 head at 38 ms a step, 21% of a Qwen3-0.6B bf16 fine-tune's busy time, as TF32 GEMMs that torch autocast runs in bf16; that and the rows above made the bf16 product the default.

The logits' rounding, 2026-10-01. The bf16 product above still kept fp32 logits, and carried their fp32 gradient into the state product as two bf16 products, a high half and the rest (`347238c7`), where torch autocast and MaxText round both to bf16. At the default `matmul_precision` Dew now rounds as they do: the logits to bf16 values and their gradient to bf16 once, read by both backward products. On the RTX 4080 (`tools/benchmark_step.py --fixed-batch`, one session against `66784383`) the 3-layer decoder (GPT-2 small widths, vocabulary 50304, 16 x 512 tokens) runs 52.17 ms against 59.95, with a planned peak of 4.53 GB against 5.36, where torch.compile runs it in 49.4-50.0; Qwen3-0.6B's widths at 1 x 1024 run 106.62 ms against 110.04. Training quality, 2000 steps of wikitext-103 Qwen3 tokens on the same card, with validation over 64 fixed windows scored with fp32 logits for both: the 3-layer decoder at vocabulary 151936 from scratch ends at 5.0604 and 5.0491 (fp32 logits, seeds 0 and 1) against 5.0604 and 5.0493, and Qwen3-0.6B fine-tuned ends at 2.71965 and 2.71996 against 2.71983 and 2.71981. At the same seed the two roundings are at most 4.5e-4 and 1.6e-3 apart at any checkpoint, where the two seeds of either rounding are 1.3e-2 and 2.9e-3 apart on average. The high half existed for layout parity: with the gradient rounded once, a 4 x RTX 3090 bf16 run read 1.75 times its bound at a dense model's final norm and 5758 times it at an MoE's expert gate_proj, against 0.47 and 0.41; why the MoE's gap is that large is not yet established. A run that compares layouts in bf16 sets `matmul_precision="highest"`, which keeps the head fp32; `tools/layout_parity.py` does so for bf16 decoders.

On sm89, the trainer compiles without XLA's Triton GEMM fusions unless the run explicitly sets that flag or the model has an SSD mixer. At Qwen3-0.6B's widths with two layers, bf16, vocabulary 151936, and a 0.9 allocator fraction on an RTX 4080 (JAX 0.11.2), this removes a 4096-token cliff: 286.0 ms per training step with the fusions, 93.4 ms without. At other shapes, the unfused step can use more temporary memory. Before tiling the head or recomputing blocks, a step that does not fit is tried with XLA's default options; at 8192 tokens only that whole-logits step fits (178.7 ms, versus 211.2 ms after tiling). When tiling is needed, sm89 uses the measured 4096-by-8192 tile. These are two-layer measurements, not full-model times.

### Generations below sm80

A T4 (sm75) rejects the `BF16_BF16_F32` dot algorithm at run time ("UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32"), cuDNN's fused attention refuses bf16 there ("SDPA FP16/BF16 requires SM80"), and Triton does not compile for it. `dew.nn.kernels.generation.bf16_dot_runs` is the one test: below sm80 bf16 attention takes the reference path for `auto` and `xla`, the bf16 operand precision keeps the caller's precision, and the grouped matmul runs XLA.

### Faster kernels not adopted

Kernel matrix, 2026-09-22, jax 0.11.2, forward plus backward medians, every cell checked against float64. Each needs a kernel or a dependency Dew does not carry yet:

| op | where it wins | numbers | why not yet |
|---|---|---|---|
| tokamax `ragged_dot` `mosaic_tpu_v2` with tokamax's own VJP | TPU v5e and v6e, 8 experts | lm-moe up 0.899 ms against XLA's 1.05 (v5e), 0.395 against 0.48 (v6e); down 0.859 against 1.19 (v5e), 0.345 against 0.383 (v6e) | tokamax 0.0.14 pins typeguard==2.13.3 and tyro needs >=4; its flax.nnx import fails on jax 0.11.2. At 128 experts XLA wins. |
| tokamax `triton` RMSNorm | sm80, sm89 | 1.07x (A100), 1.53x (L4), 1.57x (RTX 4080) over XLA | tokamax dependency. |
| fused-weight SwiGLU (tokamax `xla` formulation, one contraction for gate and up) | every GPU | 1.47x (A100), 1.47x (L4), 1.55x (RTX 4080), 1.21x (T4) over two XLA matmuls; no gain on TPU | a change to the MLP's parameter layout. |
| tokamax `xla` head plus cross entropy | sm89 speed | 73.0 ms (L4) and 36.8 ms (RTX 4080), 1.55x and 1.58x over Dew's chunked head | tokamax dependency, and it holds 1.6-3.2 GiB where the chunked head holds 131-355 MiB. |
| JAX's Pallas-Triton `mha` (`jax.experimental.pallas.ops.gpu.attention`), 2026-10-01 | sm89, training attention | forward plus backward, bf16, against cuDNN: 0.38 against 0.64 ms (batch 32, 256 tokens, 12 heads of 64), 0.43 against 0.81 (causal, batch 16, 512 tokens), 1.12 against 1.36 (causal, batch 4, 1024 tokens, 16 heads of 128); in the step, routed where a call has no bias, mask, window or lengths: the 768-wide SimpleDiT 76.10 to 74.81 ms, the small decoder 63.61 to 62.86 | deprecated in JAX 0.11 for tokamax; against an fp32 reference its dq and dk errors reach 8.5e-3 of their maximum where cuDNN's reach 5.3e-3 (causal, 512 tokens); no grouped-query heads. |
| JAX's Pallas GPU `paged_attention` | sm80 and later, decode | 1.75-1.84x (A100), 1.85-2.1x (L4), 2.1-2.8x (RTX 4080) over the XLA gather | a decode-path change; on TPU the XLA gather wins at batch 8 and up to 2k context. |

### The Mamba-2 SSD scan: `ssd_kernel_runs`

`tools/benchmark_ssd.py`, forward plus backward, batch 1, 8 heads of 64, state 128, jax 0.11.2:

| device | chunk | length | XLA | Pallas kernel |
|---|---|---|---|---|
| TPU v6e | 256 | 4096 | 0.915 ms | 0.634 ms |
| TPU v6e | 256 | 16384 | 4.505 ms | 1.911 ms |
| TPU v6e | 256 | 65536 | 17.245 ms | 7.122 ms |
| TPU v6e | 128 | 4096 | 0.632 ms | 0.705 ms |
| RTX 4080 | 256 | 4096 to 65536 | 2.28 to 33.63 ms | does not compile: 590 KB of shared memory asked, 101 KB available |

On the RTX 4080 the Triton kernel ran 6x to 12x slower than XLA wherever it compiled (chunk 64, width 32: 1.39 against 0.22 ms; at batch 8 and 16 heads, 22.7 against 2.2 ms), and every chunk of 128 or 256 asked for 131 to 590 KB of shared memory. The scan takes the kernel on TPU only.

### Packed sliding-window attention on GPU: `local_attention`

A packed batch with a sliding window has no fused-kernel flag on a GPU before Hopper: `jax.nn.dot_product_attention` takes no segment ids beside `local_window_size`, cuDNN's packed layout (`q_offsets`) raises "Packed layout requires a GPU with at least Hopper architecture" on sm89, and JAX's Pallas GPU `mha` takes segment ids but no window (and was 4-9% off in the gradient at this shape). `local_attention` therefore builds its `[W, 2W]` band mask and, where cuDNN runs, hands it to cuDNN as the additive bias; elsewhere xla takes it. Colab L4, jax 0.11.2, bf16, 16 query heads of 64 over 4 key heads, window 4096, 5 packed documents, forward plus backward:

| tokens | before (band on xla) | after (band on cuDNN) |
|---|---|---|
| 2048, window 512 | 6.15 ms, 0.32 GiB | 1.37 ms, 0.05 GiB |
| 8192 | out of memory (10.0 GiB requested) | 40.0 ms, 0.24 GiB |
| 16384 | out of memory | 76.9 ms, 0.60 GiB |
| 32768 | out of memory | 156.5 ms, 1.19 GiB |
| 65536 | out of memory | 316.7 ms, 2.38 GiB |

Against a float64 oracle at 2048 tokens the output error is 2.7e-3 relative and the gradients 3.3e-3 to 6.6e-3, the same as the xla path's. A dense `[S, S]` document mask on cuDNN is faster at 32768 tokens on an RTX 4080 (63.7 against 78.1 ms) but grows with the square of the length and ran out of memory at 65536, so the band is the path.

## Expert parallelism on 4x RTX 3090, 2026-09-23

The machine is one host with four RTX 3090s. GPU0 and GPU1 are joined by NVLink (NV4), GPU2 and GPU3 share a PCIe host bridge, and every other pair crosses the two sockets. jax 0.11.2, bf16 compute, the Pallas grouped matmul, `tools/benchmark_step.py` with 12 timed steps after 3 warmup and 3 traced. Each row's attribution is benchmark_step's reading of its trace, in milliseconds per device per step: compute kernels, each collective, and the communication no compute kernel overlapped. The model is a `causal_transformer` of 8 layers, width 1024, 16 heads and vocabulary 50304, with 32 experts of width 1024 and top-4 routing on every layer, at 4096 tokens a device (batch 16 of 1024 on four GPUs, 8 on two).

The links first, as JAX collectives of 128 MB of bf16 a device: `all_to_all` moves 33.4 GB/s over the NVLink pair, 6.9 over the PCIe pair and 6.7 across the sockets, and `all_gather` and `psum` follow (31.0, 5.8, 5.8 and 33.4, 5.7, 6.3). The PCIe pair is no faster than a cross-socket pair, so GPU0 and GPU1 are the only fast pair on this box.

| mesh | expert axis joins | dispatch | ms/step | tokens/s | compute | exposed comm | all-to-all | all-gather | reduce-scatter | all-reduce | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fsdp 4 | - | global | 904.0 | 17562 | 199.1 | 725.1 | 0 | 373.7 | 375.1 | 0.5 | 7.9 |
| expert 4 | 0123 | global | 925.5 | 17113 | 179.2 | 774.6 | 0 | 341.5 | 335.1 | 98.0 | 8.9 |
| expert 4 | 0123 | exchange | 634.0 | 26232 | 272.4 | 352.1 | 271.1 | 0 | 0 | 99.5 | 13.5 |
| expert 4 | 0123 | exchange, capacity 1.25 | 550.6 | 28829 | 163.6 | 422.2 | 253.8 | 0 | 0 | 168.3 | 9.3 |
| expert 2 x fsdp 2 | 02, 13 | exchange | 788.1 | 20916 | 248.6 | 537.4 | 281.3 | 97.7 | 102.2 | 58.9 | 10.5 |
| expert 2 x fsdp 2 | 01, 23 | exchange | 800.3 | 20598 | 247.8 | 544.5 | 109.2 | 203.7 | 212.3 | 22.2 | 10.5 |
| data 2 x expert 2 | 01, 23 | exchange | 753.8 | 21912 | 252.1 | 501.9 | 118.1 | 0 | 0 | 421.6 | 14.0 |
| data 2 x expert 2 | 02, 13 | exchange | 702.9 | 24176 | 254.6 | 442.0 | 211.8 | 0 | 0 | 306.1 | 14.0 |

One pair at a time, `MeshSpec(expert=2)` at the same 4096 tokens a device:

| pair | dispatch | ms/step | compute | exposed comm | all-to-all | all-gather | reduce-scatter | all-reduce |
|---|---|---|---|---|---|---|---|---|
| GPU0-1, NVLink | exchange | 295.8 | 245.6 | 46.9 | 40.7 | 0 | 0 | 17.8 |
| GPU0-1, NVLink | exchange, capacity 1.25 | 212.3 | 179.9 | 27.3 | 21.3 | 0 | 0 | 25.5 |
| GPU0-1, NVLink | global | 298.8 | 190.4 | 103.2 | 0 | 44.7 | 46.9 | 11.7 |
| GPU2-3, PCIe | exchange | 456.2 | 240.6 | 214.4 | 169.9 | 0 | 0 | 80.1 |
| GPU2-3, PCIe | exchange, capacity 1.25 | 332.4 | 175.5 | 162.7 | 100.4 | 0 | 0 | 81.2 |
| GPU2-3, PCIe | global | 780.4 | 188.0 | 710.4 | 0 | 309.5 | 312.4 | 88.5 |
| GPU0-2, cross-socket | exchange | 454.9 | 241.4 | 210.3 | 172.6 | 0 | 0 | 73.7 |
| GPU0-2, cross-socket | global | 795.5 | 188.0 | 620.9 | 0 | 279.9 | 273.8 | 67.2 |

What the traces say:

- The exchange beats the global dispatch wherever the link is slow: 1.46x on four GPUs, 1.71x on the PCIe pair and 1.75x across the sockets. On the NVLink pair the two tie, because the global dispatch's expert all-gather and gradient reduce-scatter cost 92 ms there, against 622 ms on the PCIe pair.
- Communication is mostly exposed. Four-way exchange spends 272 ms computing and 352 ms waiting on collectives that no compute overlaps, most of it the all-to-all. Capacity 1.25 bounds the buckets and drops the later rounds, which takes the step to 550.6 ms.
- Placement. Under data x expert the gradient all-reduce over the data axis moves more bytes than the token exchange, so the expert axis belongs across the sockets and the data axis on the pairs: 702.9 ms against 753.8. Under expert x fsdp the two placements tie (788.1 against 800.3), since fsdp's all-gather and reduce-scatter trade places with the all-to-all.

Changes, each measured before and after in one hold:

- Expert parameters enter the dispatch's `shard_map` in their stored shards and are gathered inside it, so their gradient is reduce-scattered rather than all-reduced whole. fsdp 4 goes from 1157.1 to 904.0 ms, where a 614 ms all-reduce becomes a 375 ms reduce-scatter; expert 2 x fsdp 2 goes from 856.0 to 788.1 ms.
- The exchange's first round runs outside the checkpointed scan that holds the later rounds, so the backward keeps its intermediates instead of recomputing them: expert 4 goes from 705.9 to 634.0 ms, and compute from 307.0 to 272.4.
- The exchange gathers each bucket's rows through the sort's index and scatters what returns straight to its slots, two row copies fewer a round. On the NVLink pair, dropless goes from 300.5 to 295.8 ms and capacity 1.25 from 218.5 to 212.3, with peak memory from 14.1 to 13.7 GiB and 12.6 to 12.0.

One bf16 `ExpertMLP` layer under `MeshSpec(fsdp=2)` on the NVLink pair (8192 tokens, 32 experts, top 4), forward plus backward: the Pallas kernels inside the dispatch's map take 26.2 ms (25.1 under the earlier row map), and `jax.lax.ragged_dot` inside the map takes 271.9 ms with 6.5 GiB of temporaries, since XLA runs it as a product over every expert. The same layer through the global path outside any map, where the earlier kernel selector sent fsdp-only meshes to XLA, ran out of memory on the 24 GiB cards.

### Rematerialization: the trainer's ladder

A model's `remat` is where its step starts, and the trainer moves it up one rung whenever the compiled step does not fit its devices' memory (`dew.training.trainer.step_fits`: XLA places a GPU step's temporaries in one allocation, so they, the outputs that do not reuse the donated state and the batches `fit` prefetches beside the step's own have to fit one free block of each device, not just its free bytes; the block is the BFC pool's largest, or the part of its limit a growing pool has not taken yet, and an allocator that reports no pool, cuda_async or a TPU's, is read by its free bytes; each process reads its own devices and the pool takes the tightest. Where the allocator can leave the temporaries no block to return to once a batch is prefetched beside them, the check holds room for them twice (`strands_temporaries`): cuda_async, and XLA's spatially partitioned pool, which `prepare_process` turns off): a decoder from none to `'minimal'` (MaxText's name: every projection output kept) to `'full'`, a diffusion backbone from `False` to `'dots'` (matmul outputs and the attention forward kept) to `'full'`. Each rung is slower and smaller, so the first that fits is the fastest that runs. A checkpoint records the rung its state trained on, head tile, remat and XLA options, and a resumed run compiles that rung and climbs from it only where it does not fit, saying so: a process that restores a state finds other free memory than the one that built it, and a lighter rung would run another program. The rung a step compiled under is the `remat` of the run's `StepCompiled` record and of `tools/benchmark_step.py`'s rows. Forward plus backward plus AdamW, bf16 compute, 10 timed steps, `tools/benchmark_kernels.py step --remat`:

| device | model, batch x tokens | none | minimal / dots | full |
|---|---|---|---|---|
| L4 | 359.8M decoder, 4 x 1024 | 334.7 ms, 9.93 GiB | 356.4 ms, 8.00 GiB | 401.3 ms, 6.29 GiB |
| L4 | 359.8M decoder, 8 x 1024 | 676.3 ms, 13.39 GiB | 719.6 ms, 10.00 GiB | 815.9 ms, 6.57 GiB |
| L4 | 359.8M decoder, 16 x 1024 | out of memory | out of memory | 1671.1 ms, 7.22 GiB |
| L4 | 321.8M MoE decoder, 4 x 1024 | 213.2 ms, 8.15 GiB | 235.4 ms, 6.54 GiB | 251.7 ms, 6.13 GiB |
| L4 | 321.8M MoE decoder, 8 x 1024 | 396.0 ms, 10.33 GiB | 420.1 ms, 7.82 GiB | 454.5 ms, 6.69 GiB |
| L4 | DiT-L/2, 16 x 64x64 | out of memory | out of memory | 1377.8 ms, 8.81 GiB |
| L4 | DiT-L/2, 32 x 64x64 | out of memory | out of memory | 2671.8 ms, 10.16 GiB |
| RTX 3090 | 359.8M decoder, 4 x 1024 | 225.2 ms, 9.70 GiB | 237.9 ms, 7.67 GiB | 270.0 ms, 6.06 GiB |
| RTX 3090 | 359.8M decoder, 8 x 1024 | 420.9 ms, 13.16 GiB | 438.3 ms, 9.77 GiB | 505.1 ms, 6.33 GiB |
| RTX 3090 | 321.8M MoE decoder, 4 x 1024 | 126.5 ms, 7.82 GiB | 133.6 ms, 6.26 GiB | 150.2 ms, 6.02 GiB |

In the rows where all three ran (the decoders), `'minimal'` costs 4-10% over no recomputation and `'full'` 15-21%, so a model that fits runs without either. The L4 rows are jax 0.11.2 on Colab (2026-09-23), the RTX 3090 rows one GPU of the box.
