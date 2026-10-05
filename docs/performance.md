# Performance measurements

This page records performance experiments: where a training step's time goes, which kernel each hardware generation runs and why, which XLA flags and optimizer settings I tried, and how expert parallelism and rematerialization behave. Each section states its revision and settings, and its hardware if that is not an RTX 4080; some sections use an L4, an A100, 4x RTX 3090 or a TPU v6e. A result at one shape and one revision does not settle a default for every case. [Step benchmarks](benchmarks.md) compares architectures, and [Distributed training](concepts/distributed.md) describes how to configure distributed training.

The timeline busy percentages below were taken before `e5ee70d`, which fixed the measurement window for nested kernel intervals. Before you reuse those percentages, replay the original traces. The synchronized wall-clock step times are separate measurements, and that arithmetic bug does not affect them.

The commands below are templates. Replace the angle-bracket fields with your experiment's architecture, kernel and data:

```
python tools/benchmark_attention.py --json-out attention.json
python tools/benchmark_step.py --preset small --architectures <arch> \
    --attention-impl <kernel> --warmup 3 --steps 10
XLA_FLAGS=<flags> python tools/benchmark_step.py --preset small \
    --architectures <arch> --warmup 3 --steps 10
python tools/optimizer_curve.py --dataset <tokens> --optimizer <name> \
    --learning-rate <lr> --out <json>
```

## Training scoreboard, 2026-10-03

I ran Dew and `torch.compile` on the same models with the same batch, bf16
compute over fp32 master weights and the same optimizer constants. Both time
the same warm step (forward, backward, update, and the EMA where both keep
one), with one process per row. The ratio is Dew's throughput divided by the
best reference row's, so a ratio above 1 means Dew is faster. The torch rows
come from `tools/reference_runs/torch_lm.py` (transformers models) and
`tools/benchmark_torch.py` (line-by-line ports of Dew's modules), and the
Dew rows from `tools/reference_runs/dew_lm.py` and
`tools/benchmark_step.py`. `tools/reference_runs/scoreboard.py` collects the
reference-run rows into one table.

RTX 4080 16 GiB, Dew at `f6047cf9` (jax 0.11.2.post3), torch 2.13.0+cu130,
transformers 5.17.0, SDPA attention. Each Dew row gives the range over two
processes. Each torch row is one process, or two where it gives a range. Dew
runs as installed, with cuDNN attention; where a row names tokamax, tokamax
was installed as well (docs/installation.md):

| model | step | Dew | best torch.compile | Dew / torch |
|---|---|---:|---:|---:|
| Qwen3-0.6B, pretrained | 1 x 1024 tokens, AdamW | 96.0-96.1 ms, MFU 46.8% | 112.1-112.4 ms, 40.0% | 1.17 |
| Qwen3-0.6B, pretrained | 2 x 1024 tokens, AdamW | 148.0-148.3 ms, MFU 60.6-60.8% | 168.6-169.1 ms, 53.3% | 1.14 |
| 99M Qwen3-MoE shape, 8 experts, top 2 | 8 x 1024 tokens, AdamW | 78.1-78.2 ms | 112.5 ms | 1.44 |
| decoder, GPT-2 small widths, 3 layers | 16 x 512, Adam, EMA | 49.0-49.3 ms; tokamax 48.5 | 49.4 ms (flash), 50.0 (cuDNN) | 1.00-1.01; 1.02 |
| SimpleDiT, width 384, 6 layers, 64 px | batch 16, Adam, EMA | 7.41-7.42 ms | 8.07 ms | 1.09 |
| SimpleDiT, width 768, 12 layers, 64 px | batch 32, Adam, EMA | 72.6-72.7 ms, MFU 62.4%; tokamax 70.9-71.0 | 76.4 ms (flash) | 1.05; 1.08 |
| 176M hybrid DiT (published config) | batch 16, Adam, EMA | 60.1 ms, MFU 44.4%; tokamax 60.0-60.2 | no torch port | |
| 176M hybrid DiT (published config) | batch 32 | 100.6 ms, MFU 53.1%; tokamax 100.0-100.1 | no torch port | |

The Dew decoder and SimpleDiT rows take a fresh batch from the host every
step, and their times match the fixed-batch rows. "Comparison with PyTorch"
below runs both frameworks both ways and gives the commands. With Qwen3-0.6B
at 1 x 1024 both devices stay busy (torch for 110.3 ms of its 112.1 ms step
in the second process, Dew for 96.0 of 96.2). On the MoE, torch idles
20.9 ms a step on the host, and on device time alone Dew is 1.26x faster
(78.1 against 98.3 ms busy).

From `4d392f0a` to `f6047cf9` the hybrid DiT went from 66.5 to 60.1 ms at
batch 16 and from 110.2 to 100.6 at batch 32. The merges in between include
the S5 recurrence in real arithmetic ("The hybrid DiT's SSM blocks" below)
and the DiT MLP's GELU in fp32. The other rows stayed within their spread.

These changes landed after `42ddfc14`:

- The hybrid DiT's dilated depthwise convolutions run as undilated ones over
  their interleaved grids (75.4 to 70.2 ms at batch 16).
- The vocabulary head computes its logsumexp, maximum and argmax in one
  pass, and its gradient products read one bf16 copy of the logits'
  gradient. That lets the MoE's whole logits fit in memory (119.6 to
  92.7 ms).
- Pretrained weights are placed without the hole that made Qwen3-0.6B at
  2 x 1024 recompute (185.7 to 161.2 ms).
- At the default precision, the head rounds its logits and their gradient to
  bf16 once, as torch autocast and MaxText do ("The vocabulary head" below
  has the quality comparison). The 3-layer decoder went from 59.9 to
  52.2 ms, the MoE from 92.7 to 79.2, and Qwen3-0.6B at 2 x 1024 from 161.2
  to 154.0.

From `42292e99` to `4d392f0a` (92 merges from every lane), Qwen3-0.6B went
from 100.4-101.9 to 96.2 ms at 1 x 1024 and from 154.0 to 148.1 at 2 x 1024,
the decoder from 52.2 to 49.1, and the hybrid DiT at batch 32 from
116.6-122.3 to 110.2-110.3. Two of those changes are to the training step
itself. The step keeps every bf16 rounding the program asks for
(`--xla_allow_excess_precision=false`), and it compiles without XLA's dot
merger. Dropping the dot merger took Qwen3-0.6B at 1 x 1024 from 97.7-98.0
to 94.1-94.2 ms in its own A/B. "XLA flags" below covers both.

I split Qwen3-0.6B's device kernel time per step at 1 x 1024 by kind, with
XProf for Dew and the torch profiler for torch. GEMMs take 42.2 ms in Dew
against 42.3 in torch, and attention 8.7 against 7.2. Everything else (the
update, the casts, the norms and the loss) takes 45.0 ms in Dew against 60.8
in torch. Dew's share is copies 28.0 (with the fused update), converts 10.7,
reductions 5.7 and elementwise 0.6; torch's is optimizer 39.5, copies 13.8,
elementwise 6.0, loss 1.0 and reductions 0.5. Both frameworks cast the fp32
master weights to bf16 every step and cast the gradients back, and torch's
casts are its `_to_copy` kernels.

Dew is faster in the optimizer update. XLA fuses Adam (or AdamW), the EMA
and the finiteness guard into one bandwidth-bound pass over the state. That
pass takes 6.9 ms on the 768-wide SimpleDiT, against 16.2 for torch's fused
Adam and foreach EMA, and 27.6 ms on Qwen3-0.6B, against 39.5 for torch's
fused AdamW and gradient clipping. GEMMs run at par or better (44.5 against
47.2 ms on the 768-wide SimpleDiT). Dew's host cost is at most 2.6 ms a step
on these rows, where torch.compile's reaches 25 ms, except on Qwen3-0.6B at
2 x 1024. There it is 13.0 ms, hidden behind the device.

Dew is slower in attention. cuDNN's fused kernels take 5.6 ms forward and
backward on the 768-wide SimpleDiT (head dimension 64, 256 tokens), against
4.4 for FlashAttention-2 in torch, and 8.7 against 7.2 on Qwen3-0.6B (head
dimension 128, causal, 1024 tokens). With tokamax installed, 'auto' runs
tokamax's Pallas-Triton kernel for heads up to 64 wide, which takes 4.3 ms
on the SimpleDiT ("tokamax's attention" below). At 128 wide cuDNN stays
faster; Qwen3-0.6B's widths at 2 x 1024 step in 148.8-149.0 ms with cuDNN
and 151.1 with tokamax. Kernel by kernel on Qwen3-0.6B at 1 x 1024, cuDNN
takes 2.56 ms forward and 5.25 backward (including its grouped-query head
reduction), against 2.05 and 4.56 for FlashAttention-2's kernels. No
alternative on this stack closes that gap ("At 128-wide heads" below).

A100 40 GB (Colab). The first rows come from one session at integration
`f08009d7` (c17: jax 0.11.2.post3, torch 2.13.0+cu130, transformers
5.17.0). Dew and torch.compile alternated, with two processes each where a
range is given. Each pair's losses agree to within 0.007 over 40 steps. The
remaining rows are the latest earlier records:

| model | step | Dew | reference | Dew / reference | Dew commit |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B, pretrained | 4 x 1024 tokens, AdamW | 125.2-125.3 ms, MFU 44.8-44.9% | torch.compile 139.3-139.7 ms, 40.2-40.3% | 1.11 | `f08009d7` |
| Qwen3-1.7B, pretrained | 1 x 1024 tokens, AdamW | 109.2 ms, MFU 33.1% | torch.compile 140.2 ms, 25.8% | 1.28 | `f08009d7` |
| Qwen3-1.7B, pretrained | 2 x 1024 tokens, AdamW | 179.3 ms, MFU 40.4% | torch.compile 202.3 ms, 35.8% | 1.13 | `f08009d7` |
| 99M Qwen3-MoE shape, 8 experts, top 2 | 8 x 1024 tokens, AdamW | 45.4 ms | torch.compile 118.2-118.7 ms | 2.61 | `f08009d7` |
| SimpleDiT, width 768, 12 layers, 64 px | batch 32, Adam, EMA | 38.25-38.26 ms fresh, 38.22-38.29 on the device; with tokamax 37.87-37.88 | flash: 38.15-38.23 ms on the device, 38.48-38.53 fresh | 1.00 (1.01 with tokamax) | `f08009d7` |
| 176M hybrid DiT (published config) | batch 16 / 32 | 41.0 / 62.8-62.9 ms (the same with tokamax) | no torch port | | `f08009d7` |
| Qwen3-0.6B, pretrained | 4 x 1024 tokens | 161.9 ms | MaxText 0.2.4 GPU recipe 164.9 ms | 1.02 | `157bc21a` |
| mamba2-130m | 4 x 1024 tokens | 127.9 ms | torch with mamba_ssm kernels, eager, 172.5 ms | 1.35 | `9490c9e6` |
| SimpleDiT, width 384, 8 layers, 64 px | batch 64, EMA | 21.9 ms | flaxdiff 21.4 ms | 0.98 | `9490c9e6` |

From `6a220e31` to `f08009d7`, Qwen3-0.6B at 4 x 1024 went from 128.7 to
125.3 ms and the hybrid DiT from 44.5 / 64.1 to 41.0 / 62.9 ms. SimpleDiT,
2% slower in the earlier session, now ties torch.compile with a fresh batch
on both sides and on the device on both sides. With tokamax installed, whose
Pallas kernel `auto` takes at its 64-wide heads, it is 0.8% faster.
JAX's own Pallas-Triton `mha` was no faster here either. In the step at
Qwen3-0.6B's 128-wide heads it took 131.3-131.9 against cuDNN's 125.3-126.1
ms at 4 x 1024, and 78.0-78.2 against 77.0-77.1 at 2 x 1024. One call
forward and backward took 0.37 against cuDNN's 0.39 ms at 1 x 1024, and
1.10 against 1.06 at 4 x 1024.

At `6a220e31` Qwen3-0.6B took 128.4 ms against torch's 136.1 ms busy:
GEMMs 64.2 against 70.9, attention 21.6 against 16.7 (cuDNN's sm80 kernels
against FlashAttention-2), and the update and casts 30.4 against 37.0 for
torch's copies and optimizer. The MoE took 45.1 against 78.1; torch also
idles on the host for 86 ms a step.

The older rows predate every change listed above. On Qwen3-0.6B at
`8c391009`, torch idled 18.3 ms a step on the host, so its device did 120 ms
of work against Dew's 138. Dew lost time in attention, 21.4 against 16.6 ms
(cuDNN's sm80 backward against FlashAttention-2), in its converts (17.8 ms)
and in its reductions, 10.9 against 4.2 (the norms and the fp32 head). Its
GEMMs were faster, 64.9 against 70.9, and its update, counted inside 22.7 ms
of copies, beat torch's 19.2 ms of copies plus 17.8 of optimizer. The
Mamba-2 row, and the MoE at `157bc21a`, came out ahead only because torch
idled on the host (77 to 177 ms a step). On device time Dew was 1.7 times
slower on Mamba-2 (the XLA path of the SSD scan) and 2.0 times slower on the
MoE (its expert GEMMs). The MaxText row ran on another VM, before the
whole-logits head took Dew's step from 161.9 to 141 ms.

TPU v6e (Colab, one chip), Dew against MaxText 0.2.4 on the same VM, at
integration `8895763d` (jax 0.11.2.post3, libtpu 0.0.48). Dew ran
`tools/reference_runs/dew_lm.py` with pretrained weights and the reference
corpus, and MaxText ran `tools/reference_runs/maxtext_run.py` on MaxText's
synthetic tokens. Both used bf16 compute over fp32 weights and AdamW with a
1.0 global-norm clip. Each ran 90 steps, two processes each, with steps
30-89 timed and nothing profiled. The table gives ms a step as each window's
mean (its wall time over its steps), and the ratio of the means:

| model | tokens | Dew | MaxText, minimal remat | MaxText, default remat | Dew / best MaxText |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | 8 x 1024 | 145.3, MFU 26.3% | 154.0-154.1, 24.8% | 167.9, 22.8% | 1.06 |
| Qwen3-0.6B | 16 x 1024 | 288.9, 26.4% | 299.7-299.8, 25.5% | 343.7, 22.2% | 1.04 |
| Qwen3-1.7B | 4 x 1024 | 153.3, 32.1% | 158.7-158.8, 31.0% | 179.7, 27.4% | 1.04 |

Dew's 16 x 1024 window is steps 30-63, because its two epochs of data end
there; MaxText's is 30-89. That Dew run also tiles the vocabulary head, the
first rung of the fit ladder, because the whole logits do not fit. MaxText's
minimal-remat windows contain a few slow steps. Its median step is 150.1 ms
at 0.6B 8 x 1024 and 154.9 at 1.7B, against means of 154.0 and 158.7. Dew's
runner times only the whole window, so it has no median to compare. If every
MaxText step took its median time, the ratios would be 1.03 and 1.01.

The global-norm clip costs Dew 8.1 ms a step here. Through
`tools/benchmark_step.py`, with one batch kept on the device, Qwen3-0.6B's
widths at 8 x 1024 take 137.2 ms with `optax.adam` and 145.3 with dew_lm's
optimizer (the norm, the clip, AdamW and the schedule), the same as the
reference runner's step. On the RTX 4080 the same comparison at 1 x 1024
gives 94.0 against 96.0. Compiled for the v6e, the program with the clip
writes 6.1 GiB more a step (80 to 86 GiB). The extra writes are copies of
the gradients, which the norm holds until every one is in.

## Rounding on the TPU, 2026-10-02

`import dew` turns off XLA's excess precision
(`--xla_allow_excess_precision=false`) so that every bf16 rounding a program
states is kept. I checked on one TPU v6e chip (Colab, jax 0.11.2.post3,
libtpu 0.0.48, Dew at `f24b452a`) that the policy is needed there too. With
XLA's default, a bf16 round trip before a `tanh` or a sum is computed
without the rounding, so a bf16 decoder's outputs depend on the program's
shape. At Qwen3-0.6B's widths over 256 tokens, only 10% of the elements are
equal between a batch of 4 and the same rows run one at a time. With the
policy, all of them are equal. I could not test multi-chip layouts on one
chip, so they are unverified.

The policy cost the 176M hybrid DiT's batch-16 step 8.1% on the v6e (16.95
to 18.29 ms); on an A100 it was neutral (45.31 to 45.42). In the traces the
policy added 1.39 ms, 0.93 of it in the MLP's backward, where the GELU ran
as eight bf16 elementwise steps and each kept its rounding. A further
0.27 ms is the rounding of the residual sums the norms read, which is what
the policy is for. The MLP's GELU (`dew.nn.dit`, and an ungated decoder's
`gelu`) now runs in fp32 and rounds once, as torch's bf16 GELU does and as
the gated MLPs already did (`dew.nn.moe.gated_product`). Over 2^20 bf16
values its RMS error against float64 drops from 2.19e-3 to 1.76e-3, on the
CPU and the RTX 4080 alike. The RTX 4080 steps are unchanged. The hybrid DiT
at batch 16 takes 66.66/66.54 against 66.47/66.55 ms, SimpleDiT-B at 32
takes 72.99/72.63 against 72.77/72.71, and a 3-layer GELU decoder at
16 x 512 takes 44.40/44.30 against 44.39/44.42.

On the v6e (integration `8e92a4a6`, three rounds each) the hybrid DiT's
batch-16 step now runs in 17.25-17.29 ms against 18.25, and with XLA's
excess precision it would run in 16.88-16.92. So the policy now costs 2.2%,
which is the residual roundings. Those roundings change results on the TPU
as well. An RMSNorm reading a fused bf16 residual sum matches the program
that stores the sum on 98.8% of its outputs with excess precision, and on
all of them without.

## Sampling the hybrid DiT on the CPU, 2026-10-02

The landing page's live cell samples the published 176M hybrid DiT on a
4-vCPU container: one prompt, 15 DPM-Solver++ steps under CFG 5.0, and a
bf16 SD VAE decode to 256 x 256. I reproduced it at `db1761fd` on one P-core
of an i9-12900K (two threads, `taskset -c 2,3`, jax 0.11.2.post3), traced
one warm call and summed the op time by JAX scope:

| scope | s |
|---|---:|
| MLPs (dots at about 140 GFLOP/s, YNNPACK) | 8.5 |
| 2D fusion's depthwise convolutions | 5.7 |
| VAE decode (bf16 convolutions at about 125 GFLOP/s) | 5.3 |
| S5 layers, with their output projections | 2.3 |
| attention blocks | 1.9 |
| the rest | 0.6 |

The dots and the decoder's convolutions run near the core's fp32 rate, but
the depthwise convolutions did not. XLA:CPU runs a grouped convolution
through YNNPACK, which took 6.4 ms for one 2 x 16 x 16 x 768 map of 7 MFLOP.
On the CPU, a depthwise 3x3 convolution of more than 16 features now runs as
its nine shifted products. Each product is rounded to fp32 before it is
summed in the kernel's row-major order, which is YNNPACK's arithmetic bit
for bit, and a map takes 1.4 ms. YNNPACK sums 16 features or fewer in
another order, so those keep the convolution.

I ran three alternating processes each way, three calls each, on a shared
and loaded host. The cell's median went from 24.45 s to 21.37 (fastest 22.68
to 20.51), and without the decode from 18.96 to 15.99 (17.52 to 14.91).
Every image's sha256 is the same (`923e1b09`).

## The hybrid DiT's SSM blocks, 2026-10-01

The published 176M hybrid DiT has 16 blocks, 12 of them S5 blocks with the
2D fusion convolution, and works on 32x32x4 latents with patch 2. I ran it
at batch 16 in bf16 on the RTX 4080 through `tools/benchmark_step.py`,
passing its config as `--cases`. With command buffers off, XProf names each
kernel's HLO instruction, and the optimized HLO gives that instruction's JAX
scope. Per step, at `42ddfc14`:

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
products in fp32, because cuDNN's dilated grouped kernels are slow, and its
weight gradient read the input and the output's cotangent once per tap. A
pixel's dilation-d taps are its neighbours in the grid of pixels that share
its row and column residues mod d. So the convolution now runs as cuDNN's
dilation-1 kernel over the d^2 interleaved grids. Forward and VJP take
0.097 ms at dilation 2 and 0.089 ms at dilation 3, against 0.31 and 0.33,
and the step went from 75.30 to 70.17 ms. At HIGHEST precision the fp32
output and input gradient equal those of lax's dilated convolution exactly.

In fp32 that polyphase form made the step slower (109.70 to 113.68 ms). In
fp32, cuDNN runs the polyphase convolutions (forward, input gradient and
filter gradient) with its grouped direct kernels, which took 3.8 ms of the
5.2 ms a batch-16 step spent in the dilated layers, and the interleaving
transposes took 1.3 ms more. So on CUDA the form now depends on the dtype:
bf16 uses the polyphase form, and fp32 uses the nine shifted products from
before, with the fp32 input, kernel and output held in memory. The step at
integration `f6047cf9` against the change, on the RTX 4080 with a fixed
batch and two alternating rounds:

| dtype | batch | polyphase (before) | by dtype (after) |
|---|---:|---:|---:|
| fp32 | 16 | 105.87 / 105.77 | 101.57 / 101.54 |
| fp32 | 32 | 265.81 / 265.18 | 254.71 / 254.50 |
| bf16 | 16 | 60.13 / 60.08 | 60.12 / 60.09 |
| bf16 | 32 | 100.68 / 100.60 | 100.69 / 100.62 |

fp32 is 4.0% faster at batch 16 and 4.1% at 32. bf16 compiles to the same
program, and its losses are the same to the bit. The fp32 losses move in the
8th digit (0.56920904 to 0.56920898 at batch 16), within the forms' fp32
bound (tests/test_depthwise_conv.py). In bf16 the shifted form would cost
65.4 and 107.0 ms.

The published model samples in fp32, so the live sampler's call is faster
too (15 DPM-Solver++ steps under CFG, `0964f573`, median of five, two rounds
each). Batch 1 went from 101.0-102.0 to 97.5-98.1 ms and batch 4 from
268.2-269.3 to 257.4-259.0. The denoising scan alone went from 93.0-93.2 to
90.1-90.5 and from 236.8-237.6 to 226.8-227.7. Peak memory stayed within the
rounds' spread. The latents and fp32 images move by less than twice what one
rounding of the convolutions moves them (the published-sample test).

On an A100 (c15, integration `6a220e31`, two alternating rounds) the fp32
step is faster in the same way, 66.73/66.78 to 63.44/63.37 ms at batch 16
and 117.43/117.47 to 111.03/110.97 at 32. bf16 uses the polyphase form in
both builds, so it is the same program (42.17/42.39 and 42.12/43.01 ms,
63.96/64.12 and 64.01/64.18).

The S5 layer ran its recurrence as `associative_scan` over complex states,
and the backward of that scan spent 4.5 ms of the RTX 4080's step in complex
arithmetic alone. On a GPU and on the CPU the recurrence now runs in real
arithmetic, in chunks. Inside a chunk, the states are one fp32 product of
the pole's powers with the chunk's inputs, and `associative_scan` runs only
over the chunks' last states.

The powers are built by doubling, multiplying the powers so far by the next
squared power, so each one is rounded at most 2 log2(t) times. The `[L, L]`
Toeplitz block is a one-hot product of the powers, exact at full precision,
and its transpose is a product too. Both directions' complex input products
run as one real product. A TPU still runs the layer in its first form
(`dew.nn.ssm._directions`); compiled for a v6e, its program has the same
5125 instructions and estimated cycles as before.

The rows below compare the scan with the chunks, measured on 2026-10-03. The
step columns come from `tools/benchmark_step.py`, in ms per step (the
asynchronous throughput a run sees). The sampler columns time the live
sampler's call (`dit_sample_time.py`: 15 DPM-Solver++ steps under CFG 5.0,
the denoising scan alone):

| device | step, batch 16 | step, batch 32 | sampler, 1 image | sampler, 4 images |
|---|---:|---:|---:|---:|
| RTX 4080 (integration `17e2b226`, two rounds) | 66.63/66.41 to 60.15/60.14 | 110.26/110.21 to 100.80/100.65 | | |
| A100 40 GB (Colab, `db1761fd`, a first form) | 42.38 to 40.89 | 65.67 to 62.39 | | |
| CPU (i9-12900K, 4 threads, the live sampler's `0964f573`, ABAB) | | | 17.17 to 14.19 s | |

On the CPU the row is the live sampler's call without its VAE decode, the
median of nine calls in three alternating processes on a loaded host
(14.55-20.52 s against 13.31-19.73). A second session agreed (14.35 and
14.50 against 13.07 and 13.57). With the decode, the call takes 25.47
against 22.06 s. Peak RSS is within the processes' spread (medians 4370 and
4414 MiB, ranges 3937-4478 and 4221-4829). The image's bits change with the
arithmetic (sha256 `923e1b09` to `10b80bfe` at key 0), as they do on any
backend whose recurrence changes, and both forms are held to the same error
bound.

On a v6e (Colab, three rounds each) neither form won at every shape. The
doubled chunks, at `527a32e9`, took batch 16 from 17.25 to 18.03-18.05 ms
and batch 32 from 34.39 to 32.17-32.20. Sampling went from 22.0 to 18.8 ms
for 1 image and from 63.6 to 65.2 for 4 images, with 0.11 GB more at peak.
The scan with the real input product in front of it, at `1f8d5e72`, took
batch 16 from 17.26-17.29 to 17.43-17.46 and batch 32 from 34.40-34.43 to
32.35-32.38. Sampling went from 22.0 to 16.7 for 1 image and from 63.4-63.5
to 63.0-63.6 for 4. So a TPU keeps the first form whole.

A first chunked form built the powers as a running product (`cumprod`) and
the Toeplitz block from shifted copies. Computing the powers as `exp(t
log(pole))` was cheaper, but it rounds the phase of a large-angle pole t
times over. On poles all around the unit circle its RMS error against
complex128 is 1.05e-5, against 1.57e-6 by doubling and 8.5e-7 by the running
product. Over 4096 positions, the doubled chunks' forward RMS error against
the complex128 oracle is 1.144e-6, and the old scan's is 1.156e-6.

Unless a section says otherwise, the sections below were measured on jax
0.11.1 / jaxlib 0.11.1 / jax_cuda12_plugin 0.11.1, driver 595.84, RTX 4080
16 GiB, single device, bf16 compute, adam, 3 warmup and 10 measured steps,
one architecture per process. The card was idle before each measurement;
`nvidia-smi --query-compute-apps=process_name` showed only
gnome-remote-desktop-daemon, which is the desktop itself. The card ran at
210 MHz and 30 W at rest and at 2760 MHz and 120-220 W under load. XLA reads
a flag once, when a backend opens, so every flag configuration ran in a
fresh process.

## Step time breakdown, 2026-09-05

```
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    python tools/benchmark_step.py --preset small --architectures <arch> \
    --warmup 3 --steps 30 --profile-dir /tmp/dew-trace --profile-steps 5
```

`tools/benchmark_step.py` reads back the traced window. Busy time is the
union of kernel intervals across the device's streams. The tool counts
kernels per step and sums their time by category, using kernel names to
assign categories. Dew was at `9886c20`, before the cudnn padding described
below.

| architecture | ms/step | device busy | kernels/step | gemm | elementwise | reduce | convert | attention | copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_dit | 7.0 | 100% in steady state | 532 | 3.29 | 0.69 | 1.19 | 0.79 | 0.69 | 0.16 |
| causal_transformer | 88.8 | 100% | 282 | 63.3 | 13.3 | 7.6 | 0.8 | 1.8 | 0.7 |

The trace reports 81.6% busy for the DiT over five steps. Starting the
profiler adds a 3 ms gap to each of the first two steps. After that,
the interval between steps settles at 6.9 ms, equal to the kernel time.
The device stays busy between steps once the loop is running.

The DiT's reductions compute the Dense layers' bias gradients and norm
statistics. The 6-by-64 biases of the q, k and v projections require
three full passes over each layer's activation gradient, costing 0.25 ms
a step. The convert kernels compute XLA's split-K partial sums in fp32
and cast fp32 parameters to bf16 at each use. Parameter casts account for
0.15 ms of the 0.79.

The decoder's gemm time comes from the fp32 (TF32) vocabulary head. It
uses two cutlass `s1688gemm` kernels at 12.9 and 12.5 ms, plus four Triton
tiles of 3.2 ms for the third product. At the measured TF32 ceiling of
49.5 TFLOP/s, each product needs at least 12.8 ms
(`docs/research/benchmark-parity.md`).

### Host time per step

Host times for simple_dit come from the trace's host plane and from a
dispatch loop timed while the device was deliberately allowed to lag.
The device step takes 6.9 ms.

| host work per step | ms | how measured |
|---|---:|---|
| XLA thunk execution inside PjRt Execute | 3.3 | `GpuExecutable::ExecuteThunks` on the host plane; 2.4 of it is three CUDA-graph launches |
| Python in `jax.stages.Compiled.__call__` before Execute | 1.8 | `$stages.py __call__` 6.7 ms against `PjRtCApiLoadedExecutable::Execute` 5.0 ms. The `$` events come from JAX's Python tracer, which adds time to every Python and C call, so 1.8 is an upper bound on the untraced cost |
| placing a fresh batch (`shard_batch`) | 0.25 | 200 calls timed in isolation, image plus tokens |
| the loop with a fixed device batch | 5.0 | dispatch loop time, 100 steps, device 27 steps behind |
| the loop with a fresh batch per step | 6.5 | same, device 7 steps behind |
| the loop with XLA command buffers off | 7.3 | `--xla_gpu_enable_command_buffer=`; wall 7.46 ms/step, the host is now the step |

On the smallest step, the host takes 94% of the device's time with a fresh
batch every step, and 106% without command buffers.

The Python in `Compiled.__call__` costs at most 4.5 us per leaf. This state
has 396 leaves, and more on a mesh. To avoid that cost, `Trainer.compile`
returns the jitted step starting with `de6b22c`. Since the sequence axis
was added, a mesh context wraps the jitted step. Dispatch costs 32 us on
the i9-12900K with or without that wrapper. On this card, wall time stays
the same and host time drops by 1.8 ms a step.

A fresh batch costs 1.5 ms more than a fixed one because the command buffer
must update its buffer addresses. This limits the loop to 7 steps ahead
of the device, against 27 with a fixed batch. On a faster card or smaller
model, it would limit wall-clock throughput. Placement itself takes only
0.25 ms. Keeping every consumed batch alive also changes nothing under
default preallocation. Prefetch depths 2, 8 and 32 measure the same. No
fix was adopted because the runtime controls the addresses.

Waiting on the device after every step costs 45%. The same simple_dit loop
with `block_until_ready` after each step takes 10.3 ms against 7.1.
The trainer waits only at logging ticks. Running ahead does not increase
peak allocation: 0.823 GiB at 27 steps ahead against 0.819 in lockstep,
and 3.499 against 3.495 GiB for hierarchical_mmdit.

Correction, 2026-10-01, at `42ddfc14` (jax 0.11.2.post3, same card).
The three "loop" rows above hit the runtime's limit on executions in
flight. Once the device falls a few dozen steps behind, each dispatch
waits for a step to finish. These rows therefore measure device time.
Timing eight dispatches immediately after synchronization stays under
that limit. Each costs 0.95 ms on simple_dit with a fixed device batch,
or 1.38 ms when the main thread also places a fresh batch. The small
decoder costs 0.40 and 0.63 ms. `DevicePrefetchIterator` places fresh
batches on a worker thread, as used by `Trainer.fit`. With it, every
loop runs at the device's pace:

| loop, 3 repeats of 100 steps (40 on the decoder) | simple_dit ms/step | decoder ms/step |
|---|---:|---:|
| one device batch reused | 7.41-7.46 | 63.54-63.64 |
| a fresh placement each step, `DevicePrefetchIterator` | 7.43-7.50 | 63.55-63.64 |
| `Trainer.fit` over the same host batch, logging every 100 (40) steps | 7.456-7.460 | 63.69-63.73 |

The fit row includes logging, which waits on the device once per interval.
Host time limits the cpu-smoke decoder, whose device work takes 0.3 ms.
There, `Trainer.fit` costs 0.51-0.62 ms a step, against 0.31-0.37 for the
bare loop and 0.32-0.44 with the prefetch iterator. Timing each part
separately gives 234 us for compiled-step dispatch, 85 us for the
prefetch iterator's `next`, and 22 us for jitted `bookkeep`. Checking batch
shapes and row counts, and the profiler regions, cost under 1 us each.
`fit` adds about 0.2 ms of host work a step. Device work longer than
about 0.6 ms hides this cost.

### Antipattern audit

This audit checked `src/dew` for nine classes of performance antipattern
using the small preset. Each row gives the measured cost and any fix.

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

The audit did not measure `jax_default_matmul_precision` settings, remat on
a step that fits in memory, XLA flags other than command buffers, or the
cost of the class-1 eager scalars. It gives no conclusions about them.
`bfloat16` matmul precision would change the fp32 head's numerics; the
precision rule refuses it.

Correction, 2026-09-22, checked against current main. The jitted `bookkeep`
from the first class-1 row is on main (`src/dew/training/trainer.py`, called
from `Trainer.fit`). The class-5 row about `objective.py:141` describes old
code. The diffusion objective now encodes the unconditional prompt once
at construction. Each step casts the stored encoding to the batch's
dtypes (`DiffusionObjective.blank_conditions` in
`src/dew/objectives/diffusion/objective.py`).

### Comparison with PyTorch

`tools/benchmark_torch.py` ran in a fresh venv with torch 2.14.0+cu130 and
cuDNN 9.24. The run the week before used 2.11.0+cu128 with cuDNN 9.19. The
flags were `--mode compile --warmup 20 --steps 100`, with the small presets
and one process per row. The dew columns reproduce rows from
`docs/benchmarks.md` and the table above:

| case | dew ms/step | torch compile, reference attention | torch compile, SDPA cudnn | dew against the best torch row |
|---|---:|---:|---:|---:|
| simple_dit | 7.02 | 9.28 | 8.39 | 1.19x faster |
| causal_transformer | 88.78 | 81.50 | 72.46 | 0.82x, torch faster by 18% |

The previous week's decoder ratio was 0.95x, comparing the parity
benchmark's fixed-batch decoder (75.70) with torch's 72.18. The dew row
here uses the benchmark's prefetching loop and takes 88.78. The two dew
times differ by 13 ms. The chunked head accounts for 1.9 ms, and decoder
changes since `6b0f119` account for 3.8 ms. The remaining 7.6 ms separates
the fixed-batch row from this tool's loop at the same commit: `6b0f119`
reran at 83.26 here on the same day. The DiT ratio rose from 1.16x the
week before to 1.19x. With the newer torch, its SDPA row is 0.4 ms slower
than the week before.

Correction, 2026-10-01. The 0.82x row compared different batch methods.
Dew placed a fresh batch every step, while torch without `--h2d` kept one
batch on the device. The 7.6 ms was never shown to be loop overhead.
At `42ddfc14`, both tools use matching batch methods.
`tools/benchmark_step.py --fixed-batch` is compared with
`tools/benchmark_torch.py` without `--h2d`. The default fresh-batch loop
is compared with `--h2d`, which copies pinned host memory every step.
Versions and flags: torch 2.13.0+cu130, transformers 5.17.0, `--mode compile --attention
sdpa`, `--warmup 20 --steps 100`, one process per row:

| case | Dew, fixed | Dew, fresh | torch.compile, fixed | torch.compile, `--h2d` | Dew against torch, fixed / fresh |
|---|---:|---:|---:|---:|---:|
| causal_transformer, small preset | 63.68 | 63.63 | 49.44 (flash), 50.00 (cudnn) | 49.99 (cudnn) | 0.78x / 0.79x |
| simple_dit, small preset | 7.46 | 7.43 | 8.53 (cudnn) | 8.07 (cudnn) | 1.14x / 1.09x |
| simple_dit, width 768, 12 layers, batch 32 (`--size large`) | 76.09 | 76.11 | 76.40 (flash), 78.18 (cudnn) | 76.51 (flash), 78.38 (cudnn) | 1.00x / 1.01x |

The torch decoder computes its head's product in bf16
(`--head-dtype bfloat16`, now the twin's default), as Dew's LM objective
does. With the twin's earlier fp32 head, torch took 72.38 ms. Neither a
fresh batch nor `Trainer.fit` adds measurable cost to Dew's steps in
these rows (the host-time correction above).

At `42ddfc14`, the vocabulary head accounted for the decoder's 14 ms gap.
With 8192 tokens, vocabulary 50304 and three layers, the head is most of
the step. Dew kept fp32 logits and computed the state gradient from their
fp32 cotangent as two products: a bf16 high half and the remainder. Torch
rounds both logits and cotangent to bf16.

Three later changes reduced the decoder to 52.2 ms against torch's 49.4
(the scoreboard above). The head computes log-sum-exp and argmax in one
pass, makes one bf16 cotangent copy for both gradient products, and uses
torch's rounding of logits and cotangent at default precision.

XProf with command buffers off and torch.profiler measured these costs
per step. The forward's two passes over fp32 logits for log-sum-exp and
argmax took 5.1 ms, against 1.3 for torch's fused log-softmax. The
backward wrote the logits' cotangent three times in bf16: the high half,
the remainder and the plain rounding for the head's own gradient. This
took 6.7 against 2.7. Computing the state product twice accounted for
about 3 of Dew's 39.4 ms of GEMMs, against torch's 33.9. Attention takes
1.7 against 1.3.

For the large DiT, XProf with command buffers off and torch.profiler
measured a different balance of work. GEMMs take 44.5 ms a step against
47.2. The optimizer update takes 6.9 against 16.2 because XLA fuses Adam
and EMA into one state pass. Attention takes 5.3 against 4.4 for
FlashAttention-2. CuDNN's attention time includes forward at 0.9, backward
at 2.9 and two pre-Hopper backward helpers at 1.5. Dew's reductions and
converts take 16.5 (bias gradients with GELU backward 5.4, norm statistics
1.9). Torch's elementwise and norm kernels take 5.8, and copies take 8.9.

## Attention kernels

### Splash's tiles on the TPU, 2026-10-02

Splash attention took 34 ms of Dew's 156 ms Qwen3-0.6B step at 8 x 1024
tokens on a TPU v6e. MaxText's attention took 50 ms of its step. Both ran
at about 13% of the chip's peak. This tile comparison measures forward
plus backward for Qwen3-0.6B's attention: 8 x 1024 tokens, 16 query heads
over 8, 128 wide, causal, bf16. Each time is the median of 7 rounds of
10 calls through `dew.nn.attention.splash_attention` at integration
`8e92a4a6`:

| forward tiles | backward tiles | backward kernels | ms |
|---:|---:|---|---:|
| 512 | 512 | dq and dkv apart (was the default) | 1.814 |
| 512 | 512 | one fused kernel | 1.423 |
| 1024 | 512 | one fused kernel | 1.397 |
| 1024 | 1024 | dq and dkv apart | 1.549 |
| 1024 | 1024 | one fused kernel | 1.282 |
| 512 | 256 | one fused kernel | 2.194 |
| 256 | 256 | dq and dkv apart | 3.647 |

Every configuration has the same errors against fp32 XLA at HIGHEST (out
3.7e-3, dq 5.0e-3, dk 5.1e-3, dv 2.8e-3 of their maximum). The kernel now
tiles by 1024, reduced to a divisor of each sequence, and runs backward
in one kernel. Training times in ms on the v6e (integration `527a32e9`,
three rounds each, one batch reused on the device):

| step | 512 tiles, dq and dkv apart | 1024 tiles, fused backward |
|---|---:|---:|
| Qwen3-0.6B widths, 8 x 1024 | 150.34-150.43 | 136.77-136.92 |
| Qwen3-0.6B widths, 16 x 1024 (head tiled by the fit ladder) | 320.48-320.54 | 278.63-278.66 |
| 4-layer decoder, 256-wide heads, 8 x 2048 | 80.90-80.95 | 75.36-75.45 |
| 176M hybrid DiT, batch 16 (4 attention blocks of 256 tokens) | 17.24-17.30 | 17.25 |

These rows use `tools/benchmark_step.py` with `optax.adam`. The matched
comparison against MaxText uses the reference runner's AdamW and clip
("Training scoreboard" above: 145.3 against MaxText's 154.0 at 8 x 1024).
After 45 steps, losses differ in the third or fourth significant digit
(0.003386 against 0.003389 at 8 x 1024). Reordering fp32 sums in the
backward pass changes the training trajectory, although each kernel
call has the same errors against fp32.

### tokamax's attention, 2026-10-02

This compares tokamax's Pallas-Triton flash attention (openxla/tokamax
main at `47d3d663`) with cuDNN on an RTX 4080. Inputs are bf16. Times
cover forward plus backward, as medians of 7 rounds of 10 calls.
Each call is checked against fp32 XLA at HIGHEST
(`max|err| / max|ref|` for dq and dk):

| shape | cuDNN | tokamax, its heuristic config | tokamax, best of a config grid | JAX's Pallas `mha`, best blocks |
|---|---:|---:|---:|---:|
| 32 x 256, 12 heads of 64 | 0.726 ms | 0.506 | 0.452 | 0.374 |
| 16 x 512 causal, 12 heads of 64 | 0.793 | 0.515 | 0.522 | 0.465 |
| 4 x 1024 causal, 16 heads of 64 | 0.833 | 0.592 | 0.576 | 0.523 |
| 4 x 1024 causal, 16 query heads over 4, of 64 | 0.784 | 0.608 | | 0.574 (keys repeated) |
| 4 x 1024 causal, 16 over 8, of 128 (Qwen3-0.6B) | 1.541 | 1.288 | 1.242 | 1.260 |
| 4 x 1024 causal, window of 256 | 0.579 | 0.587 | 0.535 | no window |

At 64-wide shapes, tokamax matches cuDNN's errors (dq and dk 4.8e-3 to
6.6e-3). At 128, its errors differ; see below. JAX's `mha` reaches
8.5e-3 because its backward computes `rowsum(o * do)` as a bf16 product.
An fp32 product matches cuDNN's errors at the same speed.

Tokamax reduces training time at 64-wide heads but not at 128. On SimpleDiT-B at batch
32, attention falls from 5.62 to 4.33 ms and the step from 73.1 to 71.5.
For the 3-layer decoder, attention falls from 1.82 to 1.24 ms and the
step from 50.8 to 50.2. At Qwen3-0.6B's widths with 1 x 1024, attention
increases from 9.13 to 9.24 ms and the step from 97.6 to 98.4. With
tokamax installed, 'auto' therefore selects its heuristic config for
heads up to 64 wide and calls without a window, mask or bias
(`dew.nn.attention.triton_runs`).

At 128-wide heads, 2026-10-02 (Qwen3-0.6B's: 16 query heads over 8, causal,
1024 tokens; RTX 4080, bf16; Dew at `14087252`). No kernel on this stack
beats cuDNN while matching its gradient accuracy:

- At 1 x 1024, tokamax's forward is faster and its backward slower
  (XProf, command buffers on, kernels per step). Forward takes 2.16 ms
  against cuDNN's 2.56. Backward takes 7.14 plus 0.27 for
  `rowsum(o * do)`, against cuDNN's 5.07 plus 0.18 for its head reduction.
  The step takes 95.96 against 94.07 ms. Torch's FlashAttention-2 kernels
  take 2.05 and 4.56.
- Tokamax's gradients are less accurate at these shapes. At batch 1, dk
  and dv errors reach 5.2e-3 and 4.6e-3 of their maximum, against
  cuDNN's 4.9e-3 and 2.8e-3. At batch 2, they reach 6.7e-3 and 4.8e-3
  against 4.7e-3 and 3.5e-3. Its best grid config has the same errors
  (blocks of 32, keeping openxla/tokamax#1494's constraint).
- BNTH and BTNH inputs give similar cuDNN times: 0.385 against 0.381 ms
  a call at batch 1, and 0.721 against 0.716 at batch 2. Upgrading from
  cuDNN 9.25.1 to 9.27.0 also makes no difference. Qwen3-0.6B's widths at
  1 x 1024 take 94.20 and 94.08 against 94.23 and 94.19 ms, with attention
  at 9.09 against 9.11; SimpleDiT-B and the decoder behave similarly.
  `uv pip install` resolves the newest cuDNN under 10, so a fresh install
  gets 9.27.
- JAX's Pallas-Triton `mha` repeats keys over each group's query heads
  (`jax.experimental.pallas.ops.gpu.attention`). At `0527ab19` on
  2026-10-04, forward plus backward per call, measured over 28 layered
  calls, takes 0.33 against cuDNN's 0.38 ms at 1 x 1024 and 1.30 against
  1.36 at 4 x 1024. Its gradients' RMS distance from float64 is 1.00 to
  1.04 times cuDNN's. In training, the two tie over two alternating rounds.
  At 1 x 1024, attention takes 8.49 against 8.52 ms and the step 90.6
  against 90.8. At 2 x 1024, attention takes 15.5 against 15.3 and the
  step 142.4 against 142.9. Tokamax's main has no GPU attention change
  since `47d3d663`. JAX exposes no cuDNN algorithm, workspace or determinism
  choice for fused attention; non-deterministic is already the default.
  Closing FlashAttention-2's 1.3-1.9 ms gap would require a Dew
  Pallas-Triton backward kernel on a backend JAX 0.11 deprecates. That
  would save under 2% on steps where Dew already beats torch.

At 64-wide heads, tokamax's backward is 10-18% slower than JAX's `mha`:
0.35 against 0.29 ms for the SimpleDiT-B call. The forwards take 0.093
and 0.087. None of the block sizes, warp counts or stage counts in
tokamax's grid closes the gap.

Its VJP computes wrong causal gradients when `block_m1 > block_n1` (dk
and dv off by 10^2) or `block_n2 > block_m2` (dq off by 0.7). These
configs are in its autotuning grid, but its heuristic config avoids them.
Dew therefore uses the heuristic and checks each routed shape
(`tests/test_kernels.py`). At 256-wide heads, tokamax's heuristic fails
because it requests more shared memory than sm89 has (102784 of 101376
bytes). CuDNN also refuses 256-wide heads, so those calls use XLA.

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

The reference and xla paths store the S x S logits and run out of 16 GiB
at S=4096. Where they fit, they are 3 to 12 times slower than the fused
kernel. These measurements favor cudnn for GPU forward and backward;
`'auto'` selects it wherever it can.

### Head dimension 256 through tokamax's Triton flash attention

On pre-Hopper GPUs, cudnn refuses head dimensions above 128. A Gemma 3
4B or 12B shape, with heads of 256, therefore trains through xla on
Ampere and Ada. That path stores the S x S logits. Tokamax 0.0.13
provides Pallas-Triton flash attention for compute capability 8.0 and up,
with forward and backward passes. It supports grouped query heads, causal
masks, windows and power-of-two head dimensions. JAX 0.11.1 deprecates its
own `jax.experimental.pallas.ops.gpu.attention` in favor of tokamax.

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

The Triton forward is 2.2 to 7.6 times faster than xla and uses no
temporary memory. Its error is also smaller: 0.0081 against xla's 0.0112,
on outputs of size 3.4. The backward does not run. Tokamax's Triton VJP
uses one fixed tiling for every card (`pallas_triton_vjp.py` contains
`TODO: Implement heuristics`). At head dimension 256, that tiling requests
102784 bytes of shared memory from a card with 101376. It fails with `RESOURCE_EXHAUSTED:
Shared memory size limit exceeded`.

Probing tokamax's private classes found a 32x32 tiling with one stage
that fits and is correct (gradient error 0.031, the same as xla). It
takes 1.80 ms forward and backward against xla's 2.66 at S=2048. Two
16-row tilings compile and run at the same speed, but return wrong
gradients (error 6.6 on gradients of size 6.3). Tokamax's autotuner chooses
by timing random inputs without comparing numerics. It cannot be trusted
to choose a correct tiling. At head dimension 128, Triton ties cudnn
(0.235 against 0.236 ms forward, 0.75 against 0.78 forward and backward
at S=2048), giving no gain where cudnn already runs.

The Triton forward supports Gemma 2's logit softcap (0.35 against xla's
1.01 ms at S=2048, head dimension 256), but the VJP raises
`NotImplementedError: logits_soft_cap unsupported`. Tokamax also applies
the cap after adding bias, while Gemma applies it before. On CPU, the
outputs differ by 1.4e-2 with bias and are identical without it. No
tokamax implementation supports attention sinks.

Dew does not route calls to this kernel. Training needs a backward pass,
and the only working tiling is available through private tokamax classes.
Routing needs an upstream release that selects a fitting VJP tiling, or
a public tiling setting, together with a correctness check. Installing
tokamax 0.0.13 beside Dew also pins `typeguard==2.13.3`, while tyro 1.0.16
requires `typeguard>=4.0.0`. This breaks every recipe's command line, so
these measurements called the tool's `main` function in a separate
environment.

## Odd sequence lengths on cudnn

CuDNN's fused kernel has no backward pass for odd query or key lengths.
The forward accepts any length, so the failure appeared only at the
first training step: `NotImplementedError: Unsupported sequence length Q 333, KV
333` from jax. CLIP's 77 text tokens have an odd length, as does a
concatenation of 256+77.

Until 2026-09-05, `'auto'` sent those shapes to the xla kernel. That kernel
stores the [B, H, Q, K] logits and probabilities in fp32 for backward.
`cudnn_attention` now pads odd lengths to even ones. It adds one zero query
row and slices it off the output. It also adds one zero key and hides it
with the kernel's padding mask (`key_value_seq_lengths`). Each real query
therefore attends to its original keys. On GPU, `'auto'` selects cudnn at
any sequence length, and an explicit `'cudnn'` also accepts any length.

`tests/test_kernels.py::test_cudnn_trains_odd_lengths_and_agrees_with_xla`
checks this at q1024/kv77, q9/kv7 and q333/kv333 causal. The outputs and the
three input gradients agree with the xla kernel to within two bf16 ulps of
their scale. The kernels differ by the same amount at an even length
(q256: 1.6e-2 at scale 2.9 on the output, 7.8e-2 at scale 15.6 on the
gradients, both one ulp). If the pad key is left unmasked, the q9/kv7 output
moves by 0.26 at scale 2.4 and the test fails. If the pad query row is left
in, the shape changes and the test fails.

Padding results from `--warmup 3 --steps 50` on the small preset, comparing
`'xla'` (used before padding) with `'auto'`:

| architecture | shapes | xla ms/step | cudnn ms/step | xla peak GiB | cudnn peak GiB | loss at the end, xla / cudnn |
|---|---|---:|---:|---:|---:|---|
| hierarchical_mmdit | q141, q333, q1101 | 33.86 | 20.86 | 3.50 | 1.85 | 0.551035 / 0.551038 |
| simple_mmdit | q333/kv333 | 12.86 | 11.01 | 1.43 | 1.08 | 0.584398 / 0.584407 |
| unet | q256/kv77, q1024/kv77 | 16.30 | 16.13 | 0.78 | 0.71 | 0.597518 / 0.597516 |

The xla attention on the 1101-token stage kept fp32 logits and probabilities
for backward, accounting for 1.65 GiB and 13 ms. Attention is a small
part of the unet's step, so padding saves little there. Losses after
103 steps on one fixed batch differ in the sixth digit because of the
kernels' bf16 rounding, compounded by Adam. Decoding uses one query
position at a time, an odd length. It runs on cudnn with the cache mask
as additive bias; decoding speed was not measured.

## Attention metadata and the masked conv, 2026-09-07

Before `14622ba`, supplying any `AttentionMetadata` disabled the fused kernel.
The mixer built a `[B, 1, S, S]` mask and selected xla even when the metadata
only specified rotary positions or marked every slot as valid. Those
batches computed a mask that excluded nothing.

At `14622ba`, the mixer checks whether metadata restricts key validity
or image groups on a bidirectional-image layer. It cannot read a validity
array at trace time, so even an all-true array still requires a mask.
Host code now omits the array when it knows the rows are whole. This
includes `pad_token_rows`, the processor's `from_hf`, generation input
validation, the rollout collector and episode cohorts, the PPO critic
without lengths, and every MTP depth.

The Gated DeltaNet short conv had the same kind of problem inside it.
`_masked_conv1d` convolved one token per scan step to keep a paused row's
history still. At `14622ba` it compacts each row's real tokens by
`cumsum(valid) - 1` and calls the same fp32 `causal_conv1d` once.

The chunked delta rule's inverse, 2026-10-05. Each chunk's (I - A)^-1 was
summed as the series I + A + A^2 + ..., doubling powers of A. On
Qwen3.5-0.8B's real keys (near-aligned keys, beta near one) the powers grew
as binomials while the inverse's entries stay under 1, and the cancellation
left its fifth delta-rule layer 1e24 off in fp32 on a 256-token prompt:
every later layer, the logits and every served request were NaN.
`strictly_lower_inverse` now joins diagonal-block inverses, doubling their
size, so each product is the size of its entries. Every layer's chunked rule
is within 8e-7 of a float64 token-by-token run, and the logits within 3.3e-05
of transformers' fp32 forward with every argmax equal. It is cheaper too. At
Qwen3.5-0.8B's widths (16 heads, 128 wide, 4096 tokens, bf16) one forward
plus backward took 15.1 against 12.5-12.9 ms at batch 1 and 47.1 against
32.3-32.8 ms at batch 4 on an RTX 4080, and a v6e's compiled program has 42%
of the flops with 3% more temporaries.

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

The canonical row uses the opaque row's batch with redundant validity
omitted, matching a real unpadded request. Its before column measures the
same call at `83f08e5`. The compiled HLO contains a `__cudnn$fmhaSoftmax`
custom call after the change, and none before. This confirms that routing
changed; the speedup is not clock noise. The opaque and packed rows
are unchanged by design, with differences within their spread across windows.

The process allocator's reported peaks vary by up to 20 MiB between
identical runs. Two repeats of the same packed forward gave 496.02 and
476.02 MiB, although the executable's `memory_analysis` was byte-identical.
Read the peak column at that resolution.

Canonical metadata uses the plain call. Its outputs are bitwise equal
to the no-metadata forward, and parameter gradients agree within 2.4e-06.
The opaque all-true mask stays on xla at its old cost because the
array's shape cannot establish that every entry is true.

The GDN rows time the whole mixer: projections, gates, rule and norm.
Only the masked conv changed. With a mask, the mixer is 3.3 times
faster forward and 3.0 times faster with the gradient. Kernel launches
fall 22.8 times forward (10707 to 470 a call) and 19.5 times with the
gradient (36700 to 1884). The HLO replaces the scan's `while` loop with
the unmasked path's `__cudnn$convForward`. At lengths 2048 and 1537,
outputs agree with row-by-row evaluation to 2.4e-04 (the layer's bound
is 5e-4). The padded row's input gradients and outputs are exactly zero.
Against the fp32 token scan on CPU, the largest difference is 4.8e-07
over left, right, interior and paused padding at kernels 2, 4 and 8.

Omitting the field changes the batch's pytree, so every process in a pool
must agree on its presence. Each process knows only whether its own
rows need padding. If some omit the field and others include it, they
give the same step different pytrees.

For generation, processes first agree on the signature without validity,
then on a fixed-size presence vector. They run the same collectives in
the same order regardless of their local inputs. If any process includes
the field, every process adds it. If none includes it, they omit it and
keep the fused kernel.

`shard_batch` cannot perform that agreement: `DevicePrefetchIterator` places
batches on a worker thread, while step collectives run on the caller's
thread. In a pool, it therefore adds the field to every training batch's
`ModelInputs` that lacks it. Single-process runs, as measured in the
table, are unchanged. Batches of plain token arrays are also unchanged;
they have no validity field.

The head-chunk and head-dimension-256 cases were not rerun because this
change does not affect them.

## XLA flags

`TrainerConfig.xla_flags` appends to `XLA_FLAGS`. `prepare_process` applies
it before JAX opens a backend. It also sets
`--xla_allow_excess_precision=false` unless the run explicitly sets that flag
(`dew.training.runtime.keep_roundings`). With XLA's default, a fusion may
skip bf16 rounding specified by the program. The skipped rounding depends
on layout, so one device and four produced different bf16 forwards for
the same model.

The recipes and CLI call `prepare_process`. If your script or notebook
builds a `Trainer` directly, call it first or set
`XLA_FLAGS=--xla_allow_excess_precision=false` before importing jax. On the
RTX 4080, the flag reduced step time: the 176M hybrid DiT at batch 16
fell from 69.60 to 66.62 ms, SimpleDiT-B at batch 32 from 76.00 to 73.03,
and Qwen3-0.6B's widths at 1 x 1024 from 110.52 to 109.62.

The following sweep is why `xla_flags` defaults to None. It covers three
architectures, with a fresh process per configuration. Each cell gives
the median, with a range and count for repeated configurations.

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

No flag was adopted from this sweep because the differences fell within
measurement noise. Four repeats of the same simple_dit configuration ranged
from 6.97 to 7.53 ms, or 8%, because each fresh process autotunes again.
Every simple_dit result in the table fits that distribution. The
causal_transformer varies by 0.7%, and no flag changes it by more than
0.2%. Only unet shows a measurable effect:
`--xla_gpu_triton_gemm_any=true` takes the median from 17.38 to 17.05 ms, or
1.9%, over four runs each.

The unet gains 2%, the decoder is unchanged, and simple_dit's noise
obscures any difference. A default flag must be faster on all three
architectures by more than each one's noise, so the default stays None.
To use the unet flag for a run, pass `--trainer.xla-flags`.

Two flags affect autotuning or dispatch:

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
selection and scheduling; it did not test flags that relax precision.
Such flags would not be adopted because a change must keep a fixed-seed
20-step loss trajectory within 1e-5.

## UNet batch scaling

UNet's step time is the least sensitive to batch size among these
architectures. This experiment measures how it scales; no change was adopted.

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

Increasing the batch fourfold increases step time 3.3 times. About 4 ms
of the 17.4 ms step (23%) is independent of batch size; the rest scales
at 0.84 ms per sample.
Command buffers save 1.4% at batch 16 and nothing at batch 64.

The original utilisation column reported 1.7%, which was wrong. XLA's
`cost_analysis()` cannot count operations inside the backend's cuDNN convolution
calls. It undercounted this model 22.5 times. Counting from the optimized
HLO gives 40.5% of peak, as reported in `docs/benchmarks.md`.

## Muon against AdamW at equal tokens

These CPU runs compare optimizers at equal token budgets. Accelerator
speed is not measured. One workstation CPU
can finish nine of these small runs in under an hour.

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
disjoint cores. At a given seed, every arm sees the same batches in the
same order, so differences between arms come from the solver.

The three arms are `adamw` (AdamW), `muon` (Dew's Muon with parameter
groups) and `muon-unsplit` (`optax.contrib.muon` with its ndim == 2 rule).
The 'muon' entry used the unsplit rule before parameter groups were added.
Final loss is the mean over the last 50 steps.

| arm | lr 1e-3 | lr 3e-3 | lr 1e-2 |
|---|---|---|---|
| adamw | 1.4723 | 1.4842 | 1.5885 |
| muon | 1.5229 | 1.4438 | 1.4713 |
| muon-unsplit | 1.5762 | 1.4598 | 1.4916 |

Loss at five token counts, using each arm's best learning rate and
averaging over seeds 0, 1 and 2:

| arm | 0.51M | 1.02M | 2.05M | 3.07M | 4.10M |
|---|---|---|---|---|---|
| adamw, lr 1e-3 | 2.0136 | 1.7376 | 1.5737 | 1.5015 | 1.4764 |
| muon, lr 3e-3 | 1.9885 | 1.6744 | 1.5179 | 1.4572 | 1.4386 |
| muon-unsplit, lr 3e-3 | 2.2454 | 1.7646 | 1.5559 | 1.4812 | 1.4561 |

Muon with parameter groups reaches 1.4386 against AdamW's 1.4764,
0.038 nats lower at the same token count. Loss varies by 0.007 to 0.013
across each arm's three seeds, so the AdamW gap is three times that
noise. The gap to unsplit Muon is 0.018, one and a half times the noise.
The split version is ahead on all three seeds, by 0.016, 0.020 and
0.017. Raising the learning rate from each arm's best to 1e-2 increases
Muon's loss by 0.028
(3.3 times its best rate) and AdamW's by 0.116 (10 times its best rate).

The 0.4B-parameter run requested in section 4.9 of `docs/design/plan.md`
needs a v5e-16 and is not measured here. These runs shared a machine,
so their wall-clock times are also not comparable.

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

The compiled fp8 step contains `f8e4m3fn` converts: 146 mentions in the
HLO at width 256, against 12 GPU gemm calls. Quantization therefore runs
on the device, without errors, but the converts cost more than the gemms
save at these sizes. Losses decrease to 2.44 for bf16 against 2.68 for
fp8 at width 256, and 0.009 against 0.011 at width 1024. Each is after
14 steps from the same initialization. FP8 gives no speedup to adopt
on this card at these sizes.

## Serving against vLLM, 2026-10-03

`tools/benchmark_lm_serving.py` serves Qwen3-0.6B in bf16 on the RTX 4080.
Requests have 256-token prompts and 128 greedy output tokens, with twice
as many requests as slots. The table reports output tokens a second over
three repeats, with two Dew processes a side and vLLM 0.30.0 on
2026-10-01:

| slots | vLLM | Dew `1f8d5e72` | Dew, wide cache writes gathered |
|---:|---:|---:|---:|
| 32 | 6049-6053 | 5078-5323 | 5513-5532 |
| 64 | 7614-7618 | 6869-6879 | 7171-7204 |
| 128 | 9037-9050 | 7758-7773 | 8230-8289 |

Later the same day, the benchmark alternated Dew at `88620c1d` and vLLM
0.30.0 in one session, with two processes each. That Dew revision uses
cuDNN prefill attention, stores the head's bf16 logits as bf16, and writes
caches as words; these changes are described below. The table gives the
range over every repeat. Another lane kept the host's load average near
50 during the session:

| slots | vLLM | Dew `88620c1d` | Dew / vLLM, medians |
|---:|---:|---:|---:|
| 32 | 5903-6046 | 5490-5883 | 0.96 |
| 64 | 7532-7563 | 7540-7619 | 1.01 |
| 128 | 8949-9049 | 8957-8985 | 1.00 |

Dew is level with vLLM at 64 and 128 slots and 4% behind at 32, where a
step is shortest and the host accounts for the largest share. Each process runs
`tools/benchmark_lm_serving.py --backend dew` (or `vllm-engine`, with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`) `--slots 32,64,128 --repeats 3` over the
Qwen3-0.6B checkpoint. Its JSON records each repeat's `output_tokens_per_second`.

At 64 slots, a Dew decode step takes 6.96 ms on the device. Attention
uses 4.0 ms, limited by reading the dense cache's keys and values
(2.8 GB a step at 716 GB/s). Projections and the head use 2.1 ms, near
the weight-reading limit. Each of the 16 admission steps prefilled
8 prompts alongside the other rows' decode and took 40 ms. Prefill key
and value writes accounted for 8 of those ms. XLA makes slots the minor
dimension in the fresh prefill cache, to suit attention reads. Scattering
whole tokens into that layout achieved 28 GB/s.

A write as wide as its buffer now gathers each slot's token
(`dew.nn.kv_cache.write_cache`). The bits stay the same: tokens and both
log-probability streams are identical at every slot count.

Decode attention reads each key head once for its group of query heads,
using the group as that head's query positions. Without this grouping,
cuDNN padded the lone query to two positions and ran each query head
separately. At 64 rows over 384 slots, the kernel takes 0.146 against
0.154 ms a layer. With 257 keys in every row, it takes 0.072 against
0.129, with identical bits.

Serving at integration `17e2b226` agrees with this result, but its
differences fall within the loaded host's spread. Medians over 15 runs
change from 5489-5510 to 5521-5619 tokens a second at 32 slots, from
7175 to 7215 at 64, and from 8219 to 8343 at 128. Both versions had
slow runs, the slowest at 5196 and 4685 with 32 slots. Tokens and
log-probabilities are identical.

### The remaining gap, 2026-10-03

At 64 slots, Dew serves 7146 and 7197 tokens a second (medians of five
runs, two processes, integration `6a220e31`) against vLLM 0.30.0's
7596-7609 in-process (three runs, one process). Their schedules are similar.
Dew runs 265 model steps per run; vLLM runs 267 scheduler steps, counted
in its engine process. Each admits up to 8 prompts a step alongside
the other rows' decode. The throughput gap comes from step time: 8.6 ms
wall time for Dew against 8.0 for vLLM. These traces cover whole measured
runs, using XProf for Dew at `f6047cf9` and Nsight with CUDA graph nodes
for vLLM:

| | Dew | vLLM, 249 of 267 steps traced |
|---|---:|---:|
| GEMMs | 891 ms | 907 ms |
| attention | 1016 ms, 138.7 us a decode call | 943 ms, 136.6 us a decode call |
| everything else | 313 ms | 154 ms |
| device busy | 2.22 s of 2.29 | 2.00 s of 2.18 |

Nsight warned that it missed CUDA events. Its trace contains 249 of the
267 steps, so vLLM's rows undercount by about 7%. Scaled to the whole
run, vLLM's attention time matches Dew's and its GEMMs take 80 ms longer.
Dew spends the extra device time on many small kernels, each one to
three microseconds. The residual add, split-K GEMM sum and cast total
52.5 ms in 22859 launches; norms total 80.7 ms in 32201. Cache writes
take 61.0 ms and attention transposes take 48.8 ms. vLLM fuses an add
into RMSNorm and writes K and V in one kernel. Dew's 2.29 s run also
includes 70 ms without a kernel running.

These options were measured. None was adopted; any gain was under 1% of the run:

- Writing K and V with one Pallas kernel per layer takes 0.141 ms a
  decode step for all 28 layers, against 0.166 for XLA's two scatters.
  The results are bitwise identical.
- XLA's command buffers are on by default and replay cuDNN attention in
  the decode step. Enabling every command type with no minimum graph size
  stays within the default's spread. Disabling them is slower. Tokens are
  identical in all three cases.
- JAX's Pallas split-KV decode attention (`gqa`) takes 0.43 ms a layer,
  including transposing Dew's cache, against cuDNN's 0.165. On a head-major
  cache, its best tiles take 0.155 ms, with different bits from cuDNN.
- Triton multi-output fusion and a single split-K, one traced run each:
  device time 2222.2 and 2216.2 ms against 2222.3-2224.6.
- Freeing a slot in the step that finishes it would take 263 steps
  instead of 265, counted step by step. Currently the host frees it when
  it reads the step's results.

The remaining gap comes from small kernels and idle time between steps.
Fusing add and norm would save launches worth at most about 1.5% of the
run. Rates come from `tools/benchmark_lm_serving.py`. Each trace covers
one measured run: Dew under `dew.Profiler`, and vLLM under Nsight Systems
with `--cuda-graph-trace=node`. Kernel times are summed by family.

The idle between steps, 2026-10-03. Two host costs left the device waiting.
At submission, making a request's key took three small programs: move
the seed to the device, make the key, and fold it. These ran between
steps after the device had finished its queued work. Admission also uses
fresh input buffers on every call, requiring XLA to update its CUDA
command buffer before replay. This took about 5 ms before 9 of the
run's 16 admission steps.

An integer seed now stays on the host until one admission program makes
and folds every row's key (`_row_keys`, bitwise equal to an eager key).
The admission step compiles without command buffers (`ADMISSION_OPTIONS`),
while decoding keeps them. The benchmark submits integer seeds, as clients
do. This table compares the change with its parent on the RTX 4080:
tokens a second, three alternating rounds, medians of five runs.

| slots | before | after |
|---:|---:|---:|
| 32 | 5664 / 5657 / 5663 | 3960 / 5728 / 5728 |
| 64 | 7131 / 7229 / 7222 | 7360 / 7359 / 7357 |
| 128 | 8329 / 8355 / 8346 | 8527 / 8528 / 8524 |

One process was slow in all five 32-slot runs (3846-4054), while its
64- and 128-slot runs matched the others. Four more alternating rounds
at 32 slots gave 5729-5730 after against 5564-5669 before. An earlier
session on a loaded host gave the same ordering at every slot count.
Tokens and both log-probability streams are identical in every run.

Admission already compiled separately from decoding because the step's jit
traced each separately. Cold startup is therefore unchanged. Without a
compilation cache, `Server.from_task` to the first token took 34.9 and
35.5 s before, against 38.7 and 35.2 after, in two alternating rounds.
A traced 64-slot run idles 74.7-83.2 ms before and 40.9-42.1 after. Of
five traces, one took 76.0 and one 838 during a host load spike. Most
remaining idle time comes from the traced client's own keys. Each trace
covers one measured 64-slot run of the benchmark's prompts under
`dew.Profiler`, summing idle time between kernels.

Several decode iterations per device call (`decode_steps`), measured after
that change, 2026-10-03. More iterations slow the run at every slot count
except 64 slots with two iterations, which reaches 7271 tokens a second
against 7216 with one. The table gives tokens a second on the RTX 4080,
with admission 8 rows and medians of five runs. vLLM rates come from
the table above:

| slots | vLLM | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 6049 | 5726 | 5696 | 5500 | 5106 |
| 64 | 7614 | 7216 | 7271 | 6975 | 6573 |
| 128 | 9047 | 8527 | 8357 | 7861 | 7231 |

A call admits requests only before its first iteration. Filling 64 slots
at 8 rows a call takes 8 calls of k iterations; a run takes 265, 274,
292 and 328 iterations. Admitting at every iteration could recover only
the idle time. At one iteration a call, a traced 64-slot run keeps the
device busy 2228 ms of 2242, with 20 ms idle (0.9%). Decoding already
replays CUDA graphs, and the host runs ahead.

Admitting 8k rows a call, the default, changes prefill shapes and GEMM
kernels. Its tokens differ from a run with one iteration a call. At
8 rows, tokens match at 32 and 64 slots, with log-probabilities within
9.5e-6. At 128 slots, the longer program's decoding GEMMs autotune to
different kernels. Each cell runs `tools/benchmark_lm_serving.py --decode-steps K --admission 8
--generations` and compares generations with K=1.

The decoding step's small kernels, 2026-10-03. At 64 slots on the RTX
4080, Dew uses 8.39 ms of device time per decode step against vLLM's
about 8.05. GEMMs take 3.37 against 3.64, attention 3.84 against 3.79,
and everything else 1.18 against 0.62. The largest remaining kernel took
136 us a step. It merged stored logits with the step's logits across
every slot's vocabulary in fp32 so that a newly admitted row drew from
its prompt logits.

On a step without admission, the model computes logits for every drawing
row. That step now uses the model's logits directly; only admission
steps merge them. The table reports tokens a second over three
alternating rounds, as medians of five runs. Tokens and both
log-probability streams are identical in every run:

| slots | before | after |
|---:|---:|---:|
| 32 | 5710 / 5729 / 3389 | 5780 / 5779 / 5746 |
| 64 | 7364 / 7363 / 7236 | 7410 / 6545 / 7407 |
| 128 | 8543 / 8527 / 8461 | 8610 / 8559 / 8609 |

One process on each side hit a load spike (3389 and 6545). Rounding
the logits to bf16 now takes its own kernel, at 57 us a step. Omitting
stored step logits would let XLA fuse rounding into the draw, saving
another 27 us. However, the draw's log-softmax would sum in a different
order: at 32 slots, 430 log-probabilities change by up to 3.8e-6. The
step therefore still stores its logits.

Attribution uses a run traced under `dew.Profiler` with command buffers
off (`--xla_gpu_enable_command_buffer=`) and optimized HLO dumped. Each
kernel is named by its `hlo_op` in that HLO.

The head's bf16 rounding, 2026-10-03. `dew.nn.precision.head_product` produces
bf16 values stored in fp32. The reduce-precision operation in `rounded_to`
did not fuse with XLA's Triton head GEMM. A serving step therefore wrote
fp32 logits, read them back for rounding, then read them three more ways:
the well-formedness check, argmax with the log-softmax maximum, and the
sum of exponentials. Rounding alone took 157 to 178 us at 128 rows.

On CUDA, `bf16_logits` now casts logits to bf16 behind a barrier. The
cast fuses with the GEMM's epilogue, and each reader widens the bf16
copy. Values and cotangents are bitwise equal to `rounded_to`'s
(tests/test_precision_policy.py). Served tokens and log-probabilities are
also bitwise equal at 32, 64 and 128 slots. At 128 rows with Qwen3-0.6B's
widths, the head and greedy draw fell from 1.145 to 0.981 ms. Device
busy time per serving run fell from 2211.0 to 2198.5 ms at 64 slots
and from 3803.5 to 3768.7 ms at 128, with two traced runs each.

Training's loss head rounds its own tiles (`dew.objectives.lm.chunked`);
other platforms keep `rounded_to`. Bitwise checks use
`tests/test_precision_policy.py::test_bf16_logits_round_as_rounded_to_and_are_held_as_bf16_under_cuda`
and `tools/benchmark_lm_serving.py --generations` on each tree. Busy time
comes from one traced run under `dew.Profiler`.

A fused decode prologue, measured and removed. XLA uses five kernels per
layer for q and k norms, rotated-key and value cache scatters, query
rotation and GQA folding. Each takes one to three microseconds. A Pallas
Triton kernel combined all five in one program per row. It reduced time
over 28 layers at Qwen3-0.6B's widths from 0.191 to 0.097 ms, and served
Qwen3-0.6B 2-3% faster. Tokens and log-probabilities stayed bitwise equal
in every run: 5943 / 7587 / 8765 tokens a second at 32 / 64 / 128 slots,
against vLLM's 6049 / 7614 / 9047.

Qwen3-1.7B has the same head widths, but its tokens differed from the
unfused step's: 1349 of 8192 at 32 slots. Each version was repeatable.
The first record here incorrectly described the unfused step as
unrepeatable; that run had used the kernel on both sides. The norm caused
the difference. Even with XLA's statistics from a standalone reduction,
summing over a head's 128 lanes rounded one or two elements per head
to the other bf16 neighbour. XLA's reduction order depends on the norm
fusion, and rsqrt uses the hardware approximation. There was no fixed
arithmetic for the Pallas kernel to match.

Leaving norms in XLA's fusions and combining only rotation, cache writes
and folding was bitwise equal on both models. It saved 3 of the 5
kernels, reducing device busy time per 64-slot run from 2211 to 2186 ms.
The Pallas call was outside default CUDA command buffers, so the decode
step needed `CUSTOM_CALL` in its options or the run idled 200 ms longer.
A gain under 1% did not justify a kernel on a backend JAX 0.11 deprecates.
It was removed; the rotation-only kernel remains on `perf/prologue-bitwise`.

The 128-slot gap and prefill attention, 2026-10-03. A traced run at
integration `66a1848c` disabled command buffers to attribute each kernel to
its program and HLO scope. Attention and GEMM times match vLLM: cuDNN
decode attention takes 259 us a call against vLLM FA2's 266, and GEMMs
take 4.68 against 4.74 ms a step. Small kernels account for the gap:
1.26 ms per decode step against vLLM's 0.50, and 8.1 ms per admission
across 32 admissions of 8 prompts.

Prefill attention was the largest admission cost. Admission left-pads
prompts and writes keys compactly, so cache prefill builds a cursor mask.
`kernel_for_materialized_mask` sent it to xla because cuDNN's bias backward
refuses odd lengths. This training rule caused two dense dots, a softmax
and four mask transposes, costing 3.7 ms per admission.

A cache call with more than one query now passes that mask as cuDNN
bias where cuDNN runs. Training, single-token decode, CPU and the
deterministic-ops CUDA lane are unchanged. Over 28 layers in isolation,
the call takes 4.0 against 6.2 ms. Device busy time per run fell from
2212.6 to 2179.2 ms at 64 slots and from 3807.4 to 3745.1 ms at 128.
There were two traced runs each, with the same result in both.

The results are not bitwise equal because cuDNN rounds at different points
from xla's dots and softmax. Reference checks use tests/reference_error.py's
rule over 8 benchmark prompts cut to mixed lengths and left-padded. The
same bf16 weights are computed in fp32 at the highest precision. CuDNN
prefill's RMS distance is 1.036 times xla's for Qwen3-0.6B logits and
1.088 for log-probabilities. For Qwen3-1.7B, the ratios are 0.999 and
0.987, against an allowed 2. Argmax agrees with fp32 at 96.4 against
96.8% of positions on 0.6B, and 96.9 against 97.2% on 1.7B.

Greedy generations from the benchmark's random-token prompts diverge in
56 of 128 rows at 64 slots, with each run repeatable. Every first
divergence is a bf16 near-tie. Teacher-forced through the fp32 model,
the chosen tokens' logits are a median 0.64 bf16 spacings apart at
their magnitude, with a largest gap of 2.78. Of these gaps, 77% are
under one spacing and 98% under two. The fp32 argmax matches the xla
prefill's choice in 31 rows and cuDNN's in 23.
`tests/test_causal_transformer.py::test_a_padded_prefill_attends_through_cudnn_where_it_runs`
checks routing. The comparisons apply `tests/reference_error.py`'s
`distance` to the cache prefill and to `dew.pipeline(..., dtype="float32")`
under `jax.default_matmul_precision("highest")` over the same tokens.

The cache writes as words, 2026-10-03. XLA's scatter stores one element
per thread, so bf16 caches moved two bytes per store. At 128 rows,
each decode layer's key and value writes took 3.7 us apiece; prefill
windows took 28 us per cache during admission. On GPU, `write_cache` and
admission placement now move one- or two-byte cache values' bits as uint32 words
(`dew.nn.kv_cache.as_words`). Word-wide and boolean leaves, and axes that
do not fill whole words, keep their original writes. So do TPUs, which
tile two-byte arrays on a different axis.

Over 16 caches in isolation, decode writes fell from 0.113 to 0.099 ms
and admission writes from 0.627 to 0.277. Served tokens and
log-probabilities are bitwise equal at 32, 64 and 128 slots. Device busy
time per run fell from 2165.6 to 2149.5 ms at 64 slots and from 3704.2
to 3649.5 ms at 128, with two traced runs each at integration `456a4b64`.
`tests/test_kv_cache.py::test_a_cache_write_moves_whole_words_with_the_same_bits`
checks the bits.

The cache's validity, derived, 2026-10-04. Each attention cache stored a
`[rows, capacity]` mask of filled slots beside the cursor. Because slots
fill in order, this mask always equalled the cursor's `filled_slots`.
Every decode step rewrote it in every layer, using 28 one-microsecond
kernels only to store the next state. `dew.nn.attention.cached_validity`
now derives it at each read in attention, Llama 4, MLA, the DSA pool
and DeepSeek V4. Serving no longer places or zeroes the mask.

Served tokens and log-probabilities are bitwise equal at 32, 64 and
128 slots. Device busy time per run fell from 1393.7 to 1388.3 ms at
32 slots, from 2147.5 to 2139.2 at 64, and from 3645.2 to 3626.2 at
128, with two traced runs each.

### Open loop, 2026-10-04

The closed-loop table above submits every request at once and keeps slots
full. In open loop, `tools/benchmark_lm_serving.py --rate` sends requests
over time as a Poisson process at each rate, seeded per slot count.
It sends six times as many requests as slots to a server that queues
them. Time to first token (TTFT) starts at request arrival. Token gaps
measure the time between consecutive tokens for each decoding row.

This run uses Qwen3-0.6B, the same prompts and outputs, and the RTX 4080
on a quiet host. Dew ran at integration `420ea2c1`, then with bucketed
admission in the same session as vLLM 0.30.0:

| slots | rate | Dew `420ea2c1` tok/s, TTFT p50 / p99, gap p99 (ms) | Dew bucketed | vLLM |
|---:|---:|---|---|---|
| 32 | 16 | 1985, 34.6 / 61.2, 28.5 | 1990, 12.8 / 19.5, 8.0 | 1990, 13.8 / 33.5, 8.4 |
| 32 | 24 | 2892, 55.3 / 257, 29.9 | 2929, 13.6 / 19.9, 8.5 | 2928, 15.4 / 30.1, 15.5 |
| 32 | 32 | 3216, 536 / 1216, 30.2 | 3832, 15.5 / 31.9, 11.2 | 3829, 16.3 / 33.7, 11.3 |
| 64 | 24 | 2877, 42.6 / 85.3, 30.9 | 2886, 15.0 / 25.3, 10.1 | 2893, 14.6 / 22.2, 7.0 |
| 64 | 36 | 4071, 589 / 1047, 32.4 | 4170, 18.0 / 55.2, 16.5 | 4279, 16.5 / 26.1, 8.8 |
| 64 | 48 | 4469, 1454 / 2258, 32.8 | 5580, 68.1 / 458, 55.4 | 5620, 21.3 / 33.7, 14.0 |
| 128 | 32 | 4093, 80.5 / 647, 93.8 | 4210, 21.0 / 32.3, 14.2 | 4240, 16.7 / 27.5, 9.1 |
| 128 | 44 | 5068, 1140 / 1989, 64.2 | 5701, 26.8 / 43.2, 18.6 | 5756, 20.2 / 32.4, 11.7 |
| 128 | 56 | 5320, 2217 / 4449, 83.5 | 7064, 34.6 / 55.0, 23.4 | 7090, 35.8 / 57.2, 22.4 |

The admission program prefilled every row, padding a lone request to
eight prompts. Admission steps set the token-gap p99 at 28 to 94 ms.
They reduced server capacity until the queue grew without bound at
rates where vLLM kept TTFT p50 under 40 ms.

Each admission step now pads to the smallest power of two that fits
its prompts (`dew.inference.serving.admission_share`). Each width has its
own compiled program. Closed-loop runs are unchanged because an admission
that fills its width uses the same program. Generations at 32, 64 and
128 slots are bitwise equal.

With buckets, Dew matches vLLM at 32 slots and has shorter TTFT tails.
At 64 and 128 slots, Dew's token-gap p99 is 1.4 to 2 times vLLM's.
At 64 slots and 48 requests a second, Dew reaches capacity: a traced
run was 97% device busy, and TTFT p50 varies from 21 to 447 ms
between runs. Admission accounts for the gap. Dew prefills an arriving
prompt in a separate forward alongside decoding, reading the weights twice.
With one 256-token prompt at 64 slots, admission takes 9.9 ms against
5.1 ms for a decode step. vLLM's chunked prefill puts prompt tokens in
the decode forward's batch.

A narrower prefill changes GEMM shapes, so admitting fewer than the padded
eight prompts can change the drawn bits. Serving one request at a time
at 32 slots, 6 of 16 Qwen3-0.6B rows and 8 of 16 Qwen3-1.7B rows
diverge from the padded path at bf16 near-ties. Teacher-forced in fp32,
the chosen logits are a median 0.5 bf16 spacings apart, at most 1.81.
FP32 argmax matches the padded choice in 7 rows and the bucketed choice
in 7.

Under tests/reference_error.py's rule, the 1-, 2- and 4-row prefills'
RMS distances from the same weights in fp32 are 0.96, 0.95 and 0.95
times the 8-row prefill's on Qwen3-0.6B log-probabilities. The logits'
ratios are 1.00, 1.00 and 0.99. For Qwen3-1.7B, log-probability ratios
are 1.02, 1.00 (bitwise) and 1.08, and logits ratios are 1.02, 1.00
and 1.07. The allowed ratio is 2.

The mixed admitting step, 2026-10-04. Admission now runs one forward over
each slot's last draw and the admitted prompts, arranged in one row
(`dew.nn.inputs.Admitted`). Projections, norms, MLP and head work token by
token. They read weights once for decoding rows and prompts together.
Attention scatters every token's keys into its cache row in one call.
Decoding queries read their cache rows as in a decode step. Prompts that
start a row read their own keys.

The decode-only program is unchanged. Its optimized HLO is identical to
integration's, and device time per run matches (2540.4 against 2540.0 ms
at 128 slots). This session compared Dew at integration `ab5966b1`
(bucketed admission), Dew with the mixed step, and vLLM 0.30.0 on a
quiet host:

| slots | rate | Dew `ab5966b1` TTFT p50 / p99, gap p99 (ms) | Dew mixed | vLLM |
|---:|---:|---|---|---|
| 32 | 16 | 17.3 / 36.5, 14.0 | 12.6 / 34.5, 9.4 | 13.2 / 17.7, 6.6 |
| 32 | 24 | 14.6 / 37.8, 12.1 | 13.0 / 22.0, 7.8 | 14.6 / 19.9, 6.8 |
| 32 | 32 | 14.9 / 27.9, 9.7 | 13.4 / 21.5, 8.2 | 15.3 / 21.7, 7.8 |
| 64 | 24 | 14.8 / 25.3, 9.2 | 13.5 / 20.3, 8.0 | 14.7 / 22.9, 7.6 |
| 64 | 36 | 17.0 / 27.8, 11.4 | 15.3 / 25.0, 10.3 | 17.0 / 26.6, 10.6 |
| 64 | 48 | 20.8 / 37.4, 13.8 | 18.4 / 32.0, 12.1 | 21.1 / 33.7, 14.9 |
| 128 | 32 | 20.6 / 33.9, 14.2 | 19.1 / 29.6, 13.1 | 17.1 / 34.8, 12.0 |
| 128 | 44 | 26.3 / 42.3, 17.1 | 23.9 / 39.7, 15.9 | 24.6 / 79.2, 25.1 |
| 128 | 56 | 34.7 / 55.4, 23.1 | 31.2 / 50.5, 22.5 | 38.4 / 64.6, 23.0 |

All three serve the same tokens a second at every rate, within 0.8%.
With the mixed step, Dew has the shortest median TTFT in 8 of 9 cells.
At 64 and 128 slots, its p99 TTFT and token gap are at or below
vLLM's from 36 requests a second up. At 32 slots, vLLM's token-gap p99
is shorter by 0.4 to 2.8 ms. vLLM also has shorter TTFT p99 at the
two lower rates. The 64-slot cell at 48 requests a second saturated
in the earlier session but not here; this rate is near server capacity.

In closed loop, three repeats in two alternating rounds gave 5929 to
5994 tokens a second at 32 slots, 7673 to 7766 at 64, and 9039 to
9033-9072 at 128. Traced at 128 slots, the admission program took 1068
against 1078 ms per run: GEMMs 671 against 676, attention 253 against
252.

The mixed step changes GEMM shapes, so it is not bitwise equal to two
separate forwards. Reference checks compute the same bf16 weights in fp32
at the highest precision, over 28 decoding rows and four admitted prompts
of 64 to 256 tokens. RMS distance is 0.84 times the two forwards' for
Qwen3-0.6B decoding logits and 0.97 for prompt logits. Log-probability
ratios are 0.83 and 0.95. On Qwen3-1.7B, the logits ratios are 1.09
and 0.99, and log-probability ratios are 1.14 and 0.98. The allowed
ratio in tests/reference_error.py is 2.

At 32 slots, 27 of 64 greedy rows diverge from integration's, all at
bf16 near-ties. In fp32, the chosen logits are a median 0.57 bf16
spacings apart, at most 1.44. FP32 argmax matches the two-forward choice
in 16 rows and the mixed choice in 11.

Over a paged cache, 2026-10-04. The mixed step also runs over a page
pool. It installs admitted rows' tables before writing each token to
its row's page. Decoding rows read the pool through cuDNN's paged kernel,
as in a decode step. A chunked prompt or shared prefix reads the row's
earlier keys from the pool. Without either, each piece reads only its
own keys. Dense caches support chunked prefill the same way (`chunk`);
only the mixed step serves it.

Paged decode-only programs are identical to integration's. Two alternating
rounds ran in one session at integration `ab60614a`. The table gives the
second round; the first ran under another lane's load:

| slots | rate | paged, two forwards: TTFT p50 / p99, gap p99 (ms) | paged, mixed |
|---:|---:|---|---|
| 32 | 16 | 17.8 / 23.7, 9.9 | 16.5 / 22.4, 8.8 |
| 32 | 24 | 18.2 / 27.3, 10.0 | 16.6 / 22.0, 8.9 |
| 32 | 32 | 18.1 / 26.5, 10.3 | 16.9 / 24.0, 11.0 |
| 128 | 32 | 40.3 / 58.5, 26.0 | 38.1 / 55.6, 23.7 |
| 128 | 44 | 42.9 / 63.7, 26.3 | 41.0 / 61.0, 25.7 |
| 128 | 56 | 50.6 / 231.3, 26.3 | 44.8 / 161.7, 25.8 |

In closed loop, the mixed step increased throughput from 5576-5577 to
5583-5656 tokens a second at 32 slots and from 8254-8274 to 8330-8334
at 128. A traced 32-slot run at 24 requests a second measured one-row
admission at 8.30 against 9.22 ms. Paged decoding is slower than dense:
at 128 slots and 32 requests a second, token-gap p50 is 11.7 ms paged
against 5.4 dense. Use a dense cache for speed when it fits in memory.
Served tokens diverge from the two forwards in the same 27 of 64 rows
at 32 slots as with a dense cache, all at bf16 near-ties.

Under `--xla_gpu_deterministic_ops`, the CUDA test lane's flag, a paged write
put a dropped token's keys in another head's kept slot (jax 0.11.2,
RTX 4080). The write scatters into page and offset axes past an unindexed
head axis. The expander padded out-of-range rows with 0 on that unindexed
axis, so a window offset collided with a kept index. openxla/xla#49498
fixes this (issue #49380), after the jax pin. Mapping over a group of
one makes the scatter correct, as in the paged cache's other write.
`KVStore.write_tokens` uses that form.

Open: latent attention (MLA, DSA), sliding windows, sinks, quantized or
rotated caches, a pool split into groups, hybrids whose recurrent layers
read their row's tokens in order, and prediction depths keep the two
forwards; a server names which (`Server.mixed_refusal`, logged at build).

### A hybrid: Qwen3.5-0.8B, 2026-10-05

Qwen3.5-0.8B is a multimodal wrapper around 18 gated delta net layers and
6 full-attention layers. Before 2026-10-05 Dew served it NaN (the chunked
rule's inverse, above). Then, served at 384 slots, it held caches of 8192:
`_sized` sized a model only through its own `max_seq_len` field, which the
wrapper only reads through to its language model. 4.1 GiB of cache at 32
rows ran out of memory at 128, and each decode step transposed every
full-attention layer's 8192 slots, half its time. `_sized` now sizes the
wrapper's language model, for serving and for a lone generation alike.
The same session against vLLM 0.30.0, one process each:

| slots | rate | Dew TTFT p50 / p99, gap p50 / p99 (ms) | vLLM |
|---:|---:|---|---|
| 32 | closed | 4166-4271 tokens a second | 4938-4945 |
| 32 | 8 | 25.4 / 37.1, 5.94 / 13.65 | 28.7 / 74.0, 3.50 / 22.87 |
| 32 | 16 | 26.4 / 40.8, 5.95 / 15.97 | 33.0 / 114.9, 4.11 / 31.88 |
| 32 | 24 | 28.1 / 49.2, 5.96 / 15.98 | 38.2 / 69.5, 4.52 / 26.29 |
| 128 | closed | 4203-4207 | 5410-5577 |
| 128 | 16 | 81.1 / 113.3, 22.96 / 43.98 | 41.8 / 215.7, 4.29 / 72.11 |
| 128 | 24 | 88.8 / 125.8, 25.48 / 46.88 | 42.7 / 112.3, 4.87 / 26.70 |
| 128 | 32 | 929 / 2052, 34.58 / 55.13 | 46.4 / 84.4, 6.60 / 27.22 |

At 32 slots Dew has the shorter tails and vLLM the shorter median token
gap. At 128 Dew's decode step is the gap: it runs every slot's delta-rule
state each step, 2.4 GB of fp32 at 128 rows, and reads it in separate
passes (decay, the memory read, the write, the query read), where vLLM's
fused recurrent kernel passes once over the rows it holds. Tracing 32
slots, those passes were a quarter of the decode program's device time.

## Quantized serving of the 176M text-to-image model, 2026-09-28

`TextToImage.quantized` serves the denoiser with its kernels stored as int8
or fp8 values and scales, using Qwix's post-training quantization
(`dew.training.quantization.quantize_for_serving`). `int8` and `fp8`
quantize weights and activations. A matmul of two quantized operands runs
in the quantized dtype. `int8w` and `fp8w` quantize weights only, then
dequantize to the compute dtype. The model is dewml/hybrid-dit-176m.

Each row runs in one process of `tools/benchmark_quantized_serving.py`.
Forward time measures a warm guided denoiser call over 12 prompts
(batch 24, median of 5). Sample time is warm wall time for 12 images
with 20 DPM-Solver++(2M) steps at guidance 5, including text encoding and
decoding. CLIP is mean ViT-L/14 cosine over 12 prompts at seeds 0 and
1. Memory reports denoiser weight bytes and compiled-forward temporaries
in MiB.

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

Weight-only quantization reduces kernel storage to 27% of fp32 bytes.
Forward time matches the unquantized model in the same compute dtype.
Quantizing weights and activations to int8 or fp8 reduces bf16 forward
time by 21% (28.3 ms against 36.0). It reduces fp32 forward time by
35% to 39% (34.7 and 32.5 ms against 53.3). Every quantized row keeps
CLIP within 0.002 of fp32.

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

On the A100, quantization saves memory without reducing time. Quantizing
weights and activations makes forward slower than the unquantized model
in the same compute dtype: 28.4 ms in bf16 int8 against 25.0, and
30.3 ms in fp32 int8 against 27.5. FP8 is slower still because the
A100 has no fp8 units.
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

On the v6e, int8 weights and activations halve fp32 forward time
(7.1 ms against 13.7) when depthwise convolutions are also quantized.
Keeping those convolutions in fp32 gives 14.9 ms. In bf16, weight-only
times match the unquantized forward (5.2 and 5.6 ms against 5.7).
Quantizing activations makes it slower (6.3 to 9.8 ms).

Sampling takes 28 to 31 s in every row regardless of forward time.
Something outside the denoiser's 20 steps limits this machine; these
measurements do not break down that cost. Every quantized row keeps
CLIP within 0.005 of fp32.

Before serving scaled quantized products in float32, every bf16 row with
int8 or fp8 activations sampled NaN images on the v6e. CLIP was 0.1481,
whether depthwise convolutions were quantized or not. Qwix 0.1.8 scales
the product of two quantized operands in the scales' dtype, which is
bf16 for a bf16 model. In plain JAX on the v6e, scaling an int8
depthwise convolution's int32 product in bf16 produced NaN in almost
all outputs. The model's dense and attention forms stayed finite with
the same scaling. In the served model, NaN first appeared in int8
depthwise convolutions and in an fp8 attention block. Scaling in float32
gives the bf16 model's 8-bit operations the fp32 model's result types,
and it samples as shown above.

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

On XLA:CPU, int8 weights and activations double forward time (9.8 s
against 4.9 in fp32). Weight-only int8 leaves it unchanged. Every
quantized row keeps CLIP within 0.0025 of fp32.

On GPU, Dew refuses to quantize grouped-convolution activations. These
GPU rows therefore keep spatial-fusion depthwise convolutions in float
with `--float spatial_fusion`; `TextToImage.quantized` raises without it.
XLA:GPU at jax 0.11.2 computes these convolutions incorrectly or fails
to compile them.

On the RTX 4080, int8 convolutions with one or two input channels per
group return wrong values without errors. Before Dew refused them,
whole-model int8 took 42.3 ms and scored CLIP 0.1391. In fp8, the same
convolutions fail to compile there (`Failed to get configs for: 36 out of 126 instructions`,
one per depthwise convolution). On the A100, whole-model int8 scored
CLIP 0.1373 in fp32 and failed to compile in bf16 (`UNIMPLEMENTED`).
FP8 ran on the A100, scoring CLIP 0.2443 in fp32, but Dew refuses
it there as on every other GPU.

## Kernel choices per generation, 2026-09-22

Each kernel has one place that selects its implementation by hardware
generation (`dew.nn.kernels.device_generation`: `sm80`, `sm86`, `sm89`,
`v5e`, `v6e`, ...). Generations without a measurement here use XLA.
`tools/benchmark_kernels.py` and `tools/benchmark_lm_head.py` reproduce the
rows. Each row runs in its own process with jax 0.11.1 and bf16 compute.
Hardware is a Colab NVIDIA L4 (the RTX 4080's sm_89 architecture), a
Colab TPU v6e-1, and the local RTX 4080 for kernel-level rows.

Step rows use `tools/benchmark_kernels.py step`, built on
`tools/benchmark_step.py`'s trainer, with 30 timed steps after 5 warmup.
The lm-moe model has 321.8M parameters and 8 experts with top-2 routing;
lm-dense has 359.8M. Both use sequence 1024. Batch sizes are 4 (moe)
and 1 (dense) on the L4, and 8 and 8 on the v6e. "before" means main
at c1f7e2dd.

### The MoE grouped matmul: `GROUPED_MATMUL_BY_GENERATION`

| device | path | ms/step | p50 ms | peak GiB |
|---|---|---|---|---|
| L4 | lm-moe before (xla) | 601.55 | 610.84 | 12.46 |
| L4 | lm-moe after, `auto` = pallas | 213.14 | 216.45 | 8.15 |
| L4 | lm-moe after, xla | 598.54 | 609.07 | 12.46 |
| v6e | lm-moe before (xla) | 76.04 | 76.43 | 5.05 |
| v6e | lm-moe after, `auto` = xla | 74.68 | 75.25 | 5.05 |
| v6e | lm-moe after, tokamax (`mosaic_tpu_v2`) | 75.40 | 76.04 | 5.05 |

A rerun with jax 0.11.2 used one Colab L4 session on 2026-09-22,
19:55 to 20:13 CDT. With `tools/benchmark_kernels.py step --path lm-moe --batch 4`,
`auto` (pallas) took 224.90 ms with an 8.15 GiB peak; `--implementation xla`
took 602.36 ms with 12.46 GiB. The `projection` forward plus backward
took 3.37 ms in Pallas and 25.19 ms in XLA.

On a 2x RTX 3090 mesh (jax 0.11.2), one bf16 ExpertMLP layer's forward
plus backward takes 141.9 ms in XLA against 64.4 in Pallas with fsdp 1
expert 1. With fsdp 2, it takes 83.3 against 98.1 ms. Pallas all-gathers
the fsdp-sharded expert kernel, using 128 MiB of temporaries against
XLA's 3120. Expert 2 global takes 174.0 against 102.7, and expert 2
exchange takes 93.0 against 14.9.

Whole lm-moe steps take 1024.0 against 578.3 ms with data 2, and 780.9
against 537.1 with expert 2 exchange, with the same losses. These results
first made `auto` select XLA when fsdp alone sharded experts. Dispatch
later moved every routed layer inside its row map, so both kernels see
gathered experts. Pallas wins there on the RTX 3090
([Expert parallelism on 4x RTX 3090](#expert-parallelism-on-4x-rtx-3090-2026-09-23)).

The kernel matrix measures forward plus backward with jax 0.11.2 and
checks against float64. At lm-moe's up projection, Pallas takes 1.21 ms
against XLA's 6.25 on an A100, 3.15 against 25.9 on an L4, and 1.59
against 15.7 on the RTX 4080. At 128 experts, the times are 0.43 against
15.7 (A100), 1.38 against 77.9 (L4), and 0.55 against 33.5 (RTX 4080).

XLA wins at 128 experts on TPU v5e and v6e. On v6e, it takes 0.346 ms
against `mosaic_tpu_v2`'s 0.408. On an RTX 3090 (sm86, jax 0.11.2),
Pallas takes 2.54 ms against 17.40 for up and 2.37 against 16.97 for
down, with the same forward error. The sm75 T4 cannot compile Triton
kernels. No sm90 or sm120 card was available. Those generations use XLA.

For `expert_projection` alone, the inputs are 8192 rows, 768 to 2048,
8 experts. Forward plus backward takes 26.21 ms in XLA and 3.38 ms in
Pallas on the L4; on the RTX 4080, it takes 14.84 and 1.86 ms.
Pallas's errors match or improve on XLA against a float64 oracle of
rounded operands (kernel gradient 4.2e-6 against 6.8e-6 relative).
The L4 step is 2.82x faster. JAX's stock Pallas lowering with an
out-sharding fix gave 1.97x on the same step. Its tangents run in fp32,
while Dew's backward multiplies the bf16 cotangent.

A pure-JAX loop of dense per-tile products was rejected. On the RTX
4080, the projection alone was 2.2x faster than XLA (6.67 ms), but
step temporaries doubled to 4.22 GiB against 2.17 at batch 1. On the
v6e, it was 2.2x slower than XLA (1.71 ms against 0.79).

The Pallas kernels are JAX's `gmm` and `tgmm` from the jax-v0.11.2 source
tree. They are vendored because no wheel ships them, and use a custom
VJP. JAX 0.11.2 deprecates their Pallas Triton backend and warns at
every lowering. They remain the sm80 to sm89 path because JAX's Mosaic
GPU grouped matmul (`pallas/ops/gpu/ragged_dot_mgpu.py`) uses wgmma and
fails to compile on the RTX 4080. Tokamax's sm80 Mosaic config also
exceeds Ada's shared memory. Dew leaves the warning visible; you can
filter it. Moving to Mosaic GPU on sm90 and later remains open until
Hopper hardware is available for measurements.

On a mesh, the kernels run inside `shard_map` over each device's share
of sorted rows. Parity is checked on an 8-device CPU mesh; this path
was not measured on multiple GPUs.

On TPU, tokamax's `mosaic_tpu_v2` is within 1% of XLA step time. Its
default dispatch selects the v1 kernel, which is 13x slower, so Dew
selects the kernel explicitly.

### bf16 Adam state: `OptimConfig.state_dtype`

| device | measurement | fp32 state | bf16 state, hash rounding | bf16 state, threefry rounding |
|---|---|---|---|---|
| L4 | one AdamW update, lm-dense tree | 49.10 ms | 37.20 ms | 51.54 ms |
| v6e | one AdamW update, lm-dense tree | 11.45 ms | 8.89 ms | 20.07 ms |
| L4 | lm-dense step | 138.72 ms, 7.34 GiB | 126.45 ms, 5.89 GiB | |
| L4 | lm-moe step | 224.90 ms, 8.15 GiB | 218.15 ms, 6.92 GiB | |
| v6e | lm-dense step | 123.85 ms, 5.77 GiB | 124.94 ms, 4.50 GiB | |
| v6e | lm-moe step | 74.68 ms, 5.05 GiB | 71.96 ms, 3.87 GiB | |

The two L4 step rows use jax 0.11.2 from one Colab session on
2026-09-22, 19:55 to 20:13 CDT. Update rows and v6e rows use jax
0.11.1. Rounding noise comes from a counter hash of step, leaf and
element index. Threefry noise (`jax.random.bits`) makes updates slower
than fp32 state on both devices. Bf16 state saves memory everywhere,
but slows the v6e lm-dense step by 0.9%. The option stays off by
default.

### The vocabulary head: the compute dtype's product

The head product uses the compute dtype, matching torch autocast and
MaxText (`logits_dot_in_fp32=False`). With bf16 compute, both operands
multiply as bf16 with fp32 accumulation, in the model head and chunked
loss alike. Softmax and loss stay fp32. An fp32 model keeps its fp32
head. The table measures the chunked head's forward plus backward alone,
with 8 x 1024 tokens, 1024 features and vocabulary 50304:

| device | fp32 operands (before) | bf16 operands, with argmax | bf16 operands, no argmax | fused linear cross entropy (Pallas port of Liger) |
|---|---|---|---|---|
| L4 | 206.16 ms | 134.02 ms | 133.91 ms | 142.63 ms |
| v6e | 8.04 ms | 8.03 ms | 7.26 ms | not run |

On the v6e, fp32 operands already multiplied in one bf16 pass. Only
skipping argmax (`token_accuracy=False`) reduces head time. On the L4,
argmax fuses into the head kernels. The bf16 product reduced the L4
lm-dense step from 138.72 ms to 129.36 ms. On the RTX 4080 (jax
0.11.2), the head at 4 x 1024 tokens fell from 45.25 ms to 28.00 ms.

The bf16 product changes loss by less than its own rerun spread.
`tools/lm_step_parity.py` ran the 39M-parameter decoder for 100 steps on
the RTX 4080, twice each way. Two fp32-head runs differ by at most
2.3e-4 relative at any step; two bf16-head runs differ by 7.7e-4.
Comparing bf16-head with fp32-head runs gives 3.4e-4 and 7.2e-4,
within the bf16 head's rerun spread. Final losses are 0.0078378 and
0.0078376 for fp32, and 0.0078368 and 0.0078387 for bf16.

The fused Pallas kernel was rejected because it is 6% slower than the
chunked head on the L4. Tokamax's `mosaic_tpu` head was also rejected:
it is 2.24x slower on the v6e (kernel catalog, 2026-09-22).

On an A100, the fp32 head took 38 ms a step in reference runs, 21%
of a Qwen3-0.6B bf16 fine-tune's busy time. It used TF32 GEMMs where
torch autocast uses bf16. That result and the rows above made the bf16
product the default.

The logits' rounding, 2026-10-01. The bf16 product above kept fp32 logits
and used their fp32 gradient in two bf16 state-gradient products: a high
half and the remainder (`347238c7`). Torch autocast and MaxText round
both logits and gradient to bf16. At default `matmul_precision`, Dew now
does the same, rounding the gradient once for both backward products.

On the RTX 4080, `tools/benchmark_step.py --fixed-batch` compared the change
with `66784383` in one session. The 3-layer decoder (GPT-2 small widths,
vocabulary 50304, 16 x 512 tokens) takes 52.17 ms against 59.95, with
a planned peak of 4.53 GB against 5.36. torch.compile takes 49.4-50.0.
Qwen3-0.6B's widths at 1 x 1024 take 106.62 ms against 110.04.

Quality tests ran 2000 steps on wikitext-103 Qwen3 tokens on the same
card. Validation uses 64 fixed windows, scored with fp32 logits for both
roundings. The 3-layer decoder trained from scratch at vocabulary 151936
ends at 5.0604 and 5.0491 with fp32 logits (seeds 0 and 1), against
5.0604 and 5.0493. Fine-tuned Qwen3-0.6B ends at 2.71965 and 2.71996,
against 2.71983 and 2.71981. At the same seed, roundings differ by at
most 4.5e-4 and 1.6e-3 at any checkpoint. The two seeds of either
rounding differ by 1.3e-2 and 2.9e-3 on average.

The high half was needed for layout parity. With the gradient rounded
once, a 4 x RTX 3090 bf16 run reached 1.75 times its bound at a dense
model's final norm, and 5758 times at an MoE's expert gate_proj. The
earlier values were 0.47 and 0.41. The reason for the large MoE gap
is still unknown. When comparing layouts in bf16, set
`matmul_precision="highest"` to keep the head fp32. `tools/layout_parity.py`
does this for bf16 decoders.

On sm80 and sm89, the trainer compiles without XLA's Triton GEMM fusions
unless the run explicitly sets the flag or the model has an SSD mixer
(`TRITON_GEMM_OFF_GENERATIONS`). On an A100 (jax 0.11.2, bf16), the
trainer compiled Qwen3-0.6B at 4 x 512 tokens in 27.1 s against 47.5 s,
with no Triton GEMM autotuning. Step time was 161.4 ms against 161.5.
Standalone Qwen3-0.6B at 4 x 1024 tokens fell from 162.1 to 153.1 ms,
a 99M MoE from 74.6 to 69.4 ms, and a DiT ran 5.8% faster. Mamba-2
was 7.7% slower (127.9 to 138.6 ms) because its SSD scan's small batched
dots benefit from the fusions.

On the RTX 4080, two-layer steps at 2048, 4080 and 16384 tokens fall
from 56.7 to 53.5, 103.8 to 94.0, and 420.5 to 406.3 ms. Tiled heads
run 3-6% faster. With Qwen3-0.6B's widths, two layers, bf16, vocabulary
151936 and a 0.9 allocator fraction, the RTX 4080 run at JAX 0.11.2
avoids a sharp slowdown at 4096 tokens. Each training step takes 286.0 ms
with fusions and 93.4 ms without.

At other shapes, the unfused step can need more temporary memory. If
it does not fit, the trainer tries XLA's default options before tiling
the head or recomputing blocks. At 8192 tokens, only that whole-logits
step fits (178.7 ms, versus 211.2 ms after tiling). When tiling is needed,
sm89 uses the measured 4096-by-8192 tile. These times measure two layers;
they are not full-model times.

On every GPU, the trainer also compiles its step without XLA's dot
merger. It sets `--xla_gpu_dot_merger_threshold_mb=0` in the step's own
compiler options, `step_compiler_options`. The exceptions are runs that
set this flag, or steps training beside frozen weights on 128 tokens
or fewer per device; see below.

The merger combines dots sharing an input (q, k and v; gate and up)
into one GEMM. It concatenates weights afresh every step, costing 4.0 ms
of Qwen3-0.6B's step at 1 x 1024. On the RTX 4080 (benchmark_step, two
rounds, one session), Qwen3-0.6B's widths at 1 x 1024 take 97.7-98.0 ms
against 94.1-94.2 without the merger. The 3-layer decoder takes 50.8-50.9
against 49.1-49.2, SimpleDiT-B 73.1-73.2 against 72.8, and the 176M
hybrid DiT 66.7-67.6 against 66.4-66.7. Peaks are unchanged.

On an A100 40 GB (Colab, `db1761fd`), benchmark_step reused one device
batch for two rounds. It compared XLA's merger explicitly enabled with
`--xla_gpu_dot_merger_threshold_mb=64` against the step's 0. Qwen3-0.6B's
widths at 4 x 1024 take 128.40-128.41 ms against 123.85-124.01. The
3-layer decoder at 16 x 512 takes 22.96-23.01 against 22.12-22.51, and
SimpleDiT-B at batch 32 takes 38.47-38.51 against 38.39-38.48.

Quality tests ran 2000 steps on wikitext-103 with two seeds each.
At the same seed, validation loss changed by at most 1.5e-3 for a
3-layer decoder trained from scratch, whose seeds differ by 1.2e-2 on
average. For fine-tuned Qwen3-0.6B, it changed by at most 2.2e-3, with
seeds 3.0e-3 apart. Serving keeps the merger. Qwen3-0.6B decoding at
32 slots was 4.6-14.8% slower without it; separate launches for the
32-token GEMMs cost more than concatenation.

Small training steps, 2026-10-02. These medians compare `Trainer.compile`
with and without the option on an RTX 4080, bf16, in one process.
Each used five alternating blocks of 20 steps. Full-weight training
favors separate dots from 32 tokens up. LoRA at rank 16 on Qwen3-0.6B's
seven projections, with the base frozen, is up to 0.6 ms slower with
separate dots at 128 tokens or fewer. The exception is 1 x 128,
measured in two sessions.

The rule therefore considers both token count and frozen weights. A
step with an objective's `trainable` split, as in LoRA, keeps the merger
at 128 tokens or fewer per device. Every other step uses separate dots.
Only LM objectives specify the token count, so another objective's frozen
step uses separate dots and has not been measured.

| model | tokens | merged ms | apart ms | change |
|---|---|---|---|---|
| Qwen3-0.6B widths, full | 1 x 32 | 46.83 | 44.93 | -4.1% |
| Qwen3-0.6B widths, full | 1 x 64 | 47.81 | 45.89 | -4.0% |
| Qwen3-0.6B widths, full | 1 x 128 | 49.86 | 47.98 | -3.8% |
| Qwen3-0.6B widths, full | 1 x 512 | 67.33 | 64.78 | -3.8% |
| 3-layer decoder, full | 1 x 32 | 5.01 | 4.89 | -2.4% |
| 3-layer decoder, full | 1 x 64 | 5.12 | 5.01 | -2.1% |
| 3-layer decoder, full | 1 x 128 | 5.37 | 5.27 | -1.9% |
| 3-layer decoder, full | 4 x 256 | 10.06 | 10.06 | +0.0% |
| Qwen3-0.6B, LoRA | 1 x 32 | 15.92 | 16.48 | +3.5% |
| Qwen3-0.6B, LoRA | 2 x 32 | 16.50 | 17.09 | +3.6% |
| Qwen3-0.6B, LoRA | 1 x 48 | 16.34 | 16.82 | +2.9% |
| Qwen3-0.6B, LoRA | 1 x 64 | 16.61 | 17.15 | +3.3% |
| Qwen3-0.6B, LoRA | 2 x 64 | 18.59 | 18.89 | +1.6% |
| Qwen3-0.6B, LoRA | 4 x 32 | 18.64 | 18.99 | +1.9% |
| Qwen3-0.6B, LoRA | 1 x 128 | 20.79 | 19.18 | -7.7% |
| Qwen3-0.6B, LoRA | 8 x 32 | 24.02 | 22.97 | -4.4% |
| Qwen3-0.6B, LoRA | 1 x 256 | 24.56 | 23.41 | -4.7% |
| Qwen3-0.6B, LoRA | 1 x 512 | 36.30 | 34.99 | -3.6% |
| Qwen3-0.6B, LoRA | 1 x 1024 | 60.61 | 59.19 | -2.3% |

A second session at `perf/merger-frozen` used the same measurements to
compare the rule with XLA's default, which keeps the merger on. Host
load averaged 15-27 from other work. No row is slower. LoRA steps at
128 tokens or fewer compile identical programs both ways; their -0.3%
to +0.3% spread comes from measurement noise.

| model | tokens | XLA's default ms | as shipped ms | change |
|---|---|---|---|---|
| Qwen3-0.6B, LoRA | 1 x 32 | 16.08 | 16.04 | -0.3% |
| Qwen3-0.6B, LoRA | 2 x 32 | 16.66 | 16.62 | -0.2% |
| Qwen3-0.6B, LoRA | 1 x 64 | 16.49 | 16.50 | +0.1% |
| Qwen3-0.6B, LoRA | 2 x 64 | 18.57 | 18.63 | +0.3% |
| Qwen3-0.6B, LoRA | 4 x 32 | 18.56 | 18.56 | +0.0% |
| Qwen3-0.6B, LoRA | 1 x 128 | 20.81 | 20.82 | +0.1% |
| Qwen3-0.6B, LoRA | 8 x 32 | 24.25 | 23.31 | -3.9% |
| Qwen3-0.6B, LoRA | 1 x 256 | 24.62 | 23.45 | -4.7% |
| Qwen3-0.6B, LoRA | 1 x 1024 | 60.62 | 59.38 | -2.0% |
| Qwen3-0.6B widths, full | 1 x 32 | 46.88 | 44.94 | -4.1% |
| Qwen3-0.6B widths, full | 1 x 128 | 49.83 | 47.97 | -3.7% |
| Qwen3-0.6B widths, full | 1 x 512 | 67.38 | 64.80 | -3.8% |
| 3-layer decoder, full | 1 x 32 | 5.04 | 4.91 | -2.6% |
| 3-layer decoder, full | 1 x 64 | 5.14 | 5.02 | -2.3% |
| 3-layer decoder, full | 1 x 128 | 5.33 | 5.23 | -1.8% |
| 3-layer decoder, full | 4 x 256 | 10.02 | 10.00 | -0.3% |

At 128 tokens, shape changes which option is faster. Two more sessions
gave 20.65 and 20.65 ms merged against 18.98 and 19.11 separate for
1 x 128. With 2 x 64, times were 18.72 and 18.63 against 18.96 and
19.03; with 4 x 32, they were 18.56 and 18.50 against 18.91 and 18.81.
For 1 x 96, times were 18.22 and 18.22 against 18.41 and 18.43; for
1 x 160, 22.33 and 21.95 against 20.57 and 20.37. Setting the boundary
below 128 would make 2 x 64 and 4 x 32 slower than XLA's default by
1.3-2.1%. The rule includes 128, leaving 1 x 128 at XLA's default,
1.6 ms behind separate dots.

### The forward's bf16 weights: `NARROW_COPY_GENERATIONS`, 2026-10-03

With bf16 compute over fp32 parameters, forward used a separate CUDA
kernel to cast each weight at every step. On Qwen3-0.6B at 1 x 1024
on the RTX 4080, those casts took 5.7 ms per step. Backward widened each weight's
bf16 gradient to fp32, taking 1.8 ms.

On `sm80` and `sm89`, the update now writes a bf16 copy of each such
weight from its new fp32 value (`TrainState.compute`). Forward reads
the copy; gradients reach the update in bf16 and widen when it reads
them (`dew.training.narrow`). A weight is copied only if its sole use
in the loss is that cast. A tied embedding table keeps fp32 reads
because its two cotangents must sum in fp32. Norm scales that do not
use a lone cast, and weights read by custom VJPs, also keep fp32 reads.
The table compares the step with its parent on an RTX 4080, in ms,
over two alternating rounds:

| row | before | copies | peak GiB |
|---|---:|---:|---:|
| Qwen3-0.6B, 1 x 1024, AdamW (dew_lm) | 96.03 / 96.09 | 90.88 / 90.71 | 9.87 -> 9.86 |
| Qwen3-0.6B, 2 x 1024 | 147.94 / 148.25 | 142.44 / 142.32 | 11.94 -> 11.94 |
| 176M hybrid DiT, batch 16 | 60.11 / 60.13 | 57.86 / 57.74 | 5.02 -> 5.03 |
| 176M hybrid DiT, batch 32 | 100.81 / 100.72 | 99.43 / 99.22 | 6.94 -> 6.96 |
| SimpleDiT 768, batch 32 | 72.68 / 72.69 | 71.46 / 71.56 | 4.99 -> 5.01 |
| SimpleDiT 384, batch 16 | 7.38 / 7.41 | 7.21 / 7.27 | 0.67 -> 0.67 |
| decoder, 3 layers, 16 x 512 | 49.08 / 49.12 | 48.79 / 48.80 | 3.97 -> 3.98 |
| 99M MoE, 8 x 1024 | 78.13 / 78.09 | 78.06 / 78.03 | 9.56 -> 9.56 |

No row changed memory fallback in `fit`. The MoE's grouped matmul reads
expert weights differently and gains nothing. On an A100 40 GB, copies
were off and on in one session (Colab, integration `6a220e31`, two
alternating rounds, ms):

| row | before | copies | peak GiB |
|---|---:|---:|---:|
| Qwen3-0.6B, 4 x 1024, AdamW (dew_lm) | 128.45 / 128.37 | 125.88 / 125.93 | 16.08 -> 16.08 |
| 176M hybrid DiT, batch 16 | 45.71 / 45.01 | 40.79 / 41.07 | 5.77 -> 5.78 |
| 176M hybrid DiT, batch 32 | 63.96 / 63.96 | 62.66 / 62.89 | 8.43 -> 8.45 |
| SimpleDiT 768, batch 32 | 39.26 / 39.30 | 38.24 / 38.24 | 6.12 -> 6.13 |

Forward and gradients are bitwise equal to the casts. Under plain SGD,
every parameter stays bitwise equal after three steps (tests/test_narrow.py).
Update arithmetic rounds differently because moving the widening into
the update changes contraction of fp32 multiply-adds. Adam's second
moment agrees within 1 ulp after two steps; all accumulation stays fp32.

A deterministic Qwen3-0.6B run (dew_lm, AdamW and its clip) is bitwise
equal for 13 steps. At step 40, loss differs by at most 2.7e-3, while
two default runs of the parent differ by 3.9e-3. Tests ran
`tools/lm_step_parity.py`'s decoder for 100 steps and the hybrid DiT on
one batch for 300 steps, twice each way. All four DiT runs are bitwise
equal at every step. Both decoder runs with copies equal one parent
run at every step; the parent's two runs differ by at most 6.9e-4.

The A100 runs are not repeatable. Decoder runs with and without copies
differ by at most 1.13e-3 at any step, within the pairs' differences
of 8.3e-4 and 1.13e-3. DiT runs differ by at most 2.26e-2 across
versions, against 2.23e-2 and 1.53e-2 within the pairs. Final losses
are 0.36587 and 0.36163 without copies, and 0.36583 and 0.36631 with
copies.

The TPU already fuses the cast into matmul, so copies would add writes.
At Qwen3-0.6B's widths, 8 x 1024 compiled for a v6e, they write
85.8 GiB per step against 79.8 and need 2.1 GiB more temporaries.

### Generations below sm80

A T4 (sm75) rejects the `BF16_BF16_F32` dot algorithm at run time:
"UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32".
CuDNN fused attention also refuses bf16 there ("SDPA FP16/BF16 requires SM80"),
and Triton does not compile for it. `dew.nn.kernels.generation.bf16_dot_runs`
checks support. Below sm80, bf16 attention uses the reference path for
`auto` and `xla`. Bf16 operand precision preserves the caller's precision,
and grouped matmul uses XLA.

### Faster kernels not adopted

This kernel matrix reports forward plus backward medians on 2026-09-22,
with jax 0.11.2. Every cell is checked against float64. Each option
needs a kernel or dependency not yet included in Dew:

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

On the RTX 4080, the Triton kernel was 6x to 12x slower than XLA where
it compiled. At chunk 64, width 32, it took 1.39 against 0.22 ms;
at batch 8 and 16 heads, 22.7 against 2.2 ms. Every chunk of 128 or
256 requested 131 to 590 KB of shared memory. The scan uses the
kernel only on TPU.

### Packed sliding-window attention on GPU: `local_attention`

No fused-kernel flag supports packed sliding-window attention on pre-Hopper
GPUs. `jax.nn.dot_product_attention` accepts no segment ids alongside
`local_window_size`. CuDNN's packed layout (`q_offsets`) raises "Packed layout requires a GPU with at least Hopper architecture"
on sm89. JAX's Pallas GPU `mha` accepts segment ids but no window;
its gradient was also 4-9% off at this shape.

`local_attention` therefore builds a `[W, 2W]` band mask and passes it
to cuDNN as additive bias where supported, or to xla elsewhere. The
table measures forward plus backward on a Colab L4, jax 0.11.2, bf16,
with 16 query heads of 64 over 4 key heads, window 4096 and
5 packed documents:

| tokens | before (band on xla) | after (band on cuDNN) |
|---|---|---|
| 2048, window 512 | 6.15 ms, 0.32 GiB | 1.37 ms, 0.05 GiB |
| 8192 | out of memory (10.0 GiB requested) | 40.0 ms, 0.24 GiB |
| 16384 | out of memory | 76.9 ms, 0.60 GiB |
| 32768 | out of memory | 156.5 ms, 1.19 GiB |
| 65536 | out of memory | 316.7 ms, 2.38 GiB |

Against a float64 oracle at 2048 tokens, output error is 2.7e-3 relative
and gradient errors are 3.3e-3 to 6.6e-3, matching xla. A dense
`[S, S]` document mask on cuDNN is faster at 32768 tokens on the
RTX 4080 (63.7 against 78.1 ms). However, its memory grows with the
square of length and it ran out at 65536. Dew uses the band mask.

## Expert parallelism on 4x RTX 3090, 2026-09-23

The machine has one host and four RTX 3090s. GPU0 and GPU1 connect
through NVLink (NV4). GPU2 and GPU3 share a PCIe host bridge. Every
other pair crosses the two sockets. The run uses jax 0.11.2, bf16
compute and Pallas grouped matmul, with `tools/benchmark_step.py` timing
12 steps after 3 warmup and tracing 3.

`benchmark_step` reads each row's trace to attribute ms per device per
step to compute kernels, each collective and communication without
overlapping compute. The model is a `causal_transformer` with 8 layers,
width 1024, 16 heads and vocabulary 50304. Each layer has 32 experts
of width 1024 and top-4 routing. There are 4096 tokens per device:
batch 16 of 1024 on four GPUs, or 8 on two.

JAX collectives move 128 MB of bf16 per device in these link tests.
`all_to_all` reaches 33.4 GB/s over NVLink, 6.9 over the PCIe pair and
6.7 across sockets. In the same order, `all_gather` reaches 31.0, 5.8
and 5.8, and `psum` reaches 33.4, 5.7 and 6.3. PCIe is no faster
than a cross-socket pair, leaving GPU0 and GPU1 as the only fast pair
on this box.

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

- Exchange beats global dispatch on slow links: 1.46x on four GPUs,
  1.71x on the PCIe pair and 1.75x across sockets. They tie on NVLink
  because global dispatch's expert all-gather and gradient reduce-scatter
  cost 92 ms there, against 622 ms on the PCIe pair.
- Most communication does not overlap compute. Four-way exchange spends
  272 ms computing and 352 ms waiting on collectives, mostly all-to-all.
  Capacity 1.25 bounds buckets and drops later rounds, reducing the step
  to 550.6 ms.
- With data x expert, the data-axis gradient all-reduce moves more bytes
  than token exchange. Putting the expert axis across sockets and data
  on the pairs is faster: 702.9 ms against 753.8. With expert x fsdp,
  placements tie (788.1 against 800.3) because fsdp's all-gather and
  reduce-scatter trade places with all-to-all.

Each change was measured before and after in one hold:

- Expert parameters enter the dispatch's `shard_map` as stored shards and
  are gathered inside. Their gradient is reduce-scattered rather than
  all-reduced whole. `fsdp 4` falls from 1157.1 to 904.0 ms as a
  614 ms all-reduce becomes a 375 ms reduce-scatter. Expert 2 x fsdp 2
  falls from 856.0 to 788.1 ms.
- The first exchange round runs outside the checkpointed scan of later
  rounds. Backward keeps its intermediates without recomputing them.
  Expert 4 falls from 705.9 to 634.0 ms, and compute from 307.0 to 272.4.
- Exchange gathers bucket rows through the sort index and scatters
  returned rows directly to their slots, saving two row copies per round.
  On NVLink, dropless falls from 300.5 to 295.8 ms and capacity 1.25
  from 218.5 to 212.3. Peak memory falls from 14.1 to 13.7 GiB and
  from 12.6 to 12.0.

One bf16 `ExpertMLP` layer under `MeshSpec(fsdp=2)` on NVLink uses
8192 tokens, 32 experts and top 4. Forward plus backward with Pallas
inside the dispatch map takes 26.2 ms, against 25.1 in the earlier
row map. `jax.lax.ragged_dot` inside the map takes 271.9 ms and 6.5 GiB
of temporaries because XLA computes a product over every expert. Outside
any map, the global path ran out of memory on the 24 GiB cards.
The earlier kernel selector sent fsdp-only meshes to XLA there.

### Rematerialization: the trainer's ladder

The model's `remat` sets the starting policy. If a compiled step does
not fit in device memory, the trainer moves to the next policy.
For decoders, it tries none, `'minimal'` (MaxText's name for keeping
every projection output), then `'full'`. For diffusion backbones, it
tries `False`, `'dots'` (keeping matmul outputs and the attention forward),
then `'full'`. Each policy uses less memory and takes more time, so
the first that fits is the fastest that can run.

`dew.training.trainer.step_fits` checks allocation space. XLA places GPU
step temporaries in one allocation. These temporaries, outputs that
cannot reuse donated state, and batches `fit` prefetches beside the step's own
batch must fit in one free block on each device. Total free bytes alone
are insufficient. For BFC, the block is the pool's largest free block,
or unused space within a growing pool's limit. Allocators reporting no
pool, such as cuda_async and a TPU's allocator, use free bytes. Each
process checks its devices, and the process pool uses the tightest limit.

Prefetching a batch can leave no block large enough for the temporaries
to return to. In that case, the check reserves room for them twice
(`strands_temporaries`). This applies to cuda_async and XLA's spatially
partitioned pool; `prepare_process` turns off the latter.

A checkpoint records the policy used for training, head tile, remat and
XLA options. A resumed run compiles that policy and moves up only if
it does not fit, reporting the change. A restoring process may have
different free memory from the original process, but selecting a lighter
policy would run a different program. The compiled policy appears as
`remat` in the run's `StepCompiled` record and `tools/benchmark_step.py`'s
rows. This table measures forward, backward and AdamW with bf16 compute
over 10 timed steps, using `tools/benchmark_kernels.py step --remat`:

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

For decoders where all three policies ran, `'minimal'` costs 4-10% over
no recomputation, and `'full'` costs 15-21%. Models that fit run without
either. The L4 rows use jax 0.11.2 on Colab (2026-09-23). The RTX 3090
rows use one GPU from the box.
