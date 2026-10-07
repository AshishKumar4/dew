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

## The performance gate, 2026-10-07

CI checks correctness, and `tools/perf_gate.py` checks speed. It runs one fixed
battery on two Dew trees on one machine, in alternating rounds (base then head,
then head then base). The battery is the training step of a 359.8M dense
decoder and a 321.8M MoE decoder at 4 x 1024 tokens and SimpleDiT-B at batch
32, Qwen3-0.6B served at 32 and 128 slots, causal attention forward and backward
(cudnn and xla, 2048 tokens, 128-wide heads), the Oxford Flowers input pipeline,
and the save and restore of lm-dense's training state. Each row runs in a
process of its own, with the tree's own copy of the tool where it has one. A row
regresses when the head's median is worse than the base's by more than either
tree's spread across rounds (at least 2%), and the two trees' ranges do not
overlap. `report` writes the table and exits 1 on a regression. Every commit
promoted to main that touches `src` passes the gate on a Colab A100 first, and
its table is kept in `tools/measurements/perf_gate/<head>-vs-<base>.md`:

```
python tools/perf_gate.py run --base main=<tree> --head head=<tree> --rounds 3 \
    --model <Qwen3-0.6B snapshot> --flowers <TFDS oxford_flowers102 dir> --out gate.json
python tools/perf_gate.py report gate.json --table gate.md
```

A row that one tree cannot run is reported as not compared, not gated. The
serving tool, for example, needs a newer PRNG API than main had on 2026-09-30.
The 32-slot serving row is host-bound, and its median across rounds spread 33%
on the A100's VM, so a serving sample is the best of five repeats, since noise
only slows it. The gate's CPU rows stay off armada. A small decoder's CPU
training step ran 458-735 ms across six containers on one commit, a 1.6x
spread, and within one container it varied by 1.6-10% between processes, too
wide to gate a regression of a few percent.

The first runs, on Colab A100 40 GB VMs, three rounds each. Against main a week
earlier (`94f7d773`), `6329435e` was faster in every row both trees ran. The
step rows of the week-old tree ran its own, older `benchmark_step.py`, so some of
that may be the tool. The checkpoint rows ran in a separate session:

| row | `94f7d773` | `6329435e` | change |
|---|---:|---:|---:|
| step lm-dense 4 x 1024 (ms) | 82.58-82.87 | 71.55-71.92 | -13.4% |
| step lm-moe 4 x 1024 (ms) | 57.15-57.18 | 51.68-51.70 | -9.6% |
| step SimpleDiT-B batch 32 (ms) | 41.25-41.36 | 38.92-38.98 | -5.6% |
| attention cudnn fwd+bwd (ms) | 9.89-10.36 | 10.02-11.24 | level |
| attention xla fwd+bwd (ms) | 36.67-36.69 | 36.73-36.87 | level |
| checkpoint save (ms, median) | 10518 | 6213 | -40.9% |
| checkpoint restore (ms, median) | 3504 | 2638 | -24.7% |

`85c07cfb`, the next promotion, against `6329435e` in one session, was level in
every row:

| row | `6329435e` | `85c07cfb` |
|---|---:|---:|
| step lm-dense 4 x 1024 (ms) | 71.83-72.02 | 71.76-71.94 |
| step lm-moe 4 x 1024 (ms) | 51.76-51.82 | 51.67-51.75 |
| step SimpleDiT-B batch 32 (ms) | 39.03-39.08 | 38.95-39.12 |
| serve 128 slots (tokens/s, median of 3) | 12858-14327 | 14194-14423 |
| attention cudnn fwd+bwd (ms) | 11.05-11.58 | 10.39-11.59 |
| attention xla fwd+bwd (ms) | 36.85-37.20 | 36.83-37.42 |
| checkpoint save (ms) | 5812-6402 | 6024-6394 |
| checkpoint restore (ms) | 2582-2618 | 2613-2720 |

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

RTX 4080 16 GiB, Dew at `6464cb95` (2026-10-05, jax 0.11.2.post3), torch
2.13.0+cu130, transformers 5.17.0, SDPA attention. The torch rows come from
the 2026-10-03 session; neither torch nor its models changed since. Each Dew
row gives the range over two processes. Each torch row is one process, or
two where it gives a range. Dew runs as installed, with cuDNN attention;
where a row names tokamax, tokamax was installed as well
(docs/installation.md):

| model | step | Dew | best torch.compile | Dew / torch |
|---|---|---:|---:|---:|
| Qwen3-0.6B, pretrained | 1 x 1024 tokens, AdamW | 90.65-90.69 ms, MFU 49.6% | 112.1-112.4 ms, 40.0% | 1.24 |
| Qwen3-0.6B, pretrained | 2 x 1024 tokens, AdamW | 142.6-142.8 ms, MFU 63.0-63.1% | 168.6-169.1 ms, 53.3% | 1.18 |
| 99M Qwen3-MoE shape, 8 experts, top 2 | 8 x 1024 tokens, AdamW | 78.4 ms (98.0 where the fit check tiled the head) | 112.5 ms | 1.15-1.43 |
| decoder, GPT-2 small widths, 3 layers | 16 x 512, Adam, EMA | 48.80-48.82 ms; tokamax 48.12-48.14 | 49.4 ms (flash), 50.0 (cuDNN) | 1.01; 1.03 |
| SimpleDiT, width 384, 6 layers, 64 px | batch 16, Adam, EMA | 7.21-7.28 ms; tokamax 7.19-7.20 | 8.07 ms | 1.11-1.12; 1.12 |
| SimpleDiT, width 768, 12 layers, 64 px | batch 32, Adam, EMA | 71.23-71.28 ms, MFU 63.6%; tokamax 69.82-69.92 | 76.4 ms (flash) | 1.07; 1.09 |
| 176M hybrid DiT (published config) | batch 16, Adam, EMA | 57.76-57.87 ms, MFU 46.2%; tokamax 57.58-57.69 | no torch port | |
| 176M hybrid DiT (published config) | batch 32 | 99.28-99.59 ms, MFU 53.6-53.8%; tokamax 98.69-99.28 | no torch port | |

From `f6047cf9` to `6464cb95`, most of the gain is the update writing each
weight's bf16 copy ("The forward's bf16 weights" below). Qwen3-0.6B went from
96.0 to 90.7 ms at 1 x 1024 and from 148.1 to 142.7 at 2 x 1024, and the
hybrid DiT from 60.1 to 57.8 ms at batch 16. The two MoE processes
disagreed. In one, the fit check judged that the whole logits would not fit
and tiled the head (98.0 ms, 1.15x torch); the other kept the whole logits
(1.43x), as the earlier session did. The difference is where the step came
from: the first process compiled it, and its growing pool kept the
autotuner's scratch memory, while the second loaded it from the compilation
cache ("Rematerialization: the trainer's ladder" below).

The Dew decoder and SimpleDiT rows take a fresh batch from the host every
step, and their times match the fixed-batch rows. "Comparison with PyTorch"
below runs both frameworks both ways and gives the commands. At `f6047cf9`,
with Qwen3-0.6B at 1 x 1024 both devices stayed busy (torch for 110.3 ms of
its 112.1 ms step in the second process, Dew for 96.0 of 96.2). On the MoE,
torch idles 20.9 ms a step on the host, and on device time alone Dew was
1.26x faster (78.1 against 98.3 ms busy).

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

Where the rest of the v6e step goes, profiled at integration `0773af76`
(Colab v6e, five steps traced after ten warm ones, both frameworks on the same
VM): Dew's Qwen3-0.6B step at 8 x 1024 took 144.9 ms against MaxText's 157.3
(median 149.7), and Qwen3-1.7B's at 4 x 1024 153.0 against 161.6 (median
154.2). Dew's kernels, attributed to the model through the optimized HLO's
source lines:

- The AdamW update reads and writes the fp32 weights and both moments, and
  it is bound by memory: 12.1 ms of the 0.6B step (8%), and 37.6 ms of the
  1.7B step (25%), whose batch of 4 does little else. MaxText's update fusions
  take the same.
- The 151936-word head and its loss take about 27 ms in both frameworks.
- The rotary embedding ran as separate passes: `rotate_half`'s slice and
  negate, the fp32 converts around it, and the split, about 18 ms a step
  against MaxText's 1.5. `dew.nn.rope.apply_rotary` now rotates the halves
  as `x1 cos - x2 sin` and `x2 cos + x1 sin` without the rotated copy.
  Two rounds alternating at integration `a906f011` put Qwen3-0.6B at 140.5-140.7
  ms against 145.5-145.6 (MFU 27.2% against 26.3%), and Qwen3-1.7B at
  150.0 against 153.6-153.9 (32.8% against 32.0%). The 1.7B losses and
  gradient norms were bitwise the same over 40 steps. The 0.6B losses parted
  by up to 1.7e-3, from rounding, which bf16 training then carries forward;
  each run repeated its own losses bitwise. Against float64 the two forms
  are within tests/reference_error.py's rule of each other, forward and
  backward (`test_the_rotary_rotates_the_halves_as_rotate_half_does`). On an
  RTX 4080 the two forms take the same time.
- The splash attention wrapper scales the query and transposes the
  operands to head-major and back, about 5.6 ms. Compiled for a v6e without
  the separate scale, the wrapper kept the same four kernels, which are
  splash's layout transposes, so that was left as is.

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

In fp32 that polyphase form made the step slower (109.70 to 113.68 ms).
There cuDNN runs the polyphase convolutions (forward, input gradient and
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

`tools/benchmark_step.py` reads the traced window back itself. Busy time is
the union of every kernel interval on the device's streams. The tool counts
kernels per step and sums kernel time per category, taking the category from
the kernel name. Dew was at `9886c20`, the tree before the cudnn padding
described below.

| architecture | ms/step | device busy | kernels/step | gemm | elementwise | reduce | convert | attention | copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_dit | 7.0 | 100% in steady state | 532 | 3.29 | 0.69 | 1.19 | 0.79 | 0.69 | 0.16 |
| causal_transformer | 88.8 | 100% | 282 | 63.3 | 13.3 | 7.6 | 0.8 | 1.8 | 0.7 |

The trace reports 81.6% busy for the DiT over its five steps, because
starting the profiler puts a 3 ms gap into each of the first two steps.
After that, the interval from one step to the next settles at 6.9 ms, which
equals the kernel time, so once the loop is running the device does not sit
idle between steps.

The DiT's reductions are the bias gradients of every Dense layer and the
norm statistics. The 6-by-64 biases of the q, k and v projections cost three
full passes over the activation gradient per layer, 0.25 ms a step. Its
converts are XLA's own split-K partial sums in fp32 and the casts of the
fp32 parameters to bf16 at each use (0.15 ms of the 0.79). Most of the
decoder's gemm time is the fp32 (TF32) vocabulary head. It runs as two
cutlass `s1688gemm` kernels at 12.9 and 12.5 ms, and the third product as
four Triton tiles of 3.2 ms each. At the 49.5 TFLOP/s TF32 ceiling measured
in `docs/research/benchmark-parity.md`, each product needs at least 12.8 ms.

### Host time per step

The table below shows what the host spends per step on simple_dit, from the
trace's host plane and from timing the dispatch loop while the device was
deliberately left behind. The device step is 6.9 ms.

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

The Python in `Compiled.__call__` costs up to 4.5 us per leaf, and this
state has 396 leaves (more on a mesh). That is why `Trainer.compile` has
returned the jitted step since `de6b22c`. On this card that leaves the wall
time the same and cuts host time by 1.8 ms a step. After the sequence axis
was added, the jitted step is wrapped in the mesh context, and a dispatch
costs 32 us on the i9-12900K with or without that wrapper.

A fresh batch costs 1.5 ms more than a fixed one, because the command buffer
has to be updated for the new buffer addresses. That cost limits how far the
loop runs ahead (7 steps against 27), and on a faster card or a smaller
model it would set the wall clock. The placement itself (0.25 ms) is not the
cause. Neither is freeing the consumed batch, because keeping every batch
alive changes nothing under the default preallocation. Prefetch depths 2, 8
and 32 measure the same. I adopted no fix, because the runtime controls the
addresses.

The alternative, waiting on the device every step, costs 45%. The same
simple_dit loop with `block_until_ready` after each step runs at 10.3 ms
against 7.1, so the trainer's loop does not wait between logging ticks.
Running ahead does not raise the peak allocation (0.823 GiB at 27 steps
ahead against 0.819 in lockstep, and 3.499 against 3.495 GiB for
hierarchical_mmdit).

Correction, 2026-10-01, at `42ddfc14` (jax 0.11.2.post3, same card). The
three "loop" rows above time a dispatch loop that runs into the runtime's
limit on executions in flight. Once the device is a few dozen steps behind,
each dispatch waits for a step to finish, so those rows measure the device's
pace. Eight dispatches timed right after a synchronization stay under that
limit. They cost 0.95 ms each on simple_dit with a fixed device batch and
1.38 ms when the main thread also places a fresh batch, and 0.40 and 0.63 ms
on the small decoder. When the fresh batch comes from
`DevicePrefetchIterator`, as it does in `Trainer.fit`, the placement runs on
the iterator's worker thread, and every loop runs at the device's pace:

| loop, 3 repeats of 100 steps (40 on the decoder) | simple_dit ms/step | decoder ms/step |
|---|---:|---:|
| one device batch reused | 7.41-7.46 | 63.54-63.64 |
| a fresh placement each step, `DevicePrefetchIterator` | 7.43-7.50 | 63.55-63.64 |
| `Trainer.fit` over the same host batch, logging every 100 (40) steps | 7.456-7.460 | 63.69-63.73 |

The fit row includes its logging, which waits on the device once an
interval. On the cpu-smoke decoder, with 0.3 ms of device work, the host
sets the step time. There `Trainer.fit` costs 0.51-0.62 ms a step, against
0.31-0.37 for the bare loop and 0.32-0.44 with the prefetch iterator. Timed
one by one, its pieces are the compiled step's dispatch (234 us), the
prefetch iterator's `next` (85 us), the jitted `bookkeep` (22 us), and the
batch's shapes, its row count and the profiler regions (under 1 us each). So
`fit` adds about 0.2 ms of host work a step, and any step that keeps the
device busy for longer than about 0.6 ms hides it.

### Antipattern audit

I audited `src/dew` for nine classes of performance antipattern, measuring
on the small preset. Each row names the cost found and what I did about it.

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

The audit did not measure the `jax_default_matmul_precision` settings, remat
on a step that fits in memory, XLA flags other than command buffers, or the
cost of the class-1 eager scalars. (`bfloat16` matmul precision would change
the numerics of the fp32 head, and the precision rule refuses it anyway.)

Correction, 2026-09-22, checked against current main. The jitted `bookkeep`
from the first class-1 row is on main (`src/dew/training/trainer.py`, called
from `Trainer.fit`). The class-5 row about `objective.py:141` no longer
matches the code. The diffusion objective encodes the unconditional prompt
once, when it is built, and each step only casts that stored encoding to the
batch's dtypes (`DiffusionObjective.blank_conditions` in
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

The week before, the decoder read 0.95x, but that figure compared the parity
benchmark's fixed-batch decoder row (75.70) with torch's 72.18. The dew row
here is the benchmark's own prefetching loop, at 88.78, so the two dew
numbers are 13 ms apart. Of that, 1.9 ms is the chunked head and 3.8 ms is
the decoder's own changes since `6b0f119`. The remaining 7.6 ms is the gap
between the fixed-batch row and this tool's loop at the same commit
(`6b0f119` reruns at 83.26 here on the same day). The DiT went from 1.16x
the week before to 1.19x. On the newer torch, torch's SDPA row is 0.4 ms
slower than the week before.

Correction, 2026-10-01. The 0.82x row set Dew's loop, which places a fresh
batch every step, against torch's run without `--h2d`, which keeps one batch
on the device, and nothing showed that the 7.6 ms was loop overhead. At
`42ddfc14` I compared the two tools like with like: `tools/benchmark_step.py
--fixed-batch` against `tools/benchmark_torch.py` without `--h2d`, and Dew's
default fresh-batch loop against `--h2d` (pinned host memory, copied every
step). The runs used torch 2.13.0+cu130, transformers 5.17.0, `--mode
compile --attention sdpa` and `--warmup 20 --steps 100`, with one process
per row:

| case | Dew, fixed | Dew, fresh | torch.compile, fixed | torch.compile, `--h2d` | Dew against torch, fixed / fresh |
|---|---:|---:|---:|---:|---:|
| causal_transformer, small preset | 63.68 | 63.63 | 49.44 (flash), 50.00 (cudnn) | 49.99 (cudnn) | 0.78x / 0.79x |
| simple_dit, small preset | 7.46 | 7.43 | 8.53 (cudnn) | 8.07 (cudnn) | 1.14x / 1.09x |
| simple_dit, width 768, 12 layers, batch 32 (`--size large`) | 76.09 | 76.11 | 76.40 (flash), 78.18 (cudnn) | 76.51 (flash), 78.38 (cudnn) | 1.00x / 1.01x |

The torch decoder computes its head's product in bf16 (`--head-dtype
bfloat16`, now the twin's default), as Dew's LM objective does; with the
fp32 head the twin used before, torch ran at 72.38 ms. On these rows a fresh
batch costs Dew nothing measurable, and `Trainer.fit` adds nothing on top
(the host-time correction above).

At `42ddfc14` the decoder's 14 ms gap was its vocabulary head. With 8192
tokens, vocabulary 50304 and three layers, the head is most of the step. Dew
kept the logits in fp32 and fed their fp32 cotangent into the state product
as two products, one of a bf16 high half and one of the rest; torch rounds
both the logits and the cotangent to bf16. Three changes since then took the
decoder to 52.2 ms against torch's 49.4 (the scoreboard above): the
log-sum-exp and argmax in one pass, one bf16 copy of the cotangent for both
products, and torch's rounding of the logits and their cotangent at the
default precision.

I traced both at `42ddfc14`, XProf with command buffers off against
torch.profiler. In ms per step, the forward's log-sum-exp and argmax read
the fp32 logits in two passes, 5.1 against 1.3 for torch's fused
log-softmax. The backward wrote the logits' cotangent three times in bf16
(the high half, the rest, and the plain rounding for the head's own
gradient), 6.7 against 2.7. Dew's GEMMs took 39.4 against torch's 33.9,
about 3 of it from running the state product twice. Attention is 1.7 against
1.3.

On the large DiT the two frameworks spend the step differently (same
profilers, ms per step). GEMMs take 44.5 against 47.2. The optimizer update
takes 6.9 against 16.2, because XLA fuses Adam and the EMA into one pass
over the state. Attention takes 5.3 against 4.4 for FlashAttention-2;
cuDNN's 5.3 is its forward at 0.9, its backward at 2.9 and its two
pre-Hopper backward helpers at 1.5. Dew's reductions and converts take 16.5
(the bias gradients with the GELU backward 5.4, the norm statistics 1.9),
against 5.8 for torch's elementwise and norm kernels and 8.9 for its copies.

## Attention kernels

### Splash's tiles on the TPU, 2026-10-02

Splash attention took 34 ms of Dew's 156 ms Qwen3-0.6B step at 8 x 1024
tokens on a TPU v6e, and MaxText's attention took 50 ms of its own step;
both ran at about 13% of the chip's peak. The table times the tiles on the
forward plus backward of Qwen3-0.6B's attention (8 x 1024 tokens, 16 query
heads over 8, 128 wide, causal, bf16), as the median of 7 rounds of 10 calls
through `dew.nn.attention.splash_attention` at integration `8e92a4a6`:

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
3.7e-3, dq 5.0e-3, dk 5.1e-3, dv 2.8e-3 of their maximum). So the kernel now
tiles by 1024, narrowed to a divisor of each sequence, and runs its backward
as one fused kernel. The training steps on the v6e (integration `527a32e9`,
three rounds each, one batch kept on the device), in ms:

| step | 512 tiles, dq and dkv apart | 1024 tiles, fused backward |
|---|---:|---:|
| Qwen3-0.6B widths, 8 x 1024 | 150.34-150.43 | 136.77-136.92 |
| Qwen3-0.6B widths, 16 x 1024 (head tiled by the fit ladder) | 320.48-320.54 | 278.63-278.66 |
| 4-layer decoder, 256-wide heads, 8 x 2048 | 80.90-80.95 | 75.36-75.45 |
| 176M hybrid DiT, batch 16 (4 attention blocks of 256 tokens) | 17.24-17.30 | 17.25 |

These rows are `tools/benchmark_step.py`'s step with `optax.adam`. The step
to compare against MaxText is the reference runner's, with AdamW and its
clip ("Training scoreboard" above has 145.3 against MaxText's 154.0 at
8 x 1024). Losses after the 45 steps differ in the third or fourth
significant digit (0.003386 against 0.003389 at 8 x 1024), because the
backward's fp32 sums are reordered and that moves a training run's
trajectory. Each kernel call's errors against fp32 are the same.

### tokamax's attention, 2026-10-02

I compared tokamax's Pallas-Triton flash attention (openxla/tokamax main at
`47d3d663`) with cuDNN on an RTX 4080, in bf16, timing forward plus backward
as medians of 7 rounds of 10 calls. Each call is checked against fp32 XLA at
HIGHEST (`max|err| / max|ref|` for dq and dk):

| shape | cuDNN | tokamax, its heuristic config | tokamax, best of a config grid | JAX's Pallas `mha`, best blocks |
|---|---:|---:|---:|---:|
| 32 x 256, 12 heads of 64 | 0.726 ms | 0.506 | 0.452 | 0.374 |
| 16 x 512 causal, 12 heads of 64 | 0.793 | 0.515 | 0.522 | 0.465 |
| 4 x 1024 causal, 16 heads of 64 | 0.833 | 0.592 | 0.576 | 0.523 |
| 4 x 1024 causal, 16 query heads over 4, of 64 | 0.784 | 0.608 | | 0.574 (keys repeated) |
| 4 x 1024 causal, 16 over 8, of 128 (Qwen3-0.6B) | 1.541 | 1.288 | 1.242 | 1.260 |
| 4 x 1024 causal, window of 256 | 0.579 | 0.587 | 0.535 | no window |

At the 64-wide shapes tokamax's errors are cuDNN's (dq and dk 4.8e-3 to
6.6e-3); at 128 they are not (below). JAX's `mha` reaches 8.5e-3 because its
backward forms `rowsum(o * do)` as a bf16 product, and an fp32 product gives
cuDNN's errors at the same speed.

Tokamax reduces training time at 64-wide heads but not at 128. On
SimpleDiT-B at batch 32, attention goes from 5.62 to 4.33 ms and the step
from 73.1 to 71.5; on the 3-layer decoder, attention goes from 1.82 to
1.24 ms and the step from 50.8 to 50.2. At Qwen3-0.6B's widths and 1 x 1024,
attention goes from 9.13 to 9.24 ms and the step from 97.6 to 98.4. So with
tokamax installed, 'auto' takes it for heads up to 64 wide and for calls
with no window, mask or bias (`dew.nn.attention.triton_runs`), at tokamax's
heuristic config.

At 128-wide heads, 2026-10-02 (Qwen3-0.6B's: 16 query heads over 8, causal,
1024 tokens; RTX 4080, bf16; Dew at `14087252`), I found nothing on this
stack that beats cuDNN with gradients as accurate as its own:

- In the training step at 1 x 1024 (XProf, command buffers on, kernels per
  step), tokamax's forward is faster and its backward slower. The forward
  takes 2.16 ms against cuDNN's 2.56. The backward takes 7.14 ms plus 0.27
  for `rowsum(o * do)`, against cuDNN's 5.07 plus 0.18 for its head
  reduction. The step takes 95.96 against 94.07 ms. torch's FlashAttention-2
  kernels take 2.05 and 4.56.
- tokamax's gradients are less accurate there. dk and dv reach 5.2e-3 and
  4.6e-3 of their maximum at batch 1, against cuDNN's 4.9e-3 and 2.8e-3, and
  6.7e-3 and 4.8e-3 at batch 2, against 4.7e-3 and 3.5e-3. Its best grid
  config (blocks of 32, keeping openxla/tokamax#1494's constraint) has the
  same errors.
- cuDNN runs as fast with BNTH inputs as with BTNH (0.385 against 0.381 ms a
  call at batch 1, 0.721 against 0.716 at batch 2), and cuDNN 9.27.0 as fast
  as 9.25.1. Qwen3-0.6B's widths at 1 x 1024 take 94.20 and 94.08 against
  94.23 and 94.19 ms, with attention at 9.09 against 9.11, and SimpleDiT-B
  and the decoder show the same. `uv pip install` resolves the newest cuDNN
  under 10, so a fresh install gets 9.27.
- JAX's own Pallas-Triton `mha`
  (`jax.experimental.pallas.ops.gpu.attention`, with the keys repeated over
  each group's query heads) is faster per call and ties in the step,
  measured 2026-10-04 at `0527ab19`. Timed over 28 layered calls, one call's
  forward plus backward takes 0.33 against cuDNN's 0.38 ms at 1 x 1024 and
  1.30 against 1.36 at 4 x 1024, and its gradients' RMS distance from
  float64 is 1.00 to 1.04 times cuDNN's. In the training step the two tie
  over two alternating rounds: attention takes 8.49 against 8.52 ms a step
  and the step 90.6 against 90.8 at 1 x 1024, and 15.5 against 15.3 and
  142.4 against 142.9 at 2 x 1024.
- tokamax's main has had no GPU attention change since `47d3d663`, and JAX
  exposes no cuDNN algorithm, workspace or determinism choice for its fused
  attention (non-deterministic is already the default).
- Closing FlashAttention-2's lead of 1.3-1.9 ms a step would take a
  Pallas-Triton backward of Dew's own, on the backend JAX 0.11 deprecates,
  and it would save under 2% on steps where Dew is already faster than
  torch.

At 64-wide heads tokamax's backward trails JAX's `mha` by 10-18% (0.35
against 0.29 ms of the SimpleDiT-B call; the forwards take 0.093 and 0.087),
and no block size, warp count or stage count in tokamax's own grid closes
that.

Its VJP computes wrong gradients for a causal call when `block_m1 >
block_n1` (dk and dv off by 10^2) or `block_n2 > block_m2` (dq off by 0.7),
and its autotuning grid includes such configs. Its heuristic config is not
one of them, so Dew runs the heuristic config and checks each shape it
routes (`tests/test_kernels.py`). At 256-wide heads tokamax's heuristic asks
for more shared memory than sm89 has (102784 of 101376 bytes) and fails.
cuDNN takes no 256-wide head either, so those calls run on XLA.

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

The reference and xla paths materialize the S x S logits. They run out of
16 GiB at S=4096, and wherever they fit they are 3 to 12 times slower than
the fused kernel. cudnn is the kernel to use for a GPU run, forward and
backward, and `'auto'` picks it wherever it can.

### Head dimension 256 through tokamax's Triton flash attention

Before Hopper, cudnn refuses head dimensions above 128. So a Gemma 3 4B or
12B shape (heads of 256) trains through the xla path on every Ampere and Ada
card, and that path materializes the S x S logits. tokamax 0.0.13 ships a
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
tiling for every card (`pallas_triton_vjp.py` contains a `TODO: Implement
heuristics`). At head dimension 256 that tiling asks for 102784 bytes of
shared memory, and the card has 101376, so it fails with
`RESOURCE_EXHAUSTED: Shared memory size limit exceeded`.

I probed other tilings through tokamax's private classes. A 32x32 tiling
with one stage fits and is correct (gradient error 0.031, the same as xla),
and it runs forward and backward in 1.80 ms against xla's 2.66 at S=2048.
Two 16-row tilings compile and run at the same speed, but they return wrong
gradients (error 6.6 on gradients of size 6.3). tokamax's autotuner picks a
tiling by its time on random inputs and never compares numerics, so
autotuning cannot be trusted to find the correct one. At head dimension 128
the Triton kernel ties cudnn (0.235 against 0.236 ms forward, 0.75 against
0.78 forward and backward at S=2048), so it gains nothing where cudnn
already runs.

Two other features are still missing. The first is Gemma 2's logit softcap.
The Triton forward takes it (0.35 against xla's 1.01 ms at S=2048, head
dimension 256), but the VJP raises `NotImplementedError: logits_soft_cap
unsupported`. tokamax also applies the cap after adding the bias, while
Gemma applies it before (1.4e-2 apart on CPU with a bias, identical without
one). The second is attention sinks, which no tokamax implementation takes.

Dew does not route any call to this kernel. A forward-only kernel cannot
serve training, and the one backward tiling that works is reachable only
through private tokamax classes. A route needs an upstream tokamax release
whose VJP picks a tiling that fits the card, or a public tiling setting,
with a correctness check next to it. Installing tokamax 0.0.13 next to Dew
also pins `typeguard==2.13.3`, while tyro 1.0.16 requires
`typeguard>=4.0.0`. That breaks the command line of every recipe, so for
these measurements I ran the tool through its `main` function in a separate
environment.

## Odd sequence lengths on cudnn

cudnn's fused kernel has no backward pass for an odd query or key length.
The forward pass takes any length, so the problem only showed up at the
first training step, as `NotImplementedError: Unsupported sequence length Q
333, KV 333` from jax. CLIP's 77 text tokens are an odd length, and so is
256 + 77 after concatenation.

Until 2026-09-05, `'auto'` sent those shapes to the xla kernel, which
materializes the [B, H, Q, K] logits and their probabilities in fp32 and
keeps them for the backward pass. Now `cudnn_attention` pads an odd length
to an even one. It adds one zero row to the query and slices it off the
output. It also adds one zero key and hides it with the kernel's own padding
mask (`key_value_seq_lengths`), so every real query attends to exactly the
keys it had. On a GPU, `'auto'` picks cudnn at any sequence length, and an
explicit `'cudnn'` also takes any length.

`tests/test_kernels.py::test_cudnn_trains_odd_lengths_and_agrees_with_xla`
checks this at q1024/kv77, q9/kv7 and q333/kv333 causal. The outputs and the
three input gradients agree with the xla kernel to within two bf16 ulps of
their scale. At an even length the two kernels are the same distance apart
(q256: 1.6e-2 at scale 2.9 on the output, 7.8e-2 at scale 15.6 on the
gradients, both one ulp). If the pad key is left unmasked, the q9/kv7 output
moves by 0.26 at scale 2.4 and the test fails. If the pad query row is left
in, the shape changes and the test fails.

The table measures what the padding gains, with `--warmup 3 --steps 50` on
the small preset, comparing `'xla'` (the kernel these shapes ran on before
the padding) with `'auto'`:

| architecture | shapes | xla ms/step | cudnn ms/step | xla peak GiB | cudnn peak GiB | loss at the end, xla / cudnn |
|---|---|---:|---:|---:|---:|---|
| hierarchical_mmdit | q141, q333, q1101 | 33.86 | 20.86 | 3.50 | 1.85 | 0.551035 / 0.551038 |
| simple_mmdit | q333/kv333 | 12.86 | 11.01 | 1.43 | 1.08 | 0.584398 / 0.584407 |
| unet | q256/kv77, q1024/kv77 | 16.30 | 16.13 | 0.78 | 0.71 | 0.597518 / 0.597516 |

On hierarchical_mmdit, the xla attention on the 1101-token stage kept its
fp32 logits and probabilities for the backward pass, and that is where the
1.65 GiB and the 13 ms went. Attention is a small part of the unet's step,
so the unet gains little. The losses are after 103 steps on one fixed batch
and differ in the sixth digit, which is the two kernels' bf16 rounding
compounded by Adam. Decoding asks for one query position at a time, which is
an odd length. It runs on cudnn with the cache mask as an additive bias; I
did not measure its speed.

## Attention metadata and the masked conv, 2026-09-07

Before `14622ba`, passing any `AttentionMetadata` lost the fused kernel,
whatever the metadata said. The mixer built its `[B, 1, S, S]` mask and took
the xla path as soon as any metadata arrived. So a batch that only gave
rotary positions, or one whose validity marked every slot as real, paid for
a mask that excluded nothing. At `14622ba` the mixer checks whether the
metadata restricts anything, meaning key validity, or image groups on a
bidirectional-image layer. A validity array's values are unknown at trace
time, so an all-true array still builds the mask. The host code that used to
emit one now leaves it out when it knows the rows are whole:
`pad_token_rows`, the processor's `from_hf`, generation's input validation,
the rollout collector and episode cohorts, the PPO critic without lengths,
and every MTP depth.

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

The canonical row is the opaque row's batch with the redundant validity left
out, which is what a real unpadded request looks like. Its before column is
that same call measured at `83f08e5`. The compiled HLO has a
`__cudnn$fmhaSoftmax` custom call after the change and none before, so the
route changed and the gain is not clock noise. The opaque and packed rows
are unchanged by design, and their spread across windows covers the
difference.

The peaks the process allocator reports move by up to 20 MiB between
identical runs. The packed forward gave 496.02 and 476.02 MiB on two repeats
of the same executable, whose own `memory_analysis` is byte-identical, so
read the peak column at that resolution.

With canonical metadata the call is exactly the plain call. Its outputs are
bitwise equal to the no-metadata forward, and its parameter gradients are
within 2.4e-06 of it. The opaque all-true mask stays on the xla kernel at
its old cost, because the shape of a validity array does not say that its
contents are all true.

The GDN rows time the whole mixer (projections, gates, rule and norm), and
the masked conv is the only part that changed. With a mask, the mixer is 3.3
times faster forward and 3.0 times faster with the gradient. Its kernel
launches drop 22.8 times forward (10707 to 470 a call) and 19.5 times with
the gradient (36700 to 1884), because the scan's `while` loop is gone from
the HLO and the unmasked path's `__cudnn$convForward` replaces it. On
lengths 2048 and 1537 the outputs agree with row-by-row evaluation to
2.4e-04 (the layer's bound is 5e-4), and the padded row's input gradients
and outputs are exactly zero. Against the token scan on CPU at fp32, the
largest difference over left, right, interior and paused padding at kernels
2, 4 and 8 is 4.8e-07.

Leaving the field out changes the batch's pytree, so every process in a pool
has to agree on it. Only a process itself knows whether its own rows needed
padding, and if one process leaves the field out while another includes it,
the same step gets two different pytrees. So a generation request first
agrees on the signature that ignores validity, then on one fixed-size
presence vector. Every process runs the same collectives in the same order,
whatever rows it holds. If any process includes the field, every process
materializes it; if none does, the field stays out and the call keeps the
fused kernel.

`shard_batch` cannot run that agreement, because placement runs on the
worker thread of `DevicePrefetchIterator` while the step's collectives run
on the caller's thread. So in a pool, every `ModelInputs` of a training
batch that lacks the field gets it materialized. Single-process runs, which
the table measures, are unaffected, and so are batches of plain token
arrays, which have no validity field.

I did not rerun the head-chunk and head-dimension-256 cases, because this
change does not touch them.

## fp32 attention on the CPU, 2026-10-05

On XLA:CPU, the xla path of `jax.nn.dot_product_attention` rounded fp32
attention further from float64 than torch's SDPA does. Sampling Wan, with
its 512 text keys, measured 2.07 times torch's distance. The cause is the
product of the probabilities and the values. YNNPACK, XLA:CPU's dot library
in the pinned jax, sums a contraction of up to about 1024 terms in one
chain, while torch's CPU GEMM and Eigen sum it in shorter blocks. Replacing
each stage in turn with its float64 value isolated the product: an exact
product removed four fifths of the error, while exact logits or
probabilities changed nothing.

An fp32 call on the CPU now takes the reference path whenever it has more
than `VALUE_BLOCK` (256) keys (`_xla_kernel_chains`). That path's
`weighted_values` sums the product over each block of 256 keys separately
and then adds up the blocks. Its gradients are the plain product's, since
neither gradient product sums over keys. Calls with 256 keys or fewer, bf16
calls, and GPU and TPU calls compute exactly as before. The table measures
512 to 8192 keys with cross-attention-like inputs: 128 queries, 2 heads of
width 128, and logits with standard deviation 0.36. Torch's math and flash
backends measured the same distance.

| keys | torch SDPA, RMS from float64 | jax.nn xla (before) | blocks of 256 | blocks of 128 | Eigen dots |
|---:|---:|---:|---:|---:|---:|
| 512 | 1.46e-8 | 1.34x | 0.98x | 0.76x | 0.69x |
| 2048 | 7.62e-9 | 1.75x | 0.96x | 0.74x | 0.74x |
| 8192 | 4.26e-9 | 1.63x | 0.87x | 0.66x | 0.73x |

Times are from one process on two pinned threads of an i9-12900K, with
the paths interleaved. The table gives the minimum of seven runs on a
loaded shared host:

| call (fp32, heads 128 wide) | forward before / after, ms | forward + backward before / after, ms |
|---|---:|---:|
| cross, 2048 queries x 512 keys, 12 heads | 78.1 / 83.1 | 225.7 / 236.6 |
| cross, 4096 queries x 512 keys, 12 heads | 146.9 / 151.8 | 442.2 / 433.7 |
| causal GQA 16/8 heads, 1024 | 105.9 / 115.9 | 372.8 / 383.9 |
| causal GQA 16/8 heads, 2048 | 446.1 / 419.1 | 2075.8 / 1735.6 |

Writing each block's partial product before the sum costs a little in
the forward. This run measured +3% to +9% at 512 to 1024 keys. A second
sweep, on two other pinned threads, measured -5% to +12% on the minimum and
-9% to +5% on the median. Run-to-run noise here is about 10%, so the cost is
near the noise, but I can't show it is zero; I kept the change for the
accuracy. A training step's attention runs within 5% of its old time, and
the 2048-token causal step is 16% faster because the reference path's
backward is faster.

The landing page's live sampler attends over at most 256 keys, so it is
unchanged. Its 17 compiled programs are the same apart from source
locations, its time is the same, and its image has the same sha256 before
and after (`d9ce11b0`).

The sweep covered three block sizes and three ways to sum the blocks. I
kept the cheapest choice that brings Wan's case, 512 keys, within the 2x
rule of `tests/reference_error.py`:

- Blocks of 512 or 1024 keys leave a 512-key call as it was (1.34x torch).
  At 1024, every length rounds as before, because YNNPACK's chain is about
  that long.
- The blocks' sum as the last axis of one product (kept) was faster than
  the same sum over a leading axis and than a loop of per-block products,
  which cost 10-30% more.
- Blocks of 128 keys round closer to float64 (0.66-0.76x torch), but they
  double the partial products, and their forward cost up to 18% more.
- Turning YNNPACK's dots off with
  `--xla_cpu_experimental_ynn_fusion_type=-individual_dot` gives every CPU
  dot Eigen's blocks. On two threads, Eigen ran the 176M hybrid DiT's MLP
  shapes (512 x 768 x 3072) 12% slower. It would also change every CPU
  result, including the live sampler's image.

## XLA flags

`TrainerConfig.xla_flags` appends to `XLA_FLAGS`, and `prepare_process`
applies it before JAX opens a backend. The recipes and the CLI call
`prepare_process` first; a script or notebook that builds a `Trainer` itself
calls it before its first JAX call, or sets `XLA_FLAGS` in the environment.

Separately, `import dew` sets `--xla_allow_excess_precision=false` unless
`XLA_FLAGS` already names that flag
(`dew.telemetry.devices.keep_roundings`). With XLA's default, a fusion may
skip a bf16 rounding the program states, and which roundings it skips
depends on the layout, so one device and four devices computed different
bf16 forwards of the same model. XLA reads its flags when a backend opens,
so import Dew before the first JAX computation. If the backend is already
open, Dew logs a warning and the policy does not take effect; restart with
`XLA_FLAGS=--xla_allow_excess_precision=false` set before importing JAX. On
the RTX 4080 the policy also made steps faster: the 176M hybrid DiT at batch
16 went from 69.60 to 66.62 ms, SimpleDiT-B at batch 32 from 76.00 to 73.03,
and Qwen3-0.6B's widths at 1 x 1024 from 110.52 to 109.62.

The default `xla_flags` is None because of the sweep below. It covers three
architectures, with one fresh process per configuration. Each cell is the
median of the runs, with the range and count where a configuration was
repeated.

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

I adopted no flag, because of the noise. Four repeats of the same
configuration on simple_dit spread from 6.97 to 7.53 ms, or 8%, because each
fresh process autotunes again, and against that spread every simple_dit
number in the table comes from one distribution. The causal_transformer is
the quiet measurement, with a spread of 0.7%, and no flag moves it by more
than 0.2%. The unet is the only architecture where a flag shows an effect;
`--xla_gpu_triton_gemm_any=true` takes its median from 17.38 to 17.05 ms, or
1.9%, over four runs each.

To be adopted, a flag has to be faster on all three architectures and
outside the noise on each. That one gains 2% on the unet, leaves the decoder
unchanged and is lost in simple_dit's noise, so the default stays None. A
run that wants it can pass `--trainer.xla-flags`.

Two of the rows have their own explanations:

- `--xla_gpu_autotune_level=4` changes nothing on any architecture, because
  it is already the default in this build. Level 0 turns autotuning off,
  which removes XLA's compile-time kernel choice as a source of run-to-run
  differences (see [checkpoints](guides/checkpoints.md)). On the 4080 it
  slowed the 176M hybrid DiT's step from 139 to 151 ms and a 67M decoder's
  from 79.8 to 81.5 ms, two fresh processes each.
- `--xla_gpu_enable_command_buffer=` (command buffers off) is the only
  configuration that is reliably slower: 17.90 against 17.38 on the unet
  over four runs, and slower on the other two as well. Command buffers are
  on by default and save 3% on the launch-heavy architecture. Passing a
  longer type list than the default adds nothing to that.

None of the candidate flags changes numerics; the sweep covered only kernel
selection and scheduling. I tested no flag that relaxes precision, and none
would be adopted, because an adopted change has to keep a fixed-seed 20-step
loss trajectory within 1e-5.

On a CPU with two core types, such as the i9-12900K's P-cores and E-cores,
XLA:CPU's float32 bytes can differ from one process to the next
([openxla/xla#50022](https://github.com/openxla/xla/issues/50022)). Some
convolutions and dots reach oneDNN's sgemm. It splits the rows into blocks
by the caches of the core that ran the process's first contraction, and the
result depends on the blocks. A dense layer's kernel gradient over four
tokens, `[4, 2048]^T @ [4, 32]`, ran as one 2048-row block when the first
contraction was on a P-core, and as 1032 + 1016 rows when it was on an
E-core. The two processes disagreed in the last bits, though each one
repeated its own result exactly. No XLA flag pins the blocking:
`--xla_cpu_use_onednn=false`, `--xla_cpu_use_xnnpack=false`,
`--xla_cpu_use_thunk_runtime=false` and
`--xla_cpu_multi_thread_eigen=false` each still give both results.
`ONEDNN_MAX_CPU_ISA=SSE41`, set before JAX starts, does: both orders gave the
same bytes. Timed on one thread of a P-core, two processes per setting:

| dot | default | `ONEDNN_MAX_CPU_ISA=SSE41` |
|---|---|---|
| that gradient | 41-51 µs | 48-64 µs |
| Qwen3-0.6B's MLP up-projection, [512, 1024] @ [1024, 3072] | 28.6-29.4 ms | 29.0-31.8 ms |

The up-projection never reaches oneDNN's gemm, so the variable leaves it
alone, and the differences are within the spread between two processes.
Dew does not set the variable. To compare bytes across processes on such a
machine, set it, or pin every process to one core type (`taskset -c 0-15`
for the 12900K's P-cores).

## UNet batch scaling

These numbers show where the unet, the architecture whose step is least
sensitive to batch, still has room to get faster. I adopted nothing from
them.

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

When first recorded, these rows had a utilisation column that read 1.7%, and
the FLOP counter made it wrong. XLA's `cost_analysis()` cannot see inside
the cuDNN convolution calls the backend emits, so it undercounted this model
22.5 times. Counted from the optimized HLO, the unet runs at 40.5% of peak,
as `docs/benchmarks.md` reports.

## Muon against AdamW at equal tokens

These rows ran on a CPU. They compare optimizers at equal token budgets, not
accelerator speed, and the run is small enough that one workstation CPU does
nine of them in under an hour.

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

Conditions: `causal_transformer`, 128 wide, 2 layers, 2 heads, tied head,
byte vocabulary of 256, sequence length 128, batch 16, 557,952 parameters,
bf16 compute, weight decay 0.1 on both groups, no schedule, no clipping.
2000 steps is 4,096,000 tokens, which is 3.75 passes over the 1,093,086
training tokens of the Shakespeare corpus. 12th Gen i9-12900K, jax 0.11.1,
`JAX_PLATFORMS=cpu`, six cores pinned per run, three runs at a time on
disjoint cores. Every arm sees the same batches in the same order at the
same seed, so a difference between two arms comes from the solver.

There are three arms: `adamw` is AdamW, `muon` is Dew's Muon with its
parameter groups, and `muon-unsplit` is `optax.contrib.muon` with its own
ndim == 2 rule, which is how the 'muon' entry worked before the parameter
groups. Final loss is the mean over the last 50 steps.

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
0.038 nats lower after the same tokens. The three seeds of an arm spread
0.007 to 0.013, so the gap to AdamW is three times that noise. The gap to
unsplit Muon is 0.018, one and a half times the noise, and the split version
is ahead on each of the three seeds, by 0.016, 0.020 and 0.017. Raising the
learning rate from each arm's best to 1e-2 costs Muon 0.028 (3.3 times its
best rate) and AdamW 0.116 (10 times its best rate).

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

The compiled fp8 step contains `f8e4m3fn` converts (146 mentions in the HLO
at width 256, against 12 GPU gemm calls), so the quantization does reach the
device. At these sizes the converts cost more than the gemms save. Nothing
raises an error, and the losses go down (2.44 bf16 against 2.68 fp8 at width
256, 0.009 against 0.011 at width 1024, each after 14 steps from the same
init). So on this card, at these sizes, fp8 runs but gives no speedup to
adopt.

## Serving against vLLM, 2026-10-03

I served Qwen3-0.6B in bf16 on the RTX 4080 with
`tools/benchmark_lm_serving.py`: 256-token prompts, 128 greedy output
tokens, and twice as many requests as slots. The table gives output tokens a
second over three repeats, with two Dew processes a side; vLLM 0.30.0 ran on
2026-10-01:

| slots | vLLM | Dew `1f8d5e72` | Dew, wide cache writes gathered |
|---:|---:|---:|---:|
| 32 | 6049-6053 | 5078-5323 | 5513-5532 |
| 64 | 7614-7618 | 6869-6879 | 7171-7204 |
| 128 | 9037-9050 | 7758-7773 | 8230-8289 |

Later the same day I ran the same benchmark with Dew at `88620c1d` and vLLM
0.30.0 in one session, alternating, two processes each. That Dew revision
has the prefill's cuDNN attention, keeps the head's bf16 logits as bf16 and
writes the cache as words (all described below). The table gives the range
over every repeat. Another lane held the host's load average near 50
throughout:

| slots | vLLM | Dew `88620c1d` | Dew / vLLM, medians |
|---:|---:|---:|---:|
| 32 | 5903-6046 | 5490-5883 | 0.96 |
| 64 | 7532-7563 | 7540-7619 | 1.01 |
| 128 | 8949-9049 | 8957-8985 | 1.00 |

Dew is level with vLLM at 64 and 128 slots and 4% behind at 32, where a step
is shortest and the host's share of it is largest. Each process is
`tools/benchmark_lm_serving.py --backend dew` (or `vllm-engine`, with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`) `--slots 32,64,128 --repeats 3` over the
Qwen3-0.6B checkpoint, and its JSON holds each repeat's
`output_tokens_per_second`.

At 64 slots a Dew decode step takes 6.96 ms on the device. Attention takes
4.0 ms, at the bound of reading the dense cache's keys and values (2.8 GB a
step at 716 GB/s), and the projections and head take 2.1 ms, near the bound
of reading the weights. Each of the 16 admission steps, which prefill 8
prompts next to the other rows' decode, took 40 ms, and 8 ms of that was the
prefill's key and value writes. XLA lays the prefill's fresh cache out with
its slots minor, to suit the attention that reads it, and scattering whole
tokens into that layout ran at 28 GB/s. So a write as wide as its buffer now
gathers each slot's token (`dew.nn.kv_cache.write_cache`). The output is
bit-identical; tokens and both log-probability streams match at every slot
count.

A decode step's attention reads each key head once for its group of query
heads, by passing the group in as that head's query positions. Without that,
cuDNN padded the lone query to two positions and ran each query head on its
own. The kernel timings show the gain. At 64 rows over 384 slots it takes
0.146 against 0.154 ms a layer, and 0.072 against 0.129 when every row has
257 keys, with the same bits.

Serving at integration `17e2b226` is consistent with that but shows no more,
because the host was loaded and the differences are inside its spread.
Medians of 15 runs went from 5489-5510 to 5521-5619 tokens a second at 32
slots, from 7175 to 7215 at 64, and from 8219 to 8343 at 128. Both sides had
slow runs (the slowest 5196 and 4685 at 32 slots), and the tokens and
log-probabilities are identical.

On an A100 40 GB (Colab, jax 0.11.2) at integration `e769335e`, 2026-10-07,
with the Pallas paged decode kernel, Dew's two caches and vLLM 0.30.0 ran in
one session, alternating, two rounds of two repeats, closed loop and then
open loop at 16 requests a second on 32 slots and 32 on 128:

| slots | load | Dew, dense cache | Dew, paged cache | vLLM |
|---:|---|---|---|---|
| 32 | closed, tokens a second | 6830-7797 | 6635-7372 | 6732-6917 |
| 32 | open: TTFT p50 / p99, gap p50 / p99 (ms) | 14.9-16.2 / 21.5-24.7, 3.38-3.73 / 9.5-10.2 | 13.9-16.4 / 19.9-24.7, 3.77-3.80 / 8.7-10.2 | 31.6-32.2 / 64.1-65.6, 3.07 / 23.4-24.1 |
| 128 | closed, tokens a second | 14551-14576 | 12427-12439 | 12910-13184 |
| 128 | open: TTFT p50 / p99, gap p50 / p99 (ms) | 16.2-17.4 / 24.6-27.4, 4.16-4.23 / 9.6-10.5 | 25.0-25.1 / 33.9-34.5, 7.96-7.97 / 13.2-13.4 | 50.5-53.4 / 73.8-77.8, 3.73-3.79 / 24.7-25.6 |

With the dense cache Dew serves 1.10-1.13 times vLLM's closed-loop
throughput at 128 slots and 0.99-1.16 at 32, where the first round's
repeats ran up to 12% below the second's on both caches. The paged cache
serves 0.85 of the dense one at 128 slots and 0.94-0.96 of vLLM. Under
open-loop arrivals Dew's TTFT p50 is at most 0.52 of vLLM's and its gap p99
at most 0.54, while vLLM's median token gap is shorter: by 0.3-0.7 ms
against either cache at 32 slots and the dense one at 128, and by 4.2 ms
against the paged cache at 128.

### The remaining gap, 2026-10-03

At 64 slots Dew serves 7146 and 7197 tokens a second (medians of five runs,
two processes, integration `6a220e31`) against vLLM 0.30.0's 7596-7609
in-process (three runs, one process). The two schedule alike. Dew runs 265
model steps a run and vLLM 267 (its scheduler's steps, counted in its engine
process), and each admits up to 8 prompts a step next to the other rows'
decode. So the gap is per step, 8.6 ms of wall time for Dew against 8.0 for
vLLM. I traced whole measured runs, Dew under XProf (`f6047cf9`) and vLLM
under Nsight with its CUDA graphs' nodes:

| | Dew | vLLM, 249 of 267 steps traced |
|---|---:|---:|
| GEMMs | 891 ms | 907 ms |
| attention | 1016 ms, 138.7 us a decode call | 943 ms, 136.6 us a decode call |
| everything else | 313 ms | 154 ms |
| device busy | 2.22 s of 2.29 | 2.00 s of 2.18 |

Nsight warned that it had not collected every CUDA event, and its trace
holds 249 of the run's 267 steps, so vLLM's rows are low by about 7%. Scaled
to the whole run, vLLM's attention matches Dew's and its GEMMs are 80 ms
slower. Dew's extra device time is in many small kernels of one to three
microseconds each: the residual add with the split-K GEMM's sum and its cast
(52.5 ms in 22859 launches), the norms (80.7 ms in 32201), the cache writes
(61.0 ms) and the transposes around attention (48.8 ms). vLLM fuses an add
into its RMSNorm and writes K and V in one kernel. Dew's 2.29 s run also has
70 ms with no kernel running.

I measured these and did not adopt them, because each gains under 1% of the
run:

- K and V written by one Pallas kernel a layer, in place of XLA's two
  scatters: 0.141 against 0.166 ms a decode step for all 28 layers, bitwise
  the same.
- XLA's command buffers. They are on by default, and cuDNN's attention is
  among the commands the decode step replays. Enabling every command type
  with no minimum graph size measured within the default's spread, turning
  them off is slower, and the tokens are the same all three ways.
- JAX's Pallas split-KV decode attention (`gqa`). It takes 0.43 ms a layer
  on the cache as Dew stores it, which it has to transpose, against cuDNN's
  0.165. On a head-major cache its kernel takes 0.155 ms at its best tiles,
  with different bits from cuDNN's.
- Triton multi-output fusion and a single split-K, one traced run each:
  device time 2222.2 and 2216.2 ms against 2222.3-2224.6.
- Freeing a slot in the step that finishes it, where today the slot is freed
  when the host reads that step's results. Counted step by step, the
  schedule would take 263 steps in place of 265.

What remains is the small kernels that vLLM fuses, worth at most about 1.5%
of the run by the launches a fused add and norm would save, and the idle
time between steps. The rates come from `tools/benchmark_lm_serving.py`. The
traces are one measured run each, Dew's under `dew.Profiler` and vLLM's
under Nsight Systems with `--cuda-graph-trace=node`, with kernel time summed
by family.

The idle between steps, 2026-10-03. Two host costs left the device waiting.
At submission, making a request's key took three small programs (move the
seed to the device, make a key, fold it), and they ran between steps after
the device had drained. And the admitting step's inputs are fresh buffers on
every call, so XLA updated its CUDA command buffer before it could replay,
which took about 5 ms before 9 of a run's 16 admitting steps. Now an integer
seed stays on the host until the admission's one key program makes and folds
every row's key (`_row_keys`, the same bits as an eager key), and the
admitting step compiles without command buffers (`ADMISSION_OPTIONS`); the
decoding step keeps them. The benchmark submits integer seeds, as a client
sends them. On the RTX 4080, the change against its parent, three
alternating rounds, medians of five runs, in tokens a second:

| slots | before | after |
|---:|---:|---:|
| 32 | 5664 / 5657 / 5663 | 3960 / 5728 / 5728 |
| 64 | 7131 / 7229 / 7222 | 7360 / 7359 / 7357 |
| 128 | 8329 / 8355 / 8346 | 8527 / 8528 / 8524 |

One process ran all five of its 32-slot runs slow (3846-4054) and its 64-
and 128-slot runs at the others' rate. Four more alternating rounds at 32
slots gave 5729-5730 after against 5564-5669 before, and an earlier session
on a loaded host gave the same order at every slot count. The tokens and
both log-probability streams are the same in every run.

The admitting step was already compiled on its own before (the step's jit
traced it separately from the decoding step), so a cold server's start is
unchanged. From `Server.from_task` to the first token, with no compilation
cache and two alternating rounds, it took 34.9 and 35.5 s before and 38.7
and 35.2 after. A traced 64-slot run idles 74.7-83.2 ms before and 40.9-42.1
after (one of five traces read 76.0, and one 838 during a host load spike).
Most of what is left is the traced client's own keys. Each trace is one
measured 64-slot run of the benchmark's prompts under `dew.Profiler`, with
idle time summed between kernels.

Several decode iterations a device call (`decode_steps`), measured after
that change, 2026-10-03. More iterations are slower at every slot count
except 64 slots, where two iterations read 7271 tokens a second against 7216
for one. Tokens a second on the RTX 4080, admission 8 rows, medians of five
runs, with vLLM from the table above:

| slots | vLLM | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 6049 | 5726 | 5696 | 5500 | 5106 |
| 64 | 7614 | 7216 | 7271 | 6975 | 6573 |
| 128 | 9047 | 8527 | 8357 | 7861 | 7231 |

There is little left for more iterations to gain. A call seats requests only
before its first iteration, so filling 64 slots at 8 rows a call takes 8
calls of k iterations, and a run takes 265, 274, 292 and 328 iterations.
Seating at every iteration could at most win back the idle time, and a
traced 64-slot run at one iteration a call keeps the device busy 2228 ms of
2242, idle 20 ms (0.9%), because the decoding step already replays as CUDA
graphs and the host runs ahead.

Seating 8k rows a call (the default) changes the prefill's shapes and so its
GEMM kernels, and the tokens differ from one iteration a call. At 8 rows
they are the same at 32 and 64 slots (the log-probabilities within 9.5e-6),
while at 128 slots the decoding GEMMs inside the longer program autotune to
other kernels. Each cell is `tools/benchmark_lm_serving.py --decode-steps K
--admission 8 --generations`, with its generations compared to K=1's.

The decoding step's small kernels, 2026-10-03. Per 64-slot decode step on
the RTX 4080, Dew's device spends 8.39 ms to vLLM's about 8.05: GEMMs 3.37
against 3.64, attention 3.84 against 3.79, and everything else 1.18 against
0.62. The largest item in the rest was one kernel of 136 us a step. It
merged the held logits with the step's over every slot's vocabulary in fp32,
so that a row seated in this step drew from its prompt's logits. In a step
that seats no row, every row that draws was fed through the model, so that
step now takes the model's logits whole, and only a seating step merges.
Tokens a second, three alternating rounds, medians of five runs, with the
tokens and both log-probability streams the same in every run:

| slots | before | after |
|---:|---:|---:|
| 32 | 5710 / 5729 / 3389 | 5780 / 5779 / 5746 |
| 64 | 7364 / 7363 / 7236 | 7410 / 6545 / 7407 |
| 128 | 8543 / 8527 / 8461 | 8610 / 8559 / 8609 |

One process on each side ran during a load spike (3389 and 6545). The bf16
rounding of the logits now runs as its own kernel, 57 us a step. If the step
did not store its logits at all, XLA could fuse that rounding into the draw
and save 27 us more, but the draw's log-softmax would then sum in another
order. At 32 slots, 430 log-probabilities move by up to 3.8e-6, so the step
still stores them. The attribution comes from a run traced under
`dew.Profiler` with command buffers off (`--xla_gpu_enable_command_buffer=`)
and the optimized HLO dumped, which names each kernel by its `hlo_op`.

The head's bf16 rounding, 2026-10-03. The logits are bf16 values held in
fp32 (`dew.nn.precision.head_product`), and the reduce-precision in
`rounded_to` did not fuse into XLA's Triton head GEMM. So a serving step
wrote the fp32 logits and read them back to round them, before the draw's
three readers read them again (the well-formedness check, the argmax with
the log-softmax maximum, and the sum of exponentials). The rounding was a
kernel of its own, 157 to 178 us at 128 rows. Under CUDA the logits are now
cast to bf16 behind a barrier (`bf16_logits`), so the cast fuses into the
GEMM's epilogue and each reader widens the bf16 copy itself.

The values and cotangents are bitwise those of `rounded_to`
(tests/test_precision_policy.py), and so are the served tokens and
log-probabilities at 32, 64 and 128 slots. The head and greedy draw alone
went from 1.145 to 0.981 ms at 128 rows of Qwen3-0.6B's widths, and the
serving run's device busy time went from 2211.0 to 2198.5 ms at 64 slots and
from 3803.5 to 3768.7 ms at 128 (two traced runs each). Training's loss head
rounds its tiles itself (`dew.objectives.lm.chunked`), and other platforms
keep `rounded_to`. The bits come from
`tests/test_precision_policy.py::test_bf16_logits_round_as_rounded_to_and_are_held_as_bf16_under_cuda`
and `tools/benchmark_lm_serving.py --generations` on each tree; the busy
time is one traced run under `dew.Profiler`.

A fused decode prologue, measured and removed. XLA runs a layer's q and k
norms, the rotated key's scatter into the cache, the value's scatter, and
the query's rotation and GQA fold as five kernels of one to three
microseconds each. A Pallas Triton kernel that did all five in one program a
row took 28 layers at Qwen3-0.6B's widths from 0.191 to 0.097 ms. It served
Qwen3-0.6B 2-3% faster, and the tokens and log-probabilities were bitwise
the same in every run (5943 / 7587 / 8765 tokens a second at 32 / 64 / 128
slots, against vLLM's 6049 / 7614 / 9047).

On Qwen3-1.7B, whose heads have the same widths, the kernel's served tokens
differed from the unfused step's (1349 of 8192 at 32 slots), and each side
was repeatable. (A record here first read that as the unfused step being
unrepeatable; that run had used the kernel on both sides.) The difference
came from the norm. Its sum over a head's 128 lanes, even given XLA's own
statistics from a standalone reduction, put one or two elements a head on
the other bf16 neighbour of the unfused step's value. XLA's reduction order
inside its norm fusion depends on the fusion, and its rsqrt is the
hardware's approximation, so there was no fixed arithmetic for the kernel to
match.

A kernel that left the norms to XLA's own fusions and did only the rotation,
both cache writes and the fold was bitwise on both models, but it saved only
3 of the 5 kernels. The device's busy time per 64-slot run went from 2211 to
2186 ms. The Pallas call also fell outside XLA's default CUDA command
buffers, so the decoding step's options needed `CUSTOM_CALL`, or the run
idled 200 ms more. Under 1% was not worth a kernel on a backend JAX 0.11
deprecates, so I removed it; the rotation-only kernel stays on the
`perf/prologue-bitwise` branch.

Where the gap sits at 128 slots, and the prefill's attention, 2026-10-03. I
traced a 128-slot run at integration `66a1848c` with command buffers off, so
each kernel is attributed to its program and HLO scope. Attention and GEMMs
are at parity: cuDNN's decode attention took 259 us a call against vLLM
FA2's 266, and the GEMMs 4.68 against 4.74 ms a step. The gap is the small
kernels, 1.26 ms of them a decode step against vLLM's 0.50, and 8.1 ms an
admission (32 admissions of 8 prompts).

The largest admission item was the prefill's attention. Admission left-pads
its prompts and writes their keys compactly, so the cache prefill builds the
cursor mask. `kernel_for_materialized_mask`, a rule meant for training
(cuDNN's bias backward refuses odd lengths), sent that mask to xla, which
cost two dense dots, a softmax and four mask transposes, 3.7 ms an
admission. A cache call over more than one query now gives the mask to cuDNN
as its bias wherever cuDNN runs; training, single-token decode, CPU and the
deterministic-ops CUDA lane are unchanged. Over 28 layers in isolation it
takes 4.0 against 6.2 ms, and the device's busy time per run went from
2212.6 to 2179.2 ms at 64 slots and from 3807.4 to 3745.1 ms at 128 (two
traced runs each, the same in both).

It is not bitwise, because cuDNN rounds at other points than xla's dots and
softmax. I measured the error under tests/reference_error.py's rule, against
the same bf16 weights computed in fp32 at the highest precision, over 8 of
the benchmark's prompts cut to mixed lengths and left-padded. The cuDNN
prefill's RMS distance is 1.036 times xla's on Qwen3-0.6B's logits (1.088 on
the log-probabilities) and 0.999 (0.987) on Qwen3-1.7B's, against an
allowed 2. The argmax agrees with fp32 at 96.4% of positions against 96.8% on 0.6B,
and 96.9% against 97.2% on 1.7B.

Greedy generations over the benchmark's random-token prompts part in 56 of
128 rows at 64 slots, each run repeatable, and every first divergence is a
bf16 near-tie. Teacher-forced through the fp32 model, the two chosen tokens'
logits are a median 0.64 bf16 spacings apart at their magnitude and at most
2.78; 77% are under one spacing and 98% under two. The fp32 argmax is the
xla prefill's choice in 31 rows and cuDNN's in 23.
`tests/test_causal_transformer.py::test_a_padded_prefill_attends_through_cudnn_where_it_runs`
checks the routing. The distances apply `tests/reference_error.py`'s
`distance` to the cache prefill and to `dew.pipeline(..., dtype="float32")`
under `jax.default_matmul_precision("highest")` over the same tokens.

The cache writes as words, 2026-10-03. XLA's scatter stores one element a
thread, so a bf16 cache moved two bytes a store. At 128 rows each decode
layer's key and value writes took 3.7 us apiece, and an admission's prefill
windows took 28 us a cache. On a GPU, `write_cache` and admission's
placement now move a one- or two-byte cache's bits as uint32 words
(`dew.nn.kv_cache.as_words`). A word-wide or boolean leaf and an axis that
does not fill whole words are written as they are, and so is every cache on
a TPU, which tiles two-byte arrays on another axis. Over 16 caches in
isolation the decode write went from 0.113 to 0.099 ms and admission's from
0.627 to 0.277. The served tokens and log-probabilities are bitwise the same
at 32, 64 and 128 slots, and the run's device busy time went from 2165.6 to
2149.5 ms at 64 slots and from 3704.2 to 3649.5 ms at 128 (two traced runs
each, integration `456a4b64`).
`tests/test_kv_cache.py::test_a_cache_write_moves_whole_words_with_the_same_bits`
checks the bits.

The cache's validity, derived, 2026-10-04. Each attention cache stored a
`[rows, capacity]` mask of its filled slots next to the cursor. Slots fill
in order, so the mask always equalled the cursor's `filled_slots`. A decode
step rewrote it in every layer, in 28 one-microsecond kernels, only to store
it in the state passed to the next step. The mask is now derived where it is
read (`dew.nn.attention.cached_validity`), in the attention, Llama 4, MLA,
the DSA pool and DeepSeek V4, and serving no longer places or zeroes it. The
served tokens and log-probabilities are bitwise the same at 32, 64 and 128
slots, and the device's busy time a run went from 1393.7 to 1388.3 ms at 32
slots, from 2147.5 to 2139.2 at 64 and from 3645.2 to 3626.2 at 128 (two
traced runs each).

### Open loop, 2026-10-04

The table above submits every request at once and keeps the slots full. With
`--rate`, `tools/benchmark_lm_serving.py` sends six times the slots in
requests as a Poisson process at each rate (seeded per slot count) to a
server that queues them. Time to first token (TTFT) runs from a request's
arrival, and the token gap is each decoding row's time between consecutive
tokens. I served Qwen3-0.6B with the same prompts and outputs on the RTX
4080 on a quiet host, with Dew at integration `420ea2c1` and then with
bucketed admission, in the same session as vLLM 0.30.0:

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

The admitting program prefilled every row of its admission, so a request
arriving alone was padded to eight prompts of prefill. Those admitting steps
were the p99 token gaps, 28 to 94 ms, and they cut the server's capacity
until its queue grew without bound at rates where vLLM kept a TTFT p50 under
40 ms.

Each admitting step is now padded only to the smallest power of two that
holds its prompts (`dew.inference.serving.admission_share`), and each width
is its own compiled program. The closed-loop runs are unchanged, because an
admission that fills its width is the same program, and the 32-, 64- and
128-slot generations are bitwise the same.

With buckets Dew is level with vLLM at 32 slots, with shorter TTFT tails. At
64 and 128 slots its p99 token gaps are 1.4 to 2 times vLLM's. At 64 slots
and 48 requests a second it runs at its capacity. A traced run had the
device 97% busy, and that cell's TTFT p50 swings between 21 and 447 ms from
run to run. The gap is in the admitting step. Dew prefills an arriving
prompt in a forward of its own next to the decode forward, so the weights
are read twice; for one 256-token prompt at 64 slots that prefill takes
9.9 ms against a 5.1 ms decode step. vLLM's chunked prefill puts the
prompt's tokens into the decode forward's batch.

A narrower prefill runs its GEMMs at other shapes, so a request admitted in
a narrower bucket than the padded eight can draw other bits. Served one at a
time at 32 slots, 6 of 16 Qwen3-0.6B rows and 8 of 16 Qwen3-1.7B rows part
from the padded path, each at a bf16 near-tie. Teacher-forced in fp32, the
two chosen tokens' logits are a median 0.5 bf16 spacings apart and at most
1.81, and the fp32 argmax is the padded choice in 7 rows and the bucketed
one in 7. Under tests/reference_error.py's rule, the 1-, 2- and 4-row
prefills' RMS distance from the same weights in fp32 is 0.96, 0.95 and 0.95
times the 8-row prefill's on Qwen3-0.6B's log-probabilities (1.00, 1.00 and
0.99 on the logits), and 1.02, 1.00 (bitwise) and 1.08 on Qwen3-1.7B's
(1.02, 1.00 and 1.07), against an allowed 2.

The mixed admitting step, 2026-10-04. An admitting step now runs one forward
over every token it holds, laid out in one row (`dew.nn.inputs.Admitted`):
each slot's last draw, then the admitted prompts. Projections, norms, the
MLP and the head work token by token, so they read their weights once for
the decoding rows and the prompts together. Attention writes every token's
keys into its row of the cache in one scatter. Each decoding row's query
then reads its row as a decode step does, and a prompt that starts its row
reads its own keys. The decode-only program is untouched; its optimized HLO
is identical to integration's and its device time per run is the same
(2540.4 against 2540.0 ms at 128 slots). In one session on a quiet host, I
compared Dew at integration `ab5966b1` (bucketed admission), Dew with the
mixed step, and vLLM 0.30.0:

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

The three serve the same tokens a second at every rate, within 0.8%. With
the mixed step Dew's median TTFT is the shortest of the three in 8 of 9
cells, and at 64 and 128 slots its p99 TTFT and token gap are at or below
vLLM's from 36 requests a second up. At 32 slots vLLM keeps the shorter
token-gap tails, by 0.4 to 2.8 ms at p99, and the shorter TTFT p99 at the
two lower rates. The 64-slot cell at 48 requests a second, which saturated
in the earlier session, did not saturate here; that rate is near the
server's capacity. In closed loop, over three repeats in two alternating
rounds, 32 slots went from 5929 to 5994 tokens a second, 64 slots from 7673
to 7766, and 128 slots from 9039 to 9033-9072. Traced at 128 slots, the
admitting program took 1068 against 1078 ms a run, with GEMMs at 671 against
676 and attention at 253 against 252.

The mixed step is not bitwise the same as the two forwards, because its
GEMMs run at other shapes. I compared both against the same bf16 weights
computed in fp32 at the highest precision, over 28 decoding rows and four
admitted prompts of 64 to 256 tokens. The mixed step's RMS distance is 0.84
times the two forwards' on Qwen3-0.6B's decoding logits and 0.97 on its
prompt logits (log-probabilities 0.83 and 0.95), and 1.09 and 0.99 on
Qwen3-1.7B's (1.14 and 0.98), against tests/reference_error.py's allowed 2.
At 32 slots 27 of 64 greedy rows part from integration's, all at bf16
near-ties (a median 0.57 bf16 spacings apart in fp32, at most 1.44). The
fp32 argmax is the two-forward choice in 16 rows and the mixed one in 11.

The 32-slot tails above did not repeat on 2026-10-05 (integration
`dfd31d58`). In two alternating rounds against vLLM in one session, the
second round put Dew at or below vLLM on TTFT p50 and p99 and on token gap
p50 and p99 at 16, 24 and 32 requests a second. Gap p99 was 8.16, 7.42 and
8.32 ms against 8.65, 16.4 and 9.56. The first round had a TTFT p99 outlier
at 16 requests a second on both sides: 46.4 ms for Dew and 51.6 for vLLM.

Over a paged cache, 2026-10-04. The mixed step now runs over a page pool
too. The admitted rows' tables go into the cache before the step writes,
each token is written to its row's page, and the decoding rows read the pool
through cuDNN's paged kernel as a decode step does. Pieces that continue a
row (a chunked prompt, a shared prefix's pages) read the row's earlier keys
from the pool; on a server with neither, each piece reads only its own keys.
A dense cache takes chunked prefill the same way (`chunk`), and only the
mixed step serves it. The paged decode-only programs are identical to
integration's. I ran two alternating rounds in one session at integration
`ab60614a`; the table is the second round, because the first ran under
another lane's load:

| slots | rate | paged, two forwards: TTFT p50 / p99, gap p99 (ms) | paged, mixed |
|---:|---:|---|---|
| 32 | 16 | 17.8 / 23.7, 9.9 | 16.5 / 22.4, 8.8 |
| 32 | 24 | 18.2 / 27.3, 10.0 | 16.6 / 22.0, 8.9 |
| 32 | 32 | 18.1 / 26.5, 10.3 | 16.9 / 24.0, 11.0 |
| 128 | 32 | 40.3 / 58.5, 26.0 | 38.1 / 55.6, 23.7 |
| 128 | 44 | 42.9 / 63.7, 26.3 | 41.0 / 61.0, 25.7 |
| 128 | 56 | 50.6 / 231.3, 26.3 | 44.8 / 161.7, 25.8 |

In closed loop the mixed step went from 5576-5577 to 5583-5656 tokens a
second at 32 slots and from 8254-8274 to 8330-8334 at 128. A traced 32-slot
run at 24 requests a second put a one-row admitting step at 8.30 against
9.22 ms. The paged decode step itself is slower than the dense one (at 128
slots and 32 requests a second, a token gap's p50 is 11.7 ms paged against
5.4 dense), so a dense cache stays the faster way to serve where the memory
fits. The served tokens part from the two forwards' in the same 27 of 64
rows at 32 slots as with the dense cache, all at bf16 near-ties.

Where the paged step's time goes, measured on an A100 40 GB at integration
`a906f011`. Qwen3-0.6B, two rounds alternating, 20 traced decode steps a
case. At 128 slots the paged cache served 11407-11417 tokens a second
closed loop against the dense cache's 14607-14628 (0.78), and at 32 slots
7269-7278 against 7974-7983 (0.91). At 128 slots the decode step is on the
device for 8.21 ms paged against 6.41 dense. cuDNN's paged attention kernel
accounts for 1.67 ms of the 1.80 ms difference (5.12 against 3.45 ms
dense). The paged write's scatter and a pad fusion add 0.41 ms more (0.31 +
0.23 against the dense scatter's 0.13), and the dense step's cache copies
and a concatenate (0.23 + 0.08 ms), which the paged step has none of, take
back 0.31. The GEMMs are the same (1.65 ms). The
page pools' layout costs nothing: cuDNN reads `[pages, page_size, heads,
dim]`, and XLA already assigns the donated pools that layout. Holding them
in it from the start (`perf/paged-layout`, not adopted) left the step at
8.207 ms and the generations bitwise equal. A one-layer program compiled
alone does transpose the pools, so a probe of one layer misleads here.

A GPU decode step now reads the pages through Dew's own Pallas decode
kernel (`decode_attention.attend_paged`, the dense decode kernel's
arithmetic over a page table), not cuDNN's paged attention. On the same
A100 at integration `703201b5`, two rounds alternating, the paged cache
served 12372-12387 tokens a second closed loop at 128 slots against
cuDNN's 11400-11406 (+8.6%), and 7387-7397 against 7041-7265 at 32; at 32
requests a second on 128 slots a token gap's p50 fell from 8.82-8.88 to
7.99-8.03 ms. The traced step took 7.50 against 8.19 ms, the attention
about 4.4 against 5.13. The dense cache's programs, throughput and tokens
are unchanged (14567-14592 tokens a second at 128 slots, generations
bitwise equal), so it stays the faster cache at 0.85 of it. The kernel's
error from float64 is within twice cuDNN's (`tests/reference_error.py`),
but its sums run in another order: the paged generations, which cuDNN kept
bitwise equal to the dense cache's, part from them in 28 of 64 rows at 32
slots and 102 of 256 at 128, each at a near-tie (the two tokens' log
probabilities 0.05 apart at the median, at most 0.15).

Under `--xla_gpu_deterministic_ops` (the CUDA test lane's flag), the paged
write put a dropped token's keys at another head's kept slot (jax 0.11.2, an
RTX 4080). The write is a scatter into the pool's page and offset axes past
an unindexed head axis. The scatter expander padded the out-of-range rows
with 0 on the unindexed axis, so a window offset along that axis collided
with a kept index. openxla/xla#49498 fixes it (issue #49380), in an XLA
later than Dew's jax pin. Mapped over a group of one, as the paged cache's
other write already is, the scatter is right, so `KVStore.write_tokens`
writes that way.

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

The decode token now goes through a Pallas kernel on CUDA
(`dew.nn.kernels.delta_rule`). It reads each row-head's fp32 state once and
writes it once: both readouts are products with the old state (the query's
of the update is `(q . k) delta`). At 128 rows the four XLA passes over 2.4
GB were two thirds of the decode program. In isolation, 18 layers took 9.1
against 13.0 ms at 128 rows and 2.7 against 3.7 at 32. Serving, two rounds
alternating, 128 slots went from 4208-4210 to 5819-5823 tokens a second,
above vLLM's 5410-5577 above, and 32 slots from 4267-4278 to 4109-4459. The
kernel's outputs differ from XLA's in reduction order. On Qwen3.5-0.8B's own
delta-rule inputs, decoded token by token, its RMS distance from float64 is
0.95 to 1.14 times XLA's (tests/reference_error.py allows 2). 16 of 256
greedy rows part at 128 slots, each at a bf16 near-tie: a median 0.55 bf16
spacings apart in fp32, at most 1.74, and fp32's argmax is XLA's choice in 7
rows and the kernel's in 8. The kernel's body is whole-block arithmetic, so
a Mosaic GPU kernel, the backend JAX keeps, takes it as is. On a TPU,
tokamax's Mosaic TPU `causal_conv1d_gated_delta_rule` covers the conv, the
gating and the rule over a step's tokens in one ragged call. That call is
the layout of the mixed admitting step, where it would plug in.

The six full-attention layers have 256-wide heads, and cudnn does not run
them, so their decode attention takes jax.nn's xla path. XLA's GPU dot
products split out each head's keys, so every step transposed the whole
cache, keys and values, into `[rows, heads, width, slots]`, which took 394
of the 128-slot decode program's 3673 ms. A decode-shaped xla call on a GPU
now multiplies every query head against every (slot, head) pair
(`folded_attention`, for at most 16 query positions times heads). Both
products then read the cache as `[rows, slots * heads, width]`, its own
layout, and the cross-head products are discarded. That doubles the
multiplications, but the step is bound by reading the cache. Measured
alone, six layers took 1.21 ms against 2.48 at 128 rows, and 0.33 against
0.47 at 32. In serving, over two alternating rounds, 128 slots went from
5814-5816 to 6152-6157 tokens a second (median token gap 15.25 to 14.1 ms at
16 requests a second), and 32 slots from 4245-4501 to 4637-4641. The greedy
rows that changed are near-ties: 3 of 64 at 32 slots (at most 0.93 bf16
spacings apart in fp32, and all three now pick fp32's argmax) and 20 of 256
at 128 (at most 1.74 apart; fp32's argmax sided with each version in 10).

Dew keeps the recurrent state in fp32, as transformers does. vLLM 0.30.0
keeps it in the model's dtype: for a gated delta net,
`mamba_ssm_cache_dtype="auto"` means the conv state's dtype, bf16 here
(`MambaStateDtypeCalculator._mamba_state_dtype`). At 32 slots Dew's state is
604 MB, read and written each decode step, so 1.2 GB moves against vLLM's
0.6. At the RTX 4080's bandwidth that is about 0.85 ms of the step, a large
part of the median token gap: Dew 5.6 ms against vLLM's 3.5. Dew keeps fp32
anyway. A bf16 state rounds every row's memory at every token, and the
reference does not.

Not adopted: the mixed admitting step for a gated delta net, 2026-10-05.
A hybrid server keeps two forwards for an admitting step: a prefill
forward for the admitted prompts and a decode forward for the running
rows. A branch gave `GatedDeltaNet` the mixed call (`Admitted`): the
projections over every token at once, then the conv and the rule apart
for the decoding rows (the kernel above) and for each admitted piece.
Its numerics passed the rule. At 32 slots, 3 of 64 greedy rows parted
from the two-forward path, all at ties within 1.00 bf16 spacing, and all
three took fp32's argmax. It gained nothing measurable:

- Traced closed loop: device busy 5613 against 5621 ms at 128 slots and
  1812 against 1804 at 32. Every admission in a closed loop comes as a
  wave with no row decoding, so the merge has nothing to fold in, and the
  decode program ran 241 and 253 times on both sides.
- Open loop at 32 slots, two rounds alternating: TTFT p50 3.5 ms shorter
  at 8 requests a second, level at 16 and 24, and a median token gap of
  5.6 ms on both sides.

171 added lines were not worth that.

A serving step decodes every slot, whether or not it is drawing a token,
and the decode kernel read and wrote every row's state. Under open-loop
arrivals most of the 128 slots sit idle, and the state is most of a decode
step's bytes, so a mostly idle step cost as much as a full one: a 15 ms
median token gap at 16 requests a second, against vLLM's 4.2. The kernel
now skips a row that the step marks idle (`active`, the row's validity). It
neither reads nor writes that row's state, the output state aliases the
input, and the row's output is zeros. Eighteen layers at 128 rows took 8.43
ms with every row drawing, 2.22 with a quarter and 0.20 with none. Serving
results, over two alternating rounds, with generations bitwise the same at
32 and 128 slots:

| slots | rate | before: TTFT p50 / p99, gap p50 / p99 (ms) | after |
|---:|---:|---|---|
| 32 | 8 | 23.3-23.8 / 35.0-35.2, 5.55-5.56 / 12.6-13.3 | 19.5-19.8 / 29.7-33.1, 4.05 / 10.7-10.9 |
| 32 | 16 | 24.8-25.1 / 40.5-48.4, 5.56-5.59 / 15.2-16.7 | 21.2-21.3 / 34.0-37.6, 4.36-4.38 / 13.7-14.0 |
| 32 | 24 | 25.6-25.7 / 42.2-44.3, 5.57-5.59 / 15.1-15.6 | 23.5 / 38.8-40.1, 4.86-4.88 / 14.5-14.6 |
| 128 | 16 | 57.0-57.9 / 87.4-94.0, 15.2-15.3 / 33.7-34.1 | 37.2-37.5 / 59.4-60.8, 8.94-8.96 / 21.9-22.2 |
| 128 | 24 | 62.3-62.7 / 93.3-93.8, 15.4-15.6 / 34.4-35.0 | 44.7-44.9 / 72.3-74.5, 10.6-10.7 / 25.4-26.4 |
| 128 | 32 | 66.6-66.9 / 97.3-106, 18.4-21.9 / 36.1-36.7 | 54.8-55.2 / 82.7-85.2, 13.5 / 31.6 |

Closed loop is unchanged at 32 slots (4503-4506 against 4408-4533 tokens
a second). At 128 it rose from 5801-5818 to 5966.

Idle slots still cost time. A step decodes every slot, so a 128-slot
server with 16 rows drawing stepped in 8.8-9.1 ms, against 5.5-7.0 for a
32-slot server with the same 16 (RTX 4080, capacity 384; the wall time of a
decode-only step, measured the same way for both). In a trace, the 128-slot
decode program spent 1.15 ms of its 7.4 ms in the full-attention layers'
attention, because the folded form above reads every row's keys across the
whole capacity. A decode-shaped xla call with key lengths now runs through
a Pallas kernel on CUDA (`dew.nn.kernels.decode_attention`). Each program
takes one row and one key head and reads only the key blocks below that
row's length, with an online softmax, so an idle row reads one block. At
128 rows with 16 drawing, six of the 256-wide layers took 0.28 ms, against
1.12 for the folded form and 2.40 for jax.nn; at 32 rows, 0.20 against 0.33
and 0.41. The decode program's device time per forward fell from 7.42 to
6.35 ms at 128 slots and from 4.11 to 3.87 at 32. In serving, over two
alternating rounds on a loaded host, the 128-slot median token gap went
from 8.49-8.80 to 7.13-7.62 ms at 16 requests a second, from 10.7-11.0 to
8.85-9.54 at 24, and from 14.8-15.1 to 12.9-14.1 at 32. Closed loop went
from 5593-5726 to 5713-5936 tokens a second. The 32-slot cells moved within
the base's own run-to-run spread (gap p50 4.22-4.62 against 3.93-5.44 at 8
requests a second). The kernel's RMS distance from float64 is within
`tests/reference_error.py`'s rule of jax.nn's. Four of 64 greedy rows
changed at 32 slots and 25 of 256 at 128, at a median of 0.11 and 0.59 bf16
spacings apart in fp32 (at most 1.75 and 2.04). At 128 slots, fp32's argmax
was the base's choice in 12 rows and the kernel's in 13.

Two other costs scale with the slots, traced at 128 slots with 16 drawing.
Each gated delta layer's conv state takes 0.61 ms in a select over every
row and 0.55 ms of layout copies. A one-token form of the masked conv
(`_masked_conv1d`) did not change that, because the cost is writing every
row's state. And the vocabulary head's and the MLP's GEMMs run over 128
rows, while reading their weights the same as at 32.

Here is the same session against vLLM 0.30.0 at integration `be10d331`
(with the decode kernel, the folded attention and idle rows skipped), on a
quiet host, with Dew and vLLM alternating over two rounds:

| slots | rate | Dew TTFT p50 / p99, gap p50 / p99 (ms) | vLLM |
|---:|---:|---|---|
| 32 | closed | 4657-4664 tokens a second | 4693-4938 |
| 32 | 8 | 19.3-19.5 / 27.1-28.3, 3.88-3.89 / 10.1-11.5 | 25.6-27.1 / 56.3-56.5, 3.48-3.49 / 20.7-21.9 |
| 32 | 16 | 20.6-20.9 / 33.5-40.8, 4.18-4.19 / 13.4-13.6 | 26.4-29.4 / 58.0-67.3, 4.05-4.08 / 21.3-23.0 |
| 32 | 24 | 22.6-23.0 / 38.5-39.7, 4.66-4.70 / 14.2-15.5 | 30.9-36.6 / 63.9-95.0, 4.45-4.48 / 21.8-26.6 |
| 128 | closed | 6309-6316 | 7393-7406 |
| 128 | 16 | 32.9-33.3 / 55.6-57.6, 7.62 / 19.1-20.1 | 29.0-32.7 / 70.3-71.7, 4.17-4.20 / 23.9-25.2 |
| 128 | 24 | 39.3 / 64.1-66.7, 8.83-8.88 / 22.8-23.0 | 39.4-41.5 / 81.6-86.4, 4.76-4.86 / 26.0-29.3 |
| 128 | 32 | 47.8-48.3 / 77.7-78.4, 11.7 / 28.7-28.8 | 48.1-52.4 / 81.1-176, 6.76-7.21 / 31.9-77.1 |

At 32 slots Dew is level with vLLM: 0.94-0.99 of its throughput, shorter
TTFTs and token-gap tails, and a median gap 0.1-0.4 ms longer. At 128
slots Dew serves 0.85 of vLLM's closed-loop throughput, its TTFTs and
tails are level or shorter, and its median gap is 3.4-4.9 ms longer. Two
causes of the 128-slot gap remain. The first, which its bytes account for,
is the fp32 recurrent state described above, twice vLLM's bytes per drawing
row. The second, which the trace's per-program costs suggest but no A/B has
isolated, is that a step still runs all 128 slots' rows through the
projections, the MLP, the vocabulary head and the full-attention layers'
cache reads, while vLLM batches only the rows that are running. To
reproduce the Dew side, note that `tools/benchmark_lm_serving.py` asks
`Server.from_task` for a dense cache, which a `MultimodalTransformer`
refuses because it declares no `kv_cache` layout. These runs dropped that
request; the dense cache is the default anyway.

On an A100 40 GB (Colab) at integration `0aee4c9d`, with the decode kernels
and idle rows skipped, Dew and vLLM 0.30.0 ran in one session, alternating,
two rounds, except that the session ended before vLLM's second 128-slot
round:

| slots | load | Dew: TTFT p50 / p99, gap p50 / p99 (ms) | vLLM |
|---:|---:|---|---|
| 32 | closed | 6482-6489 tokens a second | 4876-5068 |
| 32 | 8 a second | 20.3-24.4 / 35.6-43.8, 2.85-3.21 / 15.9-19.1 | 88.3-88.5 / 171-173, 2.97-2.99 / 61.0-62.2 |
| 32 | 16 | 21.1-25.4 / 41.2-52.1, 2.99-3.31 / 16.2-20.2 | 180-355 / 642-955, 3.83-3.99 / 62.7-64.6 |
| 32 | 24 | 23.7-29.4 / 41.9-50.3, 3.17-3.33 / 16.2-20.5 | 924-1032 / 1971-2088, 3.86-3.93 / 62.5-65.3 |
| 128 | closed | 9816-9851 | 7879-7951 |
| 128 | 16 | 26.0-26.3 / 52.2-54.7, 4.48-4.52 / 17.1-19.6 | 130.7 / 190.6, 3.91 / 64.3 |
| 128 | 24 | 32.3-33.1 / 55.1-55.2, 4.77-4.86 / 19.2-19.5 | 174.4 / 468.0, 7.35 / 68.4 |
| 128 | 32 | 36.7-38.3 / 60.4-64.6, 5.30-5.48 / 21.0-22.6 | 1281 / 2700, 7.81 / 70.0 |

On the A100, Dew serves 1.28-1.33 times vLLM's closed-loop throughput at 32
slots and 1.23-1.25 at 128. Under open-loop arrivals its time to first token
stays at 20-38 ms, while vLLM's grows into seconds once the arrival rate
nears its throughput. vLLM's median token gap is shorter at 128 slots and
16 requests a second (3.91 against 4.48-4.52 ms), and the two overlap at
32 slots and 8 a second (2.97-2.99 against 2.85-3.21). Its gap p99 is 61-70 ms
against Dew's 16-23. vLLM ran its defaults through
`tools/benchmark_lm_serving.py --backend vllm-engine`. A rerun at
integration `e769335e` (2026-10-07, the session of the A100 Qwen3-0.6B table
above) gave Dew 6419-6510 tokens a second against vLLM's 4868-5080 at 32
slots (1.26-1.34 times) and 9597-9861 against 7856-7927 at 128 (1.21-1.26).
Open loop at the same rates, Dew's TTFT p50 was 21-39 ms against vLLM's
90-1043, and its gap p99 16-24 ms against 62-71.

## Quantized serving of the 176M text-to-image model, 2026-09-28

`TextToImage.quantized` serves the denoiser with its kernels stored as int8
or fp8 values and their scales, through Qwix's post-training quantization
(`dew.training.quantization.quantize_for_serving`). `int8` and `fp8`
quantize weights and activations, so a matmul of two quantized operands runs
in the quantized dtype. `int8w` and `fp8w` quantize weights only and
dequantize them into the compute dtype. The model is dewml/hybrid-dit-176m.

Each row is one process of `tools/benchmark_quantized_serving.py`, and the
columns are:

- forward: the warm guided denoiser call over 12 prompts (batch 24, median
  of 5).
- sample: the warm wall time of 12 images with 20 DPM-Solver++(2M) steps at
  guidance 5, including text encoding and decoding.
- CLIP: the mean ViT-L/14 cosine over 12 prompts at seeds 0 and 1.
- weights and temporaries: the denoiser's weight bytes and the compiled
  forward's temporaries, in MiB.

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

Weight-only quantization saves memory and no time. The kernels take 27% of
their fp32 bytes, and the forward runs as fast as the unquantized one in the
same compute dtype. Quantizing weights and activations to int8 or fp8 takes
21% off the bf16 forward's time (28.3 ms against 36.0) and 35% to 39% off
the fp32 forward's (34.7 and 32.5 ms against 53.3). Every quantized row
keeps CLIP within 0.002 of fp32.

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
activations quantized, the forward is slower than unquantized in the same
compute dtype (28.4 ms in bf16 int8 against 25.0, 30.3 ms in fp32 int8
against 27.5), and slower again in fp8, which the A100 has no units for.
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

On the v6e, int8 weights and activations halve the fp32 forward (7.1 ms
against 13.7) when the depthwise convolutions are quantized too; with them
kept in fp32, the forward takes 14.9 ms. In bf16 the weight-only rows run in
the unquantized forward's time (5.2 and 5.6 ms against 5.7), and with
activations quantized the forward is slower (6.3 to 9.8 ms). Sampling takes
28 to 31 s in every row, whatever the forward's time, so on this machine
something other than the denoiser's 20 steps sets the sampling time. I did
not break that cost down. Every quantized row keeps CLIP within 0.005 of
fp32.

Before serving scaled the product of two quantized operands in float32,
every bf16 row with int8 or fp8 activations sampled NaN images on the v6e
(CLIP 0.1481, with the depthwise convolutions quantized or not). Qwix 0.1.8
scales that product in the scales' dtype, which is bf16 in a bf16 model. In
plain JAX on the v6e, an int8 depthwise convolution whose int32 product is
scaled in bf16 came out NaN in all but a few outputs, while the model's
dense and attention forms scaled the same way stayed finite. In the served
model the NaN began in the depthwise convolutions in int8 and in an
attention block in fp8. Scaled in float32, the bf16 model's 8-bit operations
have the same result types as the fp32 model's, and it samples as the table
shows.

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

On XLA:CPU, int8 weights and activations double the forward's time (9.8 s
against 4.9 in fp32), and weight-only int8 leaves it as it was. Every
quantized row keeps CLIP within 0.0025 of fp32.

On a GPU, Dew refuses to quantize the activations of a grouped convolution,
because XLA:GPU (jax 0.11.2) computes those convolutions wrongly or not at
all. So the GPU rows with quantized activations keep the spatial fusion's
depthwise convolutions in float (`--float spatial_fusion`), and
`TextToImage.quantized` raises without that option. On the RTX 4080, an int8
convolution with one or two input channels per group returns wrong values
without an error; before the refusal, the whole-model int8 row ran in
42.3 ms and scored CLIP 0.1391. In fp8 the same convolutions fail to compile
there (`Failed to get configs for: 36 out of 126 instructions`, one per
depthwise convolution). On the A100 the whole-model int8 row scored CLIP
0.1373 in fp32 and failed to compile in bf16 (`UNIMPLEMENTED`). fp8 ran on
the A100 (CLIP 0.2443 in fp32), but Dew refuses it there too, as on every
other GPU.

## Kernel choices per generation, 2026-09-22

Each kernel's implementation is chosen in one place and keyed by hardware generation (`dew.nn.kernels.device_generation`: `sm80`, `sm86`, `sm89`, `v5e`, `v6e`, ...), and a generation without a measurement here runs the XLA path. `tools/benchmark_kernels.py` and `tools/benchmark_lm_head.py` reproduce the rows. Each row is one process with jax 0.11.1 and bf16 compute, on a Colab NVIDIA L4 (the RTX 4080's architecture, sm_89), a Colab TPU v6e-1, or, for the kernel-level rows, the local RTX 4080. The step rows come from `tools/benchmark_kernels.py step` (built on `tools/benchmark_step.py`'s trainer), with 30 timed steps after 5 warmup. lm-moe has 321.8M parameters with 8 experts and top-2 routing, and lm-dense has 359.8M; both run at sequence 1024. The batch is 4 (moe) and 1 (dense) on the L4, and 8 and 8 on the v6e. "before" is main at c1f7e2dd.

### The MoE grouped matmul: `GROUPED_MATMUL_BY_GENERATION`

| device | path | ms/step | p50 ms | peak GiB |
|---|---|---|---|---|
| L4 | lm-moe before (xla) | 601.55 | 610.84 | 12.46 |
| L4 | lm-moe after, `auto` = pallas | 213.14 | 216.45 | 8.15 |
| L4 | lm-moe after, xla | 598.54 | 609.07 | 12.46 |
| v6e | lm-moe before (xla) | 76.04 | 76.43 | 5.05 |
| v6e | lm-moe after, `auto` = xla | 74.68 | 75.25 | 5.05 |
| v6e | lm-moe after, tokamax (`mosaic_tpu_v2`) | 75.40 | 76.04 | 5.05 |

A rerun at jax 0.11.2 in one Colab L4 session (2026-09-22, 19:55 to 20:13 CDT), with `tools/benchmark_kernels.py step --path lm-moe --batch 4`, gave 224.90 ms and an 8.15 GiB peak for `auto` (pallas), and 602.36 ms and 12.46 GiB for `--implementation xla`. The `projection` alone took 3.37 ms forward plus backward in Pallas and 25.19 ms in XLA.

On a mesh of 2x RTX 3090 (jax 0.11.2), I timed one bf16 ExpertMLP layer forward plus backward, XLA against Pallas. With fsdp 1 and expert 1 it took 141.9 against 64.4 ms. With fsdp 2 it took 83.3 against 98.1 ms, because there the Pallas path all-gathers the fsdp-sharded expert kernel (128 MiB of temporaries against XLA's 3120). Expert 2 global took 174.0 against 102.7, and expert 2 exchange 93.0 against 14.9. Whole lm-moe steps took 1024.0 against 578.3 ms with data 2 and 780.9 against 537.1 with expert 2 exchange, with the same losses. Those numbers made `auto` take XLA where fsdp alone sharded the experts, until the dispatch moved every routed layer inside its row map. There both kernels see gathered experts, and the Pallas kernels win on the RTX 3090 ([Expert parallelism on 4x RTX 3090](#expert-parallelism-on-4x-rtx-3090-2026-09-23)).

The kernel-matrix rows time forward plus backward at jax 0.11.2, checked against float64. At lm-moe's up projection the Pallas kernels take 1.21 ms against XLA's 6.25 on an A100, 3.15 against 25.9 on an L4 and 1.59 against 15.7 on the RTX 4080. At 128 experts they take 0.43 against 15.7 (A100), 1.38 against 77.9 (L4) and 0.55 against 33.5 (RTX 4080). On a TPU v5e and v6e XLA wins at 128 experts (on the v6e, 0.346 ms against `mosaic_tpu_v2`'s 0.408). On an RTX 3090 (sm86, jax 0.11.2) Pallas takes 2.54 ms against 17.40 for the up projection and 2.37 against 16.97 for the down projection, with the same forward error. An sm75 card (T4) cannot compile the Triton kernels, and I had no sm90 or sm120 card, so those generations run XLA.

`expert_projection` alone (8192 rows, 768 to 2048, 8 experts, forward plus backward) takes 26.21 ms in XLA and 3.38 ms in Pallas on the L4, and 14.84 and 1.86 ms on the RTX 4080. Against a float64 oracle of the rounded operands, Pallas's errors are the same as XLA's or lower (kernel gradient 4.2e-6 against 6.8e-6 relative). The L4 step is 2.82x faster. JAX's stock Pallas lowering with an out-sharding fix measured 1.97x on the same step, because its tangents run in fp32, while Dew's backward multiplies the bf16 cotangent.

I rejected a pure-JAX loop of dense per-tile products. On the RTX 4080 it was 2.2x faster than XLA for the projection alone (6.67 ms), but it doubled the step's temporaries (4.22 GiB against 2.17 at batch 1), and on the v6e it was 2.2x slower than XLA (1.71 ms against 0.79).

The Pallas kernels are JAX's own `gmm` and `tgmm` from the jax-v0.11.2 source tree, vendored because no wheel ships them and called through a custom VJP. jax 0.11.2 deprecates the Pallas Triton backend they run on and warns at every lowering. They stay the sm80 to sm89 path anyway, because JAX's Mosaic GPU grouped matmul (`pallas/ops/gpu/ragged_dot_mgpu.py`) uses wgmma and fails to compile on the RTX 4080, and tokamax's sm80 Mosaic config exceeds Ada's shared memory. Dew does not silence the warning, so filtering it is up to you. Moving this path to Mosaic GPU on sm90 and later is open, and waits for Hopper hardware to measure on. On a mesh the kernels run inside `shard_map` on each device's share of the sorted rows. That path is checked for parity on an 8-device CPU mesh and not measured on multiple GPUs.

On TPU, tokamax's `mosaic_tpu_v2` is within 1% of XLA on the step. tokamax's default dispatch picks its v1 kernel there, which is 13x slower, so Dew names the kernel.

### bf16 Adam state: `OptimConfig.state_dtype`

| device | measurement | fp32 state | bf16 state, hash rounding | bf16 state, threefry rounding |
|---|---|---|---|---|
| L4 | one AdamW update, lm-dense tree | 49.10 ms | 37.20 ms | 51.54 ms |
| v6e | one AdamW update, lm-dense tree | 11.45 ms | 8.89 ms | 20.07 ms |
| L4 | lm-dense step | 138.72 ms, 7.34 GiB | 126.45 ms, 5.89 GiB | |
| L4 | lm-moe step | 224.90 ms, 8.15 GiB | 218.15 ms, 6.92 GiB | |
| v6e | lm-dense step | 123.85 ms, 5.77 GiB | 124.94 ms, 4.50 GiB | |
| v6e | lm-moe step | 74.68 ms, 5.05 GiB | 71.96 ms, 3.87 GiB | |

The two L4 step rows are jax 0.11.2 from one Colab session (2026-09-22, 19:55 to 20:13 CDT); the update rows and the v6e rows are jax 0.11.1. The rounding noise is a counter hash of the step, the leaf and the element index. threefry noise (`jax.random.bits`) makes the update slower than fp32 state on both devices. bf16 state saves memory everywhere, but on the v6e lm-dense step it costs 0.9% in time, so the option stays off by default.

### The vocabulary head: the compute dtype's product

The head's product follows the compute dtype, as torch autocast and MaxText (`logits_dot_in_fp32=False`) run it. Under bf16 compute both operands multiply as bf16 with fp32 accumulation, in the model's head and the chunked loss alike, while the softmax and the loss stay fp32; an fp32 model keeps its fp32 head. The table times forward plus backward of the chunked head alone, at 8 x 1024 tokens, 1024 features and vocabulary 50304:

| device | fp32 operands (before) | bf16 operands, with argmax | bf16 operands, no argmax | fused linear cross entropy (Pallas port of Liger) |
|---|---|---|---|---|
| L4 | 206.16 ms | 134.02 ms | 133.91 ms | 142.63 ms |
| v6e | 8.04 ms | 8.03 ms | 7.26 ms | not run |

On the v6e the fp32 operands already multiplied in one bf16 pass, so only skipping the argmax (`token_accuracy=False`) changes the head's time. On the L4 the argmax fuses into the head's own kernels. With the bf16 product, the lm-dense step on the L4 went from 138.72 ms to 129.36 ms, and on the RTX 4080 (jax 0.11.2) the head at 4 x 1024 tokens went from 45.25 ms to 28.00 ms.

The bf16 product changes the loss by less than the loss changes between reruns. I ran `tools/lm_step_parity.py`, 100 steps of the 39M-parameter decoder on the RTX 4080, twice each way. Two fp32-head runs differ by at most 2.3e-4 relative at any step and two bf16-head runs by 7.7e-4, while a bf16-head run differs from an fp32-head run by 3.4e-4 and 7.2e-4, within the bf16 head's own rerun spread. The final losses are 0.0078378 and 0.0078376 with the fp32 head, and 0.0078368 and 0.0078387 with the bf16 head. I rejected the fused Pallas kernel, which is 6% slower than the chunked head on the L4, and tokamax's `mosaic_tpu` head, which is 2.24x slower on the v6e (kernel catalog, 2026-09-22).

On an A100, the reference runs measured the fp32 head at 38 ms a step, 21% of a Qwen3-0.6B bf16 fine-tune's busy time, running as TF32 GEMMs where torch autocast runs bf16. That and the rows above made the bf16 product the default.

The logits' rounding, 2026-10-01. The bf16 product above still kept fp32 logits, and fed their fp32 gradient into the state product as two bf16 products, one of a high half and one of the rest (`347238c7`), where torch autocast and MaxText round both to bf16. At the default `matmul_precision` Dew now rounds as they do: the logits to bf16 values, and their gradient to bf16 once, which both backward products read. On the RTX 4080 (`tools/benchmark_step.py --fixed-batch`, one session, against `66784383`) the 3-layer decoder (GPT-2 small widths, vocabulary 50304, 16 x 512 tokens) runs in 52.17 ms against 59.95, with a planned peak of 4.53 GB against 5.36; torch.compile runs it in 49.4-50.0. Qwen3-0.6B's widths at 1 x 1024 run in 106.62 ms against 110.04.

To check training quality, I trained for 2000 steps on wikitext-103 Qwen3 tokens on the same card, with validation over 64 fixed windows scored with fp32 logits for both. The 3-layer decoder at vocabulary 151936, trained from scratch, ends at 5.0604 and 5.0491 with fp32 logits (seeds 0 and 1) against 5.0604 and 5.0493 with the bf16 rounding, and Qwen3-0.6B fine-tuned ends at 2.71965 and 2.71996 against 2.71983 and 2.71981. At the same seed the two roundings are at most 4.5e-4 and 1.6e-3 apart at any checkpoint, while the two seeds of either rounding are 1.3e-2 and 2.9e-3 apart on average.

The high half existed for layout parity. With the gradient rounded once, a 4 x RTX 3090 bf16 run read 1.75 times its bound at a dense model's final norm and 5758 times it at an MoE's expert gate_proj, against 0.47 and 0.41 with the high half. Why the MoE's gap is that large is not yet established. A run that compares layouts in bf16 sets `matmul_precision="highest"`, which keeps the head fp32, and `tools/layout_parity.py` does so for bf16 decoders.

On sm80 and sm89 the trainer compiles without XLA's Triton GEMM fusions, unless the run sets that flag explicitly or the model has an SSD mixer (`TRITON_GEMM_OFF_GENERATIONS`). On an A100 (jax 0.11.2, bf16), through the trainer, Qwen3-0.6B at 4 x 512 tokens compiled in 27.1 s against 47.5 s, because there is no Triton GEMM autotuning, and stepped in 161.4 ms against 161.5. Standalone, Qwen3-0.6B at 4 x 1024 tokens went from 162.1 to 153.1 ms, a 99M MoE from 74.6 to 69.4 ms, and a DiT ran 5.8% faster. A Mamba-2 step lost 7.7% (127.9 to 138.6 ms), because its SSD scan's small batched dots gain from the fusions.

On the RTX 4080, two-layer steps at 2048, 4080 and 16384 tokens go from 56.7 to 53.5, 103.8 to 94.0 and 420.5 to 406.3 ms, and tiled heads run 3-6% faster. At Qwen3-0.6B's widths with two layers, bf16, vocabulary 151936 and a 0.9 allocator fraction on an RTX 4080 (JAX 0.11.2), this removes a cliff at 4096 tokens, where a training step took 286.0 ms with the fusions and takes 93.4 ms without. At other shapes the unfused step can use more temporary memory, so before tiling the head or recomputing blocks, the trainer tries a step that does not fit with XLA's default options. At 8192 tokens only that whole-logits step fits (178.7 ms, against 211.2 ms after tiling). When tiling is needed, sm89 uses the measured 4096-by-8192 tile. These are two-layer measurements, not full-model times.

On every GPU the trainer also compiles its step without XLA's dot merger (`--xla_gpu_dot_merger_threshold_mb=0` in the step's own compiler options, `step_compiler_options`). The exceptions are a run that sets that flag, and a step that trains next to frozen weights on 128 tokens or fewer a device (below). The merger runs dots that share an input (q, k and v; gate and up) as one GEMM over their weights, concatenated afresh every step, which cost 4.0 ms of Qwen3-0.6B's step at 1 x 1024. On the RTX 4080 (benchmark_step, two rounds, one session), Qwen3-0.6B's widths at 1 x 1024 run in 97.7-98.0 ms with the merger against 94.1-94.2 without it, the 3-layer decoder in 50.8-50.9 against 49.1-49.2, SimpleDiT-B in 73.1-73.2 against 72.8, and the 176M hybrid DiT in 66.7-67.6 against 66.4-66.7, with the peaks unchanged.

On an A100 40 GB (Colab, `db1761fd`, benchmark_step with one batch on the device, two rounds), I set XLA's merger explicitly with `--xla_gpu_dot_merger_threshold_mb=64` against the step's 0. Qwen3-0.6B's widths at 4 x 1024 run in 128.40-128.41 against 123.85-124.01 ms, the 3-layer decoder at 16 x 512 in 22.96-23.01 against 22.12-22.51, and SimpleDiT-B at batch 32 in 38.47-38.51 against 38.39-38.48. Over 2000 steps of wikitext-103, two seeds each, validation loss at the same seed moved by at most 1.5e-3 on a 3-layer decoder from scratch (whose seeds are 1.2e-2 apart on average) and 2.2e-3 on Qwen3-0.6B fine-tuned (seeds 3.0e-3 apart). Serving keeps the merger, because decoding Qwen3-0.6B at 32 slots ran 4.6-14.8% slower without it; its 32-token GEMMs lose more to separate launches than the concatenations cost.

Small training steps, 2026-10-02 (RTX 4080, bf16, `Trainer.compile` with and without the option in one process, five alternating blocks of 20 steps, medians). A full step, with every weight training, is faster with separate dots from 32 tokens up. A LoRA step (rank 16 on Qwen3-0.6B's seven projections, with the base frozen) of 128 tokens or fewer is slower with separate dots, by up to 0.6 ms, except at 1 x 128 (measured in two sessions).

The rule therefore considers both token count and frozen weights. A
step whose starting variables are split frozen, as in LoRA, keeps the merger
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

With the rule, I measured the same steps against XLA's default (the merger on) the same way, in a second session at `perf/merger-frozen`, with a load average of 15-27 from other work on the host. No row is slower. The LoRA steps of 128 tokens or fewer compile the same program both ways, so their spread of -0.3% to +0.3% is the measurement's own.

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

At 128 tokens the shapes disagree. In two more sessions, 1 x 128 ran in 20.65 and 20.65 ms merged against 18.98 and 19.11 with separate dots. 2 x 64 went the other way, 18.72 and 18.63 merged against 18.96 and 19.03, and so did 4 x 32, 18.56 and 18.50 against 18.91 and 18.81. 1 x 96 also ran faster merged (18.22 and 18.22 against 18.41 and 18.43), and 1 x 160 faster with separate dots (22.33 and 21.95 against 20.57 and 20.37). A boundary below 128 would run 2 x 64 and 4 x 32 1.3-2.1% slower than XLA's default, so the boundary includes 128, and 1 x 128 runs at XLA's default, 1.6 ms behind separate dots.

### The forward's bf16 weights: `NARROW_COPY_GENERATIONS`, 2026-10-03

A bf16 model over fp32 parameters cast each weight to bf16 in the forward,
one CUDA kernel per weight every step (5.7 ms of casts on Qwen3-0.6B at
1 x 1024 on the RTX 4080), and widened each weight's bf16 gradient back to
fp32 in the backward (1.8 ms). On `sm80` and `sm89` the update now writes
the bf16 copy of each such weight (`TrainState.compute`) from the new fp32
value. The forward reads that copy, and the gradient reaches the update in
bf16 and is widened as the update reads it (`dew.training.narrow`). Only a
weight whose one use in the loss is that cast gets a copy. A tied
embedding's table has two uses, so its cotangents sum in fp32, and it keeps
its fp32 read; so do the norms' scales, which are not read through a lone
cast, and any weight a custom VJP reads. The step against its parent on the
RTX 4080, two alternating rounds, in ms:

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

No row moved to another rung of the fit ladder. The MoE gains nothing,
because its experts run through the grouped matmul, which reads them another
way. On an A100 40 GB (Colab, integration `6a220e31`, copies off and on in
one session, two alternating rounds, ms):

| row | before | copies | peak GiB |
|---|---:|---:|---:|
| Qwen3-0.6B, 4 x 1024, AdamW (dew_lm) | 128.45 / 128.37 | 125.88 / 125.93 | 16.08 -> 16.08 |
| 176M hybrid DiT, batch 16 | 45.71 / 45.01 | 40.79 / 41.07 | 5.77 -> 5.78 |
| 176M hybrid DiT, batch 32 | 63.96 / 63.96 | 62.66 / 62.89 | 8.43 -> 8.45 |
| SimpleDiT 768, batch 32 | 39.26 / 39.30 | 38.24 / 38.24 | 6.12 -> 6.13 |

The forward and the gradients are bitwise those of the cast. Under plain SGD
every parameter is bitwise the same after three steps
(tests/test_narrow.py). The update's arithmetic rounds differently, because
with the widening inside it XLA contracts its fp32 multiply-adds another way
(Adam's second moment is within 1 ulp after two steps); every accumulation
is still fp32. A deterministic Qwen3-0.6B run (dew_lm, AdamW and its clip)
is bitwise the same for 13 steps and within 2.7e-3 of the loss at step 40,
where two default runs of the parent differ by 3.9e-3.

I also ran `tools/lm_step_parity.py`'s decoder (100 steps) and the hybrid
DiT on one batch (300 steps), twice each way. The DiT's four runs are
bitwise equal at every step, and the decoder's two runs with copies equal
one of the parent's two at every step, where the parent's pair is at most
6.9e-4 apart. On the A100, whose runs are not repeatable, the decoder's runs
with and without copies are at most 1.13e-3 apart at any step, within the
8.3e-4 and 1.13e-3 by which each side's pair differs. The DiT's are at most
2.26e-2 apart, against 2.23e-2 and 1.53e-2 within its pairs, and the final
losses are 0.36587 and 0.36163 without copies and 0.36583 and 0.36631 with
them.

A TPU fuses the cast into the matmul, so there the copies would only add
writes. Compiled for a v6e, Qwen3-0.6B's widths at 8 x 1024 write 85.8 GiB a
step with the copies against 79.8 without, with 2.1 GiB more temporaries.

### Generations below sm80

A T4 (sm75) rejects the `BF16_BF16_F32` dot algorithm at run time ("UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32"), cuDNN's fused attention refuses bf16 there ("SDPA FP16/BF16 requires SM80"), and Triton does not compile for it. `dew.nn.kernels.generation.bf16_dot_runs` is the single check for all three. Below sm80, bf16 attention takes the reference path for `auto` and `xla`, the bf16 operand precision keeps the caller's precision, and the grouped matmul runs XLA.

### Faster kernels not adopted

Kernel matrix, 2026-09-22, jax 0.11.2, forward plus backward medians, every cell checked against float64. Each of these needs a kernel or a dependency that Dew does not have yet:

| op | where it wins | numbers | why not yet |
|---|---|---|---|
| tokamax `ragged_dot` `mosaic_tpu_v2` with tokamax's own VJP | TPU v5e and v6e, 8 experts | lm-moe up 0.899 ms against XLA's 1.05 (v5e), 0.395 against 0.48 (v6e); down 0.859 against 1.19 (v5e), 0.345 against 0.383 (v6e) | tokamax 0.0.14 pins typeguard==2.13.3 and tyro needs >=4; its flax.nnx import fails on jax 0.11.2. At 128 experts XLA wins. |
| tokamax `triton` RMSNorm | sm80, sm89 | 1.07x (A100), 1.53x (L4), 1.57x (RTX 4080) over XLA | tokamax dependency. |
| fused-weight SwiGLU (tokamax `xla` formulation, one contraction for gate and up) | every GPU | 1.47x (A100), 1.47x (L4), 1.55x (RTX 4080), 1.21x (T4) over two XLA matmuls; no gain on TPU | a change to the MLP's parameter layout. |
| tokamax `xla` head plus cross entropy | sm89 speed | 73.0 ms (L4) and 36.8 ms (RTX 4080), 1.55x and 1.58x over Dew's chunked head | tokamax dependency, and it holds 1.6-3.2 GiB where the chunked head holds 131-355 MiB. |
| JAX's Pallas-Triton `mha` (`jax.experimental.pallas.ops.gpu.attention`), 2026-10-01 | sm89, training attention | forward plus backward, bf16, against cuDNN: 0.38 against 0.64 ms (batch 32, 256 tokens, 12 heads of 64), 0.43 against 0.81 (causal, batch 16, 512 tokens), 1.12 against 1.36 (causal, batch 4, 1024 tokens, 16 heads of 128); in the step, routed where a call has no bias, mask, window or lengths: the 768-wide SimpleDiT 76.10 to 74.81 ms, the small decoder 63.61 to 62.86 | deprecated in JAX 0.11 for tokamax; against an fp32 reference its dq and dk errors reach 8.5e-3 of their maximum where cuDNN's reach 5.3e-3 (causal, 512 tokens); no grouped-query heads. |
| JAX's Pallas GPU `paged_attention` | sm80 and later, decode | 1.75-1.84x (A100), 1.85-2.1x (L4), 2.1-2.8x (RTX 4080) over the XLA gather | it keeps its unnormalized sums and split partials in the query's dtype and divides by a bf16 denominator; Dew's own paged decode kernel (above) rounds once. On TPU the XLA gather wins at batch 8 and up to 2k context. |

### The Mamba-2 SSD scan: `ssd_kernel_runs`

`tools/benchmark_ssd.py`, forward plus backward, batch 1, 8 heads of 64, state 128, jax 0.11.2:

| device | chunk | length | XLA | Pallas kernel |
|---|---|---|---|---|
| TPU v6e | 256 | 4096 | 0.915 ms | 0.634 ms |
| TPU v6e | 256 | 16384 | 4.505 ms | 1.911 ms |
| TPU v6e | 256 | 65536 | 17.245 ms | 7.122 ms |
| TPU v6e | 128 | 4096 | 0.632 ms | 0.705 ms |
| RTX 4080 | 256 | 4096 to 65536 | 2.28 to 33.63 ms | does not compile: 590 KB of shared memory asked, 101 KB available |

On the RTX 4080 the Triton kernel ran 6x to 12x slower than XLA wherever it compiled (chunk 64, width 32: 1.39 against 0.22 ms; at batch 8 and 16 heads, 22.7 against 2.2 ms), and every chunk of 128 or 256 asked for 131 to 590 KB of shared memory. So the scan takes the kernel on TPU only.

### Packed sliding-window attention on GPU: `local_attention`

A packed batch with a sliding window has no fused-kernel option on a GPU before Hopper. `jax.nn.dot_product_attention` takes no segment ids next to `local_window_size`; cuDNN's packed layout (`q_offsets`) raises "Packed layout requires a GPU with at least Hopper architecture" on sm89; and JAX's Pallas GPU `mha` takes segment ids but no window (and its gradient was 4-9% off at this shape). So `local_attention` builds its `[W, 2W]` band mask and, where cuDNN runs, passes it to cuDNN as the additive bias; elsewhere xla takes it. Colab L4, jax 0.11.2, bf16, 16 query heads of 64 over 4 key heads, window 4096, 5 packed documents, forward plus backward:

| tokens | before (band on xla) | after (band on cuDNN) |
|---|---|---|
| 2048, window 512 | 6.15 ms, 0.32 GiB | 1.37 ms, 0.05 GiB |
| 8192 | out of memory (10.0 GiB requested) | 40.0 ms, 0.24 GiB |
| 16384 | out of memory | 76.9 ms, 0.60 GiB |
| 32768 | out of memory | 156.5 ms, 1.19 GiB |
| 65536 | out of memory | 316.7 ms, 2.38 GiB |

Against a float64 oracle at 2048 tokens the output error is 2.7e-3 relative and the gradients' 3.3e-3 to 6.6e-3, the same as the xla path's. A dense `[S, S]` document mask on cuDNN is faster at 32768 tokens on an RTX 4080 (63.7 against 78.1 ms), but it grows with the square of the length and ran out of memory at 65536, so Dew uses the band.

## Expert parallelism on 4x RTX 3090, 2026-09-23

The machine is one host with four RTX 3090s. GPU0 and GPU1 are joined by NVLink (NV4), GPU2 and GPU3 share a PCIe host bridge, and every other pair crosses the two sockets. The runs use jax 0.11.2, bf16 compute and the Pallas grouped matmul, through `tools/benchmark_step.py` with 12 timed steps after 3 warmup and 3 traced. Each row's attribution is benchmark_step's reading of its trace, in milliseconds per device per step: compute kernels, each collective, and the communication that no compute kernel overlapped. The model is a `causal_transformer` with 8 layers, width 1024, 16 heads and vocabulary 50304, with 32 experts of width 1024 and top-4 routing on every layer, at 4096 tokens a device (batch 16 of 1024 on four GPUs, 8 on two).

I measured the links first, with JAX collectives of 128 MB of bf16 a device. `all_to_all` moves 33.4 GB/s over the NVLink pair, 6.9 over the PCIe pair and 6.7 across the sockets; `all_gather` gives 31.0, 5.8 and 5.8, and `psum` 33.4, 5.7 and 6.3. The PCIe pair is no faster than a cross-socket pair, so GPU0 and GPU1 are the only fast pair on this box.

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

From the traces:

- The exchange beats the global dispatch wherever the link is slow: 1.46x on four GPUs, 1.71x on the PCIe pair and 1.75x across the sockets. On the NVLink pair the two tie, because the global dispatch's expert all-gather and gradient reduce-scatter cost 92 ms there, against 622 ms on the PCIe pair.
- Most of the communication is exposed. The four-way exchange spends 272 ms computing and 352 ms waiting on collectives that no compute overlaps, most of it the all-to-all. Capacity 1.25 bounds the buckets and drops the later rounds, which takes the step to 550.6 ms.
- Under data x expert, the gradient all-reduce over the data axis moves more bytes than the token exchange, so the expert axis belongs across the sockets and the data axis on the pairs (702.9 ms against 753.8). Under expert x fsdp the two placements tie (788.1 against 800.3), because fsdp's all-gather and reduce-scatter trade places with the all-to-all.

I made three changes, each measured before and after in one hold:

- Expert parameters enter the dispatch's `shard_map` in their stored shards and are gathered inside it, so their gradient is reduce-scattered and not all-reduced whole. fsdp 4 goes from 1157.1 to 904.0 ms, as a 614 ms all-reduce becomes a 375 ms reduce-scatter, and expert 2 x fsdp 2 from 856.0 to 788.1 ms.
- The exchange's first round runs outside the checkpointed scan that holds the later rounds, so the backward keeps its intermediates and does not recompute them. expert 4 goes from 705.9 to 634.0 ms, and its compute from 307.0 to 272.4.
- The exchange gathers each bucket's rows through the sort's index and scatters what returns straight to its slots, which saves two row copies a round. On the NVLink pair, dropless goes from 300.5 to 295.8 ms and capacity 1.25 from 218.5 to 212.3, with peak memory going from 14.1 to 13.7 GiB and from 12.6 to 12.0.

I also timed one bf16 `ExpertMLP` layer forward plus backward under `MeshSpec(fsdp=2)` on the NVLink pair (8192 tokens, 32 experts, top 4). The Pallas kernels inside the dispatch's map take 26.2 ms (25.1 under the earlier row map), and `jax.lax.ragged_dot` inside the map takes 271.9 ms with 6.5 GiB of temporaries, because XLA runs it as a product over every expert. The same layer through the global path outside any map, which is where the earlier kernel selector sent fsdp-only meshes to XLA, ran out of memory on the 24 GiB cards.

### Rematerialization: the trainer's ladder

A model's `remat` is where its step starts, and the trainer moves it up one rung whenever the compiled step does not fit its devices' memory. A decoder goes from none to `'minimal'` (MaxText's name: every projection output kept) to `'full'`, and a diffusion backbone from `False` to `'dots'` (matmul outputs and the attention forward kept) to `'full'`. Each rung is slower and smaller, so the first that fits is the fastest that runs.

`dew.training.trainer.step_fits` decides whether a step fits. XLA places a GPU step's temporaries in one allocation, so the temporaries, the outputs that do not reuse the donated state and the batches `fit` prefetches next to the step's own all have to fit in a single free block on each device, which can be smaller than the device's total free bytes. The block is the BFC pool's largest, or the part of its limit that a growing pool has not taken yet. An allocator that reports no pool (cuda_async, or a TPU's) is read by its free bytes. Each process reads its own devices, and the pool of processes takes the tightest. Where the allocator can leave the temporaries no block to return to once a batch is prefetched next to them, the check holds room for them twice (`strands_temporaries`). That happens with cuda_async and with XLA's spatially partitioned pool. `import dew` now turns the partitioning off wherever JAX's CUDA plugin is installed; before, only `prepare_process` did. So a Trainer built without `prepare_process`, such as `tools/reference_runs/dew_lm.py` or a notebook, held room for the 99M MoE's 8.34 GiB of temporaries twice in a preallocated pool on the RTX 4080, and tiled its head at 98.2 ms a step. With the partitioning off it keeps the whole logits at 78.3 ms, from a cold compile cache and from a warm one.

A pool that grows (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) never gives back a region it has grown (openxla/xla#50052). A process that compiles the step autotunes it, and the pool grows for the autotuner's scratch. A process that loads the step from the persistent cache does not. So the same program can fit in the second process and not in the first: the MoE kept its whole logits from a warm cache and tiled its head from a cold one. A check that ignored where the free bytes lie said it fit in the cold process, which then ran out of memory allocating its 8.34 GiB of temporaries.

A checkpoint records the rung its state trained on (head tile, remat and XLA options). A resumed run compiles that rung, climbs from it only if it does not fit, and reports any climb. A fresh run does the same with the rung that an earlier run of the same step chose. That rung is recorded beside the persistent compilation cache (`dew-rungs/`), keyed by the step's program at its starting rung, the devices' kind and allocator limit, and XLA_FLAGS. Identical runs can read different free memory: a process that compiles the step autotunes it, and a growing pool keeps the autotuner's scratch. Without the record, the 99M MoE at 8 x 1024 tiled its head in the run that compiled it (98.2 ms a step) and kept its whole logits in the next one (78.4). With it, both take the first run's rung. If a record's key does not match the run, or the objective cannot take its rung, the run refuses it and names its path. Deleting the record lets the run decide again, for example after a run that shared the device with another process. A resumed run never steps down to a lighter rung: the restoring process may see different free memory than the one that built the state, and a lighter rung would run a different program. The rung a step compiled under is the `remat` of the run's `StepCompiled` record and of `tools/benchmark_step.py`'s rows. The table times forward plus backward plus AdamW in bf16 compute over 10 timed steps, with `tools/benchmark_kernels.py step --remat`:

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

In the rows where all three ran (the decoders), `'minimal'` costs 4-10% over no recomputation and `'full'` 15-21%, so a model that fits runs without either. The L4 rows are jax 0.11.2 on Colab (2026-09-23), and the RTX 3090 rows ran on one GPU of the box.
