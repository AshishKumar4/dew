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
