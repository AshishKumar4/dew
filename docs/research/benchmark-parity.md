# Dew and PyTorch benchmark parity

## Result

Dew is not running at 1.6% utilization on the small UNet. The counter is wrong for that executable. The optimized HLO contains 646.39 GFLOP per step. XLA `cost_analysis()` reports 28.73 GFLOP. It omits 623.43 GFLOP in 131 cuDNN convolution calls. The matched step takes 16.32 ms. That is 39.6 analytic TFLOP/s, or 40.6% of Dew's 97.5 TFLOP/s denominator. Evidence: `/tmp/benchmark-parity/audit/unet_b16.rows.json` and `/tmp/benchmark-parity/jax_results.json`.

The same counter error affects the small language model. The optimized HLO contains 3.406 TFLOP per step. XLA reports 1.436 TFLOP in the audit compile. It omits 1.986 TFLOP in six cuBLAS custom calls. Three of those calls are the tied vocabulary head and its two gradients. Evidence: `/tmp/benchmark-parity/audit/causal_transformer_b16.rows.json`.

`simple_dit` is the control. Its HLO contains 292.95 GFLOP. `cost_analysis()` reports 296.89 GFLOP in the audit compile. The difference is the compiler's elementwise accounting. There is no large missing custom-call term. Evidence: `/tmp/benchmark-parity/audit/simple_dit_b16.rows.json`.

Against a matched PyTorch port, Dew is close on every case and ahead on three. Dew is 1.04x the best PyTorch UNet row (16.32 ms against 16.96 ms with `max-autotune`), 1.16x on the small DiT with fused attention (6.89 ms against 7.96 ms), 1.08x on the large DiT (72.34 ms against 78.31 ms), 0.95x on the small language model (75.70 ms against 72.18 ms), and 0.99x on the large language model (146.90 ms against 145.25 ms).

The small cases are not host-dispatch bound on this machine. The profiler found GPU-busy fractions of 97.3% for UNet, 99.1% for `simple_dit`, and 100.0% for the causal transformer. The many small kernels still matter. They do not leave enough idle time to explain the reported utilization. Evidence: `/tmp/benchmark-parity/traces/*/plugins/profile/*xplane.pb` and the parsed profiles in `/tmp/benchmark-parity/jax_results.json`.

## Scope and fixed point

The measurements were taken at Dew commit `d0d04c2`. The repository moved to `b6282df` while this ran; that range touches only `README.md`, `CONTRIBUTING.md`, `docs/index.md`, `docs/research/domain.md` and a workflow file, so no benchmarked code changed. The machine has one NVIDIA GeForce RTX 4080 16 GiB and driver 595.84. The JAX environment is Python 3.12.13, JAX and jaxlib 0.11.1, Flax 0.12.9, Optax 0.2.8, cuBLAS 12.9.2.10, and cuDNN 9.25.1.1. The PyTorch environment is `/tmp/torchbench`, Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, and cuDNN 9.19.0. Version commands and output are in `/tmp/benchmark-parity/versions.txt`.

All GPU runs started only when `nvidia-smi --query-compute-apps=pid --format=csv,noheader` showed the desktop process, PID 3243, and no training process. Every case ran in a fresh process for peak-memory measurement. The ceiling scripts sampled SM clock, board power, and temperature every 50 ms with `nvidia-smi`. Raw rows are in `/tmp/benchmark-parity/ceilings_jax.json` and `/tmp/benchmark-parity/ceilings_torch.json`.

## Fairness checklist

| item | matched setting | evidence |
|---|---|---|
| Small causal transformer | vocabulary 50,304; width 768; 3 layers; 12 query heads; 12 KV heads; head width 64; SwiGLU ratio 4; RoPE theta 10,000; q/k RMSNorm; tied embedding; sequence 512; batch 16 | Dew preset: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L180-L184). Dew model: [`causal_transformer.py`](../../src/dew/nn/backbones/causal_transformer.py#L90-L147), [`causal_transformer.py`](../../src/dew/nn/backbones/causal_transformer.py#L231-L361). Torch port: [`tools/benchmark_torch.py`](../../tools/benchmark_torch.py). Both parameter counts are 66,950,784. |
| Large causal transformer | same vocabulary and width; 12 layers; 12 heads; sequence 1,024; batch 8 | Case constructors in both harnesses. Both frameworks use the same formula and token count. |
| Small `simple_dit` | 64 px RGB; patch 4; 256 image tokens; width 384; 6 layers; 6 heads; MLP ratio 4; adaLN-Zero; RoPE; batch 16; 77 by 768 stub text context | Dew preset: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L143-L165). Dew model: [`backbones/dit.py`](../../src/dew/nn/backbones/dit.py#L13-L91), [`nn/dit.py`](../../src/dew/nn/dit.py#L63-L190), [`nn/dit.py`](../../src/dew/nn/dit.py#L216-L373). Both parameter counts are 19,835,568. |
| Large `simple_dit` | 64 px RGB; patch 4; width 768; 12 layers; 12 heads; batch 32 | Case constructors in both harnesses. |
| UNet | exact small preset, including `[64, 128, 256]` depths, two residual blocks per level, one middle pair, 4-head cross-attention at the last two levels, nearest upsampling, NHWC input, GroupNorm, and the existing upsample channel index | Dew preset: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L151-L154). Dew model: [`unet.py`](../../src/dew/nn/backbones/unet.py#L48-L219). Both parameter counts are 10,159,299. Torch's operator FLOP counter and the HLO parser both return 646.393 GFLOP for the matched forward and backward. |
| Compute and master dtype | fp32 parameters and optimizer state; bf16 Dense and Conv compute; fp32 norm statistics; fp32 loss | Dew's precision policy says parameters stay fp32: [`registry.py`](../../src/dew/registry.py#L115-L149). LM head and logits are fp32: [`causal_transformer.py`](../../src/dew/nn/backbones/causal_transformer.py#L341-L361). LM cross entropy is fp32: [`objective.py`](../../src/dew/objectives/lm/objective.py#L86-L108). Runtime leaf inspection is in the audit JSON. |
| fp32 matrix precision | TF32 is enabled in both frameworks for fp32 matmuls, matching XLA's default on this executable | A numerical probe gives maximum relative error 3.33e-4 for XLA default and 1.75e-6 for `precision='highest'`. The 8,192 by 768 by 50,304 head reaches 49.5 TFLOP/s in JAX default and 50.4 TFLOP/s in PyTorch with TF32. Commands are in `Reproduction`. |
| Attention | reference against reference; JAX cuDNN against PyTorch SDPA with the backend forced to `CUDNN_ATTENTION`; `is_causal=True` for the LM | Dew's dispatch is in [`attention.py`](../../src/dew/nn/attention.py#L97-L176). PyTorch forces the backend with `sdpa_kernel` in [`tools/benchmark_torch.py`](../../tools/benchmark_torch.py). JAX documents that `implementation='cudnn'` supports only a subset of shapes and dtypes: [JAX API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html). PyTorch documents the backend selector: [PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html). |
| Diffusion inputs | same fixed image, CFG mask, timestep, and noise arrays from NumPy `default_rng(0)`; same normalization and EDM equations; arrays stay on device | Dew's real objective is [`objectives/diffusion/objective.py`](../../src/dew/objectives/diffusion/objective.py#L55-L96). The paired fixed-input harnesses generate byte-identical arrays. The equality command prints `True` and maximum difference `0.0` for all four arrays. |
| LM input | same fixed NumPy `default_rng(0)` integer token batch; int32; one extra target token | Dew batch creation: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L316-L329). |
| Optimizer | Adam, learning rate 1e-4, betas 0.9 and 0.999, epsilon 1e-8; PyTorch uses `fused=True` | Dew builds `optax.adam(1e-4)`: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L268-L283). Torch setup is in [`tools/benchmark_torch.py`](../../tools/benchmark_torch.py). |
| EMA | full fp32 parameter copy; decay 0.999; update every optimizer step | Dew update: [`objective_trainer.py`](../../src/dew/training/objective_trainer.py#L44-L55), [`objective_trainer.py`](../../src/dew/training/objective_trainer.py#L269-L287). Torch uses one `torch._foreach_lerp_` call with weight 0.001. |
| Other step work | no gradient accumulation, no clipping, no dynamic loss scale; loss finiteness check included | Dew's step: [`objective_trainer.py`](../../src/dew/training/objective_trainer.py#L227-L297). `grad_accum_steps` defaults to one: [`objective_trainer.py`](../../src/dew/training/objective_trainer.py#L69-L125). |
| Data path | primary parity rows reuse one fixed device batch; separate H2D ablations copy every step | Dew's normal benchmark uses a depth-two device prefetch thread: [`distributed.py`](../../src/dew/training/distributed.py#L78-L126). |
| Timing | at least 20 setup steps and 100 measured warm steps; wall clock around the whole loop; one final device sync; five profiler steps after the timed window | Existing Dew timing uses asynchronous calls and one final sync: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L352-L399). The parity harness extends its ten-step window to 100 steps and adds percentiles. |
| Percentiles | PyTorch uses CUDA events around each asynchronous step; JAX uses a separate loop with one `block_until_ready` per step | These are not the same quantity. Wall-clock ms/step is the primary comparison. The p10/p50/p90 columns diagnose variance. |
| Peak memory | fresh process per row; framework allocated high-water mark, not reserved-pool size | Dew explains its monotonic allocator counter: [`tools/benchmark_step.py`](../../tools/benchmark_step.py#L332-L345). Torch reports both allocated and reserved bytes; tables use allocated. |
| Compile modes | PyTorch eager, `torch.compile` default, and `torch.compile(mode='max-autotune')`; JAX AOT compile | PyTorch documents that `max-autotune` enables CUDA graphs by default and profiles ATen, Triton, and CUTLASS candidates: [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html). |

## FLOP convention

A multiply-add is two FLOPs. Model FLOPs include the model forward and backward. They exclude Adam, EMA, RNG, loss finiteness, host copies, and scalar telemetry. This is MFU. Counting optimizer and loss operations would be hardware FLOP utilization, or HFU. The old number is neither. It mixes compiler-visible model, loss, optimizer, and telemetry operations while omitting compiler-opaque cuDNN and cuBLAS calls.

The formulas count the full attention square for causal attention. They do not halve it for the triangular mask. The same formula is used for JAX and PyTorch.

### Causal transformer

Let `B` be batch, `S` sequence length, `T = B S`, `d` width, `f` MLP width, `L` layers, and `V` vocabulary. The tied input embedding gather costs zero matmul FLOPs. The tied output head is one `d by V` matrix.

```text
N_matmul = d V + L (4 d^2 + 3 d f)
F_matmul = 6 T N_matmul
F_attention = 12 L B S^2 d
F_step = F_matmul + F_attention
```

The `3 d f` term is SwiGLU's gate, up, and down matrices. For the small case this gives 3.40644593664 TFLOP. Of that, the fp32 tied head and its gradients are 1.898912415744 TFLOP. The attention term is 0.115964116992 TFLOP.

### SimpleDiT

Let `P = (H/p)(W/p)` image tokens and `r d` MLP width.

```text
F_blocks = 6 B P L (4 d^2 + 2 r d^2)
F_attention = 12 L B P^2 d
F_step = F_blocks + F_attention + F_patch + F_output + F_conditioning + F_adaLN
```

`F_patch` and the text projection use a factor of four. Their inputs require no gradient, so each has a forward matmul and a weight-gradient matmul. Trainable paths use six. The exact small-case sum is 292.949065728 GFLOP. The independent PyTorch `FlopCounterMode` result is the same to the reported integer FLOP.

### UNet

The UNet formula is over the executed optimized operations. For each convolution:

```text
F_conv = 2 B Ho Wo Co Kh Kw Ci / groups
```

For each dot:

```text
F_dot = 2 product(output dimensions) product(lhs contracting dimensions)
```

Summing the forward, input-gradient, and weight-gradient operations gives 623.432957952 GFLOP in 131 cuDNN convolution calls and 22.960144384 GFLOP in 84 dots. The total is 646.393102336 GFLOP. PyTorch's operator FLOP counter gives the same total for the port.

## FLOP audit

| workload | XLA `cost_analysis()` | HLO analytic | analytic / XLA | opaque work found in HLO |
|---|---:|---:|---:|---|
| single bf16 3 by 3 NHWC Conv, `(16,64,64,64)` to 64 | `-1` | 4.8318 GFLOP | unknown | one `__cudnn$convForward` |
| same Conv, forward and two gradients | 0.0084 GFLOP | 14.4955 GFLOP | 1,728x | three cuDNN calls |
| bf16 GEMM, `4096 by 384 by 1536` | 4.8318 GFLOP | 4.8318 GFLOP | 1.000x | none in this compile |
| UNet train step | 28.7289 GFLOP | 646.3931 GFLOP | 22.500x | 623.4330 GFLOP in cuDNN calls |
| SimpleDiT train step | 296.8867 GFLOP | 292.9491 GFLOP | 0.987x | 0.3020 GFLOP in cuDNN calls; XLA also counts elementwise work |
| causal transformer train step | 1,436.1024 GFLOP | 3,406.4459 GFLOP | 2.372x | 1,985.8855 GFLOP in six cuBLAS calls |

The full-step XLA value changes across identical recompiles. The compiler sometimes selects a visible Triton dot and sometimes a cuBLAS custom call for the same matmul. The small LM audit compile reports 1.436 TFLOP. Another compile reports 2.069 TFLOP. The analytic HLO sum remains 3.406 TFLOP. This makes the current logged MFU dependent on compiler kernel selection even when the math and runtime are unchanged. Raw splits are in `/tmp/benchmark-parity/jax_results.json`.

JAX's own AOT documentation warns that text and cost-analysis methods are aids for manual inspection, not a stable programmatic API, and that output varies by compiler, platform, and runtime: [JAX AOT documentation](https://docs.jax.dev/en/latest/aot.html).

## Card ceilings

Every row uses ten warmup calls, five measured repeats, and one final sync per repeat. Tiny GEMMs use 200 calls per repeat. Large GEMMs use 20 or 30. The reported clock and power are medians of 50 ms samples. Short rows contain some idle samples, so they are clock records, not energy measurements.

| operation | JAX | JAX SM clock / power | PyTorch | PyTorch SM clock / power |
|---|---:|---:|---:|---:|
| bf16 GEMM `4096x384x384` | 62.1 TFLOP/s | 2,535 MHz / 32 W | 49.1 TFLOP/s | 2,535 MHz / 31 W |
| bf16 GEMM `4096x384x1536` | 79.3 TFLOP/s | 2,535 MHz / 41 W | 72.9 TFLOP/s | 2,535 MHz / 34 W |
| bf16 GEMM `8192x768x3072` | 93.3 TFLOP/s | 2,535 MHz / 56 W | 90.6 TFLOP/s | 2,535 MHz / 45 W |
| bf16 GEMM `8192x768x50304` | 103.1 TFLOP/s | 2,730 MHz / 91 W | 100.2 TFLOP/s | 2,760 MHz / 55 W |
| bf16 GEMM `8192^3` | 102.1 TFLOP/s | 2,760 MHz / 175 W | 102.5 TFLOP/s | 2,745 MHz / 299 W |
| TF32 GEMM `8192x768x50304` | 49.5 TFLOP/s | 2,745 MHz / 242 W | 50.4 TFLOP/s | 2,745 MHz / 279 W |
| strict fp32 GEMM, same shape | 32.4 TFLOP/s | 2,535 MHz / 213 W | 32.7 TFLOP/s | 2,430 MHz / 317 W |
| bf16 NHWC 3 by 3 Conv forward | 73.6 TFLOP/s | 2,715 MHz / 309 W | 88.9 TFLOP/s | 2,385 MHz / 319 W |
| same Conv, forward and backward | 63.7 TFLOP/s | 2,760 MHz / 267 W | 57.5 TFLOP/s | 2,760 MHz / 228 W |
| 1 GiB fp32 scale, read plus write | 610 GB/s | 2,760 MHz / 98 W | 610 GB/s | 2,760 MHz / 198 W |

The common measured bf16 peak used below is 102.5 TFLOP/s. It is the best sustained `8192^3` result. The 97.5 TFLOP/s column keeps Dew's existing denominator from [`instrumentation.py`](../../src/dew/telemetry/instrumentation.py#L8-L24). The bandwidth denominator is 716.8 GB/s. NVIDIA's official specification gives 22.4 Gbps GDDR6X on a 256-bit interface, which is 716.8 GB/s: [NVIDIA RTX 4080 family](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/).

The nominal 97.5 TFLOP/s denominator is not a sustained ceiling in this session. It is lower than the 102.5 TFLOP/s large-GEMM measurement because the card sustained 2,745 MHz. It is also too high for the real small shapes. The two DiT GEMMs reach 62.1 and 79.3 TFLOP/s in JAX. The LM body shape reaches 93.3 TFLOP/s. Its fp32 vocabulary head reaches only 49.5 TFLOP/s.

## Side-by-side training results

The tables use analytic model FLOPs. `MFU spec` divides by 97.5 TFLOP/s. `MFU measured` divides by 102.5 TFLOP/s. The latter is still a nominal mixed-workload number, so the shape table above is the useful ceiling for diagnosis. Every headline row reuses one on-device batch, and the diffusion rows use identical fixed input tensors on both sides.

Two coverage gaps are open. The card is shared with other work in this session, and `max-autotune` compiles for several minutes per case, so only the UNet `max-autotune` row was completed; the DiT and language-model rows under that mode are unmeasured. PyTorch eager rows for the language model are also missing at both sizes: the first attempt failed on an int32 target in eager `nll_loss`, which the harness now casts, and the rerun did not fit in the shared-GPU window. The commands are in `Reproduction`.

### UNet, 64 px, batch 16

| framework | variant | ms/step | p10 / p50 / p90 ms | samples/s | analytic TFLOP/s | MFU spec | MFU measured | GPU busy | kernels/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 16.32 | 17.42 / 17.52 / 17.73 | 981 | 39.6 | 40.6% | 38.7% | 97.3% | 1,428 | 0.76 |
| PyTorch | eager, reference | 28.06 | 28.03 / 28.06 / 28.08 | 570 | 23.0 | 23.6% | 22.5% | 94.4% | 1,776 | 1.19 |
| PyTorch | compile, reference | 18.19 | 18.17 / 18.18 / 18.20 | 880 | 35.5 | 36.5% | 34.7% | 94.9% | 1,201 | 0.72 |
| PyTorch | max-autotune, reference | 16.96 | 16.95 / 16.95 / 16.96 | 943 | 38.1 | 39.1% | 37.2% | 99.0% | 1,188 | 0.46 |

### SimpleDiT, 64 px, patch 4, batch 16

| framework | variant | ms/step | p10 / p50 / p90 ms | tokens/s | analytic TFLOP/s | MFU spec | MFU measured | GPU busy | kernels/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 7.81 | 8.30 / 8.40 / 8.52 | 524,359 | 37.5 | 38.5% | 36.6% | 99.1% | 584 | 0.90 |
| Dew (JAX) | XLA, cuDNN fused attention | 6.89 | 7.34 / 7.38 / 7.52 | 594,389 | 42.5 | 43.6% | 41.5% | 98.2% | 524 | 0.69 |
| PyTorch | eager, reference | 13.32 | 13.30 / 13.32 / 13.34 | 307,463 | 22.0 | 22.6% | 21.5% | 87.4% | 1,159 | 0.95 |
| PyTorch | compile, reference | 9.08 | 9.06 / 9.07 / 9.10 | 450,877 | 32.2 | 33.1% | 31.5% | 95.1% | 597 | 0.86 |
| PyTorch | compile, SDPA cudnn | 7.96 | 7.90 / 7.92 / 7.95 | 514,275 | 36.8 | 37.7% | 35.9% | 94.4% | 549 | 0.70 |

### Causal transformer, sequence 512, batch 16

| framework | variant | ms/step | p10 / p50 / p90 ms | tokens/s | analytic TFLOP/s | MFU spec | MFU measured | GPU busy | kernels/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 83.05 | 83.71 / 83.93 / 84.05 | 98,641 | 41.0 | 42.1% | 40.0% | 100.0% | 292 | 5.80 |
| Dew (JAX) | XLA, cuDNN fused attention | 75.70 | 76.34 / 76.48 / 76.65 | 108,213 | 45.0 | 46.2% | 43.9% | 100.0% | 254 | 4.99 |
| PyTorch | compile, reference | 80.32 | 80.20 / 80.34 / 80.39 | 101,992 | 42.4 | 43.5% | 41.4% | 99.8% | 233 | 4.30 |
| PyTorch | compile, SDPA cudnn | 72.18 | 72.14 / 72.17 / 72.21 | 113,493 | 47.2 | 48.4% | 46.0% | 99.8% | 203 | 3.52 |

### SimpleDiT, width 768, 12 layers, batch 32

| framework | variant | ms/step | p10 / p50 / p90 ms | tokens/s | analytic TFLOP/s | MFU spec | MFU measured | GPU busy | kernels/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 91.42 | 93.10 / 93.34 / 93.43 | 89,609 | 48.4 | 49.7% | 47.2% | 99.9% | 1,225 | 6.84 |
| Dew (JAX) | XLA, cuDNN fused attention | 72.34 | 73.79 / 74.04 / 74.23 | 113,247 | 61.2 | 62.8% | 59.7% | 99.9% | 1,069 | 5.15 |
| PyTorch | compile, reference | 93.98 | 93.94 / 93.97 / 94.00 | 87,166 | 47.1 | 48.3% | 45.9% | 99.2% | 1,096 | 6.32 |
| PyTorch | compile, SDPA cudnn | 78.31 | 78.27 / 78.30 / 78.34 | 104,608 | 56.5 | 58.0% | 55.1% | 99.1% | 1,000 | 5.07 |

### Causal transformer, width 768, 12 layers, sequence 1024, batch 8

| framework | variant | ms/step | p10 / p50 / p90 ms | tokens/s | analytic TFLOP/s | MFU spec | MFU measured | GPU busy | kernels/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 216.76 | 218.62 / 218.73 / 218.83 | 37,793 | 38.7 | 39.7% | 37.8% | 100.0% | 1,052 | 11.07 |
| Dew (JAX) | XLA, cuDNN fused attention | 146.90 | 148.80 / 148.88 / 149.02 | 55,765 | 57.1 | 58.6% | 55.7% | 100.0% | 881 | 8.97 |
| PyTorch | compile, reference | 206.76 | 206.70 / 206.74 / 206.81 | 39,622 | 40.6 | 41.6% | 39.6% | 99.7% | 853 | 12.43 |
| PyTorch | compile, SDPA cudnn | 145.25 | 145.21 / 145.24 / 145.27 | 56,399 | 57.8 | 59.3% | 56.4% | 99.7% | 733 | 7.35 |

### One variable at a time

| framework | model | one change from the matched row | baseline ms | changed ms | change |
|---|---|---|---:|---:|---:|
| Dew (JAX) | unet | forward and backward only | 16.51 | 15.29 | -7.4% |
| Dew (JAX) | unet | one fixed device batch | 16.51 | 17.24 | +4.4% |
| Dew (JAX) | unet | XLA command buffers requested | 16.51 | 16.74 | +1.4% |
| Dew (JAX) | simple_dit | cuDNN attention | 7.96 | 7.02 | -11.8% |
| Dew (JAX) | simple_dit | forward and backward only | 7.96 | 7.00 | -12.0% |
| Dew (JAX) | simple_dit | one fixed device batch | 7.96 | 7.89 | -0.9% |
| Dew (JAX) | simple_dit | XLA command buffers requested | 7.96 | 7.83 | -1.7% |
| Dew (JAX) | causal_transformer | cuDNN attention | 83.45 | 75.69 | -9.3% |
| Dew (JAX) | causal_transformer | bf16 LM head | 83.45 | 77.57 | -7.0% |
| Dew (JAX) | causal_transformer | no per-step accuracy or perplexity | 83.45 | 80.73 | -3.3% |
| Dew (JAX) | causal_transformer | forward and backward only | 83.45 | 77.91 | -6.6% |
| Dew (JAX) | causal_transformer | one fixed device batch | 83.45 | 83.26 | -0.2% |
| Dew (JAX) | causal_transformer | cuDNN attention, bf16 head, no accuracy | 83.45 | 67.66 | -18.9% |
| Dew (JAX) | causal_transformer | XLA command buffers requested | 83.45 | 83.09 | -0.4% |
| Dew (JAX) | causal_transformer | vocabulary 8,192 | 83.45 | 38.81 | -53.5% |
| PyTorch | unet | forward and backward only | 18.19 | 17.44 | -4.1% |
| PyTorch | unet | no EMA copy | 18.19 | 18.00 | -1.0% |
| PyTorch | unet | host-to-device copy every step | 18.19 | 18.24 | +0.3% |
| PyTorch | simple_dit | cuDNN SDPA attention | 9.08 | 7.96 | -12.3% |
| PyTorch | simple_dit | forward and backward only | 9.08 | 7.77 | -14.5% |
| PyTorch | simple_dit | no EMA copy | 9.08 | 8.67 | -4.6% |
| PyTorch | simple_dit | host-to-device copy every step | 9.08 | 9.15 | +0.8% |
| PyTorch | causal_transformer | cuDNN SDPA attention | 80.32 | 72.18 | -10.1% |
| PyTorch | causal_transformer | bf16 LM head | 80.32 | 58.03 | -27.8% |
| PyTorch | causal_transformer | no per-step accuracy or perplexity | 80.32 | 80.47 | +0.2% |
| PyTorch | causal_transformer | forward and backward only | 80.32 | 76.06 | -5.3% |
| PyTorch | causal_transformer | no EMA copy | 80.32 | 79.77 | -0.7% |
| PyTorch | causal_transformer | host-to-device copy every step | 80.32 | 81.09 | +1.0% |
| PyTorch | causal_transformer | vocabulary 8,192 | 80.32 | 39.53 | -50.8% |
| PyTorch | causal_transformer | cuDNN SDPA, bf16 head, no accuracy | 80.32 | 49.85 | -37.9% |

Dew ablation rows compare against the prefetching baseline row rather than the fixed-batch headline row. The two differ by under 1% on all three models, and every row prints its own baseline.

## Profile results

GPU busy is the union of GPU kernel intervals divided by the first-to-last kernel window. Kernel counts include compute kernels, copies, and memsets. Category times may overlap across CUDA streams. They are an attribution aid, not a second step timer.

### UNet, 64 px, batch 16

| framework | variant | gemm | conv | attention-softmax | elementwise/norm | optimizer/state | optimizer | loss/reduce | copy/gather | memcpy/memset | unmapped | kernel ms total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 1.15 | 8.24 | 0.24 | 4.20 | 0.68 | - | 0.01 | 1.38 | - | 0.03 | 15.93 |
| PyTorch | eager, reference | 1.26 | 8.31 | - | 6.94 | - | 1.19 | 0.81 | 9.16 | 0.12 | - | 27.79 |
| PyTorch | compile, reference | 1.40 | 9.15 | - | 0.39 | - | 1.20 | - | 5.95 | 0.07 | - | 18.15 |
| PyTorch | max-autotune, reference | 1.12 | 9.88 | - | 0.34 | - | 1.22 | - | 6.04 | 0.03 | - | 18.62 |

### SimpleDiT, 64 px, patch 4, batch 16

| framework | variant | gemm | conv | attention | attention-softmax | elementwise/norm | optimizer/state | optimizer | loss/reduce | copy/gather | memcpy/memset | kernel ms total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 4.25 | 0.04 | - | 0.36 | 2.22 | 0.76 | - | 0.01 | 0.19 | - | 7.82 |
| Dew (JAX) | XLA, cuDNN fused attention | 3.30 | 0.04 | 0.69 | 0.30 | 1.78 | 0.74 | - | 0.01 | 0.05 | - | 6.90 |
| PyTorch | eager, reference | 4.90 | 0.05 | - | - | 3.58 | - | 2.28 | 0.55 | 2.40 | 0.05 | 13.79 |
| PyTorch | compile, reference | 4.94 | 0.09 | - | - | 0.53 | - | 2.23 | - | 1.93 | 0.02 | 9.73 |
| PyTorch | compile, SDPA cudnn | 3.86 | 0.09 | 0.76 | - | 0.56 | - | 2.24 | - | 1.04 | 0.05 | 8.61 |

### Causal transformer, sequence 512, batch 16

| framework | variant | gemm | attention | attention-softmax | elementwise/norm | optimizer/state | optimizer | loss/reduce | copy/gather | memcpy/memset | kernel ms total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 57.11 | - | 3.62 | 4.73 | 3.87 | - | 12.97 | 0.89 | - | 83.18 |
| Dew (JAX) | XLA, cuDNN fused attention | 51.66 | 1.77 | 1.02 | 4.57 | 3.45 | - | 12.98 | 0.32 | - | 75.78 |
| PyTorch | compile, reference | 58.40 | - | - | 1.95 | - | 7.53 | 8.38 | 6.64 | 0.47 | 83.37 |
| PyTorch | compile, SDPA cudnn | 52.83 | 1.81 | - | 1.95 | - | 7.54 | 8.37 | 2.19 | 0.52 | 75.21 |

### SimpleDiT, width 768, 12 layers, batch 32

| framework | variant | gemm | conv | attention | attention-softmax | elementwise/norm | optimizer/state | optimizer | loss/reduce | copy/gather | memcpy/memset | kernel ms total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 55.95 | 0.10 | - | 7.13 | 17.87 | 8.08 | - | 0.01 | 2.43 | - | 91.57 |
| Dew (JAX) | XLA, cuDNN fused attention | 43.38 | 0.10 | 5.52 | 0.44 | 14.90 | 7.25 | - | 0.01 | 0.89 | - | 72.48 |
| PyTorch | compile, reference | 59.72 | 0.24 | - | - | 5.51 | - | 16.16 | 0.02 | 18.51 | 0.04 | 100.18 |
| PyTorch | compile, SDPA cudnn | 47.08 | 0.24 | 5.80 | - | 5.79 | - | 16.18 | 0.02 | 9.23 | 0.24 | 84.57 |

### Causal transformer, width 768, 12 layers, sequence 1024, batch 8

| framework | variant | gemm | attention | attention-softmax | elementwise/norm | optimizer/state | optimizer | loss/reduce | copy/gather | memcpy/memset | unmapped | kernel ms total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dew (JAX) | XLA, reference attention | 133.62 | - | 39.87 | 19.40 | 7.80 | - | 12.98 | 2.93 | - | 0.38 | 216.99 |
| Dew (JAX) | XLA, cuDNN fused attention | 94.53 | 10.64 | 3.38 | 18.51 | 6.58 | - | 12.97 | 0.45 | - | - | 147.06 |
| PyTorch | compile, reference | 137.23 | - | - | 7.81 | - | 17.18 | 8.68 | 42.07 | 0.50 | - | 213.47 |
| PyTorch | compile, SDPA cudnn | 98.69 | 10.83 | - | 7.81 | - | 17.14 | 8.67 | 8.18 | 0.69 | - | 152.02 |

Direct achieved DRAM bandwidth for a model step is unknown. Nsight Compute returned `ERR_NVGPUCTRPERM`. `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`. NVIDIA documents that this error means the user lacks access to GPU performance counters: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/2023.3/ProfilingGuide/index.html). An administrator must enable non-admin counter access, then rerun:

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
  /tmp/torchbench/bin/python tools/benchmark_torch.py ...
```

`cost_analysis()['bytes accessed'] / wall time` is not substituted. It is compiler logical traffic. It can exceed 716.8 GB/s because it counts HLO operand traffic before cache reuse and fusion. The measured streaming ceiling is 610 GB/s, or 85.1% of 716.8 GB/s.

## Hypothesis verdicts

| hypothesis | verdict | falsifiable result |
|---|---|---|
| H1. Small shapes are launch or bandwidth bound, so spec peak is the wrong yardstick. | **Partly confirmed.** The peak yardstick is wrong. Launch-bound is false for the measured full steps. | Small shape ceilings span 49.1 to 93.3 TFLOP/s, not 97.5. GPU busy is 94.4% to 100% in both frameworks. The remaining time is small GEMM or Conv efficiency plus elementwise, normalization, loss, optimizer, and copy work. |
| H2. XLA cost-analysis FLOPs disagree with analytic FLOPs. | **Confirmed.** | UNet is undercounted 22.50x. The LM audit compile is undercounted 2.37x. A standalone cuDNN Conv reports `-1` FLOPs while a standalone visible bf16 dot is exact. PyTorch's independent operator FLOP counter matches the HLO analytic sum for all three ports. |
| H3. The causal transformer is dominated by the 50,304-vocabulary projection and fp32 cross entropy. | **Confirmed.** | Cutting the vocabulary to 8,192 with everything else fixed drops the step from 83.05 to 38.81 ms in Dew, a 53.3% saving, and from 80.32 to 39.53 ms in PyTorch, a 50.8% saving. The Dew profile attributes 57.11 ms to GEMMs and 12.97 ms to loss and reductions at vocabulary 50,304, against 25.96 ms and 2.07 ms at 8,192. |
| H4. The UNet is launch bound at 64 px, batch 16. | **Rejected.** | It launches 1,428 kernels per step, but GPU busy is 97.3%. The HLO carries 646.39 GFLOP, not 28.7. Its 16.32 ms step reaches 39.6 analytic TFLOP/s, which is faster than PyTorch eager and `torch.compile` and within 4% of PyTorch `max-autotune`. |
| H5. The metric mixes MFU and HFU. | **Confirmed.** | The timed step includes loss, Adam, EMA, RNG, finiteness, and telemetry. The compiler count includes some of them, while omitting opaque model calls. Removing the optimizer and EMA saves 1.03 ms of 16.32 for UNet, 0.85 ms of 7.81 for DiT and 7.00 ms of 83.05 for the LM in Dew; PyTorch shows 0.75 ms, 1.31 ms and 4.26 ms for the same three changes. |
| H6. Host dispatch or per-step `device_put` leaks into timing. | **Rejected for the idle run.** | Reusing one device batch changes the Dew LM from 83.45 to 83.26 ms and the DiT from 7.96 to 7.89 ms. The input prefetcher moves batches on a background thread. In PyTorch, adding a pinned host-to-device copy per step costs 1.0% on the LM, 0.8% on UNet and 0.8% on the DiT, so the data path is not the bottleneck on either side. |
| H7. Attention kernel and fusion choice matter. | **Confirmed.** | Dew's fused path changes the small DiT from 7.81 to 6.89 ms, the small LM from 83.05 to 75.70 ms, the large DiT from 91.42 to 72.34 ms, and the large LM from 216.76 to 146.90 ms. PyTorch SDPA with the cuDNN backend changes the same four cases by -12.3%, -10.1%, -16.7% and -29.7%. |

## Utilization diagnosis

### UNet

The old 1.6% number is a counter artifact. cuDNN convolution custom calls are absent from the numerator. The Dew profile attributes 8.24 ms to convolution, 4.20 ms to elementwise and normalization kernels, 1.38 ms to copies and gathers, 1.15 ms to dots, 0.68 ms to optimizer and state work, and 0.24 ms to attention softmax. GPU busy is 97.3%.

The Conv chain is not reaching the large-GEMM ceiling. That is expected from the measured primitive. The exact `(16,64,64,64)` forward-plus-backward Conv reaches 63.7 TFLOP/s in JAX. The UNet's convolution total is 623.43 GFLOP over 8.24 ms of convolution kernels, or 75.7 TFLOP/s across the mixture of shapes. PyTorch spends 9.15 ms in convolution with `torch.compile` and 9.88 ms with `max-autotune`, so neither framework does better on the same cuDNN and CUTLASS kernels. The rest of the step is bandwidth-sensitive GroupNorm, swish, casts, residual adds, and concatenations. PyTorch pays much more of it: 5.95 ms in copies and gathers against Dew's 1.38 ms, because the NHWC-to-channels-last views and skip concatenations are separate kernels there while XLA fuses them.

cuDNN attention cannot run this UNet exactly. JAX rejects the cross-attention shape `Q=1024, KV=77` with `NotImplementedError: Unsupported sequence length Q 1024, KV 77`. The reference row is therefore the parity row. Padding the text context would change the workload and is not used here.

### SimpleDiT

The small reference profile attributes 4.25 ms to GEMMs, 2.22 ms to elementwise and normalization work, 0.76 ms to optimizer and state, 0.36 ms to attention softmax, 0.19 ms to copies and gathers, and 0.04 ms to the patch Conv. GPU busy is 99.1%.

The GEMMs are shape-limited. The real `4096x384x384` and `4096x384x1536` ceilings are 62.1 and 79.3 TFLOP/s, so the GEMM total of 292.6 GFLOP cannot go below about 4 ms. The fused attention path removes the materialized fp32 probabilities and some casts. It cuts the small step by 11.8% and the large step by 20.9%. PyTorch's optimizer is the visible difference at this size: 2.23 ms of fused Adam and EMA against Dew's 0.76 ms, and 16.18 ms against 7.25 ms at the large size, where Dew folds the update into XLA fusions.

### Causal transformer

The tied fp32 vocabulary head is the largest cost. XLA default fp32 matmul uses TF32 on this executable. Three head GEMMs each contain 632.97 GFLOP. At the measured 49.5 TFLOP/s TF32 ceiling they cost about 37.8 ms, which is 45% of the 83.05 ms step. The vocabulary ablation confirms the size of the term directly: 8,192 instead of 50,304 removes 44.2 ms in Dew and 40.8 ms in PyTorch.

The fp32 logits tensor has `8192 * 50304` elements, or 1.648 GiB. Cross entropy and accuracy make several passes over it. The Dew profile attributes 12.97 ms to loss and reductions; PyTorch attributes 8.38 ms. Removing per-step perplexity and token accuracy saves 2.72 ms in Dew and nothing measurable in PyTorch, where Inductor fuses the argmax into the same pass.

Reference attention keeps the softmax probabilities in fp32. Flax casts logits to fp32 for the softmax and returns that fp32 probability tensor, so the following probability-value dot promotes values and runs as an fp32 or TF32 dot. Source: the installed Flax 0.12.9 `flax.linen.attention.dot_product_attention_weights` and Dew's call with `force_fp32_for_softmax=True` in [`attention.py`](../../src/dew/nn/attention.py#L133-L144). The fused kernel avoids the score and probability tensors. It saves 7.35 ms at sequence 512 and 69.86 ms at sequence 1,024, where the reference softmax path alone costs 39.87 ms.

The two harnesses differ in one respect that the tables should not hide. Dew's bf16-head ablation still writes fp32 logits, so it saves only matmul time: 7.0%. The PyTorch bf16 head produces bf16 logits that Inductor keeps in bf16 until the loss, and it saves 27.8%. The gap is logits traffic, not arithmetic.

## Fixes, ordered by measured gain

1. **Use cuDNN attention by default on supported NVIDIA shapes.** It saves 32.2% on the large LM, 20.9% on the large DiT, 11.8% on the small DiT, and 8.8% on the small LM. Keep the reference fallback for unsupported shapes such as UNet cross-attention with 77 keys. The existing `attention_impl='auto'` path already selects cuDNN on GPU: [`attention.py`](../../src/dew/nn/attention.py#L146-L176). The change is a configuration default, not new code.
2. **Decide the language-model head and logits precision deliberately.** The vocabulary projection and its logits are the step. A bf16 head with fp32 logits saves 7.0%. PyTorch's bf16 head, whose logits stay bf16 until the loss, saves 27.8%, which sets the size of the prize for keeping Dew's logits in bf16 and converting inside the cross entropy. Both change numerics, so this needs a loss-parity and convergence decision before it becomes a default. The fp32 logsumexp must stay.
3. **Do not compute full-vocabulary accuracy every train step.** Removing `argmax`, token accuracy, and redundant per-step perplexity saves 2.72 ms, or 3.3%, on the small LM. Compute them at the logging interval instead. PyTorch shows no gain from the same change because Inductor fuses the argmax; XLA does not.
4. **Replace the current MFU numerator.** Use the formulas in this document, keyed by model configuration, or label `cost_analysis()` as compiler-visible FLOPs and never call it MFU. This does not make a step faster. It fixes a 22.5x reporting error for UNet and a 2.37x error for the audit LM compile, and it removes the compiler's kernel-selection noise from a reported metric.
5. **Keep prefetch and asynchronous timing.** A fixed device batch does not improve the idle Dew loop, and a per-step host copy costs PyTorch about 1%. The existing prefetch iterator is sufficient. Extend `tools/benchmark_step.py` to 100 steps with percentiles instead, since the 10-step window is a 170 ms sample for the DiT rows.
6. **Do not pursue XLA command buffers from these results.** The tested flag was accepted but produced zero command-buffer regions in `compiled.as_text()` and no consistent speed change. A future XLA version with a working CUDA graph path would need a new measurement.

## Reproduction

The retained harness is `/tmp/benchmark-parity`, a link to disk-backed storage. No repository source file is modified by the JAX audit harness.

```bash
# Environment and idle check
cd ~/Desktop/dew
nvidia-smi --query-compute-apps=pid --format=csv,noheader
.venv/bin/python -c 'import jax,jaxlib,flax,optax; print(jax.__version__,jaxlib.__version__,flax.__version__,optax.__version__,jax.devices())'
/tmp/torchbench/bin/python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name())'

# FLOP audit and HLO dumps
.venv/bin/python /tmp/benchmark-parity/audit_flops.py --probes
.venv/bin/python /tmp/benchmark-parity/audit_flops.py --arch unet simple_dit causal_transformer --steps 10

# Card ceilings
.venv/bin/python /tmp/benchmark-parity/ceilings_jax.py
/tmp/torchbench/bin/python /tmp/benchmark-parity/ceilings_torch.py

# Fair fixed-input JAX rows
.venv/bin/python /tmp/benchmark-parity/dew_bench.py --arch simple_dit \
  --fixed-batch --fixed-rng --attention reference --warmup 20 --steps 100 --profile
.venv/bin/python /tmp/benchmark-parity/dew_bench.py --arch causal_transformer \
  --fixed-batch --attention cudnn --warmup 20 --steps 100 --profile

# PyTorch rows. Use one process per case.
/tmp/torchbench/bin/python tools/benchmark_torch.py --model simple_dit \
  --mode compile --attention reference --warmup 20 --steps 100 --profile
/tmp/torchbench/bin/python tools/benchmark_torch.py --model causal_transformer \
  --mode max-autotune --attention sdpa --sdpa-backend cudnn \
  --warmup 20 --steps 100 --profile
```

JAX profile files are parsed with `jax.profiler.ProfileData`. PyTorch profiles use `torch.profiler`. Both summaries take the union of GPU event intervals for busy time. Raw logs, HLO, profiles, ceiling samples, and JSON rows remain under `/tmp/benchmark-parity`.
