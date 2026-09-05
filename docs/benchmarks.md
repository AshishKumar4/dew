# Step benchmarks

What one training step costs, per architecture, measured through
the `Trainer`'s own compiled step rather than a hand-written forward
pass. Reproduce with `tools/benchmark_step.py`; the loader is measured
separately by `tools/benchmark_data.py`.

FLOPs are counted off the compiled executable's optimized HLO through
`dew.telemetry.instrumentation.compiled_flops`: every `dot` and `convolution`,
and the cuBLAS matmul, cuDNN convolution and cuDNN fused-attention custom calls
a GPU backend hands them to, each from its own shapes. `util` is the same
figure the trainer logs as `train/mfu`: the step's measured FLOPs divided by
the step time and by one device's dense bf16 peak (97.5 TFLOP/s for this card).

## `--preset small` on one RTX 4080

```
python tools/benchmark_step.py --preset small --architectures unet --json-out bench.json
```

Run 2026-09-02, jax 0.11.1 / jaxlib 0.11.1 (CUDA), flax 0.12.9, optax 0.2.8,
driver 595.84, RTX 4080 16 GiB, dew at `6b0f119`. Every model in bf16
(`dtype=bfloat16`), single device, `MeshSpec(fsdp=1)`, adam, 2 warmup steps, 100
steps per architecture. One invocation per architecture (`--architectures
unet` and so on), so each row's peak memory is its own; `ms/step` times the
loop the way a run dispatches it, and `p10 / p50 / p90 ms` come from a second
window of the same length that waits on every step, which is where a long
tail shows up.

| architecture       | sample    | batch |      params | ms/step | p10 / p50 / p90 ms | samples/s | GFLOP/step |  util | peak GiB | compile s |
|--------------------|-----------|-------|-------------|---------|--------------------|-----------|------------|-------|----------|-----------|
| unet               | 64x64x3   |    16 |  10,159,299 |    16.4 | 18.2 / 18.4 / 19.0 |     977.2 |      646.4 | 40.5% |     0.75 |      36.4 |
| uvit               | 64x64x3   |    16 |  24,351,024 |    23.0 | 23.9 / 24.1 / 25.1 |     695.7 |      617.8 | 27.6% |     1.71 |       9.4 |
| simple_udit        | 64x64x3   |    16 |  23,381,424 |     9.4 | 10.4 / 10.9 / 12.1 |    1696.4 |      363.1 | 39.5% |     1.05 |      10.5 |
| simple_dit         | 64x64x3   |    16 |  19,835,568 |     7.9 |  8.8 / 9.1 / 10.0 |    2036.0 |      292.9 | 38.2% |     0.90 |       9.9 |
| simple_mmdit       | 64x64x3   |    16 |  36,385,584 |    12.9 | 14.7 / 15.4 / 18.4 |    1240.0 |      383.7 | 30.5% |     1.40 |      16.3 |
| hierarchical_mmdit | 64x64x3   |    16 |  55,498,188 |    32.7 | 38.1 / 38.8 / 39.2 |     489.3 |      737.4 | 23.1% |     3.38 |      36.9 |
| hybrid_dit         | 64x64x3   |    16 |  19,344,048 |     8.9 |  9.6 / 10.0 / 10.6 |    1795.5 |      244.6 | 28.2% |     0.86 |      13.1 |
| video_dit          | 8x64x64x3 |     4 |  25,155,504 |    17.3 | 18.3 / 18.7 / 19.6 |     231.8 |      760.0 | 45.2% |     1.59 |      12.6 |
| unet_3d            | 8x64x64x3 |     4 |  11,045,699 |    33.9 | 36.7 / 37.2 / 39.6 |     117.8 |     1384.1 | 41.8% |     1.72 |      46.6 |
| jepa_encoder       | 64x64x3   |    16 |  12,149,568 |     9.4 | 10.6 / 10.8 / 11.1 |    1699.1 |      297.3 | 32.4% |     0.70 |      13.7 |
| jepa_video_encoder | 8x64x64x3 |     4 |  16,143,360 |    18.3 | 20.1 / 20.3 / 22.9 |     218.2 |      758.1 | 42.4% |     1.28 |      20.5 |
| causal_transformer | 512 tokens |    16 |  66,950,784 |    83.0 | 83.7 / 83.8 / 84.0 |     192.7 |     3406.4 | 42.1% |     5.81 |      10.1 |

### Rerun 2026-09-05

```
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    python tools/benchmark_step.py --preset small --architectures <arch> --json-out <arch>.json
```

Same card, driver and library versions, dew at `9886c20` (the tree before
the cudnn padding of `3b67135`; the `simple_mmdit`, `hierarchical_mmdit`
and `unet` rows after it are in `docs/performance.md`), the host otherwise
idle, one process per architecture, 2 warmup and 100 measured steps. The
parameter counts grew by 99,840 since the last table because the condition
encoder's table (`CharTable`, 130 by 768) now lives in the state tree rather
than in the executable's constants. Last week's ms/step is in the second
column for the four architectures that were rerun.

| architecture       | 09-02 ms/step | ms/step | p10 / p50 / p90 ms | samples/s | GFLOP/step |  util | peak GiB | compile s |
|--------------------|--------------:|--------:|--------------------|----------:|-----------:|------:|---------:|----------:|
| simple_dit         |           7.9 |    7.02 |  7.5 / 7.6 / 7.9   |    2278.3 |      292.9 | 42.8% |     0.83 |       8.0 |
| hierarchical_mmdit |          32.7 |   33.95 | 37.3 / 37.6 / 39.3 |     471.2 |      737.4 | 22.3% |     3.50 |      50.4 |
| video_dit          |          17.3 |   17.06 | 18.5 / 18.9 / 20.3 |     234.5 |      760.0 | 45.7% |     1.41 |      13.0 |
| causal_transformer |          83.0 |   88.78 | 89.5 / 89.7 / 89.9 |     180.2 |     3406.4 | 39.4% |     4.51 |       9.5 |

The decoder is 7% slower than last week. Two things moved. The chunked
vocabulary head that shipped after that table costs 1.9 ms at its default
of four chunks against the full-vocabulary pass, measured in the same tree
with `--cases` naming `head_chunks` (1: 87.02 ms, 5.67 GiB; 2: 88.39 ms,
4.85 GiB; 4: 88.90 ms, 4.51 GiB; 8: 89.67 ms, 4.41 GiB; 50 steps each), so
the default trades 2.2% of the step for 1.2 GiB. The other 3.8 ms are in
the decoder itself: `6b0f119` rerun today on the same card reads 83.26 ms,
and the tree at `9886c20` with one head chunk reads 87.02. That difference
belongs to the decoder's changes between the two commits and is reported
to the architecture lane, not measured further here.

The MoE `causal_transformer` case of the preset (8 experts, top-2 on every
second layer) does not run under `XLA_PYTHON_CLIENT_PREALLOCATE=false`: its
step asks for one 4.5 GiB buffer and the BFC allocator, growing on demand
into a 12.8 GiB budget, cannot place it (`RESOURCE_EXHAUSTED ... 4.47GiB`).
The same tree runs with the default preallocation. `6b0f119`'s
full-vocabulary decoder fails the same way (4.80 GiB), so a benchmark of a
step with one buffer over about 4.5 GiB needs the default preallocation on
this card.

`jepa_predictor` has no step of its own: it is built through the registry and
trained inside the two JEPA rows. The `causal_transformer` row is GPT-2
small's width at three layers with a 50k vocabulary; its FLOPs are dominated
by the tied fp32 vocabulary projection and its gradients, which are cuBLAS
custom calls.

Every row was measured with the host otherwise idle; rows taken while
something else ran came out materially slower (`unet` read 54 ms/step under
load against 16.4 idle). The spread columns say how steady the host was: the
video rows' p90 sits 6-13% over their p50, which is the scheduler, not the
step.

### What the numbers say

- The DiT family is the efficient shape on this card: `simple_dit`,
  `simple_udit` and `hybrid_dit` all sit under 10 ms/step at 28-40% of peak,
  which is where a 64px/patch-4 (256 token) workload should be.
- `unet` does the most arithmetic of the image models relative to its time:
  646 GFLOP/step in 16 ms is 40.5% of peak, ahead of the transformers at the
  same resolution. The 1.6% an earlier version of this table reported was its
  convolution arithmetic missing from the counter, not the card idling; the
  same measurement with XLA's own `cost_analysis()` as the numerator still
  shows 28.7 GFLOP/step, which is what that table had counted.
- `unet_3d` is the slowest step in the table at 41.8%: its 3D convolutions
  carry 1.38 TFLOP/step, more than twice `video_dit`'s 760 for the same
  (8, 64, 64, 3) samples. For video, the factorized transformer buys a third
  of the step time back.
- `hierarchical_mmdit` is the largest model here (55 M) and the most expensive
  diffusion step, as its 1024-token finest stage implies.
- Compile dominates a short run: 9-47 s per architecture against 8-84 ms per
  step. A sweep is mostly XLA, and a real run should set
  `compilation_cache_dir`.

### What the counter used to say

The same executable, measured both ways on 2026-09-02, one compile each:

| architecture | `cost_analysis()` GFLOP | optimized HLO GFLOP | ratio |
|---|---:|---:|---:|
| unet | 28.7 | 646.4 | 22.50x |
| unet_3d | 145.8 | 1384.1 | 9.50x |
| causal_transformer | 1320.1 | 3406.4 | 2.58x |
| uvit | 327.4 | 617.8 | 1.89x |
| video_dit | 693.3 | 760.0 | 1.10x |
| hybrid_dit | 223.7 | 244.6 | 1.09x |
| jepa_video_encoder | 698.9 | 758.1 | 1.09x |
| simple_mmdit | 355.4 | 383.7 | 1.08x |
| jepa_encoder | 281.5 | 297.3 | 1.06x |
| hierarchical_mmdit | 715.2 | 737.4 | 1.03x |
| simple_udit | 360.8 | 363.1 | 1.01x |
| simple_dit | 296.9 | 292.9 | 0.99x |

The missing arithmetic is wherever the backend put its kernels: convolution
custom calls for the two UNets (1.8 and 9.5x), six cuBLAS calls for the tied
fp32 vocabulary head and its gradients (2.58x), and a mix of both for `uvit`.
The pure-transformer rows agree to a few percent either way, which is the
elementwise work `cost_analysis()` counts and the matmul count does not. What
remains below 1.0 (`simple_dit` at 0.99x) is the compiler's elementwise
accounting on top of the matmuls, not missing kernels. Which side of these
ratios a run lands on depends on what XLA chooses to keep visible, and it
chooses differently between recompiles of the same code; the HLO count does
not move. This matches the audit in
`docs/research/benchmark-parity.md` (22.50x, 2.372x and 0.987x there, for the
three architectures that work counted).

## `--preset cpu-smoke`

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python tools/benchmark_step.py --preset cpu-smoke --steps 2
```

Tiny models on the simulated 8-device CPU mesh, for checking the tool itself
rather than the hardware. `tests/test_benchmark_step.py` runs one case of this
preset so the tool cannot rot against the trainer internals it drives.
Utilisation and peak memory come back `null`: there is no published peak
FLOP/s for a CPU and no allocator counter behind it.

## Data loader

```
python tools/benchmark_data.py data:oxford-flowers --batch 8 \
    --data.image-size 64 --steps 100 --warmup 5 --data.loading.workers {0,8}
```

Oxford Flowers 102 from local TFDS array_record files, 8189 records, resize to
64px plus flip/jitter augmentation and CLIP tokenization per record.

| grain workers | samples/s | p50 step | p95 step |
|---------------|-----------|----------|----------|
| 0 (in-process) |     322.1 |  25.1 ms |  32.8 ms |
| 8              |     505.0 |  0.05 ms |  77.1 ms |

With workers the p50 is a queue read, so the loader only shows up in the p95.
At 8 workers the pipeline delivers 505 samples/s, which is below every image
row in the table above (668-1564 samples/s) and above the video rows
(107-207). So at 64px this dataset feeds the video models comfortably and
starves the image models: an image run whose `train/mfu` looks low should be
checked here first.

These two points are not the loader's ceiling. `benchmark_data.py` defaults to
16 read threads, where the dataset specs default to 32 workers and 64 read threads, and
Oxford Flowers is 8189 small records rather than a sharded 12M-record set.
