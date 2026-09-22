# Step benchmarks

> An AI assistant maintains this document. It is presented as-is.

These results time one complete compiled optimization step through `Trainer`. Each table states its hardware, source revision and shapes. They are records of past runs, and they do not promise the same throughput on the current checkout. To take a new measurement, run `tools/benchmark_step.py`. To time input loading on its own, run `tools/benchmark_data.py`.

The FLOP counts come from the optimized HLO of the compiled executable, through
`dew.telemetry.instrumentation.compiled_flops`. It counts every `dot` and
`convolution`, plus the cuBLAS matmul, cuDNN convolution and cuDNN
fused-attention custom calls that a GPU backend turns them into, each from its
own shapes. `util` is the number the trainer logs as `train/mfu`. It is the
step's measured FLOPs, divided by the step time and by the dense bf16 peak of
one device (97.5 TFLOP/s for this card).

## `--preset small` on one RTX 4080

```
python tools/benchmark_step.py --preset small --architectures unet --json-out bench.json
```

Run on 2026-09-02 with jax 0.11.1 / jaxlib 0.11.1 (CUDA), flax 0.12.9, optax
0.2.8, driver 595.84, RTX 4080 16 GiB, dew at `6b0f119`. Every model ran in
bf16 (`dtype=bfloat16`) on a single device, with `MeshSpec(fsdp=1)`, adam, 2
warmup steps and 100 steps per architecture. I ran one invocation per
architecture (`--architectures unet` and so on), so each row's peak memory
belongs to that row alone. `ms/step` times the loop the way a run dispatches
it. `p10 / p50 / p90 ms` come from a second window of the same length that
waits on every step, so a long tail shows up there.

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

The small preset on current main also holds `unet_2d_condition`,
`sd3_transformer`, `flux_transformer`, `multimodal_transformer` and
`diffusion_gemma` cases. These tables have no rows for them.

### Rerun 2026-09-05

```
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    python tools/benchmark_step.py --preset small --architectures <arch> --json-out <arch>.json
```

Same card, driver and library versions. Dew was at `9886c20`, the tree before
the cudnn padding of `3b67135`. The `simple_mmdit`, `hierarchical_mmdit` and
`unet` rows after that padding are in `docs/performance.md`. The host was
otherwise idle, with one process per architecture, 2 warmup steps and 100
measured steps. Each parameter count is 99,840 higher than in the first table.
The condition encoder's table (`CharTable`, 130 by 768) sits in the state
tree at this revision, where at `6b0f119` it was a constant in the
executable. The second column repeats the 09-02 ms/step for the four
architectures I reran.

| architecture       | 09-02 ms/step | ms/step | p10 / p50 / p90 ms | samples/s | GFLOP/step |  util | peak GiB | compile s |
|--------------------|--------------:|--------:|--------------------|----------:|-----------:|------:|---------:|----------:|
| simple_dit         |           7.9 |    7.02 |  7.5 / 7.6 / 7.9   |    2278.3 |      292.9 | 42.8% |     0.83 |       8.0 |
| hierarchical_mmdit |          32.7 |   33.95 | 37.3 / 37.6 / 39.3 |     471.2 |      737.4 | 22.3% |     3.50 |      50.4 |
| video_dit          |          17.3 |   17.06 | 18.5 / 18.9 / 20.3 |     234.5 |      760.0 | 45.7% |     1.41 |      13.0 |
| causal_transformer |          83.0 |   88.78 | 89.5 / 89.7 / 89.9 |     180.2 |     3406.4 | 39.4% |     4.51 |       9.5 |

The decoder is 7% slower than on 09-02, for two reasons. The first is the
chunked vocabulary head, which landed after the 09-02 table. At its default
of four chunks it costs 1.9 ms against the full-vocabulary pass. I measured
that in the same tree with `--cases` setting `head_chunks` (1: 87.02 ms, 5.67
GiB; 2: 88.39 ms, 4.85 GiB; 4: 88.90 ms, 4.51 GiB; 8: 89.67 ms, 4.41 GiB; 50
steps each). So the default gives up 2.2% of the step to save 1.2 GiB. The
other 3.8 ms are in the decoder itself. `6b0f119`, rerun the same day on the
same card, reads 83.26 ms, and the tree at `9886c20` with one head chunk reads
87.02. That difference comes from the decoder's changes between the two
commits. I reported it to the architecture lane and did not measure it
further here.

The MoE `causal_transformer` case of the preset (8 experts, top-2 on every
second layer) does not run under `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Its
step asks for one 4.5 GiB buffer. The BFC allocator grows on demand into a
12.8 GiB budget and cannot place that buffer (`RESOURCE_EXHAUSTED ...
4.47GiB`). The same tree runs with the default preallocation. The
full-vocabulary decoder at `6b0f119` fails the same way (4.80 GiB). On this
card, a benchmark of a step that needs one buffer over about 4.5 GiB has to
use the default preallocation.

`jepa_predictor` has no step of its own. The registry builds it, and it
trains inside the two JEPA rows. The `causal_transformer` row has the width of
GPT-2 small, three layers and a 50k vocabulary. Most of its FLOPs are the tied
fp32 vocabulary projection and its gradients, which run as cuBLAS custom
calls.

I measured every row with the host otherwise idle. Rows taken while something
else ran came out much slower: `unet` read 54 ms/step under load against 16.4
idle. The spread columns show how steady the host was. In the video rows the
p90 sits 6-13% above the p50. That spread comes from the scheduler; the step
itself is steady.

### What the numbers say

- The DiT family suits this card best. `simple_dit`, `simple_udit` and
  `hybrid_dit` all run under 10 ms/step at 28-40% of peak, which is the range
  a 64px, patch-4 (256 token) workload should reach.
- `unet` does the most arithmetic for its time of the image models. 646
  GFLOP/step in 16 ms is 40.5% of peak, ahead of the transformers at the same
  resolution. With XLA's own `cost_analysis()` as the numerator, the same
  measurement shows 28.7 GFLOP/step. The gap is convolution arithmetic that
  cost analysis cannot see. The card is busy.
- `unet_3d` is the slowest step in the table, at 41.8% of peak. Its 3D
  convolutions carry 1.38 TFLOP/step, more than twice the 760 of `video_dit`
  for the same (8, 64, 64, 3) samples. For video, the factorized transformer
  saves about a third of the step time.
- `hierarchical_mmdit` is the largest model here (55 M) and the most expensive
  diffusion step, which fits its 1024-token finest stage.
- Compile time dominates a short run: 9-47 s per architecture against 8-84 ms
  per step. Most of a sweep's time goes to XLA, so a real run should set
  `compilation_cache_dir`.

### `cost_analysis()` against the optimized HLO

I measured the same executable both ways on 2026-09-02, with one compile each:

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

`cost_analysis()` misses the arithmetic that the backend moves into its own
kernels. For the two UNets that is the convolution custom calls (1.8 and
9.5x). For the decoder it is six cuBLAS calls for the tied fp32 vocabulary
head and its gradients (2.58x). `uvit` has a mix of both. The pure-transformer
rows agree to within a few percent in either direction. That difference is
the elementwise work, which `cost_analysis()` counts and the matmul count
leaves out. The one row below 1.0 (`simple_dit` at 0.99x) is that elementwise
accounting on top of the matmuls; no kernels are missing there. Which side of
these ratios a run lands on depends on what XLA keeps visible, and XLA
chooses differently between recompiles of the same code. The HLO count stays
the same. This agrees with the audit in
`docs/research/benchmark-parity.md`, which found 22.50x, 2.372x and 0.987x for
the three architectures it counted.

## `--preset cpu-smoke`

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python tools/benchmark_step.py --preset cpu-smoke --steps 2
```

This preset runs tiny models on a simulated 8-device CPU mesh. It checks the
tool itself and says nothing about the hardware.
`tests/test_benchmark_step.py` runs one case of this preset, so the tool keeps
working against the trainer internals it drives. Utilisation and peak memory
come back `null`, because a CPU has no published peak FLOP/s and no allocator
counter.

## Data loader

```
python tools/benchmark_data.py data:oxford-flowers --batch 8 \
    --data.image-size 64 --steps 100 --warmup 5 --data.loading.workers {0,8} \
    --data.path <prepared version directory>
```

The dataset was Oxford Flowers 102 from local TFDS array_record files: 8189
records, resized to 64px, with flip and jitter augmentation and CLIP
tokenization per record. On current main `OxfordFlowers` reads only prepared
ArrayRecords and raises an error without `--data.path`
(`src/dew/data/images.py:358-363`), so the command above passes it.

| grain workers | samples/s | p50 step | p95 step |
|---------------|-----------|----------|----------|
| 0 (in-process) |     322.1 |  25.1 ms |  32.8 ms |
| 8              |     505.0 |  0.05 ms |  77.1 ms |

With workers, the p50 is a queue read, so the loader only shows up in the
p95. At 8 workers the pipeline delivers 505 samples/s. That is below every
image row in the table above (668-1564 samples/s) and above the video rows
(107-207). At 64px this dataset keeps up with the video models and starves
the image models. If an image run's `train/mfu` looks low, check the loader
here first.

These two points are not the loader's ceiling. `benchmark_data.py` defaulted
to 16 read threads, while the dataset specs default to 32 workers and 64 read
threads. Oxford Flowers is also only 8189 small records, far from a sharded
12M-record set.

Correction, 2026-09-22: `tools/benchmark_data.py` on current main has no
read-thread setting of its own. It reads with the dataset spec's `Loading`,
whose defaults are 32 workers and 64 threads
(`src/dew/data/dataset.py:187-188`), and `--data.loading.threads` changes it.
