# Step benchmarks

`tools/benchmark_step.py` times a complete compiled `Trainer` optimization step for each architecture. `tools/benchmark_data.py` times input loading separately. These tables record past runs with their hardware, source revisions and shapes. They do not guarantee the same throughput on the current checkout.

`dew.telemetry.instrumentation.compiled_flops` counts FLOPs from the compiled executable's optimized HLO. It counts each `dot` and `convolution` from its shapes, including operations lowered to cuBLAS matmul, cuDNN convolution and cuDNN fused-attention custom calls. `util` is logged as `train/mfu`. It divides the step's measured FLOPs by step time and one device's dense bf16 peak (97.5 TFLOP/s for the RTX 4080).

| Column | Meaning |
|---|---|
| `ms/step` | Time per step of the loop as a run dispatches it. |
| `p10 / p50 / p90 ms` | Percentiles from a second window of the same length that waits on every step, so a long tail shows up there. |
| `GFLOP/step` | FLOPs of the compiled step, counted from the optimized HLO. |
| `util` | Measured FLOPs over step time and device peak (`train/mfu`). |
| `peak GiB` | Peak device memory of the process. |
| `compile s` | Time to compile the step. |

## Small preset on one RTX 4080, 2026-09-02

```
python tools/benchmark_step.py --preset small --architectures unet --json-out bench.json
```

jax 0.11.1 / jaxlib 0.11.1 (CUDA), flax 0.12.9, optax 0.2.8, driver 595.84, RTX 4080 16 GiB, Dew at `6b0f119`. Every model ran in bf16 (`dtype=bfloat16`) on one device with `MeshSpec(fsdp=1)` and adam. Each run used 2 warmup steps and 100 measured steps. Architectures ran in separate invocations (`--architectures unet` and so on), so peak memory in each row belongs to that architecture alone. The host was otherwise idle.

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

The small preset on current main also includes `unet_2d_condition`, `sd3_transformer`, `flux_transformer`, `multimodal_transformer` and `diffusion_gemma`, which have no rows here. `jepa_predictor` trains inside the two JEPA rows, without a separate step. The `causal_transformer` row uses GPT-2 small's width, three layers and a 50k vocabulary. At this revision, most of its FLOPs came from the tied fp32 vocabulary projection and its gradients, run as cuBLAS custom calls.

Concurrent host work slowed these runs: `unet` took 54 ms/step under load and 16.4 ms/step when idle. In the video rows, p90 is 6-13% above p50. That spread comes from the host scheduler; the step itself is steady.

At these shapes:

- `simple_dit`, `simple_udit` and `hybrid_dit` each run under 10 ms/step at 28-40% of the card's dense bf16 peak.
- `unet` has the highest arithmetic throughput among the image models. Its 646 GFLOP/step in 16 ms reaches 40.5% of peak, ahead of the transformers at the same resolution. XLA's `cost_analysis()` counts only 28.7 GFLOP/step for this measurement. It misses the convolution arithmetic that accounts for the gap.
- `unet_3d` has the slowest diffusion step (33.9 ms/step, 41.8% of peak). Its 3D convolutions need 1,384 GFLOP/step. This is 1.8 times `video_dit`'s 760 for the same (8, 64, 64, 3) samples. `video_dit` takes about half as long per step, at 17.3 ms.
- `hierarchical_mmdit` is the largest diffusion model here at 55 M, with the slowest image step at 32.7 ms. This is consistent with its 1024-token finest stage. The 67 M `causal_transformer` is a language model.
- Compilation takes 9-47 s per architecture, compared with 8-84 ms per step, so it dominates short runs. Set `compilation_cache_dir` for a real run.

### Rerun, 2026-09-05

```
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    python tools/benchmark_step.py --preset small --architectures <arch> --json-out <arch>.json
```

This rerun used the same card, driver and library versions, with Dew at `9886c20`, before the cuDNN padding in `3b67135`. For `simple_mmdit`, `hierarchical_mmdit` and `unet` measurements after that padding, see [Performance measurements](performance.md).

The host was otherwise idle. Each architecture ran in its own process with 2 warmup steps and 100 measured steps. Parameter counts are 99,840 higher than in the first table because the condition encoder's table (`CharTable`, 130 by 768) is in the state tree. At `6b0f119`, it was a constant in the executable. The second column repeats the 09-02 ms/step for the four rerun architectures.

| architecture       | 09-02 ms/step | ms/step | p10 / p50 / p90 ms | samples/s | GFLOP/step |  util | peak GiB | compile s |
|--------------------|--------------:|--------:|--------------------|----------:|-----------:|------:|---------:|----------:|
| simple_dit         |           7.9 |    7.02 |  7.5 / 7.6 / 7.9   |    2278.3 |      292.9 | 42.8% |     0.83 |       8.0 |
| hierarchical_mmdit |          32.7 |   33.95 | 37.3 / 37.6 / 39.3 |     471.2 |      737.4 | 22.3% |     3.50 |      50.4 |
| video_dit          |          17.3 |   17.06 | 18.5 / 18.9 / 20.3 |     234.5 |      760.0 | 45.7% |     1.41 |      13.0 |
| causal_transformer |          83.0 |   88.78 | 89.5 / 89.7 / 89.9 |     180.2 |     3406.4 | 39.4% |     4.51 |       9.5 |

The decoder is 7% slower than on 09-02. The chunked vocabulary head, added after that table, accounts for 1.9 ms with the default four chunks compared with a full-vocabulary pass. Runs in the same tree, with `--cases` setting `head_chunks` and 50 steps each, measured: 1 chunk 87.02 ms and 5.67 GiB, 2 chunks 88.39 ms and 4.85 GiB, 4 chunks 88.90 ms and 4.51 GiB, 8 chunks 89.67 ms and 4.41 GiB. The default spends 2.2% of the step to save 1.2 GiB.

The remaining 3.8 ms are in the decoder itself. Rerun on the same card that day, `6b0f119` took 83.26 ms, while `9886c20` with one head chunk took 87.02 ms. Changes to the decoder between these commits account for this difference, which was not investigated further.

The preset's MoE `causal_transformer` case uses 8 experts, top-2 on every second layer. It fails with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The step needs one 4.5 GiB buffer, which the BFC allocator cannot place while growing on demand into a 12.8 GiB budget (`RESOURCE_EXHAUSTED ... 4.47GiB`). The same tree runs with default preallocation. The full-vocabulary decoder at `6b0f119` fails in the same way, with a 4.80 GiB buffer. On this card, use default preallocation when a step needs one buffer over about 4.5 GiB.

### cost_analysis() against the optimized HLO

These counts use the same executables compiled once each on 2026-09-02:

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

`cost_analysis()` misses arithmetic moved into backend kernels. For the two UNets, it misses convolution custom calls. For the decoder, it misses six cuBLAS calls for the tied fp32 vocabulary head and its gradients, giving the 2.58x ratio. `uvit` combines both cases.

The pure-transformer counts agree within a few percent in either direction. `cost_analysis()` includes elementwise work that the matmul count omits. This accounts for the one ratio below 1.0, `simple_dit` at 0.99x; no kernels are missing there.

The ratios depend on which operations XLA keeps visible. XLA chooses differently between recompiles of the same code, while the HLO count stays the same. The audit in `docs/research/benchmark-parity.md` found the same pattern: 22.50x, 2.372x and 0.987x for its three architectures.

## CPU smoke preset

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python tools/benchmark_step.py --preset cpu-smoke --steps 2
```

This preset runs tiny models on a simulated 8-device CPU mesh to check the benchmark tool. It does not establish hardware performance. `tests/test_benchmark_step.py` runs one case to check that the tool works with the trainer internals it uses. Utilisation and peak memory return `null` because a CPU has no published peak FLOP/s and no allocator counter.

## Data loader

```
python tools/benchmark_data.py data:oxford-flowers --batch 8 \
    --data.image-size 64 --steps 100 --warmup 5 --data.loading.workers {0,8} \
    --data.path <prepared version directory>
```

This run read Oxford Flowers 102 from local TFDS ArrayRecord files: 8189 records resized to 64px, with flip and jitter augmentation and CLIP tokenization per record. `TFDSImages` requires prepared ArrayRecords. Without `--data.path`, it raises a `ValueError`, so the command supplies the path.

| grain workers | samples/s | p50 step | p95 step |
|---------------|-----------|----------|----------|
| 0 (in-process) |     322.1 |  25.1 ms |  32.8 ms |
| 8              |     505.0 |  0.05 ms |  77.1 ms |

With workers, p50 measures a queue read; loader time shows up in p95. With 8 workers, the pipeline delivers 505 images/s. The first step table's image rows consume 489-2036 samples/s. Only `hierarchical_mmdit`, at 489, stays below 505. At 64px, this loader keeps up with that model and leaves the other image models waiting for data. If `train/mfu` is low in an image run, check the loader first. Video rows count 8-frame clips, whose throughput this image loader does not measure.

These two measurements do not establish the loader's maximum throughput. The run used 16 read threads. `tools/benchmark_data.py` uses the dataset specification's `Loading`, with no separate read-thread setting. Its defaults are no worker processes and 64 threads. Change them with `--data.loading.workers` and `--data.loading.threads`. Oxford Flowers has only 8189 small records, so it is unlike a sharded 12M-record set.
