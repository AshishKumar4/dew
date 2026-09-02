# Step benchmarks

What one training step costs, per architecture, measured through
`ObjectiveTrainer`'s own compiled step rather than a hand-written forward
pass. Reproduce with `tools/benchmark_step.py`; the loader is measured
separately by `tools/benchmark_data.py`.

FLOPs come off the compiled executable through
`dew.telemetry.instrumentation.compiled_flops`, and `util` is the same figure
the trainer logs as `train/mfu`: the step's measured FLOPs divided by the step
time and by one device's dense bf16 peak (97.5 TFLOP/s for this card).

## `--preset small` on one RTX 4080

```
python tools/benchmark_step.py --preset small --steps 10 --json-out bench.json
```

Run 2026-09-02, jax 0.11.1 / jaxlib 0.11.1 (CUDA), flax 0.12.9, optax 0.2.8,
driver 595.84, RTX 4080 16 GiB, dew at `2cc0bff`. Every model in bf16
(`dtype=bfloat16`), single device, `fsdp_size=1`, adam, 2 warmup steps.

| architecture       | sample    | batch |      params | ms/step | samples/s | GFLOP/step |  util | peak GiB | compile s | steps |
|--------------------|-----------|-------|-------------|---------|-----------|------------|-------|----------|-----------|-------|
| unet               | 64x64x3   |    16 |  10,159,299 |    18.8 |     849.2 |       28.7 |  1.6% |     0.75 |      89.4 |    10 |
| uvit               | 64x64x3   |    16 |  24,351,024 |    24.0 |     668.0 |      326.6 | 14.0% |    1.70* |      29.4 |    10 |
| simple_udit        | 64x64x3   |    16 |  23,381,424 |    10.5 |    1520.0 |      326.2 | 31.8% |    1.04* |      25.6 |    10 |
| simple_dit         | 64x64x3   |    16 |  19,835,568 |    10.4 |    1537.1 |      296.2 | 29.2% |    1.70* |      15.2 |    10 |
| simple_mmdit       | 64x64x3   |    16 |  36,385,584 |    18.9 |     846.8 |      319.1 | 17.3% |    1.40* |      29.9 |    10 |
| hierarchical_mmdit | 64x64x3   |    16 |  55,498,188 |    35.4 |     451.4 |      695.9 | 20.1% |    3.40* |     100.4 |    10 |
| hybrid_dit         | 64x64x3   |    16 |  19,344,048 |    10.2 |    1564.0 |      223.0 | 22.4% |    3.40* |      22.9 |    10 |
| video_dit          | 8x64x64x3 |     4 |  25,155,504 |    19.3 |     207.0 |      616.0 | 32.7% |    3.40* |      43.2 |    10 |
| unet_3d            | 8x64x64x3 |     4 |  11,045,699 |    37.1 |     107.7 |      145.8 |  4.0% |     1.72 |     114.1 |     5 |
| jepa_encoder       | 64x64x3   |    16 |  12,149,568 |    12.0 |    1329.0 |      261.3 | 22.3% |     0.69 |      46.2 |     5 |
| jepa_video_encoder | 8x64x64x3 |     4 |  16,143,360 |    21.5 |     186.0 |      621.6 | 29.6% |     1.26 |      76.8 |     5 |
| causal_transformer | 512 tokens |    16 |  66,950,784 |    83.8 |     190.9 |     1320.1 | 16.2% |     5.80 |      19.5 |    10 |

`jepa_predictor` has no step of its own: it is built through the registry and
trained inside the two JEPA rows. The `causal_transformer` row is GPT-2 small's width at three layers with a 50k vocabulary, measured on 2026-09-02 after the language model wave landed; its FLOPs are dominated by the vocabulary projection.

`*` marks a peak that is an upper bound rather than that row's own figure. The
allocator's high-water mark is monotonic for the life of a process and has no
reset hook, so only the first architecture in an invocation gets its own peak.
Run one architecture per invocation (`--architectures unet`) for a clean
number; the JSON also carries `case_peak_delta_bytes`.

The table was assembled from three invocations rather than one sweep, which is
why the step counts differ: 10 steps for the eight image and video diffusion
rows, 5 for the last three. Rows measured while the host was busy with
something else came out materially slower (`unet` read 54 ms/step under load
against 18.8 ms idle), so every row here is from a run with the host
otherwise idle.

### What the numbers say

- The DiT family is the efficient shape on this card: `simple_dit`,
  `simple_udit` and `hybrid_dit` all sit at 10 ms/step and 22-32% of peak,
  which is where a 64px/patch-4 (256 token) workload should be.
- `unet` is dispatch-bound, not compute-bound: 28.7 GFLOP/step is an order of
  magnitude below the transformers here, and it still takes 18.8 ms, for 1.6%
  utilisation. Convolutional stacks at this width are a long chain of small
  kernels; batch or resolution has to grow before the card is busy.
- `unet_3d` inherits the same problem and adds frames: 4% utilisation and the
  slowest step in the table, against `video_dit` at 32.7% on the same
  (8, 64, 64, 3) samples. For video, the factorized transformer is not a
  stylistic preference.
- `hierarchical_mmdit` is the largest model here (55 M) and the most expensive
  step, as its 1024-token finest stage implies.
- Compile dominates a short run: 15-114 s per architecture against 10-37 ms
  per step. A sweep is mostly XLA, and a real run should set
  `compilation_cache_dir`.

## `--preset cpu-smoke`

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python tools/benchmark_step.py --preset cpu-smoke --steps 2
```

Tiny models on the simulated 8-device CPU mesh, for checking the tool itself
rather than the hardware. `tests/test_architectures.py` runs one case of this
preset so the tool cannot rot against the trainer internals it drives.
Utilisation and peak memory come back `null`: there is no published peak
FLOP/s for a CPU and no allocator counter behind it.

## Data loader

```
python tools/benchmark_data.py --dataset oxford_flowers102 --batch-size 8 \
    --image-size 64 --steps 100 --warmup 5 --worker-count {0,8}
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
16 read threads, where `DataConfig` uses 32 workers and 140 read threads, and
Oxford Flowers is 8189 small records rather than a sharded 12M-record set.
