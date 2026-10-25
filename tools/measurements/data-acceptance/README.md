# Data acceptance receipts

The data loader's acceptance checks, measured at Dew `5e4698ff5` (main `54cf2697d` with this
tool) by `tools/benchmark_data.py`. Each arm runs in a fresh interpreter over the same produced
files: Dew's reader (`ImageData` or `TextData` through `loader`) and plain Grain over the same
record steps, two loading workers each.

| File | Host | Run |
| --- | --- | --- |
| `matrix-cpu.json`, `scaling-cpu.json` | armada container, 8 vCPU, CPU JAX | `python tools/benchmark_data.py matrix` and `scaling` |
| `matrix-a100.json`, `scaling-a100.json` | dew-a100b, A100-SXM4-40GB, 30 CPUs, whole GPU | the same, `JAX_PLATFORMS=cuda` for the matrix |

## Checks

| Check | Image, CPU | Text, CPU | Image, A100 | Text, A100 |
| --- | --- | --- | --- | --- |
| Throughput within 5% of plain Grain | 0.894, fails | 0.975, passes | 0.904, fails | 0.923, fails |
| Resident memory within Grain's + 10% | 0.996, passes | 1.022, passes | 1.035, passes | 1.144, fails |
| Warm first batch under 10 s | 6.6 s, passes | 5.5 s, passes | 5.3 s, passes | 5.1 s, passes |
| Cold first batch under 30 s | 5.6 s, passes | 5.4 s, passes | 5.4 s, passes | 5.1 s, passes |
| Host wait for data, of wait and step | 15.8% | 3.0% | 13.2% | 24.7% |

- Throughput is records per second for images and real tokens per second for text, in steady
  state after warmup. A single run moves by several percent: six alternating Dew and Grain runs
  of the image arm in one container gave Dew over Grain 0.927 on average, standard deviation
  0.045 (0.855 to 0.982). The image gap is below 0.95 in every receipt; the text gap is not
  consistent across hosts (0.975 on CPU, 0.923 on the A100 host).
- A cold first batch opens a fresh copy of the inputs Dew never provisioned, written through
  and evicted from the page cache (`POSIX_FADV_DONTNEED`), under a cache home of its own. The
  host's page cache is not dropped, which needs root.
- Host wait is the training loop's time in `next(reader)` over wait and step, with the tool's
  toy models: their steps take 1 to 25 ms, so the loader is most of the step by construction. It
  overlaps the updates still running on the device, so it is not the device's idle; these
  receipts record it as `device_idle_fraction` under the 5% check. The tool now reads device
  idle from the accelerator's timeline of traced steps (`device_idle_fraction`, the part a batch's
  copy ended as `device_input_idle_fraction`) and checks that; none of these receipts has it.

## Interleave scaling

A streamed read of 4096 encoded images in 16 shards, decoding and resizing inside each shard's
stream, by how many shards it interleaves (records per second over one shard at a time):

| Cycle | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| CPU, 8 vCPU | 1.00 | 1.46 | 1.77 | 1.30 |
| A100 host, 30 CPUs | 1.00 | 1.56 | 1.72 | 1.89 |

Scaling is sublinear on both hosts, and on 8 vCPUs a cycle of 8 is slower than one of 4.

## Not measured here

- First batch from R2, warm: needs the R2 filesystem adapter, which public Dew does not carry.
- One transform per record (fusing adjacent record steps), NUMA-aware workers and GPU decode
  are not built.

## Open

- The image read is 6 to 11% below plain Grain in every receipt (#157). Without loading workers
  the two read at the same rate (682 against 669 records per second), so the gap is in the
  worker path. A batch's caption column crossing in a shared-memory segment of its own costs
  the reader about 0.45 ms a segment, but removing it did not move the ratio.
- Text against Grain over the same planned windows (#155): 0.975 on CPU, 0.923 on the A100 host,
  where resident memory is 1.144 of Grain's.
