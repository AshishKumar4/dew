# Distributed training

Every run is sharded; one device is the degenerate case. The trainer builds a mesh over the devices it was given, derives a sharding for every leaf of the train state, and hands both to `jax.jit`. GSPMD inserts the collectives. The pieces live in `dew.training.distributed`.

## The mesh

`MeshSpec` names how many devices each sharding axis takes, and the data axis fills the rest:

```python
from dew import MeshSpec

MeshSpec()                          # data parallel: parameters replicated
MeshSpec(fsdp=4)                    # each large parameter split four ways
MeshSpec(fsdp=2, expert=4)          # experts split four ways, the rest two ways
MeshSpec(fsdp=2, tensor=2)          # widths a rule redirects split over tensor
MeshSpec(fsdp=4, sequence=2)        # token rows split over sequence
```

`build_mesh(spec)` returns the `(data, expert, fsdp, tensor, sequence)` mesh. Parameters shard over `fsdp`, an MoE layer's expert dimension over `expert`, a width the rules redirect over `tensor`, and everything replicates over `data`. The sequence axis never places a parameter; it splits the batch's sequence dimension. A product of sizes that does not divide the device count is an error.

The expert dimension has an axis of its own because it is the one dimension no dense model has. Splitting eight experts four ways leaves every expert whole, where splitting a width costs a collective on every matmul.

## How a batch is placed

`batch_shardings(mesh, batch)` gives one sharding per leaf. Rows split across every axis but sequence. A leaf of rank 2 or 3 is a sequence per row (token ids, segment ids, positions) and its second dimension splits over the sequence axis when the axis divides it; an image or a video keeps every dimension but its rows whole. `shard_batch(mesh, batch)` assembles this process's slice of each array into the global array with `jax.make_array_from_process_local_data`, which is what makes a multi-host run the same code as a single-host one.

## Logical axes

A logical axis name says what a parameter's dimension is, never where it goes. A DiT query kernel of shape `[embed, heads, head_dim]` carries `("embed", "heads", "head_dim")`.

The module that owns the parameters declares them:

```python
import flax.linen as nn
from dew.nn.sharding import logical_axes

@logical_axes({("my_q",): ("embed", "heads", "head_dim"),
               ("my_out",): ("attention", "embed")})
class MyAttention(nn.Module):
    ...
```

A key is the tail of a module path, so one entry covers every block that reuses the module, and an optimizer moment or an EMA copy inherits its parameter's names because its own path ends in the parameter's. A parameter takes the trailing names its rank can hold, so the entry above names the query kernel's three dimensions and its bias's two. Two modules that declare the same path differently are refused at import. A module whose kernels have no honest name (a convolution's taps, a raw patch projection) lists them under `heuristic=` and takes the shape rule below.

| Axis | Meaning | Where it appears |
| --- | --- | --- |
| `embed` | model width | every kernel's contract or output axis |
| `mlp` | feed-forward width | gate, up and down projections |
| `heads` | query heads | q projections, attention out with `head_dim` |
| `kv` | key and value heads | k and v projections of grouped-query attention |
| `head_dim` | width of one head | attention projections carrying both head axes |
| `vocab` | vocabulary rows | the embedding table and the untied LM head |
| `modulation` | adaLN shift, scale and gate outputs | the per-block and final ada projections |
| `output` | patch output channels | the zero-initialised output head |
| `attention` | flattened attention input | the out projection's contract axis |
| `exp` | mixture-of-experts rows | the stacked expert kernels and the router |
| `batch` | sample rows | activations only |
| `sequence` | token positions | activations only |
| `stage` | pipeline stage | reserved |

## The rules

`DEFAULT_RULES` maps each name to a mesh axis, in precedence order:

| Logical axis | Mesh axis |
| --- | --- |
| `vocab`, `mlp`, `modulation`, `attention`, `embed`, `head_dim`, `heads`, `kv`, `output` | `fsdp` |
| `exp` | `expert` |
| `sequence` | `sequence` |
| `batch`, `stage` | none |

When two dimensions of one parameter both claim `fsdp`, the earlier row wins and the other dimension stays whole, which is flax's and MaxText's semantics. The order puts the larger dimension first for every shipped shape: `mlp` over `embed` in a feed-forward kernel, `vocab` over `embed` in an embedding table, `embed` over the narrower `heads` or `kv` in a projection. A dimension the assigned axis does not divide evenly is dropped and the axis passes to the next dimension that names it; GPT-2's 50257 vocabulary rows cannot split over any mesh, so its embedding shards on `embed`.

`Layout(rules, min_shard, tolerance)` holds the table. `--trainer.layout.rules '{"heads": "tensor", "mlp": "tensor"}'` replaces rows, which is how a run moves its attention and feed-forward widths onto the tensor axis with no model edits. A name set to `null` leaves that dimension whole. Below `min_shard` elements (65536 by default) a parameter stays replicated, because below that a parameter costs more in collectives than it saves in memory.

`Layout.shardings(mesh, state)` derives the placement of the whole train state in one pass: `declared_axes` reads the names off each leaf's path, the rules map them to mesh axes, and axes of size 1 drop out. Optimizer moments and the EMA copy pick up their parameter's spec without anyone describing the optimizer's layout. The trainer then builds the state straight into that layout with `jax.jit(initial_state, out_shardings=...)`, so a model too large for one device is never materialised on one.

## Sharding tolerance

`Layout.check` runs before step one whenever a parameter axis of the mesh is above one. If more than `tolerance` (2% by default) of the shardable parameter elements are replicated, the run stops and names the fraction and the five largest replicated parameters by path, shape and element count. This is the check MaxText runs (`assert_params_sufficiently_sharded`), restricted here to parameters at or above `min_shard`, since a small parameter is replicated on purpose. `--trainer.layout.tolerance 1.0` disables it.

## The step

The training step is jitted with explicit `in_shardings` (the state's layout, a replicated loss scale, the batch placement) and `out_shardings`, and donates the train state. The loss is a mean over the batch-sharded axis, so the gradient carries its own cross-device all-reduce; there is no `pmean` in the trainer.

Loss health is watched on device. A counter of consecutive non-finite losses rides along with the step and is read on the logging cadence, so the loop never synchronises to check it. A streak of `BAD_LOSS_STEPS` (5) stops the run.

## Feeding the devices

`DevicePrefetchIterator(iterator, mesh, depth=2)` runs the host-to-device transfer a few batches ahead of the loop on a background thread. It tracks the position of the batch it most recently handed out, not the one the thread has raced ahead to, which is what makes a mid-epoch resume land on the next unseen batch. An exception in the thread is raised on the consumer's side.

A multi-process run joins the process pool before any of that. `prepare_process(multi_host=...)` calls `jax.distributed.initialize()`, which finds the coordinator from the environment on TPU pods and clusters; on a machine with no cluster environment the run continues on one process. `multi_host=True` requires the pool, `multi_host=False` never asks for it. Right after the join every process runs one collective while the processes are still together, so a process that missed the pool fails there and not minutes later inside the first step.

A validation pass scores the same number of batches on every process. The token splits are whole files strided per process, so one process can run out before another; each batch is agreed with `minimum_across_processes` before it is scored, so no process leaves the pass while the others wait in its collectives.

## Checkpoints

`Checkpoints(directory, keep=2)` writes with Orbax, asynchronously, and sharded arrays go from the devices to storage without passing through one host. `Checkpoints.wait()` blocks until the writes have landed; `fit` calls it before returning.

A checkpoint holds the train state (`step`, `params`, `opt_state`, `ema`, `key`) and `position`, every process's place in its data stream: one row per process, gathered before the save because Orbax writes a host array from process zero alone. On restore each process takes its own row. A checkpoint written by two processes refuses to resume on one, with both counts in the message, since a position per process has no meaning on a different count.

Restoring builds a template from the freshly initialised state, so shapes, dtypes and the step survive, and passes this run's shardings for the state. A checkpoint written on one mesh restores onto another.

## Throughput

`dew.telemetry.instrumentation` measures rather than estimates. `compiled_flops(compiled)` counts the matmuls and convolutions in the compiled step's optimised HLO from their own shapes, including the cuBLAS, cuDNN convolution and fused-attention custom calls a GPU backend hands them to. `model_flops_utilization(flops, step_time)` turns that into a fraction of one device's dense bf16 peak from a table covering TPU v4 to v6e, A100, H100, H200 and the RTX 4080; hardware not in the table reports nothing. The trainer logs `train/samples_per_sec` and `train/mfu` on the logging cadence.

The XLA compilation cache is on by default under `~/.cache/dew/xla` (`--trainer.compilation-cache-dir None` turns it off). On a DiT-B it takes the time to the first step from 55 s to 5 s.

`--trainer.xla-flags` appends to `XLA_FLAGS` for the run. `prepare_process` applies it as a recipe's first line, because XLA reads the variable once when it opens a backend. A library user who never runs a recipe sets `XLA_FLAGS` in the environment. `docs/performance.md` has the sweep behind the default.
