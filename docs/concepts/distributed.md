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
MeshSpec(fsdp=4, stage=2, microbatches=8)  # the layer stack in two pipeline stages
```

`build_mesh(spec)` returns the `(data, expert, fsdp, tensor, sequence, stage)` mesh. Parameters shard over `fsdp`, an MoE layer's expert dimension over `expert`, a width the rules redirect over `tensor`, and everything replicates over `data`. The sequence axis never places a parameter; it splits the batch's sequence dimension. The stage axis holds the decoder's pipeline stages and never places a stored parameter either. A product of sizes that does not divide the device count is an error.

The expert dimension has an axis of its own because it is the one dimension no dense model has. Splitting eight experts four ways leaves every expert whole, where splitting a width costs a collective on every matmul.

## How a batch is placed

`batch_shardings(mesh, batch)` gives one sharding per leaf. Rows split across every axis but sequence. A leaf of rank 2 or 3 is a sequence per row (token ids, segment ids, positions) and its second dimension splits over the sequence axis when the axis divides it; an image or a video keeps every dimension but its rows whole. `shard_batch(mesh, batch)` assembles this process's slice of each array into the global array with `jax.make_array_from_process_local_data`, so a multi-host run and a single-host run share the code.

## The sequence axis

Attention is where the sequence axis does its work. Under a mesh whose sequence axis is above one, `scaled_dot_product_attention` splits its queries over that axis, gathers the keys and values whole once per layer, and computes every query's row against the whole key set, so a layer's activations and its attention scores are divided by the axis while the parameters stay where the layout put them. The batch's token rows may or may not divide the axis (a training batch is `seq_len + 1` wide, so they usually do not); the attention claims the axis for the model's sequence either way. The trainer runs its compiled step under `jax.set_mesh`, which is how the seam sees the axis; a model called outside that context, in a test or a script, computes over whole sequences.

A causal mask makes the work uneven: the last query attends every key and the first attends one. A causal, windowed or masked call therefore reorders the queries into the striped layout of MaxText's context parallelism: the sequence is cut into `2 * shards` chunks and shard `i` holds chunks `i` and `2 * shards - 1 - i`, so every shard sees the same number of (query, key) pairs. The positions travel with the rows, so the rotary angles the caller applied and the causal, window and segment masks are exact under the reorder, and the output is put back in sequence order before it leaves the seam. This needs the model's sequence length to be a multiple of `2 * shards`, which the seam refuses by name otherwise. A call with no mask has equal work on every row and keeps its order.

On the eight-device simulated mesh, `MeshSpec(fsdp=4, sequence=2)` against `MeshSpec(fsdp=8)` trains a `causal_transformer` to the same loss (4.8e-7 over 30 steps) and the same gradients (1.2e-7 on one step, on a dense batch and on a packed one with segment ids and positions); tests/test_sequence_parallel.py holds those numbers. Decoding against the KV cache is refused under a sequence axis above one: the cache holds whole sequences, so generation runs on a mesh with `sequence=1`.

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

A key is the tail of a module path, so one entry covers every block that reuses the module, and an optimizer moment or an EMA copy inherits its parameter's names because its own path ends in the parameter's. A parameter takes the trailing names its rank can hold, so the entry above names the query kernel's three dimensions and its bias's two. Two modules that declare the same path differently are refused at import. A module whose kernels have no meaningful name (a convolution's taps, a raw patch projection) lists them under `heuristic=` and takes the shape rule below.

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
| `stage` | pipeline stage | the stacked layer weights and activations inside a pipelined step |

## The rules

`DEFAULT_RULES` maps each name to a mesh axis, in precedence order:

| Logical axis | Mesh axis |
| --- | --- |
| `vocab`, `mlp`, `modulation`, `attention`, `embed`, `head_dim`, `heads`, `kv`, `output` | `fsdp` |
| `exp` | `expert` |
| `sequence` | `sequence` |
| `batch`, `stage` | none; the mesh axis of the same name places them |

When two dimensions of one parameter both claim `fsdp`, the earlier row wins and the other dimension stays whole, which is flax's and MaxText's semantics. The order puts the larger dimension first for every shipped shape: `mlp` over `embed` in a feed-forward kernel, `vocab` over `embed` in an embedding table, `embed` over the narrower `heads` or `kv` in a projection. A dimension the assigned axis does not divide evenly is dropped and the axis passes to the next dimension that names it; GPT-2's 50257 vocabulary rows cannot split over any mesh, so its embedding shards on `embed`.

`Layout(rules, min_shard, tolerance)` holds the table. `--trainer.layout.rules '{"heads": "tensor", "mlp": "tensor"}'` replaces rows and moves the attention and feed-forward widths onto the tensor axis with no model edits. A name set to `null` leaves that dimension whole. Below `min_shard` elements (65536 by default) a parameter stays replicated, because below that a parameter costs more in collectives than it saves in memory.

`Layout.shardings(mesh, state)` derives the placement of the whole train state in one pass: `declared_axes` reads the names off each leaf's path, the rules map them to mesh axes, and axes of size 1 drop out. Optimizer moments and the EMA copy pick up their parameter's spec. The trainer builds the state straight into that layout with `jax.jit(initial_state, out_shardings=...)`, and a model larger than one device is materialised sharded.

## Sharding tolerance

`Layout.check` runs before step one whenever a parameter axis of the mesh is above one. If more than `tolerance` (2% by default) of the shardable parameter elements are replicated, the run stops and names the fraction and the five largest replicated parameters by path, shape and element count. This is the check MaxText runs (`assert_params_sufficiently_sharded`), restricted here to parameters at or above `min_shard`, since a small parameter is replicated on purpose. `--trainer.layout.tolerance 1.0` disables it.

## The stage axis

`MeshSpec(stage=N)` splits a `CausalTransformer`'s layers into N contiguous stages and runs its training step as a pipeline over the stage axis, the GPipe schedule MaxText's `layers/pipeline.py` runs. The batch splits into `microbatches` microbatches (a multiple of N; unset is one per stage). Every iteration runs all N stages at once on one microbatch each, stage s on the microbatch that stage s - 1 finished the iteration before, and hands each stage's output to the next stage with a collective permute. The microbatches wait in a stage-sharded buffer that rotates up a stage each iteration, so each one reaches stage 0 in turn and each finished one lands in the last stage's slot; nothing is gathered until the loop ends. The first N - 1 iterations and the last N - 1 run stages on nothing, and that idle share, (N - 1) / (microbatches + N - 1), is what more microbatches buy down. The embeddings, the final norm, the head and the loss run on the whole batch on every stage's devices, so the loss is the whole stack's mean over the same rows; the pipeline covers the layers alone, and a model's prediction depths run after it.

Each stage's layers sit on that stage's devices. The step stacks every stage's copy of a layer position along a leading stage dimension and constrains it to the stage axis, so the matmuls of stage s run where stage s's slice lives, sharded over `fsdp` and `tensor` within the stage as the rules place them. The stored train state stays the plain loop's tree: one subtree per layer, placed by the same rules and replicated over the stage axis, so a checkpoint written on a pipelined mesh restores on any other and a Hugging Face tree loads unchanged. The stacking costs one copy of the layer parameters each way per step, and the gradient of every layer reaches every stage's devices before the optimizer's update, an all-gather over the stage axis of the parameters' size.

The stack has to split evenly. Every stage runs one program, so the stages have the same length and layer j of each stage is the same kind of layer as layer j of the first: the same kind, the same feed-forward, the same width, the same say in keys and values. A pattern that does not repeat every `num_layers / N` layers is refused with the first two layers that differ and the fields they differ in. DeepSeek V3's dense first layer, Gemma 4's and Gemma 3n's sharing layers at the end of the stack and Gemma 3's 26 layers of a six-layer pattern all fail that test; Mixtral, Qwen3 and the dense Llamas split at any divisor of their depth, and a Gemma pattern splits into stages that are whole periods of it. Decoding under a stage axis is refused too: generation runs outside the trainer's mesh context, on the whole stack.

On the eight simulated CPU devices the pipeline's loss and gradients agree with the whole stack's to 4.8e-07 and 3.4e-07 (tests/test_layer_stack.py, dense and MoE with the balance loss), and a pool of two processes each holding half of every stage's devices trains the losses of one process to 4.8e-07 (tests/test_multiprocess.py). The MFU the trainer logs counts the bubble's matmuls, since the device runs them.

## Scanned layers

`CausalTransformer(scan_layers=True)` runs every run of consecutive layers that share a parameter shape and a computation as iterations of one body under flax's scan, and the layers between such runs unrolled. The grouping is read off the resolved layers: same kind, same feed-forward and width, same activation sparsity, the same say in keys and values, and a layer whose keys and values a later layer reads is always a run of one. Qwen3's stack is one run; Gemma's 5:1 pattern is runs of five sliding layers with a full layer between them; DeepSeek V3 is its dense first layer and then one run of routed layers.

The tree stays the plain loop's. `init` always runs the plain loop, and the scan reads the same `layers_N` leaves through a stacked view: stacked along a leading layer axis on the way in and, for what a run creates (the KV cache, the routers' sowing), unstacked per layer on the way out. A checkpoint or a Hugging Face tree loads into a scanned model unchanged and saves back unchanged, and the scanned logits, decode path and gradients hold to the plain loop's at fp32 (tests/test_layer_stack.py, with the observed differences beside each bound). The one numeric difference is under a bf16 policy, where the plain loop's first layer returns the residual stream in fp32 (the dense feed-forward computes in fp32) and both loops take the stream in fp32 from the first layer on, since a scan carries one dtype and every stage takes the dtype the one before it returns; the scanned logits sit as far from the fp32 logits as the plain loop's do (5.2e-02 against 5.0e-02 on values of order 4) and a pipelined loss lands within 3.4e-04 of the plain loop's.

What it buys is compile time at depth, measured with `tools/benchmark_stack.py` on a decoder of width 64 (the command and the cases are in the tool):

| depth | CPU (i9-12900K), fp32 | RTX 4080, bf16, cudnn |
| --- | --- | --- |
| 24, plain | 3.41 s, 77 ms/step | 8.38 s, 3.2 ms/step |
| 24, scanned | 1.92 s, 100 ms/step | 4.29 s, 5.2 ms/step |
| 48, plain | 7.38 s, 151 ms/step | 17.36 s, 11.2 ms/step |
| 48, scanned | 3.54 s, 190 ms/step | 8.19 s, 19.4 ms/step |

Compile time halves at both depths and keeps a slope with depth, since the optimizer updates one leaf per layer either way. The step time at this width rises with the stacking of the per-layer leaves into the loop's input and the unstacking of their gradients, one copy of the layer parameters each way, and with the loop's dispatch; at a width where a layer's matmuls take the step, those copies are a small share of it.

## The step

The training step is jitted with explicit `in_shardings` (the state's layout, a replicated loss scale, the batch placement) and `out_shardings`, and donates the train state. The loss is a mean over the batch-sharded axis, so the gradient carries its own cross-device all-reduce; there is no `pmean` in the trainer. `Trainer.compile` returns a function that calls that jitted step under `jax.set_mesh`, which is what the benchmarks time: the mesh context is part of jit's cache key, and entering it costs nothing beside the dispatch.

The interval's loss and a counter of consecutive non-finite losses ride along with the step on device, in one dispatch, and are read on the logging cadence. A streak of `BAD_LOSS_STEPS` (5) stops the run.

## Feeding the devices

`DevicePrefetchIterator(iterator, mesh, depth=2)` runs the host-to-device transfer a few batches ahead of the loop on a background thread. It records the position of the batch it most recently handed out, so a mid-epoch resume lands on the next unseen batch. An exception in the thread is raised on the consumer's side.

A multi-process run joins the process pool before any of that. `prepare_process(multi_host=...)` calls `jax.distributed.initialize()`, which finds the coordinator from the environment on TPU pods and clusters; on a machine with no cluster environment the run continues on one process. `multi_host=True` requires the pool, `multi_host=False` never asks for it. Right after the join every process runs one collective while the processes are still together, so a process that missed the pool fails there and not minutes later inside the first step.

A validation pass scores the same number of batches on every process. The token splits are whole files strided per process, so one process can run out before another; each batch is agreed with `minimum_across_processes` before it is scored, so no process leaves the pass while the others wait in its collectives.

## Checkpoints

`Checkpoints(directory, keep=2)` writes with Orbax, asynchronously, and sharded arrays go from the devices to storage without passing through one host. `Checkpoints.wait()` blocks until the writes have landed; `fit` calls it before returning.

A checkpoint holds the train state (`step`, `params`, `opt_state`, `ema`, `key`) and `position`, every process's place in its data stream: one row per process, gathered before the save because Orbax writes a host array from process zero alone. On restore each process takes its own row. A checkpoint written by two processes refuses to resume on one, with both counts in the message, since a position per process has no meaning on a different count.

Restoring builds a template from the freshly initialised state, so shapes, dtypes and the step survive, and passes this run's shardings for the state. A checkpoint written on one mesh restores onto another.

### Local checkpoints

`Checkpoints(directory, keep=2, local_directory="/local/ssd/run", local_every=100)` keeps one more checkpoint on every host's own disk, the newest, written every `local_every` steps beside the persistent cadence `fit`'s `checkpoint_every` sets. Every process writes the shards its devices hold and the position table under `<local_directory>/process<index>`, so the same path serves a pod and a single machine running several processes, and the persistent directory keeps its cadence, its loss metrics and its best step untouched. On a resume `latest` is the newest step every process can read, wherever it is: a pod preempted between two persistent writes comes back from its local disks. A local checkpoint restores onto the placement it was written with and nothing else, because no process holds another process's shards; a resume on another mesh, layout or process count is refused with the leaf that moved or the counts, the local directory to delete and the persistent step that resume falls back to.

This is the semantics of Orbax's emergency checkpointing (`orbax.checkpoint.experimental.emergency`), built from the pieces its local manager uses. Its own manager runs on a CPU process pool, but takes over the persistent directory with a single fixed-shape state item (no metrics, so no best step, and no per-process position) and restores local checkpoints only where each data-parallel replica is a whole set of hosts, which rules out any mesh whose fsdp axis spans hosts; tests/test_multiprocess.py runs the local checkpoints on a real two-process pool.

## Throughput

`compiled_flops(compiled)` in `dew.telemetry.instrumentation` counts the matmuls and convolutions in the compiled step's optimised HLO from their own shapes, including the cuBLAS, cuDNN convolution and fused-attention custom calls a GPU backend hands them to. `model_flops_utilization(flops, step_time)` turns that into a fraction of one device's dense bf16 peak from a table covering TPU v2 to v6e, A100, H100, H200 and the RTX 4080, matched on the start of the device kind so `NVIDIA H100 80GB HBM3` finds the H100 row; hardware not in the table reports nothing. The trainer logs `train/samples_per_sec` and `train/mfu` on the logging cadence.

The XLA compilation cache is on by default under `~/.cache/dew/xla` (`--trainer.compilation-cache-dir None` turns it off). On a DiT-B it takes the time to the first step from 55 s to 5 s.

`--trainer.xla-flags` appends to `XLA_FLAGS` for the run. `prepare_process` applies it as a recipe's first line, because XLA reads the variable once when it opens a backend. A library user who never runs a recipe sets `XLA_FLAGS` in the environment. `docs/performance.md` has the sweep behind the default.

At the end of a fit the trainer logs two of the numbers MaxText's goodput reports, computed locally: `goodput/time_to_first_step_s`, from the start of `fit` to the first step's result (the placement or restore, the first batch, the compile and the step), and `goodput/step_fraction`, the share of the fit's wall time spent in steps, everything else being that start-up, the evaluations and the checkpoint writes with the wait for them at the end. A step's own data stall counts as step time, as MaxText's start-to-start step time does.
