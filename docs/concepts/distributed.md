# Distributed training

Every run is sharded, and a single device is the degenerate case. There is one code path: the trainer builds a two-axis mesh over the devices it was given, derives a sharding for every leaf of the train state, and hands both to `jax.jit`. GSPMD infers the collectives.

The pieces live in `dew.training.distributed`.

## The mesh

`build_mesh(fsdp_size, devices=None)` returns a `(data, fsdp)` mesh of shape `(device_count // fsdp_size, fsdp_size)`. Parameters shard over `fsdp` and replicate over `data`, so `fsdp_size=1` is plain data parallelism and needs no separate branch. A `fsdp_size` that does not divide the device count is an error rather than a silent reshape.

Batches split across every device on both axes at once: `BATCH_SPEC = P(('data', 'fsdp'))`. Only parameters distinguish the two axes.

## Logical axes

A logical axis name says what a parameter's dimension is, never where it goes. A DiT attention query kernel, shape `[embed, heads, head_dim]`, carries `("embed", "heads", "head_dim")`.

`DEFAULT_LOGICAL_PARAM_AXES` declares them, keyed by the module path a parameter sits under and read outermost dimension first. A parameter takes the trailing names its rank can hold, so `("to_q",): ("embed", "heads", "head_dim")` names the query kernel's three dimensions and, from the same entry, its bias's two. A key matches the end of a path, so one entry covers every block that reuses the module, and an optimizer moment or an EMA copy inherits its parameter's names because its own path ends in the parameter's.

The names are declared here rather than on the initializers because `nn.with_partitioning` hands a parameter back inside a `Partitioned` box and `model.init` then gives that box to the caller. Parameter trees are frozen and a caller reads plain arrays under the documented names, `save_params` among them. Declaring the names here also keeps sharding vocabulary on the trainer's side of the seam, where the mesh already lives. What it costs: a module rename does not carry its entry along, so `test_every_declared_parameter_axis_names_a_module_some_model_has` fails as soon as an entry stops naming a parameter of a registry model, and a caller's own module gets the shape heuristic unless its parameters sit under a declared name.

Declared today: `CausalTransformer`'s token embedding, its attention and MLP projections and its untied head, and the DiT stack's patch embedding, attention, adaLN projections, MLP and output head. Through the shared modules that reaches `simple_dit`, `simple_udit`, `video_dit`, `hybrid_dit`, the MMDiT ada and output projections, UViT, the JEPA encoders and the attention blocks inside the U-Nets. The MMDiT and UViT blocks' own MLPs, the S5 layers and the rest of `blocks.py` fall back on shape.

The vocabulary:

| Axis | Meaning | Where it appears |
| --- | --- | --- |
| `embed` | model width | every kernel's contract or output axis |
| `mlp` | feed-forward width | gate/up/down projections |
| `heads` | query heads | q projections, attention out (with `head_dim`) |
| `kv` | key/value heads | k and v projections in grouped-query attention |
| `head_dim` | width of one head | attention projections carrying both head axes |
| `vocab` | vocabulary rows | the embedding table and the untied LM head |
| `modulation` | adaLN shift/scale/gate outputs | the per-block and final ada projections |
| `output` | patch output channels | the zero-init output head |
| `attention` | flattened attention input | the out projection's contract axis |
| `expert` | mixture-of-experts rows | reserved, no model uses it yet |
| `batch` | sample rows | activations only, never parameters |
| `sequence` | token positions | activations only, never parameters |
| `stage` | pipeline stage index | reserved, no model uses it yet |

A parameter no entry names keeps the shape heuristic below, so a family can be declared at a time. Norms and other 1-D parameters are left alone: they are replicated either way. So is any dimension whose width the model does not choose, the raw patch content of the Hilbert projection and the conditioning encoder's feature width, because there is no honest name for it and the heuristic already picks the larger side, which for CLIP-L into a 384-wide DiT is the encoder's 768.

## The rules table

`DEFAULT_LOGICAL_AXIS_RULES` maps each name to mesh axes, in precedence order. On the current mesh it is:

| Logical axis | Mesh axes |
| --- | --- |
| `vocab` | `fsdp` |
| `mlp` | `fsdp` |
| `modulation` | `fsdp` |
| `attention` | `fsdp` |
| `embed` | `fsdp` |
| `head_dim` | `fsdp` |
| `heads` | `fsdp` |
| `kv` | `fsdp` |
| `output` | `fsdp` |
| `expert` | `fsdp` |
| `batch` | none |
| `sequence` | none |
| `stage` | none |

Rule order is precedence: when two logical dimensions of one parameter both claim the single `fsdp` axis, the earlier row wins and the other dimension is left whole. That is flax's and MaxText's semantics. The order is chosen so the winner is the dimension the shape heuristic used to pick, which is what makes the default table a no-op:

| Kernel | Shape | Winner | Why it is also the largest |
| --- | --- | --- | --- |
| MLP expand/contract | `[embed, mlp]` | `mlp` | `mlp_ratio` is 2 or 4 in every shipped config |
| adaLN projection | `[embed, modulation]` | `modulation` | six modulation vectors per block |
| token embedding / LM head | `[vocab, embed]` | `vocab` | vocabularies are tens of thousands wide |
| attention out | `[attention, embed]` | `attention` | `heads * head_dim == embed`, and the tie goes left |
| q/k/v | `[embed, heads]`, `[embed, kv]` | `embed` | grouped-query `kv` is narrower; a tie goes left |
| output head | `[embed, output]` | `embed` | `patch^2 * channels` is far below the model width |

Precedence is fixed, so it cannot track the largest axis for every conceivable shape: a model narrower than its own output head, or an `mlp_ratio` of exactly 1, would land on the other dimension. That is a different split of the same parameter over the same axis, with identical memory, identical collectives and identical numbers, not a different model, and no shipped configuration reaches it. `test_default_logical_rules_match_previous_specs_for_every_registry_model` pins spec-for-spec equality with the old heuristic across every registry architecture at the sizes the suite builds, and the FSDP/DP parity tests pin the numbers.

`--trainer.logical-axis-rules` takes a JSON object and replaces the table, e.g. `{"mlp": "fsdp"}`. A name set to `null`, or absent from the table, leaves that dimension whole. Mesh axes the current mesh does not have are dropped, so one table can name `tensor` today and mean it later.

The table is the whole mechanism for future parallelism. Adding a `tensor` axis to the mesh and changing two rows (`"heads": "tensor"`, `"mlp": ["tensor"]`) moves every declared model to hybrid FSDP/tensor parallelism with no model edits, exactly as MaxText's `logical_axis_rules` does (`docs/research/google-jax-stack.md`, MaxText section). An `fsdp_transpose` axis would land in the same place.

`state_sharding_tree(mesh, abstract_state, min_shard_size, logical_axis_rules)` implements the derivation, one pass over the abstract state: `_logical_axes` reads the declared names off each leaf's path, `nn.logical_to_mesh_axes` applies the rules table, and the result drops size-1 mesh axes and replicates a parameter whose assigned dimension the mesh axes do not divide. Below `min_shard_size` elements a declared parameter stays replicated too. Deriving names and values in the same pass is what lets an optimizer state hold leaves that are not arrays at all, `optax.MaskedNode` under a masked transform among them.

## Which parameters shard

`parameter_spec(shape, fsdp_size, min_shard_size)` picks the largest axis that divides evenly by `fsdp_size` and shards it. Anything smaller than `min_shard_size` elements (`DEFAULT_MIN_SHARD_SIZE`, 65536) stays replicated: below that a parameter costs more in collectives than it saves in memory.

`state_sharding_tree` maps sharding over the whole train state, not just the params, declared and undeclared alike. Optimizer moments and the EMA copy carry the same axes as the parameters they track, so they pick up the same spec without anyone describing the optimizer's layout.

The state is then built straight into that layout: the trainer runs `jax.jit(lambda: nn.unbox(init_fn()), out_shardings=state_sharding)()`, so a model too large for one device is never materialized on one device, and any flax metadata a caller's own module attached is gone before the optimizer, the EMA or the checkpointer can see it.

## Sharding tolerance

`assert_params_sufficiently_sharded` is a startup assertion, run by the trainer whenever `fsdp_size > 1`: if more than `sharding_tolerance` of the shardable parameter elements ended up replicated, the run stops before step one, naming the fraction and the five largest replicated parameters by path, shape and element count. This is the guardrail against a mesh whose `fsdp` axis divides none of the model's dimensions, which the shape heuristic used to absorb in silence. `--trainer.sharding-tolerance 1.0` disables it.

"Shardable" means at or above `min_shard_size`. The reference (`utils/sharding.py:605` `assert_params_sufficiently_sharded`, tolerance from `configs/base.yml:672-673`) is the same ratio — replicated elements over total, raise above the tolerance, name the five largest by `jax.tree_util.keystr` path — taken over every parameter, which it can do because it has no size threshold. Here a small parameter is replicated on purpose, so counting it would make the check fire on models that are merely small rather than badly laid out. MaxText also restricts the check to mesh axes that exist and are larger than one (`_get_nontrival_mesh_axes`); on a `(data, fsdp)` mesh that is `fsdp` alone, so the check is skipped entirely when `fsdp_size == 1`. The default tolerance is MaxText's 0.02.

## The step

The training step is jitted with explicit `in_shardings` (state, replicated rng, sharded batch) and `out_shardings`, and donates the train state so its buffers are reused. Nothing may alias the donated state after the call, which is why the trainer reassigns `self.state` from the step's return value.

The loss is a mean over the batch-sharded axis, so gradients carry their cross-device all-reduce on their own. There is no explicit `pmean` anywhere in the trainer.

Loss health is watched on device: a non-finite streak counter rides along with the step and is read on the logging cadence, so the loop never synchronizes just to check. A streak of `max_bad_loss_steps` stops the run with an error instead of letting it burn through the schedule.

## Feeding the devices

`shard_batch(sharding, batch)` assembles this process's slice of each array into a globally sharded array with `jax.make_array_from_process_local_data`, which is what makes the multi-host case identical to the single-host one.

A multi-process run has to join the process pool before any of that. The recipes call `jax.distributed.initialize()`, which finds the coordinator from the environment on TPU pods and clusters; on a machine with no cluster environment the call reports that and the run continues on one process. Any other failure stops the run rather than training one process on a slice of the data and calling it a full run. `--trainer.multi-host True` requires the pool, `--trainer.multi-host False` never asks for it, and the default `None` is the behaviour above.

`DevicePrefetchIterator(iterator, sharding, depth=2)` runs that transfer a few batches ahead of the loop on a background thread. Without it the host-to-device copy sits on the critical path, because the loop only starts moving batch N+1 after step N has been dispatched. Exceptions raised in the thread are re-raised on the consumer's side rather than swallowed.

If the underlying iterator can report a position (grain's can), the prefetcher tracks the position of the batch it most recently handed out, not the one the thread has raced ahead to. That is what makes a mid-epoch resume land on the next unseen batch.

## Checkpoints

Saving is async and sharded arrays go straight to orbax; gathering them onto the host first would serialize the whole state through one process. `wait_for_checkpoints()` blocks until the writes have landed, and anything that reads a checkpoint back has to call it first.

A checkpoint holds the train state, the rng state, the best loss, the epoch, and the data iterator's position when there is one (grain reports it as JSON bytes, which ride along as a uint8 array).

Restoring builds a template from the freshly initialized state, so shapes, dtypes and the step counter survive, and it passes `ArrayRestoreArgs` with this run's sharding for the state alone. A checkpoint written on one mesh therefore restores onto a different one, and the rest of the payload stays on the host.

## Throughput

`dew.telemetry.instrumentation` measures rather than estimates. `step_flops(jitted, *args)` asks the compiler for the per-device cost analysis of the compiled step. `model_flops_utilization(flops_per_step, step_time)` turns that into a fraction of one device's peak using a small table of vendor dense bf16 numbers; hardware that is not in the table reports nothing rather than a made-up number. `enable_compilation_cache(path)` persists compiled executables so a restart skips XLA compilation.

That cache is on by default: `--trainer.compilation-cache-dir` starts at `default_compilation_cache_dir()` (`$XDG_CACHE_HOME/dew/xla`, or `~/.cache/dew/xla`), and `--trainer.compilation-cache-dir None` turns it off. On a DiT-B it takes the time to the first step from 55s to 5s and leaves the step itself alone.

The trainer logs `train/samples_per_sec` and the MFU it could compute alongside the loss, on the same logging cadence.

`--trainer.xla-flags` appends to `XLA_FLAGS` for the run. `prepare_process` applies it, which is a recipe's first line, because XLA reads that variable once when it opens a backend: a flag set after the first JAX call does nothing. A library user who never runs a recipe sets `XLA_FLAGS` in the environment instead. The default is None, and `docs/performance.md` has the sweep behind that default.
