# Distributed training

Every run is sharded, and a single device is the degenerate case. There is one code path: the trainer builds a two-axis mesh over the devices it was given, derives a sharding for every leaf of the train state, and hands both to `jax.jit`. GSPMD infers the collectives.

The pieces live in `dew.training.distributed`.

## The mesh

`build_mesh(fsdp_size, devices=None)` returns a `(data, fsdp)` mesh of shape `(device_count // fsdp_size, fsdp_size)`. Parameters shard over `fsdp` and replicate over `data`, so `fsdp_size=1` is plain data parallelism and needs no separate branch. A `fsdp_size` that does not divide the device count is an error rather than a silent reshape.

Batches split across every device on both axes at once: `BATCH_SPEC = P(('data', 'fsdp'))`. Only parameters distinguish the two axes.

## Which parameters shard

`parameter_spec(shape, fsdp_size, min_shard_size)` picks the largest axis that divides evenly by `fsdp_size` and shards it. Anything smaller than `min_shard_size` elements (`DEFAULT_MIN_SHARD_SIZE`, 65536) stays replicated: below that a parameter costs more in collectives than it saves in memory.

`state_sharding_tree(mesh, abstract_state)` maps that over the whole train state, not just the params. Optimizer moments and the EMA copy have the same shapes as the parameters they track, so they pick up the same spec without anyone describing the optimizer's layout.

The state is then built straight into that layout: the trainer runs `jax.jit(init_fn, out_shardings=state_sharding)()`, so a model too large for one device is never materialized on one device.

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
