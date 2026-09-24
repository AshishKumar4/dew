# Training on several nodes

This page shows how to start one training script on several machines, how to lay the mesh out so the slow network between them carries as little as possible, and how to rehearse all of it on one machine first. Read [distributed training](../concepts/distributed.md) before this page. It explains meshes and layouts.

## How a process pool starts

Every node runs the same script. Each copy is one process of a `jax.distributed` pool. The script joins the pool when it calls `dew.training.runtime.prepare_process()`, which every built-in recipe does on its first line. Call it yourself before you create any array if you write your own script.

`prepare_process` calls `jax.distributed.initialize`, which needs three facts: the address of process 0's coordinator, the number of processes and this process's rank. Where they come from depends on the cluster:

| Where you run | Who provides the three facts |
|---|---|
| Cloud TPU VM or pod | The TPU metadata server. JAX reads it. |
| Slurm (`srun`) | Slurm's `SLURM_*` variables. JAX reads them. One task (`SLURM_NTASKS=1`) forms no pool unless the run asks for one with `multi_host=True`; the process runs on its own. |
| Open MPI (`mpirun`) | The `OMPI_*` variables. JAX reads them, before Slurm's, so an `mpirun` inside a one-task allocation forms its pool. |
| Plain machines over ssh | `dew launch`, through `JAX_COORDINATOR_ADDRESS`, `DEW_PROCESS_COUNT` and `DEW_PROCESS_ID`. |

`--devices-per-process N` makes `dew launch` set `JAX_LOCAL_DEVICE_IDS`, so each of a host's processes takes its own N accelerators. Without it every process takes every local device, which is right for one process per host and wrong for several.

## Launch on plain machines

List the hosts, process 0's first, then the command after `--`:

```bash
dew launch --hosts node0 node1 -- /opt/dew/.venv/bin/python recipes/lm/train.py --trainer.multi-host True
```

The launcher starts the command on each host over `ssh -o BatchMode=yes`, so passwordless keys must already work. The remote side runs in the user's shell without a login profile, so a virtualenv activated there is not active: give the interpreter as an absolute path, as above. It runs a host named `localhost` directly. Each process starts in the current directory, which must exist at the same path on every host, or in the directory `--cwd` names. The coordinator listens on a port that is free on the first host when the launch starts, so pools started together on one machine never share one; name a port with `--port` when a firewall wants a fixed one, and set `--coordinator` when the other hosts reach that host by another name.

Every output line carries its rank, such as `[1] Joined the JAX process pool: process 1 of 2`. When one process exits with an error, the launcher stops the others and exits with that code. Without this, the survivors would wait in a collective for a peer that is gone.

A process of a pool that fails does not wait for its peers. It prints the error, writes it to the coordination service and exits at once, instead of sitting in `jax.distributed`'s shutdown barrier for up to 300 seconds. This holds from the moment the process joins the pool, so a process whose GPU fails to open, for example because it has no memory left, ends the launch too; otherwise its peers would wait minutes for its devices. When a rank fails between steps, for example because its data loader raised, its peers may already be inside the next step's collectives, which no GPU backend times out. The failing rank's error is written before it tries to agree with them, and a watch thread in every process ends the process when a published failure has not been heard at an agreement within 60 seconds. So the whole pool ends within about a minute, under `dew launch`, `srun` or a scheduler alike.

A rank can also stall without failing: blocked on a read, or in a compile that waits for its peers. Then every process stays alive and none reports anything, while the others wait inside a collective or a communicator's setup. For this case a pool bounds each device execution with XLA's execution watchdog. `prepare_process` sets `--xla_gpu_execution_terminate_timeout=30m` unless the run already set it. A process whose step, or whose sampling loop, runs longer than that ends, and the launcher, `srun` or the scheduler stops the rest. If you know how long your steps take, set a tighter value in `XLA_FLAGS` or through the recipe's `xla_flags`.

The watchdog bounds device executions only. Before a phase agreement's collectives, such as a checkpoint save or the end of `fit`, the ranks meet on the host, so a rank that arrives first waits there for one still busy with host work, for example process 0 uploading the final checkpoint to Weights & Biases. On the device it would sit inside a collective, which the watchdog would end. That host wait is bounded by `AGREEMENT_PATIENCE_SECONDS` in `dew.artifacts`, one day. A rank that stalls in host work before an agreement holds its peers for up to that long, so lower it when your host phases are short.

A pool keeps JAX's persistent compilation cache. jax 0.11.2 keys a cached executable by the fingerprint of the compiling process's accelerator topology, which on a GPU describes the device down to its NVLink links, and only process 0 writes entries. On one four-GPU host with an NVLink pair and a PCIe pair, ranks 0 and 1 found a step in a shared cache while ranks 2 and 3 compiled it, and that compile waited for ever for its peers' shares of the sharded autotuning. Dew pins jax to 0.11.2 with a fix (reported as jax-ml/jax#40940). A computation that spans processes hashes the fingerprints of all of them, so every rank loads the same entry, provided every process compiles for the same accelerators: the same platform and runtime, CUDA driver, cuDNN and cuBLAS, and the same device kinds, compute capability and core count. A pool whose processes differ in any of these compiles those computations without the cache, and JAX logs which processes differ. A GPU pool whose jax lacks the fix, such as an image's own jax or one installed with `--no-deps` around the pin, compiles without the persistent cache: `prepare_process` turns it off before the first compile.

To run one process per GPU instead of one per host:

```bash
dew launch --hosts node0 node1 --processes-per-host 8 --devices-per-process 1 -- /opt/dew/.venv/bin/python train.py
```

`--env NAME=VALUE` passes a variable to every process. The name must be a shell variable name, for example `--env XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`. Add `--dry-run` to print the exact commands without running them.

## Launch under Slurm

Inside an allocation, `--slurm` hands the launch to `srun`, and JAX reads the rank from Slurm. Under Slurm, JAX gives each task the one GPU at its `SLURM_LOCALID`, so run one task per GPU:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
dew launch --slurm --processes-per-host 8 -- /opt/dew/.venv/bin/python recipes/lm/train.py --trainer.multi-host True
```

This runs `srun --ntasks-per-node=8 --kill-on-bad-exit=1 --export=ALL ...`, with the `--env` variables in srun's environment. One task per node would see a single GPU, because JAX still picks the GPU at local rank 0 for it. The launcher refuses `--devices-per-process` under `--slurm` for the same reason. Plain `srun` with the same task layout works as well; the launcher adds the failure policy.

## Lay the mesh out for the network

Devices inside one node talk over NVLink or the TPU interconnect. Nodes talk over a network that is often ten times slower. Fully sharded data parallelism (`fsdp`) gathers every layer's parameters and reduce-scatters every layer's gradients, so an fsdp axis that crosses nodes pays that slow link on every layer.

Hybrid sharding keeps fsdp inside a node and replicates across nodes. Only one gradient all-reduce per step crosses the network. `MeshSpec(replicas=N)` asks for it:

```python
from dew.training import MeshSpec, Trainer

# Two nodes of eight GPUs: fsdp over the eight GPUs of each node,
# and the data axis of 2 across the two nodes.
mesh = MeshSpec(fsdp=8, replicas=2)
trainer = Trainer(objective, optimizer, key=key, mesh=mesh)
```

`replicas` counts groups of granules. A granule is whatever the devices' `slice_index` groups: a TPU slice on a multislice run, and on several GPU hosts a host or an NVLink domain, however many processes each host runs, because XLA numbers GPU slices per host boot or NVLink fabric. Where every device shares one slice, as in a CPU pool or on the hosts of one TPU slice, the process is the granule. `build_mesh` then builds the mesh with JAX's `mesh_utils.create_hybrid_device_mesh`, the same call MaxText uses for multislice runs. The data axis spans the groups. The expert, tensor, sequence and stage axes stay inside one granule. When a group holds more than one granule, fsdp is the only axis that crosses between them. `replicas` must divide both the granule count and the data axis, and fsdp must divide over the granules of one group. `build_mesh` raises with the numbers when they do not fit.

On GPU hosts whose device ids run host by host, `jax.make_mesh` already puts the data axis outermost, so for one granule per replica `replicas` states the layout rather than changing it. It changes the mesh when a replica spans several granules, when the slices do not follow the device order, and on a multislice TPU run.

With `replicas=1`, `jax.make_mesh` places the devices, which is what a single node or a single TPU slice wants. To choose which devices share an axis, pass them to `build_mesh(spec, devices)`: it fills the mesh with the list in the order you give, row-major over `MESH_AXES`, so the last axes get neighbouring entries. On a machine whose NVLink pairs are GPUs 0-1 and 2-3, `[0, 2, 1, 3]` puts a two-way tensor axis across the pairs instead of inside them. `jax.make_mesh` would sort GPU devices by id and discard that order.

Hybrid sharding keeps a full copy of the parameters and optimizer state on every replica group. Plain fsdp across all nodes divides them by the total device count instead. Choose hybrid sharding when one node's memory holds the sharded state, and plain fsdp when it does not.

## Split long sequences

`MeshSpec(sequence=N)` splits the token positions of every sequence over N devices, and each attention call exchanges data between them in one of two ways. Both give the same result as one device.

- The all-to-all follows DeepSpeed Ulysses. Each device trades its slice of the positions for a slice of the attention heads, attends the whole sequence for those heads and trades back. No device ever holds a whole key or value tensor. Causal masks, sliding windows, packed-document masks and the TPU splash kernel work as they do on one device. Grouped key and value heads are repeated only as far as the split needs.
- The gather keeps the queries split and gathers the whole keys and values on every device. It takes any head count and any key length. A causal or masked call reorders its queries so every device gets the same work, which needs the sequence length to divide by twice `sequence`.

A call runs the all-to-all when its query heads divide by `tensor` times `sequence` and its query and key lengths both divide by `sequence`. Where both exchanges can run, a causal, windowed or masked call takes the all-to-all. Its kernel sees whole sequences and the causal flag, and cuDNN and splash skip the masked blocks. The gather hands its kernel a mask for its reordered rows, which cuDNN runs as a dense bias over every logit. A call with no mask does the same work either way and takes the exchange that sends fewer bytes. Counted per device in units of sequence length times head width times (N-1)/N, with H query heads and K key heads per tensor shard, the all-to-all sends 2(H + K')/N, where K' is K repeated as far as the split needs, and the gather sends 2K. At H=32, K=8, N=2 that is 40 against 16, so unmasked grouped-query attention gathers. Joint text-and-image attention over an odd length, cross attention to a 77-token context, and head counts the split does not divide all take the gather. `dew.nn.attention.sequence_parallel_attention` holds the rule.

Measured on 4x RTX 3090 (NV4 pair + PHB pair, cross-socket), one host, bf16, forward and backward of one attention call with 16 query heads, 8 key heads and head width 128, split two ways, in milliseconds. Each pair's columns come from one run, its one-GPU column on the pair's first GPU:

| Causal, tokens | NVLink pair: one GPU | NVLink pair: all-to-all | PCIe pair: one GPU | PCIe pair: all-to-all | PCIe pair: gather |
|---|---|---|---|---|---|
| 8,192 | 18.1 | 12.0 | 16.0 | 18.5 | 37.2 |
| 16,384 | 62.3 | 41.2 | 61.9 | 46.5 | 103.7 |
| 32,768 | 245.9 | 130.4 | 244.3 | 150.1 | 350.1 |
| 65,536 | | 503.0 | | 541.8 | 1302.4 |

With 4 key heads of width 64, Rigel's attention, the gather took 2.4 to 3.9 times as long as the all-to-all on the PCIe pair over the same lengths. Unmasked, the two exchanges came within 17% of each other at 4,096 to 32,768 tokens on the NVLink pair, and the byte count picked the faster one in ten of twelve shapes, missing by at most 6%.

Two other designs were timed against the all-to-all on the same pairs, and Dew ships neither. A zigzag ring on cuDNN, timed with the kernels an exact one runs, passes key and value blocks around the devices while each attends its queries to the block it holds. Two head groups cut the all-to-all in two, so one group's exchange can run beside the other group's kernel. Each time below is relative to the all-to-all's, for the causal call of the table above:

| Tokens | Ring, NVLink pair | Ring, PCIe pair | Two head groups, NVLink pair | Two head groups, PCIe pair |
|---|---|---|---|---|
| 8,192 | 1.46 | 1.79 | 0.95 | 0.84 |
| 16,384 | 1.20 | 1.45 | 0.98 | 0.90 |
| 32,768 | 1.11 | 1.32 | 1.09 | 0.97 |
| 65,536 | 1.04 | 1.17 | 1.00 | 0.98 |

With Rigel's attention the ring took 1.02 to 1.79 times as long and two head groups 0.91 to 1.10 times; four head groups did no better than two. The ring lost at every length. The head groups gained up to 16%, on the PCIe pair at lengths one GPU trains, and came within 3% faster to 9% slower at 32,768 and 65,536 tokens.

Keep the sequence axis on the fastest links. Both exchanges run once per attention layer in the forward and the backward pass, and again when the layer is recomputed. On the 3090s the all-to-all moved 19.5 GB/s per device over the NVLink pair and 6.3 GB/s across the sockets, 64 MiB a device, and NCCL runs the PCIe pair through host memory as it runs the sockets. On the PCIe pair a 32,768-token Qwen3-0.6B-shaped training step split two ways took 7.6 s, 1.25 s of it an all-to-all that no computation overlapped. `build_mesh` puts the `sequence` axis last, on neighbouring devices, and `replicas` keeps it inside a granule.

## Rehearse on one machine

Before you book nodes, run the same launch on one machine. On a machine with several GPUs, one process per GPU with NCCL on its socket transport (no NVLink, PCIe peer access or shared memory between processes) exercises the paths that run between hosts:

```bash
dew launch --processes-per-host 4 --devices-per-process 1 \
    --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 \
    -- python tests/distribution_worker.py --out /tmp/pool.json --mesh '{"fsdp": 2, "replicas": 2}'
```

To catch code that assumes one filesystem, start each process in a directory of its own, with its own `HOME`, `HF_HOME` and compilation cache. A persistent checkpoint directory then has to stay on storage every process shares. Checkpoints refuse a directory the processes do not share (see [resuming training](checkpoints.md)).

Without GPUs, CPU devices stand in. This command starts four processes with two CPU devices each, the layout of four hosts with two accelerators, grouped into two replicas of two hosts:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 4 \
    --env XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    -- python tests/distribution_worker.py --out /tmp/pool.json \
       --mesh '{"fsdp": 2, "replicas": 2}'
```

`/tmp/pool.json` records the losses and, for every fsdp group, the processes its devices sit on. Hybrid sharding shows `"fsdp_groups": [[0, 1], [0, 1], [2, 3], [2, 3]]`: each fsdp group spans the two hosts of its replica. Without `replicas`, `jax.make_mesh` gives `[[0], [1], [2], [3]]`. Run the worker again as one process with `--env XLA_FLAGS=--xla_force_host_platform_device_count=8` and `--mesh '{"fsdp": 8}'`. The two runs print the same losses to within 1e-6.

`tests/test_distribution.py` runs this comparison for hybrid sharding and for a split sequence across processes. It also checks that a failing process stops the pool, including a rank whose loader fails in the middle of `fit` and one that stalls without failing, that a pool refuses a checkpoint directory its processes do not share, that two pools started together take a port each, and that a pool's second run loads on every process the step its first run compiled. On a GPU run, the tests marked `mesh(devices=2)` take one GPU per process.

`tools/layout_parity.py` runs every layout of every model family against one device, in one process or under `dew launch`. It compares the loss and each gradient leaf against the reference's own deviation when the batch's sums are reordered.

## What has and has not been run

These checks passed:

- on one host with four RTX 3090 GPUs (PCIe 3.0), `dew launch` pools of four processes with one GPU each, NCCL on its socket transport and a working directory, `HOME`, `HF_HOME` and compilation cache per process: `tools/layout_parity.py` over every layout of the dense, MoE (8 and 128 experts), Mamba-2 hybrid and DiT models, each layout within its floor or refused with its reason, and the pool tests of `tests/test_distribution.py`;
- real process pools launched by `dew launch`: four processes of two CPU devices for `MeshSpec(fsdp=2, replicas=2)`, and two of four for `MeshSpec(fsdp=2, sequence=2, replicas=2)`, each matching one process of plain fsdp over eight devices;
- `hybrid_devices` against stand-in devices for topologies this machine lacks: hosts as granules, and two slices of two hosts each;
- both exchanges against whole-sequence attention, forward and backward, on the simulated eight-device mesh, with the compiled trainer step showing which exchange each mesh ran, including heads split over tensor and sequence at once, four sequence shards over two key heads, packed masks, biases, sinks and the pipeline's stage axis.
- on one TPU v6e chip, `tools/qualify_sequence_exchange.py` ran the all-to-all exchange's `shard_map` around the Mosaic splash kernel, forward and backward, in fp32 and bf16. Its output and gradients equal whole-sequence splash exactly. One chip has one sequence shard, so this proves the lowering, not the exchange across chips.

Dew has not been run on two physical nodes. Throughput and memory numbers for hybrid sharding and the sequence exchanges need at least one host with eight accelerators (a TPU v5e-8 or v6e-8 slice, or eight GPUs on NVLink) for the sequence axis, and two such hosts, or a two-slice TPU run, for `replicas`. Network failures, NCCL or DCN collective tuning, shared checkpoint storage across nodes and cluster preemption are untested. A first multi-node run should compare its losses with a single-node run of the same global batch for a few hundred steps before it trains for real.
