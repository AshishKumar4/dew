# Training on several nodes

> An AI assistant maintains this document. It is presented as-is.

This page shows how to start one training script on several machines, how to lay the mesh out so the slow network between them carries as little as possible, and how to rehearse all of it on one machine first. Read [distributed training](../concepts/distributed.md) before this page. It explains meshes and layouts.

## How a process pool starts

Every node runs the same script. Each copy is one process of a `jax.distributed` pool. The script joins the pool when it calls `dew.training.runtime.prepare_process()`, which every built-in recipe does on its first line. Call it yourself before you create any array if you write your own script.

`prepare_process` calls `jax.distributed.initialize`, which needs three facts: the address of process 0's coordinator, the number of processes and this process's rank. Where they come from depends on the cluster:

| Where you run | Who provides the three facts |
|---|---|
| Cloud TPU VM or pod | The TPU metadata server. JAX reads it. |
| Slurm (`srun`) | Slurm's `SLURM_*` variables. JAX reads them. |
| Open MPI (`mpirun`) | The `OMPI_*` variables. JAX reads them. |
| Plain machines over ssh | `dew launch`, through `JAX_COORDINATOR_ADDRESS`, `DEW_PROCESS_COUNT` and `DEW_PROCESS_ID`. |

`dew launch` also sets `JAX_LOCAL_DEVICE_IDS` when a host runs more than one process, so each process takes its own accelerators.

## Launch on plain machines

List the hosts, process 0's first, then the command after `--`:

```bash
dew launch --hosts node0 node1 -- python recipes/lm/train.py --trainer.multi-host True
```

The launcher starts the command on each host over `ssh -o BatchMode=yes`, so passwordless keys must already work. It runs a host named `localhost` directly. Each process starts in the current directory, which must exist at the same path on every host, or in the directory `--cwd` names. The coordinator listens on port 43217 of the first host; change it with `--port`, and set `--coordinator` when the other hosts reach that host by another name.

Every output line carries its rank, such as `[1] Joined the JAX process pool: process 1 of 2`. When one process exits with an error, the launcher stops the others and exits with that code. Without this, the survivors would wait in a collective for a peer that is gone.

To run one process per GPU instead of one per host:

```bash
dew launch --hosts node0 node1 --processes-per-host 8 --devices-per-process 1 -- python train.py
```

`--env NAME=VALUE` passes a variable to every process, for example `--env XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`. Add `--dry-run` to print the exact commands without running them.

## Launch under Slurm

Inside an allocation, `--slurm` hands the launch to `srun`, and JAX reads the rank from Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
dew launch --slurm -- python recipes/lm/train.py --trainer.multi-host True
```

This runs `srun --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL python ...`. Plain `srun python train.py` works as well. The launcher only adds the failure policy and the same flags as the ssh path.

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

`replicas` counts groups of granules. A granule is a TPU slice on a multislice run and a process everywhere else, which on GPU clusters is usually one node. `build_mesh` then builds the mesh with JAX's `mesh_utils.create_hybrid_device_mesh`, the same call MaxText uses for multislice runs. The data axis spans the groups. The expert, tensor, sequence and stage axes stay inside one granule. When a group holds more than one granule, fsdp is the only axis that crosses between them. `replicas` must divide both the granule count and the data axis, and fsdp must divide over the granules of one group. `build_mesh` raises with the numbers when they do not fit.

With `replicas=1`, `jax.make_mesh` places the devices, which is what a single node or a single TPU slice wants.

Hybrid sharding keeps a full copy of the parameters and optimizer state on every replica group. Plain fsdp across all nodes divides them by the total device count instead. Choose hybrid sharding when one node's memory holds the sharded state, and plain fsdp when it does not.

## Split long sequences

`MeshSpec(sequence=N)` splits the token positions of every sequence over N devices, and attention exchanges data between them. `MeshSpec.sequence_exchange` picks the exchange:

- `'all_to_all'` is the default and follows DeepSpeed Ulysses. Each device trades its slice of the positions for a slice of the attention heads, attends the whole sequence for those heads and trades back. No device ever holds a whole key or value tensor. Causal masks, sliding windows, packed-document masks and the TPU splash kernel work as they do on one device. The query heads must divide by `tensor` times `sequence`. Grouped key and value heads are repeated only as far as the split needs.
- `'all_gather'` keeps the queries split and gathers the whole keys and values on every device. It takes any head count. It costs every device the full key and value tensors of each layer, and it reorders causal queries so every device gets the same work, which needs the sequence length to divide by twice `sequence`.

Keep the sequence axis inside a node. Both exchanges run once per attention layer in the forward and the backward pass. `replicas` keeps it inside a granule for you.

## Rehearse on one machine

Before you book nodes, run the same launch on one machine with CPU devices. This command starts two processes with four CPU devices each, which is the layout of two nodes with four accelerators:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 2 \
    --env XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    -- python tests/distribution_worker.py --out /tmp/pool.json \
       --mesh '{"fsdp": 4, "replicas": 2}'
```

`/tmp/pool.json` records the losses and, for every fsdp group, the processes its devices sit on. Hybrid sharding shows `"fsdp_groups": [[0], [1]]`. Run the worker again as one process with `--env XLA_FLAGS=--xla_force_host_platform_device_count=8` and `--mesh '{"fsdp": 8}'`. The two runs print the same losses to within 1e-6.

`tests/test_distribution.py` runs this comparison for hybrid sharding and for both sequence exchanges across two processes, and checks that a failing process stops the pool.

## What has and has not been run

These checks passed:

- two real processes of four CPU devices each, launched by `dew launch`, for `MeshSpec(fsdp=4, replicas=2)` and `MeshSpec(fsdp=2, sequence=2, replicas=2)` with both exchanges, each matching one process of plain fsdp over eight devices;
- both exchanges against whole-sequence attention, forward and backward, on the simulated eight-device mesh, including heads split over tensor and sequence at once, four sequence shards over two key heads, packed masks, biases, sinks and the pipeline's stage axis.

Dew has not been run on two physical nodes. Network failures, NCCL or DCN collective tuning, shared checkpoint storage across nodes and cluster preemption are untested. A first multi-node run should compare its losses with a single-node run of the same global batch for a few hundred steps before it trains for real.
