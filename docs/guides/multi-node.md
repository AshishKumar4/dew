# Multi-node training

A multi-node run executes the same training script on every machine, with one process per GPU or TPU worker. The processes join a `jax.distributed` pool, and `Trainer` uses their devices together.

`dew launch` starts the pool locally, over ssh, under Slurm or Open MPI, or on Cloud TPU VMs. `MeshSpec(replicas=N)` limits communication over the slower network between nodes to one gradient all-reduce per step. [Distributed training](../concepts/distributed.md) explains meshes and layouts.

## Process pools

Call `dew.training.runtime.prepare_process()` before creating any array in your script. It joins the process to the `jax.distributed` pool. Every built-in recipe calls it on its first line.

`prepare_process` calls `jax.distributed.initialize`, which needs process 0's coordinator address, the process count and the local process's rank. The launcher or cluster supplies these values:

| Where you run `dew launch -- python train.py` | What it does | Who provides the three facts |
|---|---|---|
| A machine with GPUs | One process per GPU, each holding its own GPU | `dew launch`, through `JAX_COORDINATOR_ADDRESS`, `DEW_PROCESS_COUNT` and `DEW_PROCESS_ID` |
| A machine with one GPU, or none | One process | `dew launch` |
| Plain machines, with `--hosts` or `--hostfile` | The same on every host, over ssh | `dew launch` |
| A Slurm allocation, such as an sbatch script | `srun`, one task per GPU | Slurm's `SLURM_*` variables, read by JAX |
| A Slurm step of several tasks, or an `mpirun` rank, `mpirun` inside a one-task allocation included | The program itself, in place | Slurm's or Open MPI's variables, read by JAX, Open MPI's first |
| A Slurm step of one task | One process per GPU of the step | `dew launch` |
| A Cloud TPU VM worker | The program itself, in place | The TPU metadata server, read by JAX |
| Anywhere, with `--tpu NAME` | One process on every worker of the TPU, through gcloud | The TPU metadata server, read by JAX |

Without `--hosts`, `--hostfile` or `--tpu`, the launcher uses JAX's cluster detection (`jax._src.clusters`). It recognizes the same Slurm, Open MPI, Cloud TPU, GKE and Kubernetes environments as `jax.distributed.initialize`. Explicit host or TPU flags override detection. Add `--dry-run` to preview the commands.

## dew launch

On one machine, `dew launch` runs the program after `--` once per GPU:

```bash
dew launch -- python recipes/lm/train.py
```

On a four-GPU machine, this starts four processes with one GPU each, selected by `JAX_LOCAL_DEVICE_IDS`. `dew launch` sets the pool size and each rank in the environment. `prepare_process` joins that pool without an extra program flag. With `--trainer.multi-host True`, the program refuses to start if no pool is configured. With `False`, it never joins a pool.

The launcher prints a `pool:` line with the process count, devices per process and coordinator address. It then prints `[r] rank r of 4 on localhost, GPU r, pid ...` for each rank. Each process confirms its join with `Joined the JAX process pool: process r of 4`.

Every output line is prefixed with its rank. `--processes-per-host N` splits the host's GPUs evenly between N processes. With `--processes-per-host 1`, one process uses every GPU. `--devices-per-process N` assigns N GPUs to each process.

Before starting any rank, the launcher checks the requested GPU count against the first host's available count. If the request exceeds it, the launcher refuses and reports both counts. If nvidia-smi is missing or fails, the launcher warns on stderr and starts without this check. `JAX_PLATFORMS=cpu`, set in the environment or with `--env`, skips GPU counting. The count covers NVIDIA GPUs, restricted to `CUDA_VISIBLE_DEVICES` when set. For other GPUs, set `--processes-per-host` yourself.

`--env NAME=VALUE` passes a variable to every process. Use a valid shell variable name, for example `--env XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`. `--cwd DIR` sets each process's starting directory. `--dry-run` prints commands without running them. On a CPU-only machine:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 2 --dry-run -- python train.py
```

```text
pool: 2 processes on localhost, each with every local device, coordinator localhost:51403
JAX_COORDINATOR_ADDRESS=localhost:51403 DEW_PROCESS_COUNT=2 DEW_PROCESS_ID=0 python train.py
JAX_COORDINATOR_ADDRESS=localhost:51403 DEW_PROCESS_COUNT=2 DEW_PROCESS_ID=1 python train.py
```

When a process exits with an error, the launcher stops its peers so they do not wait in a collective for it. The launcher exits with the failed process's code. It names the rank (`rank 2 on localhost exited 1; stopping the other 3`). After stopping the others, it prints `last lines of rank 2:` and that rank's last 20 lines. Look at the bottom of the output for the cause.

For a rank killed by a signal, the message names the signal, for example `was killed by SIGKILL`. The launch exits with 128 plus the signal number, as a shell would report it. Ctrl-C, a scheduler's SIGTERM or a closed terminal stops the whole pool. The launcher sends SIGTERM to every rank.

A pooled JAX process treats SIGTERM as a preemption notice. `Trainer.fit` checkpoints at a step agreed by every rank and exits with 143 ([Checkpoints](checkpoints.md#preemption)). Other programs keep running. If the launcher received a signal, it allows ranks 300 seconds to checkpoint before sending SIGKILL. A second signal kills them immediately. After a rank failure, it allows only 10 seconds because peers may already be waiting for that rank in a collective.

Rank 0 is killed last because it runs the coordination service. A surviving rank would otherwise abort with an XLA `Check failure`, obscuring the original failure.

## Plain machines

List process 0's host first, followed by the other hosts. Put the command after `--`:

```bash
dew launch --hosts node0,node1 -- /opt/dew/.venv/bin/python recipes/lm/train.py
```

`--hostfile FILE` reads one host per line from a file. It uses the first word of each line, accepting an MPI hostfile with `slots=8` unchanged. The launcher uses the first host's GPU count to set the process count on every host, so the hosts should match.

The launcher uses `ssh -o BatchMode=yes`, so passwordless keys must already work. The remote shell reads no login profile and does not activate your virtualenv. Give the interpreter's absolute path, as above. A host named `localhost` runs directly.

Each process starts in the current directory, or the directory specified by `--cwd`. That path must exist on every host. The coordinator chooses a free port on the first host at launch, so concurrent local pools use separate ports. For a firewall requiring a fixed port, set `--port`. If peers reach the coordinator host by a different name, set `--coordinator`.

## Slurm

Run `dew launch` inside the allocation. It starts `srun`, and JAX reads the rank from Slurm:

```bash
sbatch --nodes=2 --gpus-per-node=8 --wrap "dew launch -- /opt/dew/.venv/bin/python recipes/lm/train.py"
```

The launcher runs `srun --kill-on-bad-exit=1 --export=ALL --label --ntasks-per-node=8 ...`, including `--env` variables in srun's environment. Each output line is labeled with its task number. When a task fails, srun names it and stops the others.

JAX gives each Slurm task the GPU at its `SLURM_LOCALID`. The launcher chooses tasks per node from `--processes-per-host` if supplied, then the allocation's `--ntasks-per-node`, then its `--gpus-per-node`. Otherwise it counts GPUs visible on the node running `dew launch`. If the login node has no GPUs, request `--gpus-per-node` or pass `--processes-per-host`; `--gpus` alone is insufficient there.

Every task needs a CPU in the allocation. Request these with `--ntasks-per-node` or `--cpus-per-gpu` in sbatch. With neither, srun refuses to start more tasks than the allocation has CPUs. Each task sees a single GPU, so the launcher refuses `--devices-per-process` under Slurm. It also refuses allocations or `--processes-per-host` settings with fewer tasks than GPUs per node, which would silently leave GPUs idle. The error names `--ntasks-per-node`.

Inside an existing multi-task srun step, `dew launch` runs the program in place. `prepare_process` refuses a step with fewer tasks per node than the GPUs visible to its task. Inside a one-task step, such as a GPU wrapper script, the launcher starts one process per GPU as on a plain machine. Each process uses the launcher's placement. Without `dew launch`, the program runs as one process unless it requests a pool with `multi_host=True`.

## Open MPI

`mpirun` starts every rank itself, and JAX reads the `OMPI_*` variables. `dew launch` inside a rank runs the program in place, so both of these work:

```bash
mpirun -np 8 --hostfile hosts python train.py
mpirun -np 8 --hostfile hosts dew launch -- python train.py
```

Each rank takes the GPU at its local rank, as under Slurm.

## Cloud TPU VMs

`--tpu NAME` runs the program on every worker of a TPU VM or pod slice, from any machine where gcloud reaches the TPU:

```bash
dew launch --tpu dew-16 --cwd dew -- python recipes/lm/train.py
```

The launcher finds the zone as [`dew tpu`](../tpu.md) does, or uses `--zone`. It starts one `gcloud compute tpus tpu-vm ssh --worker=N` per worker. Each worker sources the environment written by `dew tpu setup`, so `python` uses that virtualenv. A relative `--cwd` starts under the worker's home directory, where `dew tpu sync` puts the working tree. JAX reads ranks from the TPU metadata server. Stopping the pool closes the SSH connections and hangs up the worker programs.

Name several TPUs to run one multislice pool over them:

```bash
dew launch --tpu slice-a,slice-b -- python train.py
```

The launcher sets `MEGASCALE_NUM_SLICES` and each worker's `MEGASCALE_SLICE_ID`. `MEGASCALE_COORDINATOR_ADDRESS` points to the first worker of the first slice, with `MEGASCALE_PORT=8081`. These match Ray's TPU support. JAX numbers processes slice by slice. `MeshSpec(replicas=...)` uses the devices' `slice_index` to identify slice boundaries.

On a TPU VM worker, `dew launch -- python train.py` runs only on that worker. Every worker must run the program, but workers cannot reach each other over ssh by default. On worker 0 of a pod, the launcher warns about this. To start all workers, run `dew launch --tpu NAME` from your machine or a worker with gcloud access to the TPU.

## Failures and stalls

A failed process prints its error, publishes it to the coordination service and exits immediately. It skips `jax.distributed`'s shutdown barrier, which can wait up to 300 seconds. This applies from the moment it joins the pool, including GPU startup failures such as an out-of-memory error. Peers therefore do not wait minutes for its devices.

After a failure between steps, such as a data-loader error, peers may already be in the next step's collectives. GPU backends do not time those out. The failing rank publishes its error before trying to agree with peers. A watch thread in every process exits if a published failure has not been observed at an agreement within 60 seconds. The whole pool then stops within about a minute under `dew launch`, `srun` or a scheduler.

A rank can stall on a read or a compile without reporting a failure. Its peers then remain alive, waiting in a collective or communicator setup. GPU pools use XLA's execution watchdog to bound each device execution. When the CUDA plugin is present and no timeout is set, `prepare_process` adds `--xla_gpu_execution_terminate_timeout=30m`. A step or sampling loop that exceeds it ends the process, and the launcher, `srun` or scheduler stops the rest. For known step times, set a tighter timeout in `XLA_FLAGS` or the recipe's `xla_flags`.

The watchdog covers only device execution. Before agreement collectives, such as a checkpoint save or the end of `fit`, ranks wait on the host. This lets process 0 finish host work such as uploading the final checkpoint to Weights & Biases. Waiting in a device collective during that upload could trigger the watchdog.

The host wait has its own limit: `AGREEMENT_PATIENCE_SECONDS` in `dew.artifacts`, set to one day. A rank stalled in host work can hold its peers for that long. Lower the limit when your host phases are short.

Pools retain JAX's persistent compilation cache. In jax 0.11.2, an executable's cache key uses the compiling process's accelerator topology fingerprint, including GPU NVLink links. Only process 0 writes cache entries. On one four-GPU host with an NVLink pair and a PCIe pair, ranks 0 and 1 loaded a cached step. Ranks 2 and 3 compiled it and waited indefinitely for their peers' shares of sharded autotuning.

Dew pins jax to 0.11.2 with the fix reported as jax-ml/jax#40940. For a computation spanning processes, the fix hashes all processes' fingerprints. Every rank can then load the same entry, provided they compile for matching accelerators. Platform, runtime, CUDA driver, cuDNN, cuBLAS, device kinds, compute capability and core count must match. If any differ, the pool compiles these computations without the cache, and JAX logs the differing processes.

If a GPU pool's jax lacks the fix, `prepare_process` disables the persistent cache before the first compile. This includes an image's own jax or a version installed with `--no-deps` around the pin.

## Mesh layout across nodes

Within a node, devices communicate over NVLink or the TPU interconnect. Between nodes, the network is often ten times slower. Fully sharded data parallelism (`fsdp`) gathers parameters and reduce-scatters gradients for every layer. If the fsdp axis crosses nodes, each layer uses that slower link.

Hybrid sharding keeps fsdp within a node and replicates state across nodes. Only one gradient all-reduce per step crosses the network. Request it with `MeshSpec(replicas=N)`. For two nodes of eight GPUs:

<!-- not run: needs a pool of two hosts with eight GPUs each -->
```python
import jax
import optax
from dew import Trainer
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec
from dew.training.runtime import prepare_process

prepare_process()
model = CausalTransformer(vocab_size=256, emb_features=512,
                          num_layers=8, num_heads=8, max_seq_len=1024)
# fsdp over the eight GPUs of each node, and the data axis of 2 across the nodes.
mesh = MeshSpec(fsdp=8, replicas=2)
trainer = Trainer(LMObjective(model, seq_len=1024), optax.adamw(3e-4),
                  key=jax.random.key(0), mesh=mesh)
```

`replicas` counts groups of granules. The devices' `slice_index` identifies each granule: a TPU slice in a multislice run, or a host or NVLink domain on GPU hosts. XLA numbers GPU slices by host boot or NVLink fabric, regardless of the process count per host. When all devices share one slice, as in CPU pools or on hosts within one TPU slice, each process is a granule.

`MeshSpec.build` uses JAX's `mesh_utils.create_hybrid_device_mesh`, also used by MaxText for multislice runs. The data axis spans replica groups. Expert, tensor, sequence and stage axes stay within a granule. If a group has several granules, only fsdp crosses between them. `replicas` must divide both the granule count and the data axis. FSDP must divide over the granules within a group. If the counts do not fit, `MeshSpec.build` raises and reports them.

When GPU device IDs are ordered by host, `jax.make_mesh` already places the data axis outermost. With one granule per replica, setting `replicas` confirms that layout. It changes placement when replicas span several granules, slices differ from device order, or the run uses multiple TPU slices.

For a single node or TPU slice, use `replicas=1`; `jax.make_mesh` places the devices. To choose device order yourself, pass a list to `spec.build(devices)`. It fills the mesh in that order, row-major over `MESH_AXES`. The last axes get neighbouring entries. If NVLink pairs are GPUs 0-1 and 2-3, the order `[0, 2, 1, 3]` puts a two-way tensor axis across the pairs. `jax.make_mesh` would sort GPU devices by ID and discard your order.

Each hybrid-sharding replica group holds a full copy of parameters and optimizer state, sharded within the group. Plain fsdp splits that state over the total device count. Use hybrid sharding when one node can hold the sharded state; otherwise use plain fsdp.

## Sequence parallelism

`MeshSpec(sequence=N)` splits each sequence's positions over N devices. Attention exchanges data using one of two paths, both equivalent to whole-sequence attention on one device.

- The all-to-all follows DeepSpeed Ulysses. Each device exchanges its slice of positions for a slice of attention heads, attends the whole sequence, and exchanges back. No device holds a whole key or value tensor. Causal masks, sliding windows, packed-document masks and the TPU splash kernel work as on one device. Grouped key and value heads repeat only as much as the split requires.
- The gather keeps queries split and gathers whole keys and values on every device. It supports any head count and key length. For causal or masked calls, it reorders queries to balance work across devices. The sequence length must then be divisible by twice `sequence`.

All-to-all requires the query head count to be divisible by `tensor` times `sequence`. Query and key lengths must also be divisible by `sequence`. When both paths qualify, causal, windowed or masked calls use all-to-all. Its kernel receives whole sequences and the causal flag, so cuDNN and splash skip masked blocks. The gather supplies a mask for reordered query rows, which cuDNN computes as a dense bias over every logit.

An unmasked call does the same compute work on either path and chooses the exchange sending fewer bytes. Per device, count bytes in units of sequence length times head width times (N-1)/N. With H query heads and K key heads per tensor shard, all-to-all sends 2(H + K')/N and gather sends 2K. K' is K repeated as much as the split requires. At H=32, K=8, N=2, the counts are 40 and 16, so unmasked grouped-query attention gathers.

Joint text-and-image attention over an odd length, cross attention to a 77-token context, and indivisible head counts all use gather. `dew.nn.attention.sequence_parallel_attention` implements the selection rule.

I measured forward and backward for one attention call on 4x RTX 3090 (NV4 pair + PHB pair, cross-socket), on one host. The call used bf16, 16 query heads, 8 key heads and head width 128, split two ways. Times below are in milliseconds. Each pair's columns come from one run, with its one-GPU baseline on the pair's first GPU:

| Causal, tokens | NVLink pair: one GPU | NVLink pair: all-to-all | PCIe pair: one GPU | PCIe pair: all-to-all | PCIe pair: gather |
|---|---|---|---|---|---|
| 8,192 | 18.1 | 12.0 | 16.0 | 18.5 | 37.2 |
| 16,384 | 62.3 | 41.2 | 61.9 | 46.5 | 103.7 |
| 32,768 | 245.9 | 130.4 | 244.3 | 150.1 | 350.1 |
| 65,536 | | 503.0 | | 541.8 | 1302.4 |

For Rigel's attention (4 key heads of width 64), gather took 2.4 to 3.9 times as long as all-to-all on the PCIe pair over the same lengths. For unmasked calls at 4,096 to 32,768 tokens on the NVLink pair, the two paths were within 17%. The byte count chose the faster path in ten of twelve shapes. The other choices cost at most 6%.

I also timed two other designs on the same pairs. Neither is part of Dew. The zigzag ring passes key and value blocks between devices while each attends its queries to the local block. Its cuDNN timings use the kernels required for an exact implementation. Splitting heads into two groups lets one group's exchange overlap the other group's kernel. The table gives each design's time relative to all-to-all for the causal call above:

| Tokens | Ring, NVLink pair | Ring, PCIe pair | Two head groups, NVLink pair | Two head groups, PCIe pair |
|---|---|---|---|---|
| 8,192 | 1.46 | 1.79 | 0.95 | 0.84 |
| 16,384 | 1.20 | 1.45 | 0.98 | 0.90 |
| 32,768 | 1.11 | 1.32 | 1.09 | 0.97 |
| 65,536 | 1.04 | 1.17 | 1.00 | 0.98 |

For Rigel's attention, the ring took 1.02 to 1.79 times as long as all-to-all. Two head groups took 0.91 to 1.10 times as long; four did no better than two. The ring was slower at every length. Head grouping improved time by up to 16% on the PCIe pair at lengths trainable on one GPU. At 32,768 and 65,536 tokens, it ranged from 3% faster to 9% slower.

Keep the sequence axis on the fastest links. Both exchanges run once per attention layer in the forward and backward passes, and again on recomputation. On the 3090s, a 64 MiB all-to-all per device moved 19.5 GB/s over the NVLink pair and 6.3 GB/s across sockets. NCCL routes the PCIe pair through host memory too.

On the PCIe pair, a 32,768-token Qwen3-0.6B-shaped training step split two ways took 7.6 s. All-to-all took 1.25 s with no compute overlap. `MeshSpec.build` puts `sequence` last, on neighbouring devices; `replicas` keeps it within a granule.

## Single-machine rehearsal

Before booking nodes, rehearse on one machine. On four GPUs, launch one process per GPU and force NCCL's socket transport. Disabling NVLink, PCIe peer access and shared memory between processes exercises the paths used between hosts:

```bash
dew launch --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 \
    -- python tests/distribution_worker.py --out /tmp/pool.json --mesh '{"fsdp": 2, "replicas": 2}'
```

To catch assumptions about shared filesystems, give each process its own working directory, `HOME`, `HF_HOME` and compilation cache. Keep persistent checkpoints on storage shared by every process. `Checkpoints` refuses an unshared directory ([Checkpoints](checkpoints.md)).

Without GPUs, use simulated CPU devices. This command starts four processes with two devices each. It models four hosts with two accelerators each, grouped into two replicas of two hosts:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 4 \
    --env XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    -- python tests/distribution_worker.py --out /tmp/pool.json \
       --mesh '{"fsdp": 2, "replicas": 2}'
```

`/tmp/pool.json` records the mesh, losses and the processes holding each fsdp group's devices:

```text
{"processes": 4, "devices": 8, "mesh": {"data": 4, "expert": 1, "fsdp": 2, "tensor": 1, "sequence": 1, "stage": 1}, "fsdp_groups": [[0, 1], [0, 1], [2, 3], [2, 3]], "partition": {"count": 4, "readers": 1}, "placed_whole": false, "losses": [4.891427993774414, 3.6097896099090576, 2.8838775157928467], "loss_steps": [1, 2, 3]}
```

Each fsdp group spans its replica's two hosts. With `--mesh '{"fsdp": 2}'` and no `replicas`, `jax.make_mesh` gives `"fsdp_groups": [[0], [1], [2], [3]]`. The one-process reference uses `XLA_FLAGS=--xla_force_host_platform_device_count=8` and `--mesh '{"fsdp": 8}'`. Its losses are 4.891427993774414, 3.609790086746216 and 2.8838775157928467, within 1e-6 of the pool's.

`tests/test_distribution.py` compares hybrid sharding and split sequences across processes with that reference. It checks failed-process shutdown, including loader failures during `fit`, and that stalls without errors stop the pool. It also checks rejection of unshared checkpoint directories, separate ports for concurrent pools, and cache reuse on every process in a second run. GPU tests marked `mesh(devices=2)` use one GPU per process.

`tools/layout_parity.py` compares every layout of every model family with one device, in a single process or under `dew launch`. Its loss and gradient tolerances come from the reference's own deviation when batch sums are reordered. It also compares per-device FLOPs with an even split of the one-device work.

Unsupported layouts, such as a stage axis over a DiT, raise `dew.nn.sharding.LayoutRefused` with a reason and a supported layout. The tool lists these separately. It exits nonzero only for a mismatch, repeated work or another error. To prepare references once, run `--prepare` on one device with a `--references` directory. Subsequent layout runs use those references without recomputing them on every device.

## Tested configurations

These checks passed:

- One host with four RTX 3090 GPUs (PCIe 3.0), using `dew launch` with one GPU per process and NCCL's socket transport. Each process had its own working directory, `HOME`, `HF_HOME` and compilation cache. `tools/layout_parity.py` covered every layout of dense, MoE (8 and 128 experts), Mamba-2 hybrid and DiT models. Each met its numerical floor or was refused with a reason. The pool tests in `tests/test_distribution.py` passed too.
- CPU pools launched by `dew launch`: four processes with two devices each for `MeshSpec(fsdp=2, replicas=2)`, and two with four devices each for `MeshSpec(fsdp=2, sequence=2, replicas=2)`. Both matched one process using plain fsdp over eight devices.
- `hybrid_devices` with stand-in devices for unavailable topologies: hosts as granules, and two slices of two hosts each.
- Both exchanges compared with whole-sequence attention, forward and backward, on eight simulated devices. Compiled trainer steps showed which exchange each mesh used. Coverage included heads split over tensor and sequence together, four sequence shards over two key heads, packed masks, biases, sinks and the pipeline stage axis.
- One TPU v6e chip running `tools/qualify_sequence_exchange.py`. The all-to-all exchange's `shard_map` wrapped the Mosaic splash kernel, forward and backward, in fp32 and bf16. Output and gradients matched whole-sequence splash exactly. With one sequence shard, this checks lowering only. Exchange across chips remains untested.

Dew has not run on two physical nodes. Measuring sequence-exchange throughput and memory needs at least one host with eight accelerators: a TPU v5e-8 or v6e-8 slice, or eight GPUs on NVLink. Measuring hybrid sharding with `replicas` needs two such hosts or a two-slice TPU run. Network failures, NCCL or DCN collective tuning, shared checkpoint storage across nodes and cluster preemption remain untested.

Before a full multi-node run, compare a few hundred steps with a single-node run using the same global batch.
