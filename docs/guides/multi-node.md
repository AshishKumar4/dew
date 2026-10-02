# Multi-node training

A multi-node run starts the same training script on every machine, one process per GPU (or per TPU worker), and joins the processes into one `jax.distributed` pool that `Trainer` then treats as one set of devices. `dew launch` starts the pool on one machine, over ssh on plain hosts, under Slurm or Open MPI, or on Cloud TPU VMs. `MeshSpec(replicas=N)` lays the mesh out so the slow network between nodes carries one gradient all-reduce per step. [Distributed training](../concepts/distributed.md) explains meshes and layouts.

## Process pools

Every node runs the same script, and each copy is one process of a `jax.distributed` pool. The script joins the pool when it calls `dew.training.runtime.prepare_process()`, which every built-in recipe does on its first line; a script of your own calls it before creating any array.

`prepare_process` calls `jax.distributed.initialize`, which needs three facts: the address of process 0's coordinator, the number of processes and this process's rank. `dew launch` starts the processes wherever you run it, and each cluster supplies the facts in its own way:

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

Without `--hosts`, `--hostfile` or `--tpu`, `dew launch` asks JAX's own cluster detection (`jax._src.clusters`) where it runs, so it recognizes the same Slurm, Open MPI, Cloud TPU, GKE and Kubernetes environments that `jax.distributed.initialize` does. A flag that names hosts or TPUs wins over what it detects. Add `--dry-run` to see what it would run.

## dew launch

On one machine, `dew launch` runs the program after `--` once per GPU:

```bash
dew launch -- python recipes/lm/train.py
```

On a four-GPU machine this starts four processes, each holding one GPU through `JAX_LOCAL_DEVICE_IDS`. The program needs no flag to join: `dew launch` leaves the pool's size and each rank in the environment, and `prepare_process` joins the pool it finds there. `--trainer.multi-host True` makes a run refuse to start alone, where no pool is configured; `False` never joins one. The launcher prints one `pool:` line with the process count, the devices per process and the coordinator address, then one `[r] rank r of 4 on localhost, GPU r, pid ...` line per rank, and each process prints `Joined the JAX process pool: process r of 4` when it joins.

Every output line carries its rank. `--processes-per-host N` runs N processes a host and splits the GPUs evenly between them, so `--processes-per-host 1` runs one process that holds every GPU. `--devices-per-process N` gives each process N GPUs. A pool that would need more GPUs than the first host shows is refused before any rank starts, and the message names both counts. If the launcher cannot count a host's GPUs, because nvidia-smi is missing there or fails, it says so on stderr and starts the pool without the check. `JAX_PLATFORMS=cpu`, in the environment or through `--env`, stops the launcher from counting GPUs. The launcher counts NVIDIA GPUs, those in `CUDA_VISIBLE_DEVICES` when it is set; on other GPUs, set `--processes-per-host`.

`--env NAME=VALUE` passes a variable to every process. The name must be a shell variable name, for example `--env XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`. `--cwd DIR` names the directory each process starts in. `--dry-run` prints the commands without running them; on a CPU-only machine:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 2 --dry-run -- python train.py
```

```text
pool: 2 processes on localhost, each with every local device, coordinator localhost:51403
JAX_COORDINATOR_ADDRESS=localhost:51403 DEW_PROCESS_COUNT=2 DEW_PROCESS_ID=0 python train.py
JAX_COORDINATOR_ADDRESS=localhost:51403 DEW_PROCESS_COUNT=2 DEW_PROCESS_ID=1 python train.py
```

When one process exits with an error, the launcher stops the others and exits with that code; without this, the survivors would wait in a collective for a peer that is gone. It names the failed rank (`rank 2 on localhost exited 1; stopping the other 3`) and, after the others have stopped, prints `last lines of rank 2:` followed by that rank's last 20 lines, so the cause is at the bottom of the output.

A rank killed by a signal reads as `was killed by SIGKILL`, and the launch exits 128 plus the signal, as a shell reports it. Ctrl-C, a scheduler's SIGTERM and a closed terminal stop the whole pool the same way. The launcher sends each rank SIGTERM. A JAX process of a pool takes SIGTERM as a preemption notice: `Trainer.fit` checkpoints at the step every rank agrees on and exits with 143 ([Checkpoints](checkpoints.md#preemption)), and any other program keeps running. When the launcher was itself signalled it gives the ranks 300 seconds for that before SIGKILL, and a second signal kills them at once; when a rank failed, the others wait in a collective for it, so they get 10 seconds. Rank 0 is killed last: its process holds the pool's coordination service, and a rank that outlived it would abort with an XLA `Check failure` that reads as a crash of its own.

## Plain machines

List the hosts, process 0's first, then the command after `--`:

```bash
dew launch --hosts node0,node1 -- /opt/dew/.venv/bin/python recipes/lm/train.py
```

`--hostfile FILE` reads the hosts from a file instead, one per line. It takes the first word of each line, so an MPI hostfile with `slots=8` works as it is. The launcher counts the GPUs on the first host and runs as many processes on every host, so the hosts should match.

The launcher starts the command on each host over `ssh -o BatchMode=yes`, so passwordless keys must already work. The remote side runs in the user's shell without a login profile, so a virtualenv activated there is not active: give the interpreter as an absolute path, as above. It runs a host named `localhost` directly. Each process starts in the current directory, which must exist at the same path on every host, or in the directory `--cwd` names. The coordinator listens on a port that is free on the first host when the launch starts, so pools started together on one machine never share one. Name a port with `--port` when a firewall wants a fixed one, and set `--coordinator` when the other hosts reach that host by another name.

## Slurm

Run `dew launch` inside the allocation. It starts `srun`, and JAX reads the rank from Slurm:

```bash
sbatch --nodes=2 --gpus-per-node=8 --wrap "dew launch -- /opt/dew/.venv/bin/python recipes/lm/train.py"
```

This runs `srun --kill-on-bad-exit=1 --export=ALL --label --ntasks-per-node=8 ...`, with the `--env` variables in srun's environment. Each line carries its task number, and when a task fails srun names it and stops the others. JAX gives each Slurm task the one GPU at its `SLURM_LOCALID`, so a node runs one task per GPU: `--processes-per-host` when you give it, else the allocation's own `--ntasks-per-node`, else its `--gpus-per-node`, else the GPUs the node running `dew launch` sees. On a cluster whose login node has no GPUs, ask for `--gpus-per-node` rather than `--gpus`, or pass `--processes-per-host`. Every task needs a CPU of its own in the allocation, which `--ntasks-per-node` or `--cpus-per-gpu` in the sbatch request provides; with neither, srun refuses to start more tasks than the allocation has CPUs. One task per node would see a single GPU, so the launcher refuses `--devices-per-process` under Slurm. Fewer tasks a node than GPUs a node would leave the rest idle with nothing reporting it, so the launcher refuses such an allocation, or such a `--processes-per-host`, and names `--ntasks-per-node`. Inside a step of several tasks that `srun` already started, `dew launch` runs the program in place, and `prepare_process` refuses the same way a step with fewer tasks on a node than the GPUs its task sees. Inside a step of one task, such as a GPU wrapper script that runs its command under `srun`, it starts one process per GPU of the step, as on a plain machine, and each process takes its placement from the launcher, not from Slurm's variables. A program that such a step runs without `dew launch` runs as one process, unless it asks for a pool with `multi_host=True`.

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

The launcher finds the TPU's zone the way [`dew tpu`](../tpu.md) does, or takes `--zone`, and starts one `gcloud compute tpus tpu-vm ssh --worker=N` per worker. Each worker sources the environment `dew tpu setup` wrote, so `python` is the setup's virtualenv. A relative `--cwd` is under the worker's home directory, where `dew tpu sync` puts the working tree. JAX reads each worker's rank from the TPU metadata server. Closing the connections, which stopping the pool does, hangs up the programs on the workers.

Name several TPUs to run one multislice pool over them:

```bash
dew launch --tpu slice-a,slice-b -- python train.py
```

Each worker then gets `MEGASCALE_NUM_SLICES`, its `MEGASCALE_SLICE_ID`, and the first worker of the first slice as `MEGASCALE_COORDINATOR_ADDRESS`, with `MEGASCALE_PORT=8081`. These are the values Ray's TPU support sets. JAX numbers the processes slice by slice, and the devices' `slice_index` tells `MeshSpec(replicas=...)` where the slices meet.

On a TPU VM worker, `dew launch -- python train.py` runs the program on that worker only, because every worker has to run it and the workers cannot reach each other over ssh by default. On worker 0 of a pod it says so. Run `dew launch --tpu NAME` from your machine, or from a worker whose gcloud can reach the TPU, to start them all.

## Failures and stalls

A process of a pool that fails does not wait for its peers. It prints the error, writes it to the coordination service and exits at once, instead of sitting in `jax.distributed`'s shutdown barrier for up to 300 seconds. This holds from the moment the process joins the pool, so a process whose GPU fails to open, for example because it has no memory left, ends the launch too; otherwise its peers would wait minutes for its devices. When a rank fails between steps, for example because its data loader raised, its peers may already be inside the next step's collectives, which no GPU backend times out. The failing rank's error is written before it tries to agree with them, and a watch thread in every process ends the process when a published failure has not been heard at an agreement within 60 seconds. So the whole pool ends within about a minute, under `dew launch`, `srun` or a scheduler alike.

A rank can also stall without failing: blocked on a read, or in a compile that waits for its peers. Then every process stays alive and none reports anything, while the others wait inside a collective or a communicator's setup. For this case a GPU pool bounds each device execution with XLA's execution watchdog: `prepare_process` sets `--xla_gpu_execution_terminate_timeout=30m` when the CUDA plugin is present and the run has not set it. A process whose step, or whose sampling loop, runs longer than that ends, and the launcher, `srun` or the scheduler stops the rest. If you know how long your steps take, set a tighter value in `XLA_FLAGS` or through the recipe's `xla_flags`.

The watchdog bounds device executions only. Before a phase agreement's collectives, such as a checkpoint save or the end of `fit`, the ranks meet on the host, so a rank that arrives first waits there for one still busy with host work, for example process 0 uploading the final checkpoint to Weights & Biases. On the device it would sit inside a collective, which the watchdog would end. That host wait is bounded by `AGREEMENT_PATIENCE_SECONDS` in `dew.artifacts`, one day. A rank that stalls in host work before an agreement holds its peers for up to that long, so lower it when your host phases are short.

A pool keeps JAX's persistent compilation cache. jax 0.11.2 keys a cached executable by the fingerprint of the compiling process's accelerator topology, which on a GPU describes the device down to its NVLink links, and only process 0 writes entries. On one four-GPU host with an NVLink pair and a PCIe pair, ranks 0 and 1 found a step in a shared cache while ranks 2 and 3 compiled it, and that compile waited for ever for its peers' shares of the sharded autotuning. Dew pins jax to 0.11.2 with a fix (reported as jax-ml/jax#40940). A computation that spans processes hashes the fingerprints of all of them, so every rank loads the same entry, provided every process compiles for the same accelerators: the same platform and runtime, CUDA driver, cuDNN and cuBLAS, and the same device kinds, compute capability and core count. A pool whose processes differ in any of these compiles those computations without the cache, and JAX logs which processes differ. A GPU pool whose jax lacks the fix, such as an image's own jax or one installed with `--no-deps` around the pin, compiles without the persistent cache: `prepare_process` turns it off before the first compile.

## Mesh layout across nodes

Devices inside one node talk over NVLink or the TPU interconnect. Nodes talk over a network that is often ten times slower. Fully sharded data parallelism (`fsdp`) gathers every layer's parameters and reduce-scatters every layer's gradients, so an fsdp axis that crosses nodes pays that slow link on every layer.

Hybrid sharding keeps fsdp inside a node and replicates across nodes, so only one gradient all-reduce per step crosses the network. `MeshSpec(replicas=N)` asks for it. For two nodes of eight GPUs:

<!-- not run: needs a pool of two hosts with eight GPUs each -->
```python
import jax
import optax
from dew import Trainer, models
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec
from dew.training.runtime import prepare_process

prepare_process()
model = models.build("causal_transformer", vocab_size=256, emb_features=512,
                     num_layers=8, num_heads=8, max_seq_len=1024)
# fsdp over the eight GPUs of each node, and the data axis of 2 across the nodes.
mesh = MeshSpec(fsdp=8, replicas=2)
trainer = Trainer(LMObjective(model, seq_len=1024), optax.adamw(3e-4),
                  key=jax.random.key(0), mesh=mesh)
```

`replicas` counts groups of granules. A granule is whatever the devices' `slice_index` groups: a TPU slice on a multislice run, and on several GPU hosts a host or an NVLink domain, however many processes each host runs, because XLA numbers GPU slices per host boot or NVLink fabric. Where every device shares one slice, as in a CPU pool or on the hosts of one TPU slice, the process is the granule. `build_mesh` then builds the mesh with JAX's `mesh_utils.create_hybrid_device_mesh`, the same call MaxText uses for multislice runs. The data axis spans the groups. The expert, tensor, sequence and stage axes stay inside one granule. When a group holds more than one granule, fsdp is the only axis that crosses between them. `replicas` must divide both the granule count and the data axis, and fsdp must divide over the granules of one group. `build_mesh` raises with the numbers when they do not fit.

On GPU hosts whose device ids run host by host, `jax.make_mesh` already puts the data axis outermost, so for one granule per replica `replicas` states the layout rather than changing it. It changes the mesh when a replica spans several granules, when the slices do not follow the device order, and on a multislice TPU run.

With `replicas=1`, `jax.make_mesh` places the devices, which is what a single node or a single TPU slice wants. To choose which devices share an axis, pass them to `build_mesh(spec, devices)`: it fills the mesh with the list in the order you give, row-major over `MESH_AXES`, so the last axes get neighbouring entries. On a machine whose NVLink pairs are GPUs 0-1 and 2-3, `[0, 2, 1, 3]` puts a two-way tensor axis across the pairs instead of inside them. `jax.make_mesh` would sort GPU devices by id and discard that order.

Hybrid sharding keeps a full copy of the parameters and optimizer state on every replica group. Plain fsdp across all nodes divides them by the total device count instead. Choose hybrid sharding when one node's memory holds the sharded state, and plain fsdp when it does not.

## Sequence parallelism

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

## Single-machine rehearsal

Before you book nodes, run the same launch on one machine. On a machine with four GPUs, the default of one process per GPU, with NCCL on its socket transport (no NVLink, PCIe peer access or shared memory between processes), exercises the paths that run between hosts:

```bash
dew launch --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 \
    -- python tests/distribution_worker.py --out /tmp/pool.json --mesh '{"fsdp": 2, "replicas": 2}'
```

To catch code that assumes one filesystem, start each process in a directory of its own, with its own `HOME`, `HF_HOME` and compilation cache. A persistent checkpoint directory then has to stay on storage every process shares; `Checkpoints` refuses a directory the processes do not share ([Checkpoints](checkpoints.md)).

Without GPUs, CPU devices stand in. This command starts four processes with two CPU devices each, the layout of four hosts with two accelerators, grouped into two replicas of two hosts:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 4 \
    --env XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    -- python tests/distribution_worker.py --out /tmp/pool.json \
       --mesh '{"fsdp": 2, "replicas": 2}'
```

`/tmp/pool.json` records the mesh, the losses and, for every fsdp group, the processes its devices sit on:

```text
{"processes": 4, "devices": 8, "mesh": {"data": 4, "expert": 1, "fsdp": 2, "tensor": 1, "sequence": 1, "stage": 1}, "fsdp_groups": [[0, 1], [0, 1], [2, 3], [2, 3]], "partition": {"count": 4, "readers": 1}, "placed_whole": false, "losses": [4.891427993774414, 3.6097896099090576, 2.8838775157928467], "loss_steps": [1, 2, 3]}
```

Each fsdp group spans the two hosts of its replica. With `--mesh '{"fsdp": 2}'` and no `replicas`, `jax.make_mesh` gives `"fsdp_groups": [[0], [1], [2], [3]]`. Run as one process with `XLA_FLAGS=--xla_force_host_platform_device_count=8` and `--mesh '{"fsdp": 8}'`, the worker records losses of 4.891427993774414, 3.609790086746216 and 2.8838775157928467, within 1e-6 of the pool's.

`tests/test_distribution.py` runs this comparison for hybrid sharding and for a split sequence across processes. It also checks that a failing process stops the pool, including a rank whose loader fails in the middle of `fit` and one that stalls without failing, that a pool refuses a checkpoint directory its processes do not share, that two pools started together take a port each, and that a pool's second run loads on every process the step its first run compiled. On a GPU run, the tests marked `mesh(devices=2)` take one GPU per process.

`tools/layout_parity.py` runs every layout of every model family against one device, in one process or under `dew launch`. It compares the loss and each gradient leaf against the reference's own deviation when the batch's sums are reordered, and the devices' FLOPs against one device's split evenly. A layout Dew refuses by design raises `dew.nn.sharding.LayoutRefused` with the reason and a layout that runs, such as a stage axis over a DiT; the tool lists those rows apart and exits nonzero only on a mismatch, repeated work or another error. `--prepare` computes each model's one-device reference into a `--references` directory in a job of one device, so a run of layouts on every device computes none.

## Tested configurations

These checks passed:

- on one host with four RTX 3090 GPUs (PCIe 3.0), `dew launch` pools of four processes with one GPU each, NCCL on its socket transport and a working directory, `HOME`, `HF_HOME` and compilation cache per process: `tools/layout_parity.py` over every layout of the dense, MoE (8 and 128 experts), Mamba-2 hybrid and DiT models, each layout within its floor or refused with its reason, and the pool tests of `tests/test_distribution.py`;
- real process pools launched by `dew launch`: four processes of two CPU devices for `MeshSpec(fsdp=2, replicas=2)`, and two of four for `MeshSpec(fsdp=2, sequence=2, replicas=2)`, each matching one process of plain fsdp over eight devices;
- `hybrid_devices` against stand-in devices for topologies this machine lacks: hosts as granules, and two slices of two hosts each;
- both exchanges against whole-sequence attention, forward and backward, on the simulated eight-device mesh, with the compiled trainer step showing which exchange each mesh ran, including heads split over tensor and sequence at once, four sequence shards over two key heads, packed masks, biases, sinks and the pipeline's stage axis.
- on one TPU v6e chip, `tools/qualify_sequence_exchange.py` ran the all-to-all exchange's `shard_map` around the Mosaic splash kernel, forward and backward, in fp32 and bf16. Its output and gradients equal whole-sequence splash exactly. One chip has one sequence shard, so this proves the lowering, not the exchange across chips.

Dew has not been run on two physical nodes. Throughput and memory numbers for hybrid sharding and the sequence exchanges need at least one host with eight accelerators (a TPU v5e-8 or v6e-8 slice, or eight GPUs on NVLink) for the sequence axis, and two such hosts, or a two-slice TPU run, for `replicas`. Network failures, NCCL or DCN collective tuning, shared checkpoint storage across nodes and cluster preemption are untested. A first multi-node run should compare its losses with a single-node run of the same global batch for a few hundred steps before it trains for real.
