# Multi-node training

A multi-node run starts the same training script on every machine, one process per GPU (or per TPU worker), and joins the processes into one `jax.distributed` pool. `Trainer` then treats the pool's devices as one set. `dew launch` starts the pool on one machine, over ssh on plain hosts, under Slurm or Open MPI, or on Cloud TPU VMs. `MeshSpec(replicas=N)` lays out the mesh so that only one gradient all-reduce per step crosses the slow network between nodes. [Distributed training](../concepts/distributed.md) explains meshes and layouts.

## Process pools

Every node runs the same script, and each copy is one process of a `jax.distributed` pool. The script joins the pool when it calls `dew.training.runtime.prepare_process()`. Every built-in recipe does this on its first line, and a script of your own must call it before it creates any array.

`prepare_process` calls `jax.distributed.initialize`, which needs three facts: the address of process 0's coordinator, the number of processes and this process's rank. `dew launch` starts the processes wherever you run it, and each kind of cluster supplies the three facts in its own way:

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

Without `--hosts`, `--hostfile` or `--tpu`, `dew launch` uses JAX's own cluster detection (`jax._src.clusters`) to find out where it is running, so it recognizes the same Slurm, Open MPI, Cloud TPU, GKE and Kubernetes environments that `jax.distributed.initialize` does. A flag that names hosts or TPUs takes precedence over what it detects. Add `--dry-run` to see what it would run.

## dew launch

On one machine, `dew launch` runs the program after `--` once per GPU:

```bash
dew launch -- python recipes/lm/train.py
```

On a four-GPU machine this starts four processes, each holding one GPU through `JAX_LOCAL_DEVICE_IDS`. The program needs no flag to join the pool, because `dew launch` puts the pool's size and each process's rank in the environment, and `prepare_process` joins the pool it finds there. With `--trainer.multi-host True`, a run refuses to start alone when no pool is configured; with `False`, it never joins one.

The launcher prints one `pool:` line with the process count, the devices per process and the coordinator address, then one `[r] rank r of 4 on localhost, GPU r, pid ...` line per rank. Each process prints `Joined the JAX process pool: process r of 4` when it joins.

Every output line starts with its rank. `--processes-per-host N` runs N processes on each host and splits the GPUs evenly between them, so `--processes-per-host 1` runs one process that holds every GPU. `--devices-per-process N` gives each process N GPUs.

If a pool would need more GPUs than the launcher finds on the first host, it refuses the pool before any rank starts, and the message gives both counts. If it cannot count a host's GPUs, because nvidia-smi is missing there or fails, it says so on stderr and starts the pool without the check. `JAX_PLATFORMS=cpu`, in the environment or through `--env`, stops the launcher from counting GPUs. It counts only NVIDIA GPUs (those in `CUDA_VISIBLE_DEVICES`, when that is set), so on other GPUs, set `--processes-per-host` yourself.

`--env NAME=VALUE` passes a variable to every process. NAME must be a valid shell variable name, as in `--env XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`. `--cwd DIR` sets the directory each process starts in. `--dry-run` prints the commands without running them. On a CPU-only machine:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 2 --dry-run -- python train.py
```

```text
pool: 2 processes on localhost, each with every local device, coordinator localhost:51403
JAX_COORDINATOR_ADDRESS=localhost:51403 DEW_PROCESS_COUNT=2 DEW_PROCESS_ID=0 python train.py
JAX_COORDINATOR_ADDRESS=localhost:51403 DEW_PROCESS_COUNT=2 DEW_PROCESS_ID=1 python train.py
```

When one process exits with an error, the launcher stops the others and exits with that process's code. Otherwise the survivors would wait in a collective for a peer that is gone. The launcher names the failed rank (`rank 2 on localhost exited 1; stopping the other 3`), and once the others have stopped, it prints `last lines of rank 2:` followed by that rank's last 20 lines, so the cause is at the bottom of the output.

A rank killed by a signal is reported as `was killed by SIGKILL`, and the launch exits with 128 plus the signal number, as a shell reports it. Ctrl-C, a scheduler's SIGTERM and a closed terminal all stop the whole pool the same way, by sending each rank SIGTERM. A JAX process in a pool takes SIGTERM as a preemption notice, so `Trainer.fit` writes a checkpoint at the step every rank agrees on and exits with 143 ([Checkpoints](checkpoints.md#preemption)). Any other program keeps running until the launcher kills it.

When the launcher itself received the signal, it gives the ranks 300 seconds to checkpoint and exit before it sends SIGKILL, and a second signal kills them at once. When a rank failed, the others are waiting in a collective for it, so they get 10 seconds. Rank 0 is killed last, because its process holds the pool's coordination service, and a rank that outlived it would abort with an XLA `Check failure` that looks like a crash of its own.

## Plain machines

List the hosts, process 0's first, then the command after `--`:

```bash
dew launch --hosts node0,node1 -- /opt/dew/.venv/bin/python recipes/lm/train.py
```

`--hostfile FILE` reads the hosts from a file instead, one per line. It takes the first word of each line, so an MPI hostfile with `slots=8` works as it is. The launcher counts the GPUs on the first host and runs as many processes on every host, so the hosts should match.

The launcher starts the command on each host over `ssh -o BatchMode=yes`, so passwordless keys must already work. The remote command runs in your shell without a login profile, so a virtualenv activated in a profile is not active there; give the interpreter as an absolute path, as above. A host named `localhost` runs the command directly.

Each process starts in the current directory, which must exist at the same path on every host, or in the directory that `--cwd` names. The coordinator listens on a port that is free on the first host when the launch starts, so pools started together on one machine never share a port. If a firewall needs a fixed port, name it with `--port`, and if the other hosts reach the first host by another name, set `--coordinator`.

## Slurm

Run `dew launch` inside the allocation. It starts `srun`, and JAX reads the rank from Slurm:

```bash
sbatch --nodes=2 --gpus-per-node=8 --wrap "dew launch -- /opt/dew/.venv/bin/python recipes/lm/train.py"
```

This runs `srun --kill-on-bad-exit=1 --export=ALL --label --ntasks-per-node=8 ...`, with the `--env` variables in srun's environment. Each output line starts with its task number, and when a task fails, srun names it and stops the others.

JAX gives each Slurm task the one GPU at its `SLURM_LOCALID`, so a node runs one task per GPU. The launcher takes the number of tasks per node from `--processes-per-host` when you give it, else from the allocation's own `--ntasks-per-node`, else from its `--gpus-per-node`, else from the GPUs that the node running `dew launch` sees. So on a cluster whose login node has no GPUs, ask for `--gpus-per-node` rather than `--gpus`, or pass `--processes-per-host`. Every task needs a CPU of its own in the allocation, which `--ntasks-per-node` or `--cpus-per-gpu` in the sbatch request provides; with neither, srun refuses to start more tasks than the allocation has CPUs.

Because each task sees a single GPU, the launcher refuses `--devices-per-process` under Slurm. Fewer tasks per node than GPUs per node would leave the remaining GPUs idle with nothing reporting it, so the launcher refuses such an allocation, or such a `--processes-per-host`, and its message names `--ntasks-per-node`. Inside a step of several tasks that `srun` already started, `dew launch` runs the program in place, and `prepare_process` refuses a step with fewer tasks on a node than the GPUs its task sees, for the same reason. Inside a step of one task, such as a GPU wrapper script that runs its command under `srun`, `dew launch` starts one process per GPU of the step, as on a plain machine, and each process gets its placement from the launcher, not from Slurm's variables. A program that such a step runs without `dew launch` runs as one process, unless it asks for a pool with `multi_host=True`.

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

The launcher finds the TPU's zone the way [`dew tpu`](../tpu.md) does, or uses `--zone`, and starts one `gcloud compute tpus tpu-vm ssh --worker=N` per worker. Each worker sources the environment that `dew tpu setup` wrote, so `python` is the setup's virtualenv. A relative `--cwd` is taken under the worker's home directory, where `dew tpu sync` puts the working tree. JAX reads each worker's rank from the TPU metadata server. Stopping the pool closes the connections, which hangs up the programs on the workers.

Name several TPUs to run one multislice pool over them:

```bash
dew launch --tpu slice-a,slice-b -- python train.py
```

Each worker then gets `MEGASCALE_NUM_SLICES`, its `MEGASCALE_SLICE_ID`, and the first worker of the first slice as `MEGASCALE_COORDINATOR_ADDRESS`, with `MEGASCALE_PORT=8081`. These are the values Ray's TPU support sets. JAX numbers the processes slice by slice, and the devices' `slice_index` tells `MeshSpec(replicas=...)` where the slices meet.

On a TPU VM worker, `dew launch -- python train.py` runs the program on that worker only. Every worker has to run the program, but by default the workers cannot reach each other over ssh, so the launcher cannot start the others from there. On worker 0 of a pod it prints a reminder that every worker has to run it. To start them all, run `dew launch --tpu NAME` from your machine, or from a worker whose gcloud can reach the TPU.

## Failures and stalls

A process in a pool that fails does not wait for its peers. It prints the error, writes it to the coordination service and exits at once, where it would otherwise sit in `jax.distributed`'s shutdown barrier for up to 300 seconds. This holds from the moment the process joins the pool. So a process whose GPU fails to open, for example because it has no memory left, ends the launch too, and its peers do not wait minutes for its devices.

When a rank fails between steps, for example because its data loader raised, its peers may already be inside the next step's collectives, and no GPU backend times those out. So the failing rank writes its error before it tries to agree with them, and a watch thread in every process ends that process 60 seconds after a failure is published, unless an agreement has picked the failure up by then. The whole pool then ends within about a minute, whether it runs under `dew launch`, `srun` or a scheduler.

A rank can also stall without failing, for example when it is blocked on a read, or in a compile that waits for its peers. Then every process stays alive and none of them reports anything, while the others wait inside a collective or a communicator's setup. For this case a GPU pool bounds each device execution with XLA's execution watchdog: `prepare_process` sets `--xla_gpu_execution_terminate_timeout=30m` when the CUDA plugin is present and the run has not set that flag. A process whose step, or whose sampling loop, runs longer than that ends, and the launcher, `srun` or the scheduler stops the rest. If you know how long your steps take, set a tighter value in `XLA_FLAGS` or through the recipe's `xla_flags`.

The watchdog bounds device executions only. Before the collectives of a phase agreement, such as a checkpoint save or the end of `fit`, the ranks meet on the host. So a rank that arrives first waits on the host for one that is still busy with host work, for example process 0 uploading the final checkpoint to Weights & Biases. If it waited on the device instead, it would sit inside a collective and the watchdog would end it. `AGREEMENT_PATIENCE_SECONDS` in `dew.coordination` bounds that host wait at one day. A rank that stalls in host work before an agreement can hold its peers for that long, so lower the value when your host phases are short.

A pool keeps JAX's persistent compilation cache. jax 0.11.2 keys a cached executable by a fingerprint of the compiling process's accelerator topology, which on a GPU describes the device down to its NVLink links, and only process 0 writes entries. On one four-GPU host with an NVLink pair and a PCIe pair, ranks 0 and 1 found a step in a shared cache while ranks 2 and 3 compiled it, and that compile waited forever for its peers' shares of the sharded autotuning.

Dew pins jax to 0.11.2 with a fix (reported as jax-ml/jax#40940). With the fix, a computation that spans processes hashes the fingerprints of all of them, so every rank loads the same entry, as long as every process compiles for the same accelerators: the same platform and runtime, CUDA driver, cuDNN and cuBLAS, and the same device kinds, compute capability and core count. A pool whose processes differ in any of these compiles those computations without the cache, and JAX logs which processes differ. A GPU pool whose jax lacks the fix, such as an image's own jax or one installed with `--no-deps` around the pin, compiles without the persistent cache, because `prepare_process` turns the cache off before the first compile.

## Mesh layout across nodes

Devices inside one node talk over NVLink or the TPU interconnect, and nodes talk over a network that is often ten times slower. Fully sharded data parallelism (`fsdp`) gathers every layer's parameters and reduce-scatters every layer's gradients, so an fsdp axis that crosses nodes goes over the slow link for every layer.

Hybrid sharding keeps fsdp inside a node and replicates across nodes, so only one gradient all-reduce per step crosses the network. `MeshSpec(replicas=N)` asks for it. For two nodes of eight GPUs:

<!-- not run: needs a pool of two hosts with eight GPUs each -->
```python
import jax
from dew import Trainer
from dew.config import OptimConfig
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec
from dew.training.runtime import prepare_process

prepare_process()
model = CausalTransformer(vocab_size=256, emb_features=512,
                          num_layers=8, num_heads=8, max_seq_len=1024)
# fsdp over the eight GPUs of each node, and the data axis of 2 across the nodes.
mesh = MeshSpec(fsdp=8, replicas=2)
trainer = Trainer(LMObjective(model, seq_len=1024), OptimConfig(learning_rate=3e-4),
                  key=jax.random.key(0), mesh=mesh)
```

`replicas` counts groups of granules. A granule is whatever the devices' `slice_index` groups together. On a multislice run that is a TPU slice. On several GPU hosts it is a host or an NVLink domain, however many processes each host runs, because XLA numbers GPU slices per host boot or NVLink fabric. Where every device shares one slice, as in a CPU pool or on the hosts of one TPU slice, the process is the granule.

`MeshSpec.build` then builds the mesh with JAX's `mesh_utils.create_hybrid_device_mesh`, the same call MaxText uses for multislice runs. The data axis spans the groups, and the expert, tensor, sequence and stage axes stay inside one granule. When a group holds more than one granule, fsdp is the only axis that crosses between them. `replicas` must divide both the granule count and the data axis, and fsdp must divide evenly over the granules of one group. When they do not fit, `MeshSpec.build` raises with the numbers.

On GPU hosts whose device ids run host by host, `jax.make_mesh` already puts the data axis outermost. So with one granule per replica, `replicas` gives the mesh you would get anyway. It does change the mesh when a replica spans several granules, when the slices do not follow the device order, and on a multislice TPU run.

With `replicas=1`, `jax.make_mesh` places the devices, which is what you want on a single node or a single TPU slice. To choose which devices share an axis, pass them to `spec.build(devices)`. It fills the mesh with the list in the order you give, row-major over `MESH_AXES`, so the last axes get neighbouring entries. For example, on a machine whose NVLink pairs are GPUs 0-1 and 2-3, `[0, 2, 1, 3]` puts a two-way tensor axis across the pairs instead of inside them. `jax.make_mesh` would sort GPU devices by id and lose that order.

Hybrid sharding keeps a full copy of the parameters and optimizer state on every replica group. Plain fsdp across all nodes divides them by the total device count instead. Choose hybrid sharding when one node's memory holds the sharded state, and plain fsdp when it does not.

## Sequence parallelism

`MeshSpec(sequence=N)` splits the token positions of every sequence over N devices, and each attention call exchanges data between them in one of two ways. Both give the same result as one device.

- The all-to-all follows DeepSpeed Ulysses. Each device trades its slice of the positions for a slice of the attention heads, attends the whole sequence for those heads and trades back. No device ever holds a whole key or value tensor. Causal masks, sliding windows, packed-document masks and the TPU splash kernel work as they do on one device. Grouped key and value heads are repeated only as far as the split needs.
- The gather keeps the queries split and gathers the whole keys and values on every device. It takes any head count and any key length. A causal or masked call reorders its queries so every device gets the same work, so its sequence length must be a multiple of twice `sequence`.

A call runs the all-to-all when `tensor` times `sequence` divides its number of query heads and `sequence` divides both its query and key lengths. Where both exchanges can run, a causal, windowed or masked call uses the all-to-all. Its kernel sees whole sequences and the causal flag, so cuDNN and splash skip the masked blocks. The gather gives its kernel a mask for its reordered rows, and cuDNN runs that mask as a dense bias over every logit.

A call with no mask does the same work either way, so it uses the exchange that sends fewer bytes. Count bytes per device in units of sequence length times head width times (N-1)/N. With H query heads and K key heads per tensor shard, the all-to-all sends 2(H + K')/N, where K' is K repeated as far as the split needs, and the gather sends 2K. At H=32, K=8, N=2 that is 40 against 16, so unmasked grouped-query attention uses the gather. Joint text-and-image attention over an odd length, cross attention to a 77-token context, and head counts the split does not divide all use the gather too. The rule is in `dew.nn.attention.sequence_parallel_attention`.

The table gives the time, in milliseconds, of the forward and backward pass of one attention call with 16 query heads, 8 key heads and head width 128, in bf16, split two ways. I measured it on one host with 4x RTX 3090 (NV4 pair + PHB pair, cross-socket). Each pair's columns come from one run, and its one-GPU column ran on the pair's first GPU.

| Causal, tokens | NVLink pair: one GPU | NVLink pair: all-to-all | PCIe pair: one GPU | PCIe pair: all-to-all | PCIe pair: gather |
|---|---|---|---|---|---|
| 8,192 | 18.1 | 12.0 | 16.0 | 18.5 | 37.2 |
| 16,384 | 62.3 | 41.2 | 61.9 | 46.5 | 103.7 |
| 32,768 | 245.9 | 130.4 | 244.3 | 150.1 | 350.1 |
| 65,536 | | 503.0 | | 541.8 | 1302.4 |

With Rigel's attention (4 key heads of width 64), the gather took 2.4 to 3.9 times as long as the all-to-all on the PCIe pair over the same lengths. Without a mask, the two exchanges came within 17% of each other at 4,096 to 32,768 tokens on the NVLink pair. The byte count picked the faster one in ten of twelve shapes, and where it picked the slower one, that one was at most 6% slower.

I also timed two other designs against the all-to-all on the same pairs, and Dew ships neither. A zigzag ring passes key and value blocks around the devices while each device attends its queries to the block it holds; I timed it on cuDNN with the kernels an exact ring would run. Two head groups cut the all-to-all in two, so that one group's exchange can run while the other group's kernel computes. Each time below is relative to the all-to-all's, for the causal call in the table above.

| Tokens | Ring, NVLink pair | Ring, PCIe pair | Two head groups, NVLink pair | Two head groups, PCIe pair |
|---|---|---|---|---|
| 8,192 | 1.46 | 1.79 | 0.95 | 0.84 |
| 16,384 | 1.20 | 1.45 | 0.98 | 0.90 |
| 32,768 | 1.11 | 1.32 | 1.09 | 0.97 |
| 65,536 | 1.04 | 1.17 | 1.00 | 0.98 |

With Rigel's attention, the ring took 1.02 to 1.79 times as long, and two head groups 0.91 to 1.10 times; four head groups did no better than two. So the ring was slower at every length. The head groups saved up to 16% on the PCIe pair at lengths that one GPU can also train, and at 32,768 and 65,536 tokens they ranged from 3% faster to 9% slower.

Keep the sequence axis on the fastest links, because both exchanges run once per attention layer in the forward and the backward pass, and again when the layer is recomputed. On the 3090s, an all-to-all of 64 MiB per device moved 19.5 GB/s per device over the NVLink pair and 6.3 GB/s across the sockets, and NCCL runs the PCIe pair through host memory, the same way it runs the sockets. On the PCIe pair, a 32,768-token Qwen3-0.6B-shaped training step split two ways took 7.6 s, and 1.25 s of that was all-to-all that no computation overlapped. `MeshSpec.build` puts the `sequence` axis last, on neighbouring devices, and `replicas` keeps it inside a granule.

## Single-machine rehearsal

Before you book nodes, run the same launch on one machine. On a machine with four GPUs, the default of one process per GPU, with NCCL on its socket transport (no NVLink, PCIe peer access or shared memory between processes), exercises the same paths that run between hosts:

```bash
dew launch --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 \
    -- python tests/distribution_worker.py --out /tmp/pool.json --mesh '{"fsdp": 2, "replicas": 2}'
```

To catch code that assumes one filesystem, start each process in a directory of its own, with its own `HOME`, `HF_HOME` and compilation cache. A persistent checkpoint directory then has to stay on storage every process shares; `Checkpoints` refuses a directory the processes do not share ([Checkpoints](checkpoints.md)).

Without GPUs, CPU devices stand in for them. This command starts four processes with two CPU devices each, which is the layout of four hosts with two accelerators each, grouped into two replicas of two hosts:

```bash
JAX_PLATFORMS=cpu dew launch --processes-per-host 4 \
    --env XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    -- python tests/distribution_worker.py --out /tmp/pool.json \
       --mesh '{"fsdp": 2, "replicas": 2}'
```

`/tmp/pool.json` records the mesh, the losses and, for every fsdp group, the processes its devices are on:

```text
{"processes": 4, "devices": 8, "mesh": {"data": 4, "expert": 1, "fsdp": 2, "tensor": 1, "sequence": 1, "stage": 1}, "fsdp_groups": [[0, 1], [0, 1], [2, 3], [2, 3]], "partition": {"count": 4, "readers": 1}, "placed_whole": false, "losses": [4.891427993774414, 3.6097896099090576, 2.8838775157928467], "loss_steps": [1, 2, 3]}
```

Each fsdp group spans the two hosts of its replica. With `--mesh '{"fsdp": 2}'` and no `replicas`, `jax.make_mesh` gives `"fsdp_groups": [[0], [1], [2], [3]]`. Run as one process with `XLA_FLAGS=--xla_force_host_platform_device_count=8` and `--mesh '{"fsdp": 8}'`, the worker records losses of 4.891427993774414, 3.609790086746216 and 2.8838775157928467, within 1e-6 of the pool's.

`tests/test_distribution.py` runs this comparison for hybrid sharding and for a sequence split across processes. It also checks that:

- a failing process stops the pool, including a rank whose loader fails in the middle of `fit` and a rank that stalls without failing;
- a pool refuses a checkpoint directory its processes do not share;
- two pools started together get a port each;
- a pool's second run loads, on every process, the step its first run compiled.

On a GPU run, the tests marked `mesh(devices=2)` take one GPU per process.

`tools/layout_parity.py` runs every layout of every model family against one device, in one process or under `dew launch`. It compares the loss and each gradient leaf with the one-device reference, within the reference's own deviation when the batch's sums are reordered, and it compares the devices' FLOPs with one device's FLOPs split evenly. A layout that Dew refuses by design, such as a stage axis over a DiT, raises `dew.nn.sharding.LayoutRefused` with the reason and a layout that does run. The tool lists those rows separately, and it exits nonzero only on a mismatch, repeated work or another error. `--prepare` computes each model's one-device reference into a `--references` directory in a one-device job, so a later run of the layouts on every device does not compute the references itself.

## Tested configurations

These checks passed:

- On one host with four RTX 3090 GPUs (PCIe 3.0), `dew launch` ran pools of four processes with one GPU each, NCCL on its socket transport, and a working directory, `HOME`, `HF_HOME` and compilation cache per process. In those pools `tools/layout_parity.py` ran every layout of the dense, MoE (8 and 128 experts), Mamba-2 hybrid and DiT models, and each layout came within its floor or was refused with its reason. The pool tests of `tests/test_distribution.py` passed as well.
- Real process pools launched by `dew launch`: four processes of two CPU devices for `MeshSpec(fsdp=2, replicas=2)`, and two of four for `MeshSpec(fsdp=2, sequence=2, replicas=2)`, each matching one process of plain fsdp over eight devices.
- `hybrid_devices` against stand-in devices for topologies this machine lacks: hosts as granules, and two slices of two hosts each.
- Both exchanges against whole-sequence attention, forward and backward, on the simulated eight-device mesh. The compiled trainer step showed which exchange each mesh ran. The cases included heads split over tensor and sequence at once, four sequence shards over two key heads, packed masks, biases, sinks and the pipeline's stage axis.
- On one TPU v6e chip, `tools/qualify_sequence_exchange.py` ran the all-to-all exchange's `shard_map` around the Mosaic splash kernel, forward and backward, in fp32 and bf16, and its output and gradients equal whole-sequence splash exactly. One chip has only one sequence shard, so this proves the lowering but not the exchange across chips.

Dew has not been run on two physical nodes. To measure throughput and memory for the sequence exchanges, you need at least one host with eight accelerators (a TPU v5e-8 or v6e-8 slice, or eight GPUs on NVLink); for hybrid sharding with `replicas`, you need two such hosts or a two-slice TPU run. Network failures, NCCL or DCN collective tuning, shared checkpoint storage across nodes and cluster preemption are untested. So on your first multi-node run, compare the losses with a single-node run of the same global batch for a few hundred steps before you train for real.
