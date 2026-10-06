# Distributed training

To train on several devices, you give `Trainer` two objects. A `MeshSpec` says how many devices each of six named mesh axes gets, and `MeshSpec.build` arranges the devices into a `jax.sharding.Mesh` with those axes. A `Layout` maps the logical axis names that Dew's modules give their parameters, such as `embed` and `mlp`, to mesh axes. `Trainer` uses both to initialize, place and update the training state, and XLA compiles the collectives that placement needs. The model, the objective and the data stay the same whatever mesh you choose.

![A mesh of 8 devices with fsdp=2 and tensor=2, so data=2. The batch's 16 rows split into 4 blocks over data and fsdp, each held by a tensor pair. An MLP kernel of shape (256, 1024) splits its rows over fsdp and its columns over tensor, and each block is held by one device of each data index.](../assets/mesh-light.svg)
![A mesh of 8 devices with fsdp=2 and tensor=2, so data=2. The batch's 16 rows split into 4 blocks over data and fsdp, each held by a tensor pair. An MLP kernel of shape (256, 1024) splits its rows over fsdp and its columns over tensor, and each block is held by one device of each data index.](../assets/mesh-dark.svg)

`docs/assets/figures.py` draws the figure from the placements that `MeshSpec.build`, `Layout().shardings` and `batch_shardings` return.

## Example

This trains a small decoder on eight simulated CPU devices. `--xla_force_host_platform_device_count` must be set before JAX is imported.

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import itertools

import jax
import numpy as np

from dew import Dataset, Trainer
from dew.config import OptimConfig
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec

model = CausalTransformer(vocab_size=512, emb_features=256, num_layers=2,
                          num_heads=4, mlp_features=1024, max_seq_len=64)
rows = np.random.default_rng(0).integers(0, 512, (16, 65), dtype=np.int32)
data = Dataset(train=lambda partition: itertools.repeat({"text": rows}), val=None,
               records=16, batch=16)
trainer = Trainer(LMObjective(model, seq_len=64), OptimConfig(learning_rate=1e-3), key=jax.random.key(0),
                  mesh=MeshSpec(fsdp=2, tensor=2))
state = trainer.fit(data, steps=4, log_every=2)

kernel = state.variables["params"]["layers_0"]["mlp"]["up_proj"]["kernel"]
print(kernel.shape, kernel.sharding.spec)
print(kernel.addressable_shards[0].data.shape)
```

```text
Training CausalTransformer from step 0 to 4: 2.2M parameters, on 8 × cpu, mesh data 2 × fsdp 2 × tensor 2, batch 16, float32
step 2/4  loss 4.741  ce 4.741  perplexity 114.5  token_accuracy 18.8%  step_time_ms 302.9  samples_per_sec 52.82  accepted 100.0%  0:00:01 left
step 4/4  loss 2.140  ce 2.140  perplexity 8.499  token_accuracy 92.5%  step_time_ms 154.2  samples_per_sec 103.8  accepted 100.0%
Trained 4 steps in 0:00:02: first step after 1.32 s, then 5.9 step/s
27.7% of the wall time in steps, final loss 2.140
(256, 1024) P('fsdp', 'tensor')
(128, 512)
```

Each device holds a `(128, 512)` block of the `(256, 1024)` kernel, because the rows are split over the two `fsdp` devices and the columns over the two `tensor` devices. The kernel's Adam moments are placed the same way.

## Mesh

Each `MeshSpec` field sets how many devices one axis gets, and the `data` axis gets the rest. The product of the fields must divide the device count, or `MeshSpec.build` raises `LayoutRefused`.

| Axis | `MeshSpec` field | Splits |
|---|---|---|
| `data` | (the remainder) | Batch rows; parameters are replicated over it |
| `expert` | `expert` | The expert dimension of mixture-of-experts layers, and batch rows |
| `fsdp` | `fsdp` | Parameter dimensions the layout assigns to it, and batch rows |
| `tensor` | `tensor` | The MLP hidden width, attention heads and vocabulary (Megatron's split) |
| `sequence` | `sequence` | Token positions, for sequence-parallel attention |
| `stage` | `stage` | The decoder's layer stack, into pipeline stages |

```python
from dew.training import MeshSpec

print(dict(MeshSpec(fsdp=4).build().shape))
print(dict(MeshSpec(fsdp=2, expert=4).build().shape))
print(dict(MeshSpec(fsdp=4, sequence=2).build().shape))
```

```text
{'data': 2, 'expert': 1, 'fsdp': 4, 'tensor': 1, 'sequence': 1, 'stage': 1}
{'data': 1, 'expert': 4, 'fsdp': 2, 'tensor': 1, 'sequence': 1, 'stage': 1}
{'data': 1, 'expert': 1, 'fsdp': 4, 'tensor': 1, 'sequence': 2, 'stage': 1}
```

`MeshSpec` also has `microbatches` (pipeline microbatches, below) and `replicas` (host groups the data axis spans; see [Multiple hosts](../guides/multi-node.md)).

## Parameter placement

Modules declare logical axes such as `embed`, `mlp`, `heads`, `kv`, `vocab` and `exp`. `Layout.rules` maps each name to mesh axes in precedence order, and the default table, `dew.nn.sharding.DEFAULT_RULES`, puts large dense dimensions on `fsdp` and expert dimensions on `expert`. Under `MeshSpec(tensor=N)`, the widths that Megatron splits are also split over the tensor axis: the MLP hidden width, the query and key-value heads, the attention width that `o_proj` reads, and the vocabulary. The residual width is not split over `tensor`. When a rule's mesh axes do not divide a dimension, Dew tries the next rule for that name, and if none fits, the dimension stays whole.

| `Layout` field | Default | Meaning |
|---|---|---|
| `rules` | `DEFAULT_RULES` | Logical axis name to mesh axes, in precedence order |
| `min_shard` | `2 ** 16` | Parameters with fewer elements stay replicated |
| `tolerance` | `0.02` | The fraction of shardable parameter elements that may stay replicated before `Layout.check` raises |
| `host` | `()` | Train-state fields kept in pinned host memory between steps: `"opt_state"`, `"ema"`, or `"variables"` for a CPU-owned step |
| `host_parameters` | `()` | Globs of variables an inference placement keeps in pinned host memory |

A parameter that no module declares logical axes for is split on its largest dimension that the `fsdp` size divides. `Layout.check` raises `LayoutRefused` when more than `tolerance` of the shardable elements stay replicated, and it lists the largest replicated parameters. Raising `tolerance` only hides the problem, so fix the placement. A rule that puts a parameter on `data`, `sequence` or `stage` is refused.

Optimizer moments and EMA copies have the same shapes as their parameters and are placed the same way. When a checkpoint is restored, Dew builds the restore template for the layout you asked for, so a run can resume on a different mesh as long as the parameter names and shapes match the saved state.

## Activation placement

The same table places activations. Its `activation_` names put a batch's rows on the `data`, `expert` and `fsdp` axes and its positions on `sequence`, and they put an activation's heads, MLP hidden width and vocabulary on `tensor`. The model pins the residual stream, the attention heads and the MLP hidden width to these names with `dew.nn.sharding.constrain`, and the trainer compiles its step under `flax.linen.logical_axis_rules(layout.rules)`. So when you change a rule, the activations move with the parameters.

Without those pins, GSPMD chooses each activation's placement from the weights around it. On an fsdp mesh it split the residual width and all-reduced the partial products of every projection, and on a tensor mesh it gathered every weight, as fsdp does. A batch's rows are never split over `tensor`. As in Megatron, every tensor shard reads the same rows and computes its own part of the width for them. When a rule's mesh axes do not divide an activation's dimension, the next rule for that name applies, or the dimension stays whole, the same as for parameters.

Some per-token work does not split any width that the tensor axis splits. Multi-head latent attention's down-projections into its query and key-value latents, and DeepSeek-V4's query latent and shared key head, read the whole residual, so under a tensor axis every tensor shard computed them over every token. At DeepSeek-V3's shape under `tensor=8`, that made a step cost 1.133 times one device's matmul FLOPs.

So where the tensor axis's link is fast enough, these projections run on each tensor shard's own tokens. Their input is placed as `SPREAD`, whose `activation_spread` rule splits positions over the tensor axis as well as the sequence axis, and the latents are gathered back for the up-projections, which are split by head. That saves (T - 1)/T of the projections' FLOPs. It adds gathers of the latents and of the residual's gradient, plus a sum of the projections' weight gradient over the tensor axis.

`dew.nn.sharding.down_projection` decides for each projection while the step is traced. It spreads when the link can move those extra bytes in no more time than the device, running at its peak rate, would take for the FLOPs saved. The peak rate is the fastest the device can run those FLOPs, so at any real utilisation they take at least that long, and spreading cannot slow the step. The trainer measures the link once per mesh with an all-gather over the tensor axis (`dew.training.distributed.link_bandwidth`). Every process uses the lowest figure in the pool, so they all compile the same program. For each axis it measured, `StepCompiled` records that figure and whether a projection spread.

At DeepSeek-V3's widths in bf16, with 16384 tokens per microbatch, an RTX 3090 needs 20.3 GB/s. On 4x RTX 3090 (PCIe 3.0, one host), the all-gather over the NVLink pair moved 31.0 GB/s, so that pair spreads; the PCIe pair moved 5.8 GB/s, so it does not. An H100 SXM needs 283 GB/s, under NVLink 4's nominal 450 GB/s per direction. No spread step has been timed on NVLink or on a TPU. A CPU mesh always spreads, and a device that the peak table in `dew.telemetry.instrumentation` does not name never does. To keep these projections on every token whatever the link, map `activation_spread` to the sequence axis alone in your rules.

The sequence axis has the same problem with a cross-attention's context, such as SD's 77 text tokens. The sequence axis does not divide a length like that, so every sequence shard projects the whole context into keys and values. On a CPU mesh under `sequence=4`, that made the conditional UNet's step cost 1.57 times one device's FLOPs. `dew.nn.sharding.split_positions` decides by the same rule, using the sequence axis's measured link. Where splitting pays, each shard projects its share of the positions, padded with zero rows to a multiple of the shard count.

The split adds a sum of the projections' weight gradients, which is a lot of bytes next to the FLOPs it saves, so a GPU's link rarely pays for it. At SDXL's widths, with 8 rows per sequence group and no head sharding, an H100 would need 2.47 TB/s once the extra work on padded positions is subtracted from the savings. So in that configuration every sequence shard projects the whole context, which costs FLOPs but no collectives. The link calculation uses the output head width that the active layout rules leave on each device.

The cross entropy scores each device's own tokens. It runs in a `shard_map` over the axes that hold the tokens, plus any other axis whose size divides the token count, and every device holds the whole head. Only the head's gradient and the loss's sums cross devices.

## Host memory

`Layout(host=("opt_state", "ema"))` keeps the optimizer state and the EMA copy in pinned host memory between steps. The compiled step fetches them to the device, runs the same update and writes them back. So the values match a device-only layout, and between steps the state takes host memory in place of device memory. Checkpoints save and restore this placement. The transfers add time to every step, so measure the step time with and without them.

When a checkpoint of host-resident state is saved, training waits until it is written. Orbax copies a device array to host memory before its asynchronous write, but it writes an array that is already in pinned host memory straight from the array's own buffer. The next step donates that buffer, and a GPU refuses the donation while the write is still reading it. So the step after a save waits for the write, and no extra memory is used.

Handing Orbax a copy would make the step wait only for the copy. But the copy would stay in pinned host memory until the write finished, and for fp32 Adam moments and EMA it is 12 bytes per parameter, or 84 GB at 7B parameters, on a host that keeps the state there because it has no room elsewhere. On an RTX 4080 host with a consumer NVMe drive (657 MB/s direct writes), writing a 6 GiB host-resident state took 18.4 s, where the copy would have taken 3.6 s and 8.4 GiB more peak memory. At 7B parameters that is about 4 minutes per checkpoint on that drive, against 20 to 50 s for the copy, and proportionally less on a faster drive. Choose `checkpoint_every` with that in mind. On that host, Orbax's write path itself held about 0.9 bytes of transient memory per byte written, whether or not it was given a copy.

`Layout(host_parameters=("params/layers_*",))` keeps a root decoder stack in pinned host memory for inference. Each model declares every physical stack with a `DecoderBank`, which records the stack's layer namespace (the same below every variables collection) and its `StackView`. `MultimodalTransformer` puts its decoder under `language_model`, and `DiffusionGemma` declares `text` once for the scope its encoder and decoder share. To select a nested stack, use a path such as `params/language_model/layers_*` or `params/text/layers_*`.

A scanned decoder declares one bank per run of like layers, and an unscanned decoder declares one bank per layer. Both stream through the same fetch loop, which holds the current layer and the next one. A `LayerBanks` source's `place(model, layout=...)` reads each physical bank once and rejects two different views of the same namespace. Selected leaves keep their FSDP and tensor PartitionSpecs, and external cache arrays stay on the device under their logical per-layer paths. GPU peak-memory bounds, and whether the transfers overlap with compute, have not been measured. `Trainer.place` rejects host-parameter layouts, because Dew has no backward pass or optimizer path for them.

On a TPU, pinned banks keep their layer axis major-most in physical memory. A fetch keeps that axis until the copy reaches device memory and squeezes it out there, which avoids incompatible host tile bitcasts and still transfers one layer at a time. Bank sources must honor any JAX `Format` given in their placement; both built-in sources do.

`dew.inference.LayerBanks` has two adapters for real use:

- `CheckpointBanks` reads Dew run checkpoints bank by bank.
- `HeldBanks` borrows an existing variables tree without donating or deleting it. The whole source stays in memory while the destination banks fill up, so it can hold two copies of the model at once.

Both read the entry leaves, the ones outside the declared stacks, and both accept namespaces relative to a collection for bank reads. Media, embeddings and heads stay entries even when they share a parent module with a decoder. Each entry or bank is placed before the next read starts. `CausalTransformer(bank_layers=N)` limits how large a bank is when it is built; it does not cap the pinned allocator or free the source's own storage. `StackView` still exposes the logical `layers_N` paths for saving and export. Synthetic generation is only in `tools/benchmark_host_offload.py`.

Loading a published checkpoint onto a mesh streams it leaf by leaf, though it does not stream banks into pinned host memory. `Pretrained.load(source, mesh=MeshSpec(...), layout=Layout(...))`, which `dew.pipeline(source)` calls, never builds the translated model on the host. Each decoder leaf is a recipe over the memory-mapped checkpoint that says which stored tensors to read, whether to transpose them or stack experts, and the storage dtype. `jax.make_array_from_callback` then reads only the part each device holds, and casts and transposes that part alone. The mapped pages are released once the leaf is placed. So the host holds at most one device shard of one leaf at a time, plus the towers, projectors and any dequantized quantized tensors, which are still built whole.

On one RTX 4080, loading Qwen3-0.6B in float32 this way peaks at 3.9 GB RSS, against 6.2 GB for loading first and placing afterwards. On an 8-device CPU mesh the figures are 4.0 GB and 5.9 GB, and 2.4 GB of that is the placed model itself. For Qwen3.8-27B (55.6 GB in bf16) on an 8-device mesh, the recipes cover 54.6 GB of the checkpoint, the largest single read is 0.32 GB, each device holds 6.95 GB, and the vision tower built on the host is 0.92 GB.

Host-parameter placement works only for decoder stack variables. Dew rejects a selection that includes an embedding table, a head or a prediction depth. Matching leaves of every layer in a bank must have the same memory kind and PartitionSpec. The stage pipeline and training through a banked store are rejected. In JAX 0.11.1, reverse-mode autodiff through the fetch loop does not lower, because its transpose asks for a `dynamic_update_slice` whose operands are in different memory spaces. A small forward-mode probe agrees with resident placement, but that probe does not show that backward re-fetch or training works.

Every host-resident weight you select is copied to the device on each forward pass, including every decode step, so PCIe transfers can take most of the time per token. `tools/benchmark_host_offload.py` reports local addressable-shard bytes, the device and host compiled-memory fields, process RSS and execution times. Its bytes per token divided by the decode latency is an end-to-end effective rate, not a measurement of PCIe bandwidth alone.

An earlier synthetic BF16 bank load of 18.00 GiB (144 layers, width 2048, `bank_layers=8`) reported 18.74 GiB final RSS and 37.9 GiB peak RSS. The saved load probe synchronized every bank, so a backlog of asynchronous work alone cannot explain that peak. The numbers do not separate temporary storage, allocator reservation and duplicate driver mappings. There is no completed result yet for generating on a GPU with a model larger than device memory, and no throughput result. Looking further into the allocator needs a small process tree with a hard memory cap, swap accounting and RSS/PSS measurements. The earlier oversized runs with an RSS watchdog are not a safe way to do it.

## Rematerialization

`CausalTransformer(remat=...)` sets which activations of each decoder block the backward pass recomputes and which it keeps. The value is a policy. The default, `None`, recomputes nothing, and `"full"` keeps only the block inputs. The other names in `dew.nn.backbones.decoder_block.REMAT_POLICIES` are `minimal`, `minimal_with_context`, `save_dot_except_mlp`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_qkv_proj`, `save_out_proj`, `minimal_offloaded` and `qkv_proj_offloaded`. Each keeps some of the named projection outputs (`q_proj`, `k_proj`, `v_proj`, `kv_proj`, `context`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) or offloads them to host memory. They trade recompute time for memory the same way MaxText's recipes of the same names do. You can also pass a record with your own lists, such as `{"save": ["q_proj", "k_proj", "v_proj"], "offload": ["gate_proj", "up_proj"]}`. Every policy trains the same model. `tools/benchmark_decoder_remat.py --remat <name>` reports the residual and compiler memory of one configuration.

## Batches

`Dataset.batch` is the global batch size. `Dataset.train(partition)` and `Dataset.val(partition)` read one share of every global batch, and `DataPartition.of(mesh)` returns this process's `DataPartition(index, count, readers, reader)`. Processes whose devices hold the same rows read the same share. For example, when a `stage` or `sequence` axis spans processes, several processes hold one row shard, and they count as `readers` of one share. A loader takes its records `index::count`, so global batch *k* holds the same records at every process count. `shard_batch` assembles the shares into global arrays of whole rows.

For custom data, read only the share that the partition you are given describes. Two readers of one share must read the same records in the same order, so a source that returns rows in whatever order its fetches finish, such as `UrlStream`, refuses a partition with more than one reader. If each process repeats the full dataset on its own, the run trains on a different distribution.

The placement helper treats rank-two and rank-three arrays as sequences, and it can split their second dimension when the sequence factor divides it. It keeps the non-batch dimensions of image and video tensors whole. Check custom rank-three data yourself, because the rank of an array does not tell the helper whether its second dimension really holds token positions.

## Sequence parallelism

`MeshSpec(sequence=N)` splits the token positions of every sequence over N devices, and each attention call uses one of two exact exchanges.

The all-to-all follows DeepSpeed Ulysses. Each device trades its slice of the positions for a slice of the heads, attends over the whole sequence and trades back, so no device holds a whole key or value. It runs when `tensor` times `sequence` divides the number of query heads and `sequence` divides both the query and key lengths. Every other call gathers the whole keys and values next to split queries, which works for any head count and any key length.

Where both can run, a causal, windowed or masked call uses the all-to-all, because its kernel skips the masked blocks. The gather gives its kernel an explicit mask, which no kernel skips, and it took two to four times as long for causal attention. A call with no mask uses whichever exchange sends fewer bytes. On the gather path, a causal or masked call stripes query chunks to balance the work, so its sequence length must be a multiple of twice the number of sequence shards. [Multiple hosts](../guides/multi-node.md) gives the measurements and the byte count.

A window that cuDNN accepts as its window flag runs through the all-to-all like any causal call, and the kernel skips the blocks outside the window. Local attention runs banded in the other cases: packed documents or sinks on cuDNN, a validity mask, a window on XLA, and chunks. A banded layer whose span fits in one shard's slice needs no exchange. Each device attends its own positions, and one shift brings its first block the previous device's last `window` keys and values, with their positions, segment IDs and validity. A layer with a wider span uses the all-to-all, like a dense call.

A token window has `seq_len + 1` IDs, and the model reads `seq_len` positions. Check the model length as well as the shape of the batch array. Packed segments, windows and rotary positions must stay aligned.

Mamba-2 layers split the sequence the same way, so a hybrid of Mamba-2 and attention layers trains under `MeshSpec(sequence=N)`. Each device convolves its own slice, reading the last `conv_kernel - 1` tokens of the previous device's slice, and runs its scan from a zero state. It then needs the state that the earlier slices leave. The devices pass each slice's total decay and final state, `[batch, heads, head_dim, state_size]` in fp32, between them in `ceil(log2 N) + 1` shifts (a parallel prefix scan), and each device adds that state's decayed contribution to each of its outputs. Packed documents reset the state and the convolution at every segment change, including a change that falls on a slice boundary. The sequence length must be a multiple of `N`, and each slice needs at least `conv_kernel - 1` tokens. The tensor axis does not split the Mamba-2 scan, so every tensor shard runs the scan over the whole width.

Cached autoregressive generation needs `sequence=1`. A mesh that trains with sequence parallelism may still be unable to run the decode cache.

## Layer scan and pipeline

`scan_layers=True` groups compatible consecutive decoder layers into a Flax scan, and the stored variables stay per layer. Scanning can cut compile time, but it adds stacking and loop overhead, and it does not promise a flat compile time or faster steps. Measure it at the model size and on the backend you plan to use.

Whatever `scan_layers` says, `init` draws each run of two or more like layers under one scan, so its program has one draw per run instead of one per layer. At a given seed those layers get new values from the same distribution, and the draw depends on `bank_layers`, which splits the runs. So a run recorded with a Dew version that initialized layer by layer starts from different weights at the same seed. Init under a stage mesh or with `decode=True` still runs the plain loop.

`MeshSpec(stage=N, microbatches=M)` turns on the GPipe-style pipeline. The layer pattern must repeat across stages and split evenly between them. Only a decoder's layer stack is pipelined; embeddings, the output head and other operations outside the stack are not split across stages. The trainer refuses a stage axis on a model that runs no pipeline, such as a DiT, because every stage would compute the whole step. That refusal raises `dew.nn.sharding.LayoutRefused`, a `ValueError` that names the axis and a layout that does run. Any other error on a layout is a defect.

Microbatch m takes rows m, m + M, m + 2M and so on, so each microbatch keeps its rows on the batch shards that already hold them. For that to work, M must divide the number of rows each device holds. For example, a batch of 8 rows over 4 row shards gives each device 2 rows. With four microbatches, some devices would have no rows in a given microbatch and would compute another microbatch's rows again, so a stage × fsdp step would cost 1.43 times one device's matmul FLOPs, where the pipeline bubble accounts for 1.25. The pipeline refuses that schedule. It names a microbatch count that divides the rows a device holds, and a batch size that fits the count you asked for. With 16 rows the same layout costs 1.20 times.

The stored master parameters are replicated over the stage axis, and each step builds a view that is partitioned by stage. Count that copy and its communication when you estimate memory savings. Some mixed layer patterns and KV-sharing configurations cannot use this pipeline. Cached decoding also needs `stage=1`.

Dew does not implement a 1F1B schedule.

## Kernels and precision

With `"auto"`, a call that asks for arithmetic no fused kernel performs takes the reference path on every backend: a matmul precision above the default, a softmax outside fp32, or a compute dtype other than the inputs'. A bf16 call on a GPU older than sm80 takes the reference path too. Otherwise, on a GPU, attention uses cuDNN when the shape, dtype and requested features fit its supported path, and XLA for other calls. On a TPU it uses the Pallas splash kernel when the call qualifies, and XLA otherwise. A call qualifies with bf16 or fp32, query and key lengths that are multiples of 128 and at least 512, a mask that splash can describe, no additive bias, and whole sequences at the kernel. The sequence-parallel all-to-all gives the kernel whole sequences; the key/value gather does not. Packed masks and optional attention features can change which kernel runs and how much memory it uses.

Qwix quantization is an optional, experimental training path with int8 and fp8 matmuls. It keeps full-precision master parameters and changes the numerics. In the RTX 4080 measurements on [Performance](../performance.md), fp8 did not make steps faster at the sizes tested.

## Several hosts

The JAX process pool must be initialized before any device array or model is created; the built-in recipes do this first. Each host needs the same software, access to the data and a reachable coordinator. `dew launch` starts a pool over ssh or under Slurm, and `MeshSpec(replicas=N)` keeps `fsdp` inside a node while the data axis spans the nodes. [Multiple hosts](../guides/multi-node.md) covers both and how to rehearse them on one machine, and [Cloud TPUs](../tpu.md) covers TPU provisioning.

The local process-pool tests run real `jax.distributed` processes. They do not cover network failures, remote storage, TPU collectives or cluster preemption.

## Checking a distributed run

Compare a small global computation under each placement you plan to use: the loss and gradients, record identity, optimizer and EMA state, and a save and restore. Record the compiler and backend versions and the tolerances used. A device count and a finite loss do not show that two runs train the same way. The CPU tests on global arrays cover loss normalization with unequal masks and restarting partway through an accumulation window. [Checkpoints](../guides/checkpoints.md) and [Evaluation and tracking](../guides/evaluation.md) describe what a resumed run and a reported metric guarantee.
