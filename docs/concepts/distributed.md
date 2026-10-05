# Distributed training

To train on several devices, pass `Trainer` a `MeshSpec` and a `Layout`. `MeshSpec` sets the device count for each of six named axes. `MeshSpec.build` arranges those devices into a `jax.sharding.Mesh`. `Layout` maps the parameters' logical axes, such as `embed` and `mlp`, onto the mesh axes.

`Trainer` uses these settings to initialize, place and update the state. XLA compiles the collective operations needed for that placement. You can change the mesh without changing the model, objective or data.

![A mesh of 8 devices with fsdp=2 and tensor=2, so data=2. The batch's 16 rows split into 4 blocks over data and fsdp, each held by a tensor pair. An MLP kernel of shape (256, 1024) splits its rows over fsdp and its columns over tensor, and each block is held by one device of each data index.](../assets/mesh-light.svg)
![A mesh of 8 devices with fsdp=2 and tensor=2, so data=2. The batch's 16 rows split into 4 blocks over data and fsdp, each held by a tensor pair. An MLP kernel of shape (256, 1024) splits its rows over fsdp and its columns over tensor, and each block is held by one device of each data index.](../assets/mesh-dark.svg)

`docs/assets/figures.py` computes the figure from the placements returned by `MeshSpec.build`, `Layout().shardings` and `batch_shardings`.

## Example

This example trains a small decoder on eight simulated CPU devices. Set `--xla_force_host_platform_device_count` before importing JAX.

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import itertools

import jax
import numpy as np
import optax

from dew import Dataset, Trainer
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec

model = CausalTransformer(vocab_size=512, emb_features=256, num_layers=2,
                          num_heads=4, mlp_features=1024, max_seq_len=64)
rows = np.random.default_rng(0).integers(0, 512, (16, 65), dtype=np.int32)
data = Dataset(train=lambda partition: itertools.repeat({"text": rows}), val=None,
               records=16, batch=16)
trainer = Trainer(LMObjective(model, seq_len=64), optax.adamw(1e-3), key=jax.random.key(0),
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

Each device holds a `(128, 512)` block of the `(256, 1024)` kernel. Its rows split over the two `fsdp` devices; its columns split over the two `tensor` devices. The kernel's Adam moments use the same placement.

## Mesh

Each `MeshSpec` field sets the size of one axis. The remaining devices form the `data` axis. If the product of the fields does not divide the device count, `MeshSpec.build` raises `LayoutRefused`.

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

`MeshSpec.microbatches` sets the number of pipeline microbatches, described below. `MeshSpec.replicas` sets the number of host groups across which the data axis is split. See [Multiple hosts](../guides/multi-node.md).

## Parameter placement

Modules declare logical axes such as `embed`, `mlp`, `heads`, `kv`, `vocab` and `exp`. `Layout.rules` maps each name to mesh axes in precedence order. The default table, `dew.nn.sharding.DEFAULT_RULES`, assigns large dense dimensions to `fsdp` and expert dimensions to `expert`.

With `MeshSpec(tensor=N)`, Dew also splits the dimensions used by Megatron's tensor parallelism. These are the MLP hidden width, query and key-value heads, the attention width read by `o_proj`, and the vocabulary. The residual width stays whole over `tensor`. If an axis size does not divide a dimension, Dew tries the next rule or leaves the dimension whole.

| `Layout` field | Default | Meaning |
|---|---|---|
| `rules` | `DEFAULT_RULES` | Logical axis name to mesh axes, in precedence order |
| `min_shard` | `2 ** 16` | Parameters with fewer elements stay replicated |
| `tolerance` | `0.02` | The fraction of shardable parameter elements that may stay replicated before `Layout.check` raises |
| `host` | `()` | Train-state fields kept in pinned host memory between steps: `"opt_state"`, `"ema"`, or `"variables"` for a CPU-owned step |
| `host_parameters` | `()` | Globs of variables an inference placement keeps in pinned host memory |

For a parameter with no declared path, Dew splits the largest dimension divisible by the `fsdp` size. If more than `tolerance` of the shardable elements stay replicated, `Layout.check` raises `LayoutRefused` and lists the largest replicated parameters. Raising `tolerance` permits more replication; it does not improve placement. Dew also refuses rules that put parameters on `data`, `sequence` or `stage`.

Optimizer moments and EMA copies have the same shapes and placement as their parameters. On restore, Dew builds a template for the requested layout. This lets you resume on a different mesh, provided the parameter names and shapes match the saved state.

## Activation placement

The layout table also places activations. Its `activation_` names assign batch rows to `data`, `expert` and `fsdp`, and token positions to `sequence`. Activation heads, MLP hidden width and vocabulary split over `tensor`.

The model constrains the residual stream, attention heads and MLP hidden width with `dew.nn.sharding.constrain`. `Trainer` compiles the step under `flax.linen.logical_axis_rules(layout.rules)`. Changing a rule therefore changes both parameter and activation placement.

Without these constraints, GSPMD chooses activation placement from the surrounding weights. In the unconstrained runs, it split the residual width on an fsdp mesh and all-reduced every projection's partial products. On a tensor mesh, it gathered every weight as fsdp does. Dew keeps batch rows whole over `tensor`, so every tensor shard reads the rows needed for its part of the width, as in Megatron. If an axis size does not divide a dimension, Dew tries the next rule or leaves the dimension whole.

Some projections read the whole residual without splitting their output width over `tensor`. These include multi-head latent attention's query and key-value down-projections, and DeepSeek-V4's query latent and shared key head. Computing them over every token on every tensor shard repeated work. At DeepSeek-V3's shape with `tensor=8`, a step used 1.133 times one device's matmul FLOPs.

When the link is fast enough, each tensor shard computes these projections on its own tokens. The input placement is `SPREAD`: its `activation_spread` rule splits positions over both the tensor and sequence axes. Dew then gathers the latents for the up-projections split by head. This saves (T - 1)/T of the projections' FLOPs. It adds gathers of the latents and residual gradient, and a sum of projection weight gradients over the tensor axis.

During tracing, `dew.nn.sharding.down_projection` compares transfer time with the time saved at the device's peak compute rate. It spreads only when the link can move the extra bytes within that time. The peak rate is the fastest the device can run the saved FLOPs, so at any real utilisation they take at least that long. The trainer measures the link once per mesh with a tensor-axis all-gather (`dew.training.distributed.link_bandwidth`). Every process uses the pool's lowest result so they compile the same program. For each measured axis, `StepCompiled` records that result and whether a projection spread.

At DeepSeek-V3's widths in bf16, with 16384 tokens per microbatch, an RTX 3090 needs 20.3 GB/s. On 4x RTX 3090 (PCIe 3.0, one host), the NVLink pair's all-gather moved 31.0 GB/s, enough to spread. The PCIe pair's 5.8 GB/s was not enough. An H100 SXM needs 283 GB/s, below NVLink 4's nominal 450 GB/s per direction. No spread step has been timed on NVLink or a TPU.

A CPU mesh always spreads. A device absent from the peak table in `dew.telemetry.instrumentation` never does. To compute these projections over every token regardless of the link, map `activation_spread` to the sequence axis alone.

Cross-attention can repeat work on the sequence axis too. For a context length such as SD's 77 text tokens, the axis size does not divide the length. Every sequence shard then projects the whole context into keys and values. On a CPU mesh with `sequence=4`, this made the conditional UNet's step use 1.57 times one device's FLOPs.

`dew.nn.sharding.split_positions` makes the same transfer-time comparison using the measured sequence-axis link. When the link is fast enough, each shard projects its share of positions. Zero rows pad the context to a multiple of the shard count. The split adds a sum of projection weight gradients, which requires many bytes relative to the FLOPs saved. A GPU link rarely meets that bound.

At SDXL's widths, with 8 rows per sequence group and no head sharding, an H100 would need 2.47 TB/s. This estimate subtracts the extra work on padded positions from the savings. Each sequence shard therefore projects the whole context in that configuration, repeating compute to avoid the collectives. The link calculation uses the output head width on each device under the active layout rules.

Cross entropy scores each device's own tokens in a `shard_map`. It maps over the axes holding the tokens, plus any other axis whose size divides the token count. Every device holds the whole head. Only the head gradient and loss sums cross devices.

## Host memory

`Layout(host=("opt_state", "ema"))` keeps optimizer state and EMA weights in pinned host memory between steps. Each compiled step fetches them to the device, updates them and writes them back. The values match a device-only layout, with less device memory used between steps. Checkpoints save and restore this placement. The transfers add time to every step, so measure step time with and without them.

Training waits for a checkpoint of host-resident state to finish writing. For device arrays, Orbax first makes a host copy and writes it asynchronously. For arrays already in pinned host memory, it reads their existing buffer. The next step donates that buffer, and a GPU refuses donation while the write is reading it. Waiting avoids an extra copy of the state.

Passing Orbax a copy would let the next step start after the copy finishes. It would also hold a second copy in pinned memory until the write finishes. For fp32 Adam moments and EMA, that costs 12 bytes per parameter, or 84 GB at 7B parameters. A host using this placement may have no room for that copy.

On an RTX 4080 host with a consumer NVMe drive (657 MB/s direct writes), writing a 6 GiB host-resident state took 18.4 s. Copying it would have taken 3.6 s and raised peak memory by 8.4 GiB. At 7B parameters, the estimated cost on that drive is about 4 minutes per checkpoint, compared with 20 to 50 s for a copy. A faster drive reduces the write time proportionally. Choose `checkpoint_every` with these costs in mind. Orbax's write path used about 0.9 bytes of transient memory per byte written on that host, regardless of whether it received a copy.

For inference, `Layout(host_parameters=("params/layers_*",))` keeps a root decoder stack in pinned host memory. Each model declares its physical stacks with `DecoderBank`. A bank holds its namespace within each variables collection and its `StackView`.

`MultimodalTransformer` puts its decoder under `language_model`. `DiffusionGemma` declares `text` once for the scope shared by its encoder and decoder. To select a nested stack, use a path such as `params/language_model/layers_*` or `params/text/layers_*`.

A scanned decoder declares one bank per run of like layers; an unscanned decoder declares one per layer. Both use the same fetch loop, which holds the current and next layer. A `LayerBanks` source's `place(model, layout=...)` reads each physical bank once. It rejects different views of the same namespace.

GPU peak-memory bounds and transfer overlap with compute have not been measured. Selected leaves keep their FSDP and tensor PartitionSpecs. External cache arrays stay on the device at their logical per-layer paths. `Trainer.place` rejects host-parameter layouts because they have no backward-pass or optimizer implementation.

On a TPU, the layer axis is major-most in a pinned bank's physical memory. A fetch keeps that axis during the copy to the device, then squeezes it on the device. This avoids incompatible host tile bitcasts and still transfers one layer at a time. Bank sources must honor any JAX `Format` supplied in their placement. Both built-in sources do.

`dew.inference.LayerBanks` has two source adapters:

- `CheckpointBanks` reads Dew run checkpoints bank by bank.
- `HeldBanks` borrows an existing variables tree without donating or deleting it. The whole source stays in memory while the destination banks fill up, so it can hold two copies of the model at once.

Both adapters read the entry leaves outside the declared stacks. For bank reads, they accept namespaces relative to a collection. Media, embeddings and heads remain entries even when they share a parent module with a decoder. Each entry or bank finishes placement before the next read starts.

`CausalTransformer(bank_layers=N)` limits the size of each bank when it is built. It does not cap the pinned allocator or free the source's storage. `StackView` exposes logical `layers_N` paths for saving and export. The synthetic generation example is only in `tools/benchmark_host_offload.py`.

Published checkpoints load onto a mesh leaf by leaf. This loader does not stream banks into pinned host memory. `Pretrained.load(source, mesh=MeshSpec(...), layout=Layout(...))`, also used by `dew.pipeline(source)`, avoids building the translated model on the host.

For each decoder leaf, it records which memory-mapped tensors to read, any transpose or expert stacking, and the storage dtype. `jax.make_array_from_callback` reads, casts and transposes only the part held by each device. The mapped pages are released after the leaf is placed. The host holds at most one device shard of one leaf at a time. Towers, projectors and dequantized quantized tensors are still built whole on the host.

On one RTX 4080, loading Qwen3-0.6B in float32 this way peaked at 3.9 GB RSS. Loading first and placing afterward peaked at 6.2 GB. On an 8-device CPU mesh, the figures were 4.0 GB and 5.9 GB; the placed model itself accounted for 2.4 GB. For Qwen3.8-27B (55.6 GB in bf16) on an 8-device mesh, the leaf recipes cover 54.6 GB of the checkpoint. The largest single read is 0.32 GB, each device holds 6.95 GB, and the vision tower built on the host is 0.92 GB.

Host-parameter placement supports only decoder stack variables. Dew rejects selections that include an embedding table, head or prediction depth. Matching leaves in every layer of a bank must share a memory kind and PartitionSpec. Dew also rejects the stage pipeline and training through a banked store.

In JAX 0.11.1, reverse-mode autodiff through the fetch loop does not lower. Its transpose requires a `dynamic_update_slice` with operands in different memory spaces. A small forward-mode probe matches resident placement. Backward re-fetch and training remain unverified by that probe.

Each forward pass copies all selected host-resident weights to the device, including on every decode step. PCIe transfers can take most of the time per token. `tools/benchmark_host_offload.py` reports local addressable-shard bytes, device and host compiled-memory fields, process RSS and execution times. Dividing bytes per token by decode latency gives an end-to-end effective rate, which includes more than PCIe transfers.

A synthetic BF16 bank load of 18.00 GiB (144 layers, width 2048, `bank_layers=8`) reported 18.74 GiB final RSS and 37.9 GiB peak RSS. The saved probe synchronized every bank, so queued asynchronous work alone cannot explain the peak. These numbers do not distinguish temporary storage, allocator reservation and duplicate driver mappings.

There is no completed GPU generation or throughput result for a model larger than device memory. Further allocator investigation needs a small process tree with a hard memory cap, swap accounting and RSS/PSS measurements. The earlier oversized runs with an RSS watchdog were unsafe for this purpose.

## Rematerialization

`CausalTransformer(remat=...)` sets which activations to keep and which to recompute during the backward pass. `"full"` keeps only the block inputs. The default, `None`, recomputes nothing.

The other policies in `dew.nn.backbones.decoder_block.REMAT_POLICIES` are `minimal`, `minimal_with_context`, `save_dot_except_mlp`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_qkv_proj`, `save_out_proj`, `minimal_offloaded` and `qkv_proj_offloaded`. Each saves selected projection outputs (`q_proj`, `k_proj`, `v_proj`, `kv_proj`, `context`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) or offloads them to host memory. The policies trade computation for memory as MaxText's recipes of the same names do.

You can also supply your own lists, such as `{"save": ["q_proj", "k_proj", "v_proj"], "offload": ["gate_proj", "up_proj"]}`. Every policy trains the same model. To measure residual and compiler memory for a configuration, run `tools/benchmark_decoder_remat.py --remat <name>`.

## Batches

`Dataset.batch` is the global batch size. `Dataset.train(partition)` and `Dataset.val(partition)` read one share of each global batch. `DataPartition.of(mesh)` returns this process's `DataPartition(index, count, readers, reader)`.

When a `stage` or `sequence` axis spans processes, several processes can hold the same row shard. They all read the same share and count as its `readers`. A loader selects records with `index::count`, so global batch *k* contains the same records at every process count. `shard_batch` assembles the shares into global arrays of whole rows.

For custom data, use the supplied partition to read only your process's share. Readers of the same share must return the same records in the same order. Sources such as `UrlStream`, which return rows in fetch-completion order, refuse partitions with more than one reader. Repeating the full dataset independently on every process changes the training distribution.

The placement helper treats rank-two and rank-three arrays as sequences. It can split their second dimension when its length is divisible by the sequence factor. Image and video tensors keep their non-batch dimensions whole. Check custom rank-three data yourself, since rank alone cannot identify token positions.

## Sequence parallelism

`MeshSpec(sequence=N)` splits each sequence's token positions over N devices. Attention uses one of two exact exchanges.

The all-to-all follows DeepSpeed Ulysses. Each device exchanges its slice of positions for a slice of heads, attends the whole sequence, and exchanges back. No device holds a whole key or value tensor. This path requires the query head count to be divisible by `tensor` times `sequence`. Query and key lengths must also be divisible by `sequence`.

Other calls gather the whole keys and values while keeping queries split. The gather supports any head count and key length. When both paths qualify, a causal, windowed or masked call uses all-to-all so the kernel can skip masked blocks. The gather uses an explicit mask whose blocks the kernels do not skip. For causal attention it took two to four times as long. Unmasked calls use whichever exchange sends fewer bytes.

On the gather path, causal or masked calls stripe query chunks to balance the work. Their sequence length must be divisible by twice the number of sequence shards. [Multiple hosts](../guides/multi-node.md) gives the measurements and byte counts.

If cuDNN accepts the window as its window flag, attention uses all-to-all and skips blocks outside the window. Local attention uses banded computation for packed documents or sinks on cuDNN, validity masks, windows on XLA, and chunks.

When a banded layer's span fits in one shard's slice, it needs no all-to-all or gather. Each device attends its own positions. One shift supplies its first block with the previous device's last `window` keys and values, including positions, segment IDs and validity. A wider span uses all-to-all as a dense call.

A token window has `seq_len + 1` IDs, and the model reads `seq_len` positions. Check the model length as well as the shape of the batch array. Packed segments, windows and rotary positions must stay aligned.

Mamba-2 layers split sequence positions too, so Mamba-2/attention hybrids train under `MeshSpec(sequence=N)`. Each device convolves its own slice, using the previous device's last `conv_kernel - 1` tokens. It first scans from a zero state.

A parallel prefix scan then computes the state left by earlier slices. Each device exchanges its total decay and final fp32 state, shaped `[batch, heads, head_dim, state_size]`, in `ceil(log2 N) + 1` shifts. It adds the decayed contribution of that earlier state to each output. Packed documents reset both state and convolution at every segment change, including slice boundaries.

The sequence length must be divisible by `N`. Each slice needs at least `conv_kernel - 1` tokens. The Mamba-2 scan is not split over `tensor`, so every tensor shard scans the whole width.

Cached autoregressive generation needs `sequence=1`. A mesh that trains with sequence parallelism may still be unable to run the decode cache.

## Layer scan and pipeline

`scan_layers=True` groups compatible consecutive decoder layers into a Flax scan while keeping stored variables per layer. Scanning can reduce compile time, at the cost of stacking and loop overhead. Compile time can still grow with model size, and steps may be slower. Measure it on the model size and backend you plan to use.

Regardless of `scan_layers`, `init` draws each run of two or more like layers under one scan. Its program therefore contains one draw per run. `bank_layers` splits these runs and affects the draw. The distribution is unchanged, but a given seed produces different weights from Dew versions that initialized each layer separately. On a stage mesh or with `decode=True`, initialization still uses the plain loop.

`MeshSpec(stage=N, microbatches=M)` enables a GPipe-style pipeline. The layer pattern must repeat across stages and split evenly between them. Only the decoder's layer stack is pipelined. Embeddings, the output head and other operations outside that stack stay unsplit over stages.

For a model without a pipeline, such as a DiT, the trainer refuses a stage axis because each stage would repeat the whole step. It raises `dew.nn.sharding.LayoutRefused`, a `ValueError` naming the axis and a supported layout. Any other error on a layout is a defect.

Microbatch m takes rows m, m + M, m + 2M and so on, retaining their batch-shard placement. M must divide the row count on each device. For example, a batch of 8 rows over 4 row shards has 2 rows per device. With four microbatches, some devices would have no rows for a microbatch and would repeat another microbatch's work. A stage × fsdp step would then use 1.43 times one device's matmul FLOPs, although the pipeline bubble accounts for only 1.25.

Dew refuses that schedule and suggests a microbatch count that divides each device's row count, plus a batch size for the requested count. With 16 rows, the same layout uses 1.20 times one device's matmul FLOPs.

Master parameters are stored replicated over `stage`. Each step builds a view partitioned by stage, so include that copy and its communication when estimating memory savings. Some mixed layer patterns and KV-sharing configurations are unsupported. Cached decoding needs `stage=1`.

Dew does not implement a 1F1B schedule.

## Kernels and precision

With `"auto"`, attention uses the reference path for arithmetic unavailable in fused kernels. This includes matmul precision above default, softmax outside fp32, or a compute dtype different from the inputs'. A bf16 call on a GPU older than sm80 also uses the reference path.

Other GPU calls use cuDNN when it supports their shape, dtype and requested features, or XLA otherwise. TPU calls use the Pallas splash kernel when they qualify: bf16 or fp32, query and key lengths that are multiples of 128 and at least 512, a supported mask, no additive bias, and whole sequences at the kernel. The sequence-parallel all-to-all supplies whole sequences; the key/value gather does not. Other TPU calls use XLA. Packed masks and optional attention features can change the kernel and memory use.

Qwix quantization is an optional, experimental training path with int8 and fp8 matmuls. It keeps full-precision master parameters and changes the numerics. In the RTX 4080 measurements on [Performance](../performance.md), fp8 did not make steps faster at the sizes tested.

## Several hosts

Initialize the JAX process pool before creating device arrays or models. The built-in recipes do this first. Each host needs the same software, data access and a reachable coordinator.

`dew launch` starts a pool over ssh or under Slurm. `MeshSpec(replicas=N)` keeps `fsdp` inside a node and splits the data axis across nodes. [Multiple hosts](../guides/multi-node.md) explains launch and layout, including a one-machine rehearsal. [Cloud TPUs](../tpu.md) covers TPU provisioning.

The local process-pool tests run real `jax.distributed` processes. They do not cover network failures, remote storage, TPU collectives or cluster preemption.

## Checking a distributed run

For each planned placement, compare a small global computation with a reference run. Check the loss and gradients, record identity, optimizer and EMA state, and save/restore behavior. Record compiler and backend versions and the tolerances used. Device count and finite loss alone do not establish equivalent training.

The CPU tests on global arrays cover loss normalization with unequal masks and restart partway through an accumulation window. [Checkpoints](../guides/checkpoints.md) and [Evaluation and tracking](../guides/evaluation.md) describe the guarantees for resumed runs and reported metrics.
