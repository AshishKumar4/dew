# Distributed training

This page assumes you can train a model on one device and know JAX arrays and Flax variables. Work through [the single-device tutorial](../getting-started.md) before you change placement. You should know your global batch size and the shapes of your model's parameters.

A mesh gives names to groups of devices. A layout maps parameter dimensions to those mesh axes. `Trainer` uses the mesh and the layout to initialize and update sharded state, and JAX compiles the collective communication this needs.

## Inspect a single-device mesh

This example runs on any device, CPU included, and allocates no model:

```python
import jax
from dew.training import MeshSpec, build_mesh

mesh = build_mesh(MeshSpec())
print("Visible devices:", len(jax.devices()))
print("Mesh dimensions:", dict(mesh.shape))
assert mesh.size == len(jax.devices())
```

`MeshSpec()` puts all visible devices on data parallelism. The other fields reserve a factor of the device count for their own axis, and data parallelism gets what is left. The product of the fields must divide the number of devices.

| Mesh axis | Role |
|---|---|
| `data` | Partition batch rows while replicating model parameters |
| `fsdp` | Partition parameter dimensions selected by the layout |
| `expert` | Partition the expert dimension in sparse layers |
| `tensor` | Partition widths that layout rules explicitly assign to tensor parallelism |
| `sequence` | Partition token positions for sequence-parallel attention |
| `stage` | Place pipeline stages in the layer-stack execution |

On eight devices, `MeshSpec(fsdp=4)`, `MeshSpec(fsdp=2, expert=4)` and `MeshSpec(fsdp=4, sequence=2)` are valid configurations. Constructing one of these values tests nothing. Building and running its mesh needs enough visible devices.

## Describe parameter placement

Modules declare logical axes such as `embed`, `mlp`, `heads`, `kv`, `vocab` and `exp`. `Layout` maps those names to mesh axes. The default rules put many large dense dimensions on `fsdp` and expert dimensions on `expert`. Under `MeshSpec(tensor=N)` the widths of Megatron's split take the tensor axis as well: the mlp's hidden width, the query and key-value heads, the attention width `o_proj` reads, and the vocabulary. The residual width stays whole on the tensor axis.

The same table places activations. Its `activation_` names put a batch's rows on the `data`, `expert` and `fsdp` axes, its positions on `sequence`, and the heads, the mlp hidden width and the vocabulary of an activation on `tensor`. The model pins the residual stream, the attention's heads and the mlp's hidden width with `dew.nn.sharding.constrain`, and the trainer compiles its step under `flax.linen.logical_axis_rules(layout.rules)`, so a rule you change moves the activations with the parameters. Without those pins GSPMD chooses each activation's placement from the weights around it. On an fsdp mesh it split the residual width and all-reduced every projection's partial products, and on a tensor mesh it gathered every weight, as fsdp does. A batch's rows never split over `tensor`: every tensor shard reads the rows it computes its part of the width for, as in Megatron. A name whose axes do not divide a dimension falls back to its next rule, or stays whole.

The cross entropy scores each device's own tokens: it runs in a `shard_map` over the axes that hold the tokens, plus any other axis the token count divides, with the head whole on every device. Only the head's gradient and the loss's sums cross devices.

Parameters smaller than `min_shard` elements stay replicated. `Layout.check` raises when too many of the parameters that should be sharded end up replicated. If it raises, look at the parameter paths and dimensions it lists before you change the rules. Raising `tolerance` only turns the check off. It does not make the placement any better.

Optimizer moments and EMA variables take their placement from the matching parameter paths. When a checkpoint is restored, Dew builds a restore template for the layout you ask for. The model's shapes and parameter names still have to match the saved state.

`Layout(host=("opt_state", "ema"))` keeps the optimizer state and the EMA copy in pinned host memory between steps. The compiled step fetches them to the device, runs the same update and writes them back. The values are the same as with a device-only layout. Only the device memory they take up between steps changes. Checkpoints save and restore this placement. Parameters stay on the device. The transfer adds time to every step, so measure step time with and without it.

`Layout(host_parameters=("params/layers_*",))` keeps a root decoder stack in pinned host memory for inference. Each model declares every physical stack with a `DecoderBank`, which holds a namespace below every variables collection and its `StackView`. `MultimodalTransformer` puts its decoder under `language_model`. `DiffusionGemma` declares `text` once for the scope its encoder and decoder share. To select a nested stack, use a path such as `params/language_model/layers_*` or `params/text/layers_*`.

A scanned decoder declares one bank per run of like layers. An unscanned decoder declares one bank per layer. Both stream through the same fetch loop. `dew.inference.host_banked(model, source, layout=...)` reads each physical bank once and rejects two different views of the same namespace. The fetch loop holds the current layer and the next one. I have not yet established GPU peak-memory bounds or whether the transfers overlap with compute. Selected leaves keep their FSDP and tensor PartitionSpecs. External cache arrays stay on the device under their logical per-layer paths. `Trainer.place` rejects host-parameter layouts, because the backward pass and optimizer paths for them do not exist.

`dew.inference.LayerBanks` has two adapters for real use:

- `CheckpointBanks` reads Dew run checkpoints bank by bank.
- `HeldBanks` borrows an existing variables tree without donating or deleting it. The whole source stays in memory while the destination banks fill up, so it can hold two copies of the model at once.

Both read the entry leaves outside the declared stacks, and both accept namespaces relative to a collection for bank reads. Media, embeddings and heads stay entries even when they share a parent module with a decoder. Each entry and bank placement finishes before the next read starts. `CausalTransformer(bank_layers=N)` limits how large a bank is when it is built. It does not cap the pinned allocator or free storage that the source owns. `StackView` still exposes the logical `layers_N` paths for saving and export. Synthetic generation exists only in `tools/benchmark_host_offload.py`.

Loading a published checkpoint onto a mesh streams it leaf by leaf, but it does not stream banks into pinned host memory. `load_pretrained(source, mesh=MeshSpec(...), layout=Layout(...))`, which `dew.pipeline(source)` calls, never builds the translated model on the host. Each decoder leaf is a recipe over the memory-mapped checkpoint: which stored tensors, whether to transpose, whether to stack experts, and the storage dtype. `jax.make_array_from_callback` reads only the part each device holds, casting and transposing that part alone. The mapped pages are released once the leaf lands. The host holds at most one device shard of one leaf at a time, plus the towers, projectors and any dequantized quantized tensors, which are still built whole. Qwen3-0.6B loaded in float32 peaks at 3.9 GB RSS this way, against 6.2 GB for loading first and placing after, on one RTX 4080. On an 8-device CPU mesh the figures are 4.0 GB against 5.9 GB, and 2.4 GB of that is the placed model itself. For Qwen3.8-27B (55.6 GB in bf16) on an 8-device mesh, the recipes cover 54.6 GB of the checkpoint, the largest single read is 0.32 GB, each device holds 6.95 GB, and the vision tower built on the host is 0.92 GB.

Host-parameter placement works only for decoder stack variables. Dew rejects a selection that includes an embedding table, a head or a prediction depth. Matching leaves of every layer in a bank must have the same memory kind and PartitionSpec. The stage pipeline and training through a banked store are rejected. In JAX 0.11.1, reverse-mode autodiff through the fetch loop does not lower: its transpose asks for a `dynamic_update_slice` whose operands sit in different memory spaces. A small forward-mode probe agrees with resident placement. That probe does not show that backward re-fetch or training works.

Every host-resident weight you select is copied to the device on each forward pass, including every decode step. PCIe transfers can take most of the time per token. `tools/benchmark_host_offload.py` reports local addressable-shard bytes, the device and host compiled-memory fields, process RSS and execution times. Its bytes-per-token divided by decode latency is an end-to-end effective rate. It is not a measurement of PCIe bandwidth alone.

An earlier synthetic BF16 bank load of 18.00 GiB (144 layers, width 2048, `bank_layers=8`) reported 18.74 GiB final RSS and 37.9 GiB peak RSS. The saved load probe synchronized every bank, so a backlog of asynchronous work alone cannot explain that peak. The numbers do not separate temporary storage, allocator reservation and duplicate driver mappings. I have no completed result for generating on the GPU with a model larger than device memory, and no throughput result. Looking further into the allocator needs a small process tree with a hard memory cap, swap accounting and RSS/PSS measurements. The earlier oversized runs with an RSS watchdog are not a safe way to do it.

## Recompute block activations

`CausalTransformer(remat=...)` recomputes each decoder block in the backward pass instead of keeping its activations. The value is a policy. `"full"` keeps only the block inputs. The other names in `dew.nn.backbones.causal_transformer.REMAT_POLICIES` are `minimal`, `minimal_with_context`, `save_dot_except_mlp`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_qkv_proj`, `save_out_proj`, `minimal_offloaded` and `qkv_proj_offloaded`. Each keeps some of the named projection outputs (`q_proj`, `k_proj`, `v_proj`, `kv_proj`, `context`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) or offloads them to host memory. They trade recompute time for memory the same way MaxText's recipes of the same names do. You can also pass a record with your own lists, such as `{"save": ["q_proj", "k_proj", "v_proj"], "offload": ["gate_proj", "up_proj"]}`. The default, `None`, recomputes nothing. Every policy trains the same model. `tools/benchmark_decoder_remat.py --remat <name>` reports the residual and compiler memory of one configuration.

## Feed global batches

`Dataset.batch` is the global batch. `Dataset.train(partition)` and `Dataset.val(partition)` open a stream over one share of every global batch, and `dew.training.data_partition(mesh)` says which share a process reads: a `DataPartition(index, count, readers)`. The processes whose devices hold the same rows read the same share. Under `stage` or `sequence` axes that span processes, several processes hold one row shard, so they read one share and the partition counts them as `readers`. A loader cuts its records `index::count`, so global batch k holds the same records at every count. `shard_batch` assembles each share into the global arrays, whole rows from each share.

For custom data, take the partition you are handed and read that share alone. Two readers of one share must read the same records in the same order. A source that returns rows in whatever order its fetches finish, such as `ImageStream`, refuses a partition with more than one reader. If each process repeats the full dataset on its own, the run trains on a different distribution.

The placement helper treats rank-two and rank-three arrays as sequences. It can split their second dimension when that dimension divides by the sequence factor. Image and video tensors keep their non-batch dimensions in that helper. Check custom rank-three data yourself: the rank of an array does not tell the helper whether its second dimension really is token positions.

## Use sequence-parallel attention

`MeshSpec(sequence=N)` splits the token positions of every sequence over N devices. Each attention call picks one of two exact exchanges. The all-to-all follows DeepSpeed Ulysses: each device trades its slice of the positions for a slice of the heads, attends the whole sequence and trades back, so no device holds a whole key or value. A call takes it when its query heads divide by `tensor` times `sequence`, its query and key lengths both divide by `sequence`, and it moves fewer bytes than the gather, which is the case for causal attention. Every other call gathers the whole keys and values beside split queries, which takes any head count and any key length. A causal or masked call on the gather path stripes query chunks to balance the work, so its sequence length must divide by twice the number of sequence shards. [Training on several nodes](../guides/multi-node.md#split-long-sequences) gives the byte count.

A token window has `seq_len + 1` IDs, and the model reads `seq_len` positions. Check the model length as well as the shape of the batch array. Packed segments, windows and rotary positions must stay aligned.

Mamba-2 layers split the sequence the same way, so a hybrid of Mamba-2 and attention layers trains under `MeshSpec(sequence=N)`. Each device convolves its own slice, reading the last `conv_kernel - 1` tokens of the previous device's slice, and runs its scan from a zero state. It then gets the state the earlier slices leave by passing each slice's total decay and final state, `[batch, heads, head_dim, state_size]` in fp32, between devices in `ceil(log2 N) + 1` shifts (a parallel prefix scan), and adds that state's decayed contribution to each of its outputs. Packed documents reset the state and the convolution at every segment change, including a change that falls on a slice boundary. The sequence must divide by `N`, and each slice needs at least `conv_kernel - 1` tokens. The tensor axis does not split the Mamba-2 scan: every tensor shard runs the scan on the whole width.

Cached autoregressive generation needs `sequence=1`. A mesh that trains with sequence parallelism may still be unable to run the decode cache.

## Scan layers and use a pipeline

`scan_layers=True` groups compatible consecutive decoder layers into a Flax scan. The stored variables stay per layer. Scanning can cut compile time, but it adds stacking and loop overhead. It does not promise a flat compile time or faster steps. Measure it at the model size and on the backend you plan to use.

`MeshSpec(stage=N, microbatches=M)` turns on the GPipe-style pipeline. The layer pattern must repeat across stages and split evenly between them. Embeddings, the output head and other operations outside the layer stack are not split across pipeline stages. Only a decoder's layer stack pipelines: the trainer refuses a stage axis under a model that runs no pipeline, such as a DiT, because every stage would compute the whole step. Microbatch m takes rows m, m + M, m + 2M and so on, so each microbatch keeps its rows on the batch shards that hold them.

The stored master parameters are replicated over the stage axis, and each step builds a view that is partitioned by stage. Count that copy and its communication when you estimate memory savings. Some mixed layer patterns and KV-sharing configurations cannot use this pipeline. Cached decoding also needs `stage=1`.

Dew does not implement 1F1B. Earlier experiments tried one approach. They do not show that a compiled 1F1B design is impossible. It is still an open gap in the implementation and design.

## Select precision and kernels

On GPU, attention can use cuDNN when the shape, dtype, mask and requested features fit its supported path. Other calls use XLA. On TPU, attention uses Pallas. Packed masks and optional attention features can change which kernel runs and how much memory it uses.

Qwix quantization is an optional, experimental training path with int8 and fp8 computation. It keeps the master parameters but changes the numerics. In the local RTX 4080 measurements, fp8 did not make steps faster at the sizes tested. See the [performance measurements](../performance.md), and check accuracy and throughput on your own configuration.

## Run across hosts

Initialize the JAX process pool before you create any device arrays or models. The built-in recipes call their process setup function early. Each host needs compatible software, access to the data and a coordinator it can reach. `dew launch` starts a pool over ssh or under Slurm, and `MeshSpec(replicas=N)` keeps fsdp inside a node while the data axis spans the nodes. [Training on several nodes](../guides/multi-node.md) covers both and how to rehearse them on one machine. Test remote checkpoint storage and iterator partitioning separately.

The local process-pool tests run real `jax.distributed` processes. They do not test network failures, remote storage, TPU collectives or cluster preemption. [TPU setup](../tpu.md) describes provisioning, which can cost money on the cloud.

## Verify a distributed run

Compute a small global operation under each placement you plan to use and compare the results. Check the loss and gradients, record identity, optimizer and EMA state, and saving and restoring. Write down the compiler and backend versions and the tolerances you used. A device count and a finite loss do not show that two runs train the same way.

CPU tests on global arrays cover loss normalization with unequal masks and restarting partway through a window. Read the [checkpoint guide](../guides/checkpoints.md) and the [evaluation limits](../guides/evaluation.md) to see what a resumed run and a reported metric each guarantee.
