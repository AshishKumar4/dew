# Distributed training

> An AI assistant maintains this document. It is presented as-is.

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

Modules declare logical axes such as `embed`, `mlp`, `heads`, `kv`, `vocab` and `exp`. `Layout` maps those names to mesh axes. The default rules put many large dense dimensions on `fsdp` and expert dimensions on `expert`.

Parameters smaller than `min_shard` elements stay replicated. `Layout.check` raises when too many of the parameters that should be sharded end up replicated. If it raises, look at the parameter paths and dimensions it lists before you change the rules. Raising `tolerance` only turns the check off. It does not make the placement any better.

Optimizer moments and EMA variables take their placement from the matching parameter paths. When a checkpoint is restored, Dew builds a restore template for the layout you ask for. The model's shapes and parameter names still have to match the saved state.

`Layout(host=("opt_state", "ema"))` keeps the optimizer state and the EMA copy in pinned host memory between steps. The compiled step fetches them to the device, runs the same update and writes them back. The values are the same as with a device-only layout. Only the device memory they take up between steps changes. Checkpoints save and restore this placement. Parameters stay on the device. The transfer adds time to every step, so measure step time with and without it.

`Layout(host_parameters=("params/layers_*",))` keeps a root decoder stack in pinned host memory for inference. Each model declares every physical stack with a `DecoderBank`, which holds a namespace below every variables collection and its `StackView`. `MultimodalTransformer` puts its decoder under `language_model`. `DiffusionGemma` declares `text` once for the scope its encoder and decoder share. To select a nested stack, use a path such as `params/language_model/layers_*` or `params/text/layers_*`.

A scanned decoder declares one bank per run of like layers. An unscanned decoder declares one bank per layer. Both stream through the same fetch loop. `dew.inference.host_banked(model, source, layout=...)` reads each physical bank once and rejects two different views of the same namespace. The fetch loop holds the current layer and the next one. I have not yet established GPU peak-memory bounds or whether the transfers overlap with compute. Selected leaves keep their FSDP and tensor PartitionSpecs. External cache arrays stay on the device under their logical per-layer paths. `Trainer.place` rejects host-parameter layouts, because the backward pass and optimizer paths for them do not exist.

`dew.inference.LayerBanks` has two adapters for real use:

- `CheckpointBanks` reads Dew run checkpoints bank by bank.
- `HeldBanks` borrows an existing variables tree without donating or deleting it. The whole source stays in memory while the destination banks fill up, so it can hold two copies of the model at once.

Both read the entry leaves outside the declared stacks, and both accept namespaces relative to a collection for bank reads. Media, embeddings and heads stay entries even when they share a parent module with a decoder. Each entry and bank placement finishes before the next read starts. `CausalTransformer(bank_layers=N)` limits how large a bank is when it is built. It does not cap the pinned allocator or free storage that the source owns. `StackView` still exposes the logical `layers_N` paths for saving and export. Synthetic generation exists only in `tools/benchmark_host_offload.py`.

Loading a published checkpoint does not stream banks yet. `load_pretrained(..., dtype="bfloat16")` picks the compute precision, but the loader keeps float32 weights, and the source pipeline loads the full tree before placing it on the mesh. Dividing the checkpoint size by the device count does not bound that temporary storage. Measurements on synthetic BF16 banks say nothing about how loading a published checkpoint scales.

Host-parameter placement works only for decoder stack variables. Dew rejects a selection that includes an embedding table, a head or a prediction depth. Matching leaves of every layer in a bank must have the same memory kind and PartitionSpec. The stage pipeline and training through a banked store are rejected. In JAX 0.11.1, reverse-mode autodiff through the fetch loop does not lower: its transpose asks for a `dynamic_update_slice` whose operands sit in different memory spaces. A small forward-mode probe agrees with resident placement. That probe does not show that backward re-fetch or training works.

Every host-resident weight you select is copied to the device on each forward pass, including every decode step. PCIe transfers can take most of the time per token. `tools/benchmark_host_offload.py` reports local addressable-shard bytes, the device and host compiled-memory fields, process RSS and execution times. Its bytes-per-token divided by decode latency is an end-to-end effective rate. It is not a measurement of PCIe bandwidth alone.

An earlier synthetic BF16 bank load of 18.00 GiB (144 layers, width 2048, `bank_layers=8`) reported 18.74 GiB final RSS and 37.9 GiB peak RSS. The saved load probe synchronized every bank, so a backlog of asynchronous work alone cannot explain that peak. The numbers do not separate temporary storage, allocator reservation and duplicate driver mappings. I have no completed result for generating on the GPU with a model larger than device memory, and no throughput result. Looking further into the allocator needs a small process tree with a hard memory cap, swap accounting and RSS/PSS measurements. The earlier oversized runs with an RSS watchdog are not a safe way to do it.

## Recompute block activations

`CausalTransformer(remat=...)` recomputes each decoder block in the backward pass instead of keeping its activations. The value is a policy. `"full"` keeps only the block inputs. The other names in `dew.nn.backbones.causal_transformer.REMAT_POLICIES` are `minimal`, `minimal_with_context`, `save_dot_except_mlp`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_qkv_proj`, `save_out_proj`, `minimal_offloaded` and `qkv_proj_offloaded`. Each keeps some of the named projection outputs (`q_proj`, `k_proj`, `v_proj`, `kv_proj`, `context`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) or offloads them to host memory. They trade recompute time for memory the same way MaxText's recipes of the same names do. You can also pass a record with your own lists, such as `{"save": ["q_proj", "k_proj", "v_proj"], "offload": ["gate_proj", "up_proj"]}`. The default, `None`, recomputes nothing. Every policy trains the same model. `tools/benchmark_decoder_remat.py --remat <name>` reports the residual and compiler memory of one configuration.

## Feed global batches

`Dataset.batch` is the global batch. Each process supplies its own records and Dew assembles the global arrays. For custom data, check that the process slices do not overlap and come out the same every time. If each process repeats the full dataset on its own, the run trains on a different distribution.

The placement helper treats rank-two and rank-three arrays as sequences. It can split their second dimension when that dimension divides by the sequence factor. Image and video tensors keep their non-batch dimensions in that helper. Check custom rank-three data yourself: the rank of an array does not tell the helper whether its second dimension really is token positions.

## Use sequence-parallel attention

`MeshSpec(sequence=N)` splits the token positions of every sequence over N devices. `MeshSpec.sequence_exchange` picks how attention exchanges data between them. The default, `'all_to_all'`, follows DeepSpeed Ulysses: each device trades its slice of the positions for a slice of the heads, attends the whole sequence and trades back, so no device holds a whole key or value. The query heads must divide by `tensor` times `sequence`. `'all_gather'` gathers the whole keys and values beside split queries and takes any head count. Its masked paths stripe query chunks to balance the work, so the model's sequence length must divide by twice the number of sequence shards. [Training on several nodes](../guides/multi-node.md#split-long-sequences) compares the two.

A token window has `seq_len + 1` IDs, and the model reads `seq_len` positions. Check the model length as well as the shape of the batch array. Packed segments, windows and rotary positions must stay aligned.

Cached autoregressive generation needs `sequence=1`. A mesh that trains with sequence parallelism may still be unable to run the decode cache.

## Scan layers and use a pipeline

`scan_layers=True` groups compatible consecutive decoder layers into a Flax scan. The stored variables stay per layer. Scanning can cut compile time, but it adds stacking and loop overhead. It does not promise a flat compile time or faster steps. Measure it at the model size and on the backend you plan to use.

`MeshSpec(stage=N, microbatches=M)` turns on the GPipe-style pipeline. The layer pattern must repeat across stages and split evenly between them. Embeddings, the output head and other operations outside the layer stack are not split across pipeline stages.

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
