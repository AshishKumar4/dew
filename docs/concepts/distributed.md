# Distributed training

This page assumes you can train a model on one device and understand JAX arrays and Flax variables. Start with [the single-device tutorial](../getting-started.md) before changing placement. You should know your global batch size and model parameter shapes.

A mesh assigns names to groups of devices. A layout maps parameter dimensions to those mesh axes. `Trainer` uses the mesh and layout to initialize and update sharded state. JAX compiles the necessary collective communication.

## Inspect a single-device mesh

This complete example works on CPU and allocates no model:

```python
import jax
from dew.training import MeshSpec, build_mesh

mesh = build_mesh(MeshSpec())
print("Visible devices:", len(jax.devices()))
print("Mesh dimensions:", dict(mesh.shape))
assert mesh.size == len(jax.devices())
```

`MeshSpec()` assigns the visible devices to data parallelism. Additional fields reserve device factors for other axes; data parallelism uses the remaining factor. The product must divide the available device count.

| Mesh axis | Role |
|---|---|
| `data` | Partition batch rows while replicating model parameters |
| `fsdp` | Partition parameter dimensions selected by the layout |
| `expert` | Partition the expert dimension in sparse layers |
| `tensor` | Partition widths that layout rules explicitly assign to tensor parallelism |
| `sequence` | Partition token positions for sequence-parallel attention |
| `stage` | Place pipeline stages in the layer-stack execution |

On eight devices, examples of configurations are `MeshSpec(fsdp=4)`, `MeshSpec(fsdp=2, expert=4)`, or `MeshSpec(fsdp=4, sequence=2)`. Constructing such a value is not a distributed test; building and running its mesh requires enough visible devices.

## Describe parameter placement

Modules declare logical axes such as `embed`, `mlp`, `heads`, `kv`, `vocab`, and `exp`. `Layout` maps those names to mesh axes. Default rules place many large dense dimensions on `fsdp` and expert dimensions on `expert`.

The `min_shard` threshold keeps small parameters replicated. `Layout.check` detects excessive replication among parameters considered shardable. If it raises, inspect the listed parameter paths and dimensions before changing the rules. Increasing the tolerance only removes that check; it does not make the placement more efficient.

Optimizer moments and EMA variables inherit placement from the corresponding parameter paths. Persistent checkpoints use a restore template for the requested layout. A model's shape and parameter naming still need to match the saved state.

`Layout(host=("opt_state", "ema"))` keeps the optimizer state and the EMA copy in pinned host memory between steps. The compiled step fetches them to the device, runs the same update, and writes them back, so the values are identical to a device-only layout and only the device memory they occupy between steps changes; checkpoints save and restore the placement. Parameters stay on the device. The transfer costs time on every step, so measure a run's step time with and without it.

## Recompute block activations

`CausalTransformer(remat=...)` recomputes each decoder block in the backward pass instead of keeping its activations. The value is a policy: `"full"` keeps only the block inputs, and the other names in `dew.nn.backbones.causal_transformer.REMAT_POLICIES` (`minimal`, `minimal_with_context`, `save_dot_except_mlp`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_qkv_proj`, `save_out_proj`, `minimal_offloaded`, `qkv_proj_offloaded`) keep or offload to host memory the named projection outputs (`q_proj`, `k_proj`, `v_proj`, `kv_proj`, `context`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), trading recompute time for memory as MaxText's recipes of the same names do. A record such as `{"save": ["q_proj", "k_proj", "v_proj"], "offload": ["gate_proj", "up_proj"]}` names its own lists. `None`, the default, recomputes nothing. Every policy trains the same model; `tools/benchmark_decoder_remat.py --remat <name>` reports the residual and compiler memory of one configuration.

## Feed global batches

`Dataset.batch` describes the global batch. Each process supplies its local records, and Dew assembles global arrays. For custom data, validate that process slices are disjoint and deterministic. Repeating the full dataset independently on each process changes the effective training distribution.

The current placement helper treats rank-two and rank-three arrays as sequences and can split their second dimension when divisible by the sequence factor. Image and video tensors preserve non-batch dimensions in that helper. Custom rank-three data may need explicit scrutiny; shape rank alone does not establish semantic token positions.

## Use sequence-parallel attention

Sequence-parallel attention divides queries between devices and gathers the key/value context needed for those queries. Masked paths stripe query chunks to balance work. Their model sequence length must be divisible by twice the sequence-shard count.

A token window has `seq_len + 1` IDs, while the model consumes `seq_len` positions. Check the model length as well as the batch array shape. Packed segments, windows, and rotary positions must remain aligned.

Cached autoregressive generation currently requires `sequence=1`. A successful training run with sequence parallelism does not imply the same mesh can run its decode cache.

## Scan layers and use a pipeline

`scan_layers=True` groups compatible consecutive decoder layers into a Flax scan. The stored variables remain per-layer. Scanning can reduce compilation cost, but it also introduces stacking and loop overhead. It is not a guarantee of flat compilation time or faster steps. Measure the model size and backend you intend to use.

`MeshSpec(stage=N, microbatches=M)` enables the implemented GPipe-style pipeline. The layer pattern must repeat compatibly across stages and split evenly. Embeddings, the output head, and other operations outside the layer stack are not automatically pipeline-partitioned.

The stored master parameters replicate over the stage axis; the step constructs a stage-partitioned execution view. Account for the copy and communication cost when estimating memory savings. Some heterogeneous layer patterns and KV-sharing configurations cannot use this pipeline. Cached decoding also requires `stage=1`.

Dew does not implement 1F1B. Earlier experiments explored one approach; they do not prove that no compiled 1F1B design is possible. This remains an implementation and design gap.

## Select precision and kernels

GPU attention can select cuDNN when the requested shape, dtype, mask, and features fit its supported path. Other calls use XLA; the TPU implementation uses Pallas. Packed masks and optional attention behavior can change the selected kernel and its memory use.

Qwix quantization is an optional experimental training path with int8 and fp8 computation. It keeps master parameters but changes numerical behavior. The local RTX 4080 measurements did not show an fp8 step-time improvement at the tested sizes. Consult [performance measurements](../performance.md) and validate accuracy and throughput on your configuration.

## Run across hosts

Initialize the JAX process pool before creating device arrays or models. Built-in recipes call their process-setup function early. Each host needs compatible software, data access, and a reachable coordinator. Remote checkpoint storage and iterator partitioning need their own validation.

Local process-pool tests exercise real `jax.distributed` processes, but do not validate network failures, remote storage, TPU collectives, or cluster preemption. [TPU setup](../tpu.md) describes provisioning; it can incur cloud costs.

## Verify a distributed run

Compare a small global mathematical operation across the intended placements. Check loss and gradients, record identity, optimizer and EMA state, and save/restore behavior. Capture compiler/backend versions and tolerances. Device count and a finite loss alone are insufficient evidence of equivalent training.

CPU global-array tests cover unequal-mask loss normalization and partial-window restart. These checks do not establish large-model throughput or accelerator cross-host recovery. Read the [capability reference](../reference/support.md), [checkpoint guide](../guides/checkpoints.md), and [evaluation limits](../guides/evaluation.md) before making a production claim.
