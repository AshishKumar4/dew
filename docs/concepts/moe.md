# Mixture of experts

A mixture-of-experts (MoE) layer replaces one feed-forward network with several networks, called experts. For each token, a router selects the `top_k` highest-scoring experts, and the layer sums their outputs weighted by the router scores. To use MoE in a `CausalTransformer`, set `mixture` to a `Mixture` from `dew.nn.backbones`.

## Example

```python
import jax
import jax.numpy as jnp

from dew.nn.backbones import CausalTransformer, Mixture

model = CausalTransformer(
    vocab_size=32, emb_features=16,
    num_layers=2, num_heads=2, mlp_features=32, max_seq_len=8,
    mixture=Mixture(experts=4, top_k=2),
    dtype=jnp.float32, attention_impl="xla",
)
tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
variables = model.init(jax.random.key(0), tokens)
logits, routed = model.apply(variables, tokens, mutable=["router"])
print("logits:", logits.shape)
chosen = routed["router"]["layers_0"]["mlp"]["gate"]["indices"][0]
print("experts per token, layer 0:", chosen[0].tolist())
```

```text
logits: (1, 4, 32)
experts per token, layer 0: [[1, 3], [3, 1], [1, 0], [3, 2]]
```

Every layer here routes to two of four experts. The logits keep the usual `(batch, sequence, vocabulary)` shape. `mutable=["router"]` records each router's choices (`indices`, `[batch, sequence, top_k]`), scores and log partition. Without it, nothing is recorded. Flax stores sown values in a tuple, which is why the example uses the first `[0]`. The figure reads this model's router collection over 64 random tokens:

![Router choices of a four-expert, top-2 mixture at layer 0 for 32 tokens, with each chosen expert's normalized weight, and the number of tokens each expert received in both layers.](../assets/moe-routing-light.svg)
![Router choices of a four-expert, top-2 mixture at layer 0 for 32 tokens, with each chosen expert's normalized weight, and the number of tokens each expert received in both layers.](../assets/moe-routing-dark.svg)

## Mixture

| Field | Default | Meaning |
|---|---|---|
| `experts` | required | Number of experts. |
| `top_k` | `2` | Experts per token. |
| `layers` | `None` | Sparse layers by index; `None` makes every layer sparse. A checkpoint's cadence (Qwen3-MoE's `decoder_sparse_step`, Llama 4's `interleave_moe_layer_step`) translates to these indices. |
| `score_function` | `'softmax'` | How router logits become scores. |
| `norm_topk_prob` | `True` | Divide a token's selected weights by their sum. |
| `scaling` | `1.0` | Routed output scale. |
| `groups`, `groups_per_token`, `group_score` | `1`, `1`, `'top2'` | DeepSeek's group-limited routing. |
| `bias` | `False` | Keep an aux-loss-free balancing bias (DeepSeek's `e_score_correction_bias`). |
| `expert_features` | `None` | Expert width; `None` uses `mlp_features`. |
| `shared_features`, `shared_gate` | `0`, `False` | Width of one dense shared expert every token takes, and a learned sigmoid gate on it. |
| `parallel` | `False` | Gemma 4's placement: experts beside the dense MLP. |
| `implementation` | `'auto'` | Grouped matmul kernel: `'auto'`, `'xla'`, `'pallas'` or `'tokamax'`. |
| `dispatch` | `'global'` | `'global'` or `'exchange'` (expert parallelism). |
| `capacity_factor` | `None` | Per-expert slot capacity; `None` keeps every selected slot. |
| `hash_layers`, `latent_features`, `latent_norm`, `media_bias` | | DeepSeek V4 hash routing, Kimi K3 latent experts, DeepSeek-V4.1's media bias. |

A published checkpoint's configuration sets these values. Changing them changes the architecture and can make the weights unusable.

## Routing

Model families differ in how they score and select experts. Their choices include softmax or sigmoid scores, normalized or raw selected weights, output scaling, group-limited routing, shared experts and a selection bias. Two routers with matching tensor shapes can still behave differently. The router's gate projection always runs in float32, regardless of the activation dtype.

To balance routing without an auxiliary loss, `bias=True` adds a per-expert selection bias. This changes the chosen experts while leaving their weights unchanged. Each step, `LMObjective(balance_rate=...)` adjusts this non-parameter state against each expert's load. An auxiliary balancing loss (`aux_loss_alpha`, `router_z_loss`) is a separate term of the objective ([Language models](language_models.md)). Check your model family's algorithm and configuration before enabling either.

Routing replay trains on the experts used by a rollout engine (R3, arXiv 2510.11370). Pass `routes=(routed_experts, routed)` to `LMObjective.token_scores`. For GRPO, pack sessions whose calls recorded `routed_experts` ([Post-training](post_training.md)). The record has shape `[batch, tokens, layers, top_k]`, indexed by decoder layer as vLLM and SGLang return it.

Each router selects the recorded experts, then gets their weights from the current forward pass's scores. This preserves the router's gradient. If a token has no record (`routed` false, such as the last sampled ID), the router makes its own choice. A balancing bias counts the replayed experts.

## Expert kernels

Dew sorts tokens into expert order and runs the experts as one grouped matrix multiplication. `implementation` picks the kernel:

| Value | Kernel |
|---|---|
| `'auto'` | The one measured fastest on the hardware generation (`dew.nn.kernels.device_generation`, keyed in `dew.nn.moe.GROUPED_MATMUL_BY_GENERATION`): `'pallas'` on sm80 (A100), sm86 (RTX 3090) and sm89 (L4, RTX 4080), `'xla'` on TPU v5e and v6e. Every other generation runs `'xla'`: GPUs older than sm80 cannot compile the kernels, and sm90 and later are unmeasured. |
| `'xla'` | `jax.lax.ragged_dot`. On a GPU, XLA runs it as a product over every expert. |
| `'pallas'` | JAX's own Pallas/Triton grouped-matmul kernels, `gmm` and `tgmm`, vendored in `dew.nn.kernels.ragged_dot` from the jax 0.11.2 source tree because no wheel ships them. |
| `'tokamax'` | `tokamax.ragged_dot` with the kernel named per generation (`TOKAMAX_KERNEL_BY_GENERATION`): Triton on sm80 and sm89, `mosaic_tpu_v2` on v5e and v6e, tokamax's XLA path elsewhere. Only the forward runs on tokamax; the backward differentiates on XLA. |

On an L4, the Pallas kernels reduce the lm-moe training step from 601.6 ms to 213.1 ms ([Performance measurements](../performance.md)). jax 0.11.2 deprecates their Pallas Triton backend. Dew still uses them on compute capability 8.0 to 8.9 because JAX's Mosaic GPU grouped matmul does not compile there. On sm89, it fails for lack of wgmma. Dew leaves the deprecation warning to the user's warning filters.

The Pallas kernels support first-order reverse mode, as used by an ordinary `Trainer` step. For forward mode or higher-order derivatives, such as meta-learning, use `'xla'`. Dew also falls back to `'xla'` when Pallas would change the product: float64, fp32 operands at a precision above the default, or an x64 run.

On a mesh, the same kernel choice applies. The experts run inside the dispatch's `shard_map` on each device's rows, with the required weights gathered there. On 2x RTX 3090, one fsdp-sharded expert layer took 26.2 ms with Pallas and 271.9 ms with `'xla'`. Every routed expert module follows `implementation`, including GPT OSS's.

tokamax defaults to its v1 TPU kernel, which is 13 times slower than XLA on a v6e. Dew therefore selects a kernel explicitly. If you choose `'tokamax'` and the package cannot be imported, initialization fails. It does not fall back to XLA.

tokamax 0.0.13 and 0.0.14 pin `typeguard==2.13.3`. tyro 1.0.16, which parses every recipe's command line, needs `typeguard>=4.0.0`. Installing either tokamax release downgrades typeguard. `uv pip check` reports the conflict, and every recipe fails to parse its arguments with `AttributeError: module 'typeguard' has no attribute 'TypeCheckError'`. Install tokamax with `-c constraints.txt` to use a commit from its main branch that dropped typeguard ([Installation](../installation.md)). The grouped-matmul numbers here were measured on 0.0.14.

## Dispatch and expert parallelism

The `expert` mesh axis splits the expert dimension across devices. Dense parameter dimensions can use FSDP or tensor placement on their own; [Distributed training](distributed.md) covers the global batch and layout requirements.

`dispatch='global'`, the default, sorts and gathers tokens on their current devices. Each device routes its own tokens through every expert, so every layout computes each token once.

`dispatch='exchange'` uses expert parallelism. JAX `all_to_all` collectives send selected tokens to the devices holding their experts and return the results. The `expert` axis must be larger than one and divide the number of experts. The data, fsdp and sequence axes may split the tokens further.

The exchange keeps every selected token, even when all traffic goes to one shard. The first round sends each shard a device's balanced share of tokens. Skewed routing sends the remaining tokens in later rounds of the same size. The backward pass recomputes these rounds instead of storing them.

You can initialize the model outside a mesh, but applying the exchange needs the mesh. `TextGeneration` rejects exchange dispatch ([Inference](inference.md)); use `dispatch='global'` to compute the same layer. Gated experts and GPT OSS's interleaved biased experts use the same transport. Each keeps its own activation and output-weight arithmetic.

To limit capacity by dropping tokens, as GShard and MaxText do, set `capacity_factor`. Each sequence keeps `max(ceil(length * top_k / experts) * capacity_factor, capacity_factor)` slots per expert in token order. Dropped slots add nothing to a token's output. Both dispatch modes drop the same slots on every placement because the count uses whole sequences. With a capacity limit, the exchange runs one round.

## Precision

Both dispatch modes use `moe.expert_projection` to keep the routed layer's training arithmetic consistent across activation dtypes and placements:

- each contraction accumulates in at least fp32 and rounds once to the compute dtype;
- a kernel gradient sums each device's and each exchange round's share in at least fp32, then rounds once to the master dtype. The summation order depends on the mesh, so gradients on different meshes agree within fp32 rounding but can differ bit for bit;
- kernel gradients keep the master dtype, and input gradients their input's dtype;
- the exact GELU rounds once.

If a model has no `dtype`, it computes experts in their stored dtype when that is narrower than the stream (`moe.expert_compute_dtype`). For an fp32 residual stream with bf16 expert kernels, it rounds expert inputs to bf16. It multiplies bf16 by bf16 with fp32 accumulation and returns bf16. The dense layers still promote to fp32.

Widening the experts to fp32 copied each layer's kernels. When serving gpt-oss-20b on four RTX 3090s, those copies needed 14.35 GiB of live temporaries. The forward pass reads kernels in their stored dtype. Only gradients widen them for fp32 cross-device sums.

JAX's x64 mode widens default integer counts. The TPU ragged-dot kernel cannot lower int64 indices, so the XLA grouped-matmul path converts signed int64 group sizes to int32 when the row domain fits. Expert values and gradients keep the dtypes described above.

Forward-mode and reverse-mode differentiation follow the same rules. `tests/test_moe_precision.py` compares them with float64 arithmetic on the rounded operands. It also checks three Adam steps in bf16 with both dispatch modes. `tools/moe_exchange_probe.py` compares working memory for exchange and global dispatch on CPU. It does not measure throughput on several accelerators.

GPT OSS's per-expert biases use `moe.gather_expert_bias`. It accumulates bias gradients at master or compute precision, then converts them to the parameter dtype. Padding and idle experts add no bias gradient. The stored fused kernel and bias leaves, router choices, clipping limits and SwiGLU scaling are unchanged. `tests/test_moe_biased_exchange.py` checks the router and experts against pinned transformers fixtures. It also compares forward passes, backward passes and optimizer updates between dispatch modes.

## Cost and validation

More experts need more parameter storage, even with a fixed `top_k`. Routing, communication, shared experts and load imbalance all add to runtime. Estimate the optimizer and EMA storage along with the parameters, and time a representative forward and backward step on the topology you plan to use.

Against a reference model, compare the router's selections and weights, the sparse layer output, the loss and the parameter updates. For distributed training, check that the balancing statistics cover the global batch and that replicated state stays identical across shards. [Supported models](../models.md) lists which sparse checkpoints load and which have no export writer.
