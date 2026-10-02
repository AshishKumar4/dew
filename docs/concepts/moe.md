# Mixture of experts

A mixture-of-experts (MoE) layer holds several feed-forward networks, called experts, in place of one. For each token, a router scores the experts, picks the `top_k` best and sums their outputs weighted by the router's scores. In Dew a `causal_transformer` becomes sparse through its `mixture` field, a `Mixture` value (`dew.nn.backbones.causal_transformer`) or a mapping of its fields.

## Example

```python
import jax
import jax.numpy as jnp
from dew import models

model = models.build(
    "causal_transformer", vocab_size=32, emb_features=16,
    num_layers=2, num_heads=2, mlp_features=32, max_seq_len=8,
    mixture={"experts": 4, "top_k": 2, "every": 1},
    dtype="float32", attention_impl="xla",
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

Every layer here routes (`every=1`) to two of four experts. The logits keep the usual `(batch, sequence, vocabulary)` shape. Opening the `router` collection makes each router record its choices (`indices`, `[batch, sequence, top_k]`), its scores and its log partition; without it nothing is recorded. Flax keeps sown values in a tuple, hence the first `[0]`. The figure reads the same collection from this model over 64 random tokens:

![Router choices of a four-expert, top-2 mixture at layer 0 for 32 tokens, with each chosen expert's normalized weight, and the number of tokens each expert received in both layers.](../assets/moe-routing-light.svg)
![Router choices of a four-expert, top-2 mixture at layer 0 for 32 tokens, with each chosen expert's normalized weight, and the number of tokens each expert received in both layers.](../assets/moe-routing-dark.svg)

## Mixture

| Field | Default | Meaning |
|---|---|---|
| `experts` | required | Number of experts. |
| `top_k` | `2` | Experts per token. |
| `layers` / `every` | `None` | Sparse layers by index, or every nth layer counting from the end of the first group (Qwen3-MoE's `decoder_sparse_step`). With neither, every layer is sparse. Setting both is refused. |
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

For a published checkpoint, its configuration fixes these values; changing them changes the architecture and can make the weights unusable.

## Routing

Model families route in different ways: softmax or sigmoid scores, normalized or raw selected weights, output scaling, group-limited routing, shared experts and a selection bias. Matching tensor shapes do not make two routers interchangeable. The router's gate projection runs in float32 whatever the activation dtype.

Balancing without an auxiliary loss uses a per-expert bias (`bias=True`) that changes which experts are picked but not the weights they get; `LMObjective(balance_rate=...)` moves it against each expert's load every step through non-parameter state. An auxiliary balancing loss (`aux_loss_alpha`, `router_z_loss`) is a separate term of the objective ([Language models](language_models.md)). Check the algorithm and configuration of your model family before turning on either.

Routing replay trains on the experts a rollout engine used (R3, arXiv 2510.11370). Pass `routes=(routed_experts, routed)` to `LMObjective.token_scores`, or pack sessions whose calls recorded `routed_experts` for GRPO ([Post-training](post_training.md)). The record is `[batch, tokens, layers, top_k]`, indexed by decoder layer as vLLM and SGLang return it. Each router uses the recorded experts instead of its own top-k and still gathers their weights from this forward's scores, so the router keeps its gradient. A token the record does not cover (`routed` false, such as the last sampled ID) keeps the router's own choice. A balancing bias counts the replayed experts.

## Expert kernels

Dew sorts tokens into expert order and runs the experts as one grouped matrix multiplication. `implementation` picks the kernel:

| Value | Kernel |
|---|---|
| `'auto'` | The one measured fastest on the hardware generation (`dew.nn.kernels.device_generation`, keyed in `dew.nn.moe.GROUPED_MATMUL_BY_GENERATION`): `'pallas'` on sm80 (A100), sm86 (RTX 3090) and sm89 (L4, RTX 4080), `'xla'` on TPU v5e and v6e. Every other generation runs `'xla'`: GPUs older than sm80 cannot compile the kernels, and sm90 and later are unmeasured. |
| `'xla'` | `jax.lax.ragged_dot`. On a GPU, XLA runs it as a product over every expert. |
| `'pallas'` | JAX's own Pallas/Triton grouped-matmul kernels, `gmm` and `tgmm`, vendored in `dew.nn.kernels.ragged_dot` from the jax 0.11.2 source tree because no wheel ships them. |
| `'tokamax'` | `tokamax.ragged_dot` with the kernel named per generation (`TOKAMAX_KERNEL_BY_GENERATION`): Triton on sm80 and sm89, `mosaic_tpu_v2` on v5e and v6e, tokamax's XLA path elsewhere. Only the forward runs on tokamax; the backward differentiates on XLA. |

On an L4 the Pallas kernels take the lm-moe training step from 601.6 ms to 213.1 ms ([Performance measurements](../performance.md)). jax 0.11.2 deprecates the Pallas Triton backend they run on; Dew keeps them on compute capability 8.0 to 8.9, where JAX's Mosaic GPU grouped matmul does not compile (on sm89 it fails for lack of wgmma), and leaves the deprecation warning to the user's warning filters. They support first-order reverse mode only, which is what the ordinary `Trainer` step uses; forward mode and higher-order derivatives, as in meta-learning, need `'xla'`. They fall back to `'xla'` where they would change the product: float64, fp32 operands at a precision above the default, and an x64 run.

A mesh does not change the choice: the experts run inside the dispatch's `shard_map` on each device's rows, with the weights they need gathered there. On 2x RTX 3090 one fsdp-sharded expert layer took 26.2 ms on the Pallas kernels against 271.9 ms on `'xla'`. Every routed expert module the decoder builds follows `implementation`, GPT OSS's included.

tokamax's own default picks its v1 TPU kernel, 13 times slower than XLA on a v6e, so Dew names the kernel. If a model names `'tokamax'` and the package cannot be imported, initialization fails; there is no fallback to XLA under that name. tokamax's releases cannot be installed cleanly beside Dew: tokamax 0.0.13 (and 0.0.14) pins `typeguard==2.13.3`, while tyro 1.0.16, which parses every recipe's command line, needs `typeguard>=4.0.0`. Installing such a release downgrades typeguard, `uv pip check` reports the conflict, and every recipe fails while parsing its arguments with `AttributeError: module 'typeguard' has no attribute 'TypeCheckError'`. The `kernels` extra installed with `-c constraints.txt` takes tokamax's main at a commit that dropped typeguard ([Installation](../installation.md)); the grouped-matmul numbers here were measured on 0.0.14.

## Dispatch and expert parallelism

The `expert` mesh axis splits the expert dimension across devices. Dense parameter dimensions can use FSDP or tensor placement on their own; [Distributed training](distributed.md) covers the global batch and layout requirements.

`dispatch='global'`, the default, sorts and gathers tokens where they are: on a mesh each device routes its own tokens through every expert, so every layout computes each token once. `dispatch='exchange'` is expert parallelism: each device sends its selected tokens to the devices that hold their experts with JAX `all_to_all` collectives and gets the results back. It needs an `expert` mesh axis larger than one that divides the number of experts; the data, fsdp and sequence axes may split the tokens further. Every selected token is kept, even when all traffic goes to one shard. The first round sends each shard a device's balanced share of its tokens, and what a skewed routing leaves over follows in later rounds of the same size, which the backward pass recomputes rather than keeps. The model can be initialized outside a mesh, but applying the exchange needs the mesh. `TextGeneration` refuses the exchange ([Inference](inference.md)); `dispatch='global'` computes the same layer. Gated experts and GPT OSS's interleaved biased experts use the same transport and keep their own activation and output-weight arithmetic.

`capacity_factor` drops tokens instead, as GShard and MaxText do. Each sequence keeps `max(ceil(length * top_k / experts) * capacity_factor, capacity_factor)` slots per expert, taken in token order, and a dropped slot adds nothing to its token's output. Both dispatch modes drop the same slots on every placement, because the count runs over whole sequences. Under a capacity the exchange runs one round.

## Precision

Both dispatch modes run their expert projections through `moe.expert_projection`, which fixes the arithmetic of a routed layer during training, whatever the activation dtype and placement:

- each contraction accumulates in at least fp32 and rounds once to the compute dtype;
- a kernel gradient sums every device's and every exchange round's share in at least fp32 and rounds to the master dtype once, rather than rounding each share; the fp32 summation order still depends on the mesh, so gradients on different meshes agree to fp32 rounding, not bit for bit;
- kernel gradients keep the master dtype, and input gradients their input's dtype;
- the exact GELU rounds once.

A model that names no `dtype` computes its experts in their stored dtype when that dtype is narrower than the stream (`moe.expert_compute_dtype`). An fp32 residual stream over bf16 expert kernels rounds each expert's input to bf16, multiplies bf16 by bf16 with fp32 accumulation, and returns bf16; the dense layers of such a model still promote to fp32. Widening the experts instead copied each layer's kernels to fp32, 14.35 GiB of live temporaries serving gpt-oss-20b on four RTX 3090s. The forward reads the kernels as stored; only a gradient, whose cross-device sums need fp32, widens them.

JAX's x64 mode also widens default integer counts. The XLA grouped-matmul path narrows signed int64 group sizes to int32 when the row domain fits, because the TPU ragged-dot kernel cannot lower int64 indices. This does not widen the expert values or their gradients; they keep the dtypes above.

Forward-mode and reverse-mode differentiation follow the same rules. `tests/test_moe_precision.py` checks this against float64 arithmetic on the rounded operands, and through three Adam steps of both dispatch modes in bf16. `tools/moe_exchange_probe.py` measures the exchange's working memory against the global path on CPU. It says nothing about throughput on several accelerators.

GPT OSS's per-expert biases go through `moe.gather_expert_bias`, which accumulates the bias gradients at master or compute precision and converts them back to the parameter dtype. Padding and idle experts add no bias gradient. The stored fused kernel and bias leaves, router choices, clipping limits and SwiGLU scaling are unchanged. `tests/test_moe_biased_exchange.py` checks the full router and experts against pinned transformers fixtures, and compares the forward pass, backward pass and optimizer updates of both dispatch modes.

## Cost and validation

More experts need more parameter storage, even with a fixed `top_k`. Routing, communication, shared experts and load imbalance all add to runtime. Estimate the optimizer and EMA storage along with the parameters, and time a representative forward and backward step on the topology you plan to use.

Against a reference model, compare the router's selections and weights, the sparse layer output, the loss and the parameter updates. For distributed training, check that the balancing statistics cover the global batch and that replicated state stays identical across shards. [Supported models](../models.md) lists which sparse checkpoints load and which have no export writer.
