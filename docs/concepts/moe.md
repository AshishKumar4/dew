# Mixture of experts

This page assumes you know transformer feed-forward layers and have read the [language model guide](language_models.md). A mixture-of-experts layer holds several feed-forward networks, called experts. For each token, a router scores the experts, picks a few of them, and mixes their outputs with routing weights.

## Construct a small sparse decoder

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
logits = model.apply(variables, tokens)
assert logits.shape == (1, 4, 32)
assert bool(jnp.all(jnp.isfinite(logits)))
print("Sparse decoder logits:", logits.shape)
```

The example builds four experts in each routed layer and sends each token to two of them. The logits keep the usual `(batch, sequence, vocabulary)` shape. The shape check shows the interface only. To check that a released model family computes the same numbers as its reference, you need the reference weights and computation.

`mixture` takes a configuration record or a `Mixture` value. `experts` is the number of experts and `top_k` the number picked per token. `layers` or `every` chooses which layers are routed. For a checkpoint, its configuration fixes these values. Changing them changes the architecture and can make the weights unusable.

## Router behavior

Model families route in different ways. They differ in softmax or sigmoid scores, whether the selected weights are normalized, output scaling, group-limited routing, shared experts and a selection bias. Two routers are not interchangeable just because their tensor shapes match.

For balancing without an auxiliary loss, a per-expert bias changes which experts are picked. The router reads the bias, and the objective updates it through non-parameter state. An auxiliary balancing loss is a separate term in the objective. Check the algorithm and configuration of your model family before you turn on either one.

Routing replay trains on the experts a rollout engine used (R3, arXiv 2510.11370). Pass `routes=(routed_experts, routed)` to `LMObjective.token_scores`, or pack sessions whose calls recorded `routed_experts` for GRPO. The record is `[batch, tokens, layers, top_k]`, indexed by decoder layer as vLLM and SGLang return it. Each router uses the recorded experts instead of its own top-k and still gathers their weights from this forward's scores, so the router keeps its gradient. A token the record does not cover (`routed` false, such as the last sampled id) keeps the router's own choice. A balancing bias counts the replayed experts.

## Expert computation and placement

Dew sorts tokens into expert order and runs the experts as one grouped matrix multiplication. The mixture's `implementation` field picks the kernel. `"auto"`, the default, picks the one measured fastest on the hardware generation (`dew.nn.kernels.device_generation`, keyed in `dew.nn.moe.GROUPED_MATMUL_BY_GENERATION`): `"pallas"` on sm80 (A100), sm86 (RTX 3090) and sm89 (L4, RTX 4080), and `"xla"` on a TPU v5e or v6e. Every other generation runs `"xla"`: GPUs older than sm80, which cannot compile the kernels, and sm90 and later, which are unmeasured. A mesh leaves the choice alone: the experts run inside the dispatch's `shard_map` on each device's rows, with the weights they need gathered there, and on 2x RTX 3090 one fsdp-sharded expert layer took 26.2 ms on the Pallas kernels against 271.9 ms on `"xla"` ([performance](../performance.md#expert-parallelism-on-4x-rtx-3090-2026-09-23)). `"xla"` is `jax.lax.ragged_dot`; on a GPU, XLA runs it as a product over every expert. `"pallas"` is JAX's own Pallas/Triton grouped-matmul kernels, `gmm` and `tgmm`, vendored in `dew.nn.kernels.ragged_dot` from the jax 0.11.2 source tree because no wheel ships them. jax 0.11.2 deprecates the Pallas Triton backend they run on; Dew keeps them on compute capability 8.0 to 8.9, where JAX's Mosaic GPU grouped matmul does not compile (on sm89 it fails: no wgmma); the deprecation warning it raises is left to the user's warning filters. On an L4 it takes the lm-moe training step from 601.6 ms to 213.1 ms ([performance](../performance.md#kernel-choices-per-generation-2026-09-22)). It supports first-order reverse mode only, which is all training needs; forward mode and higher-order derivatives need `"xla"`. It falls back to `"xla"` where the kernels would change the product: float64, fp32 operands at a precision above the default, and an x64 run. Under a mesh the kernels run on each device's share of the sorted rows, inside `shard_map`. `"tokamax"` uses `tokamax.ragged_dot` with the kernel named per generation: Triton on sm89, `mosaic_tpu_v2` on a v6e, tokamax's XLA path elsewhere. tokamax's own default picks its v1 TPU kernel, 13 times slower than XLA on a v6e. Every routed expert module the decoder builds follows this field, GPT OSS's included. If a model names `"tokamax"` and the package cannot be imported, initialization fails. Dew does not fall back to the XLA kernel under that name.

tokamax is not a Dew dependency, and its current release cannot be installed cleanly next to Dew. tokamax 0.0.13 pins `typeguard==2.13.3`. tyro 1.0.16, which parses every recipe's command line, needs `typeguard>=4.0.0`. Installing tokamax into a Dew environment downgrades typeguard, `uv pip check` reports the conflict, and every recipe fails while parsing its arguments with `AttributeError: module 'typeguard' has no attribute 'TypeCheckError'`. Until a tokamax release relaxes the pin, use the tokamax kernel from a separate environment that drives Dew from Python rather than from a recipe command line. `tools/benchmark_attention.py` does this, as described in [performance](../performance.md). The conflict is between the two packages' declared requirements. Dew does not pin around it or fall back at runtime.

The `expert` mesh axis splits the expert dimension across devices. Dense parameter dimensions can use FSDP or tensor placement on their own. See [distributed training](distributed.md) for the global batch and layout requirements.

The mixture's `dispatch` field defaults to `"global"`, which sorts and gathers tokens where they are: on a mesh each device routes its own tokens through every expert, so every layout computes each token once. `"exchange"` is expert parallelism: each device sends its selected tokens to the devices that hold their experts with JAX `all_to_all` collectives, and gets the results back. It needs an `expert` mesh axis larger than one that divides the number of experts, and the data, fsdp and sequence axes may split the tokens further. Every selected token is kept, even when all traffic goes to one shard. The first round sends each shard a device's balanced share of its tokens, and what a skewed routing leaves over follows in later rounds of the same size, which the backward pass recomputes rather than keeps. You can initialize the model outside a mesh, but applying the exchange model needs the mesh. Gated experts and GPT OSS's interleaved biased experts use the same transport and keep their own activation and output-weight arithmetic.

`capacity_factor` drops tokens instead, as GShard and MaxText do. Each sequence keeps `max(ceil(length * top_k / experts) * capacity_factor, capacity_factor)` slots per expert, taken in token order, and a dropped slot adds nothing to its token's output. Both dispatch modes drop the same slots on every placement, because the count runs over whole sequences. Under a capacity the exchange runs one round. None, the default, keeps every slot.

Both dispatch modes run their expert projections through `moe.expert_projection`. It fixes the arithmetic of a routed layer during training, whatever the activation dtype and placement:

- each contraction accumulates in at least fp32 and rounds once to the compute dtype;
- a kernel gradient sums every device's and every exchange round's share in at least fp32, and rounds to the master dtype once, so a bf16 master gets the same gradient on one device and on any mesh;
- kernel gradients keep the master dtype;
- input gradients keep their input's dtype;
- the exact GELU rounds once.

A model that names no `dtype` computes its experts in their stored dtype when that dtype is narrower than the stream (`moe.expert_compute_dtype`). An fp32 residual stream over bf16 expert kernels rounds each expert's input to bf16, multiplies bf16 by bf16 with fp32 accumulation, and returns bf16. The dense layers of such a model still promote to fp32. Widening the experts instead copied each layer's kernels to fp32, 14.35 GiB of live temporaries serving gpt-oss-20b on four RTX 3090s. The forward reads the kernels as stored; only a gradient, whose cross-device sums need fp32, widens them.

Forward-mode and reverse-mode differentiation follow the same rules. `tests/test_moe_precision.py` checks this against float64 arithmetic on the rounded operands, and through three Adam steps of both dispatch modes in bf16. `tools/moe_exchange_probe.py` measures the exchange's working memory against the global path on CPU. It says nothing about throughput on several accelerators.

GPT OSS's per-expert biases go through `moe.gather_expert_bias`. It accumulates the bias gradients at master or compute precision, then converts them back to the parameter dtype. Padding and idle experts add no bias gradient. The stored fused kernel and bias leaves, router choices, clipping limits and SwiGLU scaling are unchanged. `tests/test_moe_biased_exchange.py` checks the full router and experts against pinned transformers fixtures, and compares the forward pass, backward pass and optimizer updates of both dispatch modes.

More experts need more parameter storage, even with a fixed `top_k`. Routing, communication, shared experts and load imbalance all add to runtime. Estimate the optimizer and EMA storage along with the parameters, and time a representative forward and backward step on the topology you plan to use.

## Validate a sparse run

Against a reference model, compare the router's selections and weights, the sparse layer output, the loss and the parameter updates. For distributed training, check that the balancing statistics cover the global batch and that replicated state stays identical across shards.

The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) says which sparse checkpoints load and which have no export writer.
