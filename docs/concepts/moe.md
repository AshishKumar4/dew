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

## Expert computation and placement

Dew sorts tokens into expert order and runs the experts as one grouped matrix multiplication. The mixture's `implementation` field picks the kernel. `"auto"`, the default, picks the one measured fastest on the backend: `"pallas"` on a GPU and `"xla"` everywhere else. `"xla"` is `jax.lax.ragged_dot`; on a GPU, XLA runs it as a product over every expert. `"pallas"` is JAX's own Pallas/Triton grouped-matmul kernels, `gmm` and `tgmm`, vendored in `dew.nn.kernels.ragged_dot` from the jax 0.11.1 source tree because the wheel does not ship them. On an L4 it takes the lm-moe training step from 601.6 ms to 213.1 ms ([performance](../performance.md#kernel-choices-per-backend-2026-09-22)). It supports first-order reverse mode only, which is all training needs; forward mode and higher-order derivatives need `"xla"`. It falls back to `"xla"` where the kernels would change the product: float64, fp32 operands at a precision above the default, and a mesh that splits the arrays outside `shard_map`. `"tokamax"` uses `tokamax.ragged_dot` with the kernel named per backend: Triton on a GPU, `mosaic_tpu_v2` on a TPU. tokamax's own default picks its v1 TPU kernel, 13 times slower than XLA on a v6e. Every routed expert module the decoder builds follows this field, GPT OSS's included. If a model names `"tokamax"` and the package cannot be imported, initialization fails. Dew does not fall back to the XLA kernel under that name.

tokamax is not a Dew dependency, and its current release cannot be installed cleanly next to Dew. tokamax 0.0.13 pins `typeguard==2.13.3`. tyro 1.0.16, which parses every recipe's command line, needs `typeguard>=4.0.0`. Installing tokamax into a Dew environment downgrades typeguard, `uv pip check` reports the conflict, and every recipe fails while parsing its arguments with `AttributeError: module 'typeguard' has no attribute 'TypeCheckError'`. Until a tokamax release relaxes the pin, use the tokamax kernel from a separate environment that drives Dew from Python rather than from a recipe command line. `tools/benchmark_attention.py` does this, as described in [performance](../performance.md). The conflict is between the two packages' declared requirements. Dew does not pin around it or fall back at runtime.

The `expert` mesh axis splits the expert dimension across devices. Dense parameter dimensions can use FSDP or tensor placement on their own. See [distributed training](distributed.md) for the global batch and layout requirements.

The mixture's `dispatch` field defaults to `"global"`, which sorts and gathers tokens globally. `"exchange"` sends tokens to their experts through a bounded exchange with public JAX `all_to_all` collectives. It needs an `expert` mesh axis larger than one that divides the number of experts. Every selected token is kept, even when all traffic goes to one shard: later rounds empty that shard's bucket. You can initialize the model outside a mesh, but applying the exchange model needs the mesh. Gated experts and GPT OSS's interleaved biased experts use the same transport and keep their own activation and output-weight arithmetic.

Both dispatch modes run their expert projections through `moe.expert_projection`. It fixes the arithmetic of a routed layer during training, whatever the activation dtype and placement:

- each contraction accumulates in at least fp32 and rounds once to the compute dtype;
- kernel gradients keep the master dtype;
- input gradients keep their input's dtype;
- the exact GELU rounds once.

Forward-mode and reverse-mode differentiation follow the same rules. `tests/test_moe_precision.py` checks this against float64 arithmetic on the rounded operands, and through three Adam steps of both dispatch modes in bf16. `tools/moe_exchange_probe.py` measures the exchange's working memory against the global path on CPU. It says nothing about throughput on several accelerators.

GPT OSS's per-expert biases go through `moe.gather_expert_bias`. It accumulates the bias gradients at master or compute precision, then converts them back to the parameter dtype. Padding and idle experts add no bias gradient. The stored fused kernel and bias leaves, router choices, clipping limits and SwiGLU scaling are unchanged. `tests/test_moe_biased_exchange.py` checks the full router and experts against pinned transformers fixtures, and compares the forward pass, backward pass and optimizer updates of both dispatch modes.

More experts need more parameter storage, even with a fixed `top_k`. Routing, communication, shared experts and load imbalance all add to runtime. Estimate the optimizer and EMA storage along with the parameters, and time a representative forward and backward step on the topology you plan to use.

## Validate a sparse run

Against a reference model, compare the router's selections and weights, the sparse layer output, the loss and the parameter updates. For distributed training, check that the balancing statistics cover the global batch and that replicated state stays identical across shards.

The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) says which sparse checkpoints load and which have no export writer.
