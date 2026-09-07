# Mixture of experts

This page assumes basic transformer feed-forward layers and the [language model guide](language_models.md). A mixture-of-experts layer contains several feed-forward networks. A router scores the experts for each token, selects a subset, and combines their outputs with routing weights.

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

The example initializes four experts per selected layer and routes each token to two experts. The logits retain the usual `(batch, sequence, vocabulary)` shape. The shape check explains the interface; numerical parity of a particular released family requires its reference weights and computation.

`mixture` accepts a configuration record or a `Mixture` value. `experts` sets the number of experts, `top_k` the number selected per token, and `layers` or `every` chooses the routed layers. A checkpoint's configuration determines these choices; changing them changes the architecture and may invalidate its weights.

## Router behavior

Router conventions vary by family. Important choices include softmax versus sigmoid scores, normalization of selected weights, output scaling, group-limited routing, shared experts, and a selection bias. These settings are not interchangeable merely because the tensor shapes agree.

For auxiliary-loss-free balancing, a per-expert bias affects selection. The router reads it; the objective updates it through non-parameter state. An auxiliary balancing loss is a separate objective term. Check the selected family's algorithm and configuration before enabling either.

## Expert computation and placement

Dew gathers tokens into expert order and uses grouped matrix multiplication through `jax.lax.ragged_dot`. An optional tokamax implementation provides another kernel path. Kernel availability and speed depend on the installed package and device; an optional path is not automatically faster.

tokamax is not a Dew dependency, and its current release does not install cleanly next to one. tokamax 0.0.13 pins `typeguard==2.13.3`; tyro 1.0.16, which parses every recipe's command line, requires `typeguard>=4.0.0`. Installing tokamax into a Dew environment downgrades typeguard, `uv pip check` reports the conflict, and each recipe fails while parsing its arguments with `AttributeError: module 'typeguard' has no attribute 'TypeCheckError'`. Use the tokamax kernel path from a separate environment that drives Dew through Python rather than a recipe command line, as `tools/benchmark_attention.py` documents in [performance](../performance.md), until a tokamax release relaxes the pin. The dependency conflict is between two packages' declared requirements, and Dew neither pins around it nor falls back at runtime.

The `expert` mesh axis partitions the expert dimension. Dense parameter dimensions can use FSDP or tensor placement independently. See [distributed training](distributed.md) for the global batch and layout requirements.

More experts increase parameter storage even when `top_k` is fixed. Routing, communication, shared experts, and load imbalance still contribute to runtime. Estimate optimizer and EMA storage as well as the parameters, and measure a representative forward/backward step on the intended topology.

## Validate a sparse run

For a reference model, compare router selections and weights, the sparse layer output, the loss, and parameter updates. For distributed training, verify that the balancing statistics represent the global batch and that replicated state remains identical across shards.

The current documentation does not claim a completed large-scale sparse-model training run. Consult [family translation coverage](../reference/model-families.md) and [capabilities and limitations](../reference/support.md) for the implemented and measured scope.
