# Mixture of experts

A sparse layer replaces one feed-forward with many: a router scores every expert for a token, the token goes through the `top_k` best ones, and their outputs are summed with the router's weights. The parameter count grows with the expert count while the work per token stays at `top_k` experts, which is why the large open decoders from Mixtral to DeepSeek are built this way.

The modules live in `dew.nn.moe` and the mesh axis in `dew.training.distributed`.

## The layer

`SparseMLP` goes where `GatedMLP` goes in a decoder block and holds two submodules, `gate` for the `Router` and `experts` for the `ExpertMLP`:

```python
from dew.nn.moe import SparseMLP

layer = SparseMLP(num_experts=8, top_k=2, hidden_features=2048, out_features=512)
```

`CausalTransformer` builds them from one value, `mixture`, because none of its dials means anything without experts to route to. `experts` and `top_k` size the routing, and the sparse layers are `layers` by index, or every `every`-th layer, or all of them when neither is set:

```python
from dew import models

model = models.build("causal_transformer", vocab_size=50304, emb_features=64, num_layers=1,
                     num_heads=2, num_kv_heads=1,
                     mixture={"experts": 8, "top_k": 2, "every": 1})
```

The value takes a record from a config or a `Mixture` from code, and the two build the same model. That cadence is the same rule Qwen3-MoE's `decoder_sparse_step` means, counting from the end of the first group, and DeepSeek's first dense layers are the mixture's `layers`. A dense layer keeps every leaf it had, so a checkpoint of the dense model still loads into the dense layers of the sparse one.

The registry name is `moe`, which is `CausalTransformer` with a mixture set.

## The router

`Router` computes its logits in fp32 whatever dtype the activations carry, which is where DeepSeek's router runs and what the frontier configs ask for. Then:

| Field | Meaning | Mixtral, Qwen3.5 | DeepSeek V3, V4 |
| --- | --- | --- | --- |
| `score_function` | how a logit becomes an affinity | `softmax` over the experts | `sigmoid` per expert |
| `normalize_weights` | divide the chosen weights by their sum | yes | yes |
| `routed_scaling_factor` | multiply the weights | 1.0 | 2.5 |
| `expert_groups`, `groups_per_token` | the node limit | 1, 1 | 8 groups, 4 kept |
| `expert_bias` | a per-expert selection bias | no | yes |

The node limit scores each group of experts by its two best members and lets a token choose only inside the best `groups_per_token` groups, which is what bounds how many nodes a token's experts are spread over.

`expert_bias` is DeepSeek's aux-loss-free balancing bias (arXiv 2408.15664). It lives in the `moe` variable collection as `e_score_correction_bias`, it is fp32, and it enters the selection only: the weights are gathered from the unbiased scores, so balancing changes which experts a token gets and never what they contribute. The router reads it and never writes it, which is where transformers keeps it (`nn.Buffer`) and how MaxText hands the update back to its caller. The update itself is a function:

```python
import jax.numpy as jnp
from dew.nn.moe import calculate_load_balance_updates

indices = jnp.asarray([[0, 3], [1, 3], [2, 3]], jnp.int32)
update = calculate_load_balance_updates(indices, num_experts=8, rate=0.001)
```

It is `+rate` for every expert below the average load, `-rate` for every expert above it. `LMObjective(balance_rate=...)` applies it every step and hands the moved bias back to the trainer through `Aux.variables`, the one channel for a collection a step updates without a gradient; under the compiled step the load count is global, so the bias stays one replicated value on every shard. A mixture routes on the scores alone unless it asks for the bias, `mixture={"experts": 8, "bias": True}`, which is what a DeepSeek checkpoint needs. The mixture also carries the rest of the router's choices: `score_function`, `scaling`, `groups` and `groups_per_token`.

## The grouped matmul

`ExpertMLP` sorts the tokens by expert, runs the three projections as grouped matmuls over that order, puts the rows back in token order and sums each token's `top_k` results with its router weights in fp32. The expert weights are stacked on an expert dimension, `[exp, embed, mlp]` for `gate_proj` and `up_proj` and `[exp, mlp, embed]` for `down_proj`, which is the layout the grouped matmul takes and the `expert` mesh axis shards.

`implementation` picks the kernel:

| Value | Kernel |
| --- | --- |
| `xla` (default) | `jax.lax.ragged_dot`, which lowers on every backend |
| `tokamax` | `tokamax.ragged_dot`, the same call against tokamax's kernels |

tokamax is not a dependency, so its import sits inside the branch that needs it and its test skips where it is absent.

## Parameter layout

The leaves follow the Hugging Face layout of a sparse decoder layer, with the experts stacked:

| Dew leaf | Shape | Checkpoint tensors |
| --- | --- | --- |
| `layers_N/mlp/gate/kernel` | `[embed, exp]` | `model.layers.N.mlp.gate.weight` |
| `layers_N/mlp/experts/gate_proj/kernel` | `[exp, embed, mlp]` | `model.layers.N.mlp.experts.E.gate_proj.weight` |
| `layers_N/mlp/experts/up_proj/kernel` | `[exp, embed, mlp]` | `model.layers.N.mlp.experts.E.up_proj.weight` |
| `layers_N/mlp/experts/down_proj/kernel` | `[exp, mlp, embed]` | `model.layers.N.mlp.experts.E.down_proj.weight` |

The translation is the one every Dew kernel takes, a transpose of each matrix, plus a stack over the experts. transformers 5.16.1 merges the same tensors into one `mlp.experts.gate_up_proj` while loading (`transformers/core_model_loading.py:1545`), gate rows first, so a DeepSeek V4 or Qwen3.5-MoE checkpoint maps either way.

## Expert parallelism

`MeshSpec(fsdp=2, expert=4)` gives the expert dimension its own mesh axis, and one rules row (`exp` to `expert`) puts it there. On that eight-device mesh an 8-expert `[8, 32, 64]` kernel holds two experts and half a width per device, and a 50-step run there matches the same run at `expert=1` step for step. See [distributed training](distributed.md) for the mesh and the rules table.

## What is measured

Everything below ran on CPU at fp32 with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_moe.py -q`, against fixtures written by `tools/moe_reference.py` from transformers 5.16.1.

| Check | Reference | Largest difference |
| --- | --- | --- |
| Router indices and gate values | `MixtralSparseMoeBlock` | indices equal, weights 1.79e-07 |
| Router indices and gate values | the router of `DeepseekV3MoE`, bias and node limit on | indices equal, weights 2.38e-07 |
| The whole sparse layer's output | `MixtralSparseMoeBlock` | 7.63e-06 on outputs reaching 24.7 |
| Grouped matmul against a per-expert loop | written out one expert at a time | 1.19e-07 |
| 50 training steps, `expert_size` 1 against 4 | the same run at the same seed | equal on every step |
| `tokamax` against `xla` in `ExpertMLP` | `jax.lax.ragged_dot` | bitwise equal, tokamax 0.0.13 on CPU |

The tokamax row needs the package, which Dew does not depend on, so its test skips without it. It was run once with tokamax 0.0.13 and its dependencies staged on `PYTHONPATH`.

The 2000-step run of the 8-expert decoder on FineWeb-Edu, with its load-balance band and its loss curve against a dense model of the same active parameter count, needs a TPU slice and has not been run.
