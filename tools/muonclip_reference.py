#!/usr/bin/env python3
"""Reference QK-Clip math for MuonClip, and the MaxText MLA fixture.

Kimi K2 (arXiv 2507.20534) trains with Muon plus QK-Clip: after the optimizer
step, every attention head whose maximum pre-softmax logit `s` exceeds a
threshold `tau` (100.0) has its query and key projections rescaled by
`gamma = min(1, tau / s)`, queries and keys by `sqrt(gamma)` each so their
product carries `gamma`. For MLA the rotary slice of the query carries the
full `gamma` and the shared rotary key is untouched, since it has no
per-head parameters.

MaxText implements exactly this for MLA in `maxtext/utils/qk_clip_utils.py`
(`_scale_from_max_logits`, `_clip_mla_weight`), refused for any other
attention type, and applies it post-step in the training loop
(`trainers/pre_train/train.py:704`). `write_maxtext_fixture` runs its
kernel rule as published at the 0.2.4 release (`QK_CLIP`) in float64 on
fixed-seed inputs and Dew's float64 per-head scale, and stores that truth
under `tests/fixtures/muonclip/`. Dew's port is tested against that
fixture, not against a live maxtext install. Regenerate with:

```
python tools/muonclip_reference.py
```

For grouped-query and multi-head attention, which MaxText refuses, the
applicable implementation is Megatron Core's (`SelfAttention.clip_qk` and
`_clip_linear_qkv`, megatron/core/transformer/attention.py at core_v0.19.2):
per query group, eta = min(1, tau / the group's largest logit), the group's
query heads scaled by eta^alpha and its key head by eta^(1 - alpha), alpha
0.5. `write_megatron_fixture` runs those two methods as published, fetched
at that commit, on a stand-in layer whose fused `linear_qkv` holds Dew-layout
query, key and value kernels, in float32 and in float64, and stores the
clipped kernels in Dew's layout in `tests/fixtures/muonclip/megatron.npz`:

```
python tools/muonclip_reference.py megatron
```
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

FIXTURE = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
           / "muonclip" / "maxtext_mla.json")
QK_CLIP = ("https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/"
           "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45/src/maxtext/utils/qk_clip_utils.py")
"""MaxText's QK-Clip at its 0.2.4 release (tag maxtext-v0.2.4)."""
MEGATRON = ("https://raw.githubusercontent.com/NVIDIA/Megatron-LM/"
            "4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/transformer/attention.py")
MEGATRON_CASES = {"gqa": (8, 2, [250.0, 10.0, 40.0, 120.0, 10.0, -5.0, 30.0, 99.0]),
                  "mha": (4, 4, [250.0, 10.0, 30.0, 150.0])}
"""Heads, key heads and per-head maxima at tau 100: under grouped queries
one group fires on two heads and the other stays below, a negative maximum
among its quiet ones; under multi-head attention two heads fire and two
hold. A group whose largest maximum is not positive is left out: Megatron's
tau over it goes negative and its square root NaN, where Dew holds the
group (`_clip_scale`)."""
HIDDEN, HEAD_DIM = 16, 4


def clip_scale(s_max: jax.Array, tau: float, guard: float = 1e-6) -> jax.Array:
    """Per-head rescale `gamma`: `min(1, tau / (s + guard))` where the head
    exceeded the threshold, 1.0 elsewhere, including heads whose every logit
    is non-positive. The guard is MaxText's 1e-6 by default (its latent clip
    reads `minimum(1, tau / (s + 1e-6))`, which goes negative for such a
    head; Dew clips nothing there instead, since logits below zero need no
    bounding), and 0 for the paper's and Megatron Core's grouped clip,
    `min(1, tau / s)`."""
    s_max = jnp.asarray(s_max)
    # At least float32, as the clip computes it; a float64 truth's maxima stay float64.
    s_max = s_max.astype(jnp.promote_types(s_max.dtype, jnp.float32))
    return jnp.where(s_max > 0, jnp.minimum(1.0, tau / (s_max + guard)), 1.0)


def clip_qk_kernel(kernel: jax.Array, scale: jax.Array, heads: int) -> jax.Array:
    """A stepped query/key kernel rescaled per head, the paper's update."""
    tail = kernel.shape[-1] // heads
    scaled = (kernel.reshape(kernel.shape[:-1] + (heads, tail))
              * jnp.sqrt(scale)[:, None].astype(kernel.dtype))
    return scaled.reshape(kernel.shape)


def maxtext_clip_mla_weight():
    """MaxText 0.2.4's `_clip_mla_weight`, as published (`QK_CLIP`)."""
    from reference_source import definitions

    return definitions(QK_CLIP, ("_clip_mla_weight",), {"jax": jax, "jnp": jnp})["_clip_mla_weight"]


def write_maxtext_fixture() -> None:
    cases = []
    for seed_case, (batches, heads, nope) in enumerate([(4, 3, 16), (2, 8, 32)]):
        case_rng = np.random.default_rng(100 + seed_case)
        max_logits = (case_rng.normal(60, 60, (batches, heads))
                      .astype(np.float32))
        for layer, width in (("wq_b", nope + 8), ("wkv_b", nope + 16)):
            param = (case_rng.normal(0, 0.02, (5, heads, width))
                     .astype(np.float32))
            # The float64 truth: MaxText's kernel rule run in float64 on the float64 per-head scale Dew's clip
            # takes (`clip_scale`, which clips nothing for a head whose logits are all non-positive, where
            # MaxText's scale goes negative).
            clip = maxtext_clip_mla_weight()
            with jax.enable_x64(new_val=True):
                wide = clip_scale(np.max(max_logits.astype(np.float64), axis=0), 100.0)
                truth = np.asarray(clip(
                    layer, jnp.asarray(param, jnp.float64), jnp.asarray(wide, jnp.float64), nope))
            cases.append({
                "layer": layer, "tau": 100.0, "qk_nope": nope,
                "max_logits": max_logits.tolist(),
                "param": param.tolist(),
                "truth": truth.tolist(),
            })
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(
        {"maxtext": "0.2.4", "source": "maxtext/utils/qk_clip_utils.py:85-101",
         "note": "fixed-seed inputs and _clip_mla_weight's output, as published "
                 "at maxtext-v0.2.4, in float64 on Dew's float64 per-head scale",
         "cases": cases}, indent=2) + "\n")
    print(f"wrote {FIXTURE} with {len(cases)} cases")


def megatron_methods() -> dict:
    """`SelfAttention.clip_qk` and `_clip_linear_qkv`, as published."""
    import torch
    from reference_source import definitions

    names = ("clip_qk", "_clip_linear_qkv")
    scope = definitions(MEGATRON, names, {"torch": torch}, within="SelfAttention")
    return {name: scope[name] for name in names}


def megatron_clip(methods: dict, kernels: dict, heads: int, kv_heads: int, maxima, dtype) -> dict:
    """The clipped query, key and value kernels, `[hidden, width]` each, of a
    layer whose fused `linear_qkv` weight Megatron lays out group by group:
    the group's query heads, then its key head, then its value head."""
    import torch

    group = heads // kv_heads
    rows = {name: torch.as_tensor(kernel.T, dtype=dtype) for name, kernel in kernels.items()}
    fused = torch.cat([torch.cat([rows["q"][g * group * HEAD_DIM:(g + 1) * group * HEAD_DIM],
                                  rows["k"][g * HEAD_DIM:(g + 1) * HEAD_DIM],
                                  rows["v"][g * HEAD_DIM:(g + 1) * HEAD_DIM]]) for g in range(kv_heads)])
    layer = types.SimpleNamespace(
        config=types.SimpleNamespace(qk_clip=True, qk_clip_threshold=100.0, qk_clip_alpha=0.5),
        core_attention=types.SimpleNamespace(current_max_attn_logits=torch.tensor(maxima, dtype=dtype)),
        num_attention_heads_per_partition=heads, num_query_groups_per_partition=kv_heads,
        query_projection_size=heads * HEAD_DIM, kv_projection_size=kv_heads * HEAD_DIM,
        linear_qkv=types.SimpleNamespace(weight=torch.nn.Parameter(fused, requires_grad=False)))
    layer._clip_linear_qkv = types.MethodType(methods["_clip_linear_qkv"], layer)
    methods["clip_qk"](layer)
    clipped = layer.linear_qkv.weight.data.view(kv_heads, (group + 2) * HEAD_DIM, HIDDEN)
    return {"q": clipped[:, :group * HEAD_DIM].reshape(-1, HIDDEN).T.numpy(),
            "k": clipped[:, group * HEAD_DIM:(group + 1) * HEAD_DIM].reshape(-1, HIDDEN).T.numpy(),
            "v": clipped[:, (group + 1) * HEAD_DIM:].reshape(-1, HIDDEN).T.numpy()}


def write_megatron_fixture() -> None:
    import torch

    methods = megatron_methods()
    rng = np.random.default_rng(7)
    arrays = {}
    for name, (heads, kv_heads, maxima) in MEGATRON_CASES.items():
        kernels = {"q": rng.normal(0, 0.5, (HIDDEN, heads * HEAD_DIM)).astype(np.float32),
                   "k": rng.normal(0, 0.5, (HIDDEN, kv_heads * HEAD_DIM)).astype(np.float32),
                   "v": rng.normal(0, 0.5, (HIDDEN, kv_heads * HEAD_DIM)).astype(np.float32)}
        arrays[f"{name}/max_logits"] = np.asarray(maxima, np.float32)
        arrays.update({f"{name}/{part}": kernel for part, kernel in kernels.items()})
        for dtype, tail in ((torch.float32, ""), (torch.float64, "_f64")):
            clipped = megatron_clip(methods, kernels, heads, kv_heads, maxima, dtype)
            arrays.update({f"{name}/clipped_{part}{tail}": kernel for part, kernel in clipped.items()})
    path = FIXTURE.parent / "megatron.npz"
    np.savez(path, **arrays)
    print(f"wrote {path}: {', '.join(MEGATRON_CASES)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["megatron"]:
        write_megatron_fixture()
    else:
        write_maxtext_fixture()
