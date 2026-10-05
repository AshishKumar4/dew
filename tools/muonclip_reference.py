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
(`trainers/pre_train/train.py:704`). `clip_mla_reference` below transcribes
that file's two functions; `write_maxtext_fixture` runs the real ones from
a pip-installable maxtext checkout against fixed-seed inputs and stores the
outputs under `tests/fixtures/muonclip/`. Dew's port is tested against that
fixture, not against a live maxtext install.
Regenerate with a maxtext checkout on the path:

```
PYTHONPATH=/path/to/maxtext-parent python tools/muonclip_reference.py
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

import ast
import importlib
import json
import sys
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

FIXTURE = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
           / "muonclip" / "maxtext_mla.json")
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


def clip_scale(s_max: jax.Array, tau: float) -> jax.Array:
    """Per-head rescale `gamma`: `min(1, tau / s)` where the head exceeded
    the threshold, 1.0 elsewhere, including heads whose every logit is
    non-positive. MaxText's formula reads `minimum(1, tau / (s + 1e-6))`,
    which goes negative for such a head; Dew clips nothing there instead,
    since logits below zero need no bounding."""
    s_max = jnp.asarray(s_max, jnp.float32)
    return jnp.where(s_max > 0, jnp.minimum(1.0, tau / (s_max + 1e-6)), 1.0)


def clip_qk_kernel(kernel: jax.Array, scale: jax.Array, heads: int) -> jax.Array:
    """A stepped query/key kernel rescaled per head, the paper's update."""
    tail = kernel.shape[-1] // heads
    scaled = (kernel.reshape(kernel.shape[:-1] + (heads, tail))
              * jnp.sqrt(scale)[:, None].astype(kernel.dtype))
    return scaled.reshape(kernel.shape)


def clip_mla_reference(param: np.ndarray, scale: np.ndarray, qk_nope: int,
                       layer: str) -> np.ndarray:
    """MaxText's `_clip_mla_weight` (`maxtext/utils/qk_clip_utils.py:91`),
    transcribed: the head slice of a `wq_b`/`wkv_b` kernel by `sqrt(scale)`,
    the tail of `wq_b` (rotary queries) by `scale`, the tail of `wkv_b`
    (values) left alone."""
    scale_b = np.expand_dims(np.asarray(scale, np.float32), -1)
    head, tail = param[..., :qk_nope], param[..., qk_nope:]
    head_new = head * np.sqrt(scale_b).astype(param.dtype)
    if layer == "wq_b":
        tail_new = tail * scale_b.astype(param.dtype)
    else:
        tail_new = tail
    return np.concatenate([head_new, tail_new], axis=-1)


def write_maxtext_fixture() -> None:
    sys.path.insert(0, "/tmp/mt")
    qk_clip_utils = importlib.import_module("maxtext.utils.qk_clip_utils")
    cases = []
    for seed_case, (batches, heads, nope) in enumerate([(4, 3, 16), (2, 8, 32)]):
        case_rng = np.random.default_rng(100 + seed_case)
        max_logits = (case_rng.normal(60, 60, (batches, heads))
                      .astype(np.float32))
        for layer, width in (("wq_b", nope + 8), ("wkv_b", nope + 16)):
            param = (case_rng.normal(0, 0.02, (5, heads, width))
                     .astype(np.float32))
            scale = np.asarray(qk_clip_utils._scale_from_max_logits(
                jnp.asarray(max_logits), 100.0))
            expected = np.asarray(qk_clip_utils._clip_mla_weight(
                layer, jnp.asarray(param), jnp.asarray(scale), nope))
            check = clip_mla_reference(param, scale, nope, layer)
            assert np.max(np.abs(check - expected)) == 0.0
            cases.append({
                "layer": layer, "tau": 100.0, "qk_nope": nope,
                "max_logits": max_logits.tolist(),
                "param": param.tolist(),
                "expected": np.asarray(expected).tolist(),
            })
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(
        {"maxtext": "0.2.4", "source": "maxtext/utils/qk_clip_utils.py:85-101",
         "note": "fixed-seed outputs of _scale_from_max_logits and "
                 "_clip_mla_weight; the generator asserts the transcription "
                 "in this file reproduces them bitwise",
         "cases": cases}, indent=2) + "\n")
    print(f"wrote {FIXTURE} with {len(cases)} cases")


def megatron_methods() -> dict:
    """`SelfAttention.clip_qk` and `_clip_linear_qkv`, as published."""
    text = urllib.request.urlopen(MEGATRON).read().decode()
    attention = next(node for node in ast.parse(text).body
                     if isinstance(node, ast.ClassDef) and node.name == "SelfAttention")
    methods = [node for node in attention.body if isinstance(node, ast.FunctionDef)
               and node.name in ("clip_qk", "_clip_linear_qkv")]
    import torch

    scope = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "attention.py", "exec"), scope)
    return {name: scope[name] for name in ("clip_qk", "_clip_linear_qkv")}


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
