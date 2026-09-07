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
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

FIXTURE = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
           / "muonclip" / "maxtext_mla.json")


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


if __name__ == "__main__":
    write_maxtext_fixture()
