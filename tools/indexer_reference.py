#!/usr/bin/env python3
"""Write the sparse-indexer training fixtures tests/test_indexer_training.py
checks against.

The reference is MaxText 0.2.4 (commit 538fe7a3), the release that trains
DeepSeek-V3.2's lightning indexer: `MLA.calculate_indexer_loss`
(src/maxtext/layers/attention_mla.py:1117-1229) is the KL of the indexer's
softmax from the head-summed, L1-normalised attention distribution, over
every allowed key in the dense warm-up (`indexer_sparse_training=False`)
and over the indexer's own top-k in sparse training; `Indexer.generate_mask`
(:228-268) is the exact top-k the mask keeps, ties to the earliest key.
Both are transcribed here in NumPy, line for line with the masks the
reference adds (`DEFAULT_MASK_VALUE`, common_types.py:75; `EPS`,
utils/globals.py:42), and run on fixed-seed tiny tensors. Dew's
`dew.nn.mla.indexer_kl` and `top_k_keys` are tested against what lands
under tests/fixtures/indexer, so the suite needs no maxtext install.

    python tools/indexer_reference.py

What it writes:

- indexer_loss.npz: the inputs (the main heads' query and key, the
  indexer's scores with exact-zero ties at the boundary, the causal and the
  packed additive masks) and the outputs (per-query KL and its mean for the
  dense and the sparse loss, the exact top-k mask).
- meta.json: the reference release and lines, and the tensor shapes.
- maxtext.npz: `MLA.calculate_indexer_loss` and `Indexer.generate_mask`
  themselves, fetched at that commit and run as published under JAX (EPS
  and DEFAULT_MASK_VALUE read from the same commit's globals.py and
  common_types.py, the config at base.yml's defaults: exact top-k, no head
  chunking, default matmul precision; the float64 run reads the loss's
  float32 casts as float64), behind `Indexer.__call__`'s masking
  and `jax.lax.top_k`, on larger tensors: the dense loss over a causal and
  a packed mask and the sparse loss over the causal one, each with its
  gradient in the indexer's scores, in float32 and in float64, and the
  exact top-k masks.
"""

import ast
import json
import types
import urllib.request
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "indexer"
COMMIT = "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
MAXTEXT = f"https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/{COMMIT}/src/maxtext/"
PUBLISHED = {"batch": 2, "length": 24, "heads": 4, "head_dim": 16, "top_k": 6, "seed": 29}
"""maxtext.npz's sizes: 1152 scores per mask, which the float64 rule's RMS
settles over."""

BATCH, LENGTH, HEADS, HEAD_DIM = 2, 7, 3, 8
TOP_K = 3
# MaxText common_types.py:75 and utils/globals.py:42.
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
EPS = 1e-8


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def generate_mask(indexer_score: np.ndarray, topk_values: np.ndarray) -> np.ndarray:
    """`Indexer.generate_mask` with `indexer_mask_exact_topk` (:245-264):
    0.0 on the kept keys, DEFAULT_MASK_VALUE elsewhere."""
    cutoff = topk_values[..., -1][..., None]
    k = topk_values.shape[-1]
    strictly_greater = indexer_score > cutoff
    equal = indexer_score == cutoff
    sg_rank = np.cumsum(strictly_greater.astype(np.int32), axis=-1)
    eq_rank = np.cumsum(equal.astype(np.int32), axis=-1)
    sg_kept = strictly_greater & (sg_rank <= k)
    num_sg_kept = np.minimum(sg_rank[..., -1:], k)
    num_eq_to_keep = k - num_sg_kept
    selected = sg_kept | (equal & (eq_rank <= num_eq_to_keep))
    return np.where(selected, np.float32(0.0), np.float32(DEFAULT_MASK_VALUE))


def indexer_call(indexer_score: np.ndarray, attention_mask: np.ndarray, top_k: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The tail of `Indexer.__call__` (:428-452): the attention mask goes
    on before the top-k and again after it, and the masked score is what
    the loss receives."""
    indexer_score = indexer_score + attention_mask
    topk_values = -np.sort(-indexer_score, axis=-1)[..., :top_k]
    indexer_mask = generate_mask(indexer_score, topk_values)
    indexer_mask = indexer_mask + attention_mask
    return indexer_mask, indexer_score


def calculate_indexer_loss(indexer_score: np.ndarray, query: np.ndarray, key: np.ndarray,
                           attention_mask: np.ndarray, indexer_mask: np.ndarray,
                           sparse_loss: bool) -> tuple[np.ndarray, np.ndarray]:
    """`MLA.calculate_indexer_loss` (:1117-1229) at scaling factor 1, the
    native (unchunked) head path; returns the per-query KL sum with the
    mean the reference reports."""
    if sparse_loss:
        indexer_score = indexer_score + indexer_mask
    indexer_probs = softmax(indexer_score.astype(np.float32), axis=-1)
    attention_scores = np.einsum("bthd,bshd->bhts", query, key)
    if sparse_loss:
        attention_scores = attention_scores + indexer_mask[:, None, :, :]
    else:
        attention_scores = attention_scores + attention_mask[:, None, :, :]
    attention_probs = np.sum(softmax(attention_scores.astype(np.float32), axis=-1), axis=1)
    attention_probs = attention_probs / (np.sum(attention_probs, axis=-1, keepdims=True) + EPS)
    log_attention_probs = np.log(attention_probs + EPS)
    log_indexer_probs = np.log(indexer_probs + EPS)
    kl_per_token = attention_probs * (log_attention_probs - log_indexer_probs)
    per_query = np.sum(kl_per_token, axis=-1)
    return per_query, np.asarray(np.mean(per_query), np.float32)


def causal_mask() -> np.ndarray:
    keep = np.tril(np.ones((LENGTH, LENGTH), bool))
    return np.broadcast_to(np.where(keep, 0.0, DEFAULT_MASK_VALUE).astype(np.float32),
                           (BATCH, LENGTH, LENGTH)).copy()


def packed_mask(segments: np.ndarray) -> np.ndarray:
    """Block-diagonal and causal over `[B, S]` segment ids, segment 0 a
    padding query that may attend nothing."""
    inside = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    keep = inside & np.tril(np.ones((LENGTH, LENGTH), bool))[None]
    return np.where(keep, 0.0, DEFAULT_MASK_VALUE).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(23)
    scale = np.float32(HEAD_DIM ** -0.5)
    query = rng.normal(size=(BATCH, LENGTH, HEADS, HEAD_DIM)).astype(np.float32)
    key = rng.normal(size=(BATCH, LENGTH, HEADS, HEAD_DIM)).astype(np.float32)
    # ReLU-shaped scores with exact zeros, so the top-k boundary ties and
    # the exact top-k's earliest-key rule decides the mask.
    scores = np.maximum(rng.normal(scale=1.5, size=(BATCH, LENGTH, LENGTH)), 0).astype(np.float32)
    # Two documents in the first row and one plus padding in the second;
    # the padding queries of the second row attend nothing.
    segments = np.array([[1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 1, 0, 0]], np.int32)
    arrays = {"query": query, "key": key, "scores": scores, "segments": segments,
              "scale": scale}
    # MaxText's query carries the softmax scale (attention_mla.py:955).
    scaled = query * scale
    for name, mask in (("causal", causal_mask()), ("packed", packed_mask(segments))):
        indexer_mask, masked_scores = indexer_call(scores, mask, TOP_K)
        dense_kl, dense_loss = calculate_indexer_loss(
            masked_scores, scaled, key, mask, indexer_mask, sparse_loss=False)
        arrays[f"{name}_dense_kl"] = dense_kl
        arrays[f"{name}_dense_loss"] = dense_loss
        arrays[f"{name}_selected"] = indexer_mask == 0.0
        if name == "causal":
            # The sparse branch adds the mask twice, which overflows a fully
            # masked (padding) row to nan in fp32; the reference never
            # scores such a row sparsely, so only the causal case does.
            sparse_kl, sparse_loss = calculate_indexer_loss(
                masked_scores, scaled, key, mask, indexer_mask, sparse_loss=True)
            arrays["causal_sparse_kl"] = sparse_kl
            arrays["causal_sparse_loss"] = sparse_loss
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES / "indexer_loss.npz", **arrays)
    meta = {
        "maxtext": "0.2.4",
        "commit": "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45",
        "source": {
            "calculate_indexer_loss": "src/maxtext/layers/attention_mla.py:1117-1229",
            "generate_mask": "src/maxtext/layers/attention_mla.py:228-268",
            "indexer_call_masking": "src/maxtext/layers/attention_mla.py:428-452",
            "query_scale": "src/maxtext/layers/attention_mla.py:955",
        },
        "batch": BATCH, "length": LENGTH, "heads": HEADS, "head_dim": HEAD_DIM,
        "top_k": TOP_K, "seed": 23,
    }
    (FIXTURES / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, value in arrays.items():
        print(f"{name}: {value.shape} {value.dtype}"
              + (f" = {float(value):.6f}" if value.ndim == 0 else ""))


def fetched(path: str) -> str:
    return urllib.request.urlopen(MAXTEXT + path).read().decode()


def constant(text: str, name: str, scope: dict):
    """A module-level constant of the fetched source, evaluated as published."""
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return eval(compile(ast.Expression(node.value), name, "eval"), scope)


class Numpy(types.ModuleType):
    """jax.numpy with `float32`, which the loss casts its softmaxes to, as
    the run's precision: the float64 run is the same code in float64."""

    def __init__(self, dtype):
        super().__init__("jax.numpy")
        self.float32 = dtype

    def __getattr__(self, name):
        import jax.numpy as jnp

        return getattr(jnp, name)


def published_methods(dtype) -> tuple[dict, float]:
    """`Indexer.generate_mask` and `MLA.calculate_indexer_loss` from
    attention_mla.py, with the module scope they read, at `dtype`."""
    import jax
    from jax.ad_checkpoint import checkpoint_name

    scope = {"jax": jax, "jnp": Numpy(dtype), "checkpoint_name": checkpoint_name, "np": np,
             "Array": jax.Array, "Optional": __import__("typing").Optional}
    scope["EPS"] = constant(fetched("utils/globals.py"), "EPS", scope)
    mask_value = constant(fetched("common/common_types.py"), "DEFAULT_MASK_VALUE", scope)
    scope["DEFAULT_MASK_VALUE"] = mask_value
    text = fetched("layers/attention_mla.py")
    methods = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name in ("Indexer", "MLA"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {"generate_mask",
                                                                       "calculate_indexer_loss"}:
                    exec(compile(ast.Module(body=[item], type_ignores=[]), "attention_mla.py", "exec"), scope)
                    methods[item.name] = scope[item.name]
    assert set(methods) == {"generate_mask", "calculate_indexer_loss"}
    return methods, mask_value


def published_loss(methods: dict, dtype, raw, attention_mask, query, key, sparse: bool):
    """The published loss on the indexer's raw scores behind
    `Indexer.__call__`'s tail (:428-452): the attention mask goes on before
    `jax.lax.top_k` and again after `generate_mask`, and the masked score is
    what the loss receives. Returns the loss and the indexer's mask."""
    import jax

    layer = types.SimpleNamespace(config=types.SimpleNamespace(
        indexer_mask_exact_topk=True, mla_qk_head_chunk_size=0, matmul_precision="default"), dtype=dtype)
    indexer_score = raw + attention_mask
    topk_values, _ = jax.lax.top_k(indexer_score, k=PUBLISHED["top_k"])
    indexer_mask = methods["generate_mask"](layer, indexer_score, topk_values) + attention_mask
    value = methods["calculate_indexer_loss"](layer, indexer_score, query, key, attention_mask, indexer_mask,
                                              sparse_loss=sparse, scaling_factor=1.0)
    return value, indexer_mask


def write_published() -> None:
    """maxtext.npz: the published loss, gradient and mask in both precisions."""
    import jax
    import jax.numpy as jnp

    batch, length, heads = PUBLISHED["batch"], PUBLISHED["length"], PUBLISHED["heads"]
    rng = np.random.default_rng(PUBLISHED["seed"])
    scale = PUBLISHED["head_dim"] ** -0.5
    query = rng.normal(size=(batch, length, heads, PUBLISHED["head_dim"])).astype(np.float32)
    key = rng.normal(size=(batch, length, heads, PUBLISHED["head_dim"])).astype(np.float32)
    scores = np.maximum(rng.normal(scale=1.5, size=(batch, length, length)), 0).astype(np.float32)
    segments = np.repeat(np.array([[1, 2, 3], [1, 2, 0]], np.int32), length // 3, axis=1)
    arrays = {"query": query, "key": key, "scores": scores, "segments": segments,
              "scale": np.float64(scale)}
    keeps = {"causal": np.broadcast_to(np.tril(np.ones((length, length), bool)), (batch, length, length)),
             "packed": ((segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
                        & np.tril(np.ones((length, length), bool))[None])}
    cases = {"causal_dense": ("causal", False), "packed_dense": ("packed", False),
             "causal_sparse": ("causal", True)}
    for dtype, tail in ((jnp.float32, ""), (jnp.float64, "_f64")):
        methods, mask_value = published_methods(dtype)
        with jax.enable_x64(new_val=True):
            arrays.update(published_case(methods, mask_value, dtype, tail, query, key, scores, scale,
                                         keeps, cases))
    np.savez(FIXTURES / "maxtext.npz", **arrays)
    print(f"{FIXTURES / 'maxtext.npz'}: " + ", ".join(
        f"{name} {float(arrays[f'{name}/loss_f64']):.6f}" for name in cases))


def published_case(methods, mask_value, dtype, tail, query, key, scores, scale, keeps, cases) -> dict:
    """Every case's published loss and gradient at `dtype`, and at float32
    its mask."""
    import jax
    import jax.numpy as jnp

    arrays = {}
    for name, (mask, sparse) in cases.items():
        attention_mask = jnp.where(keeps[mask], 0.0, mask_value).astype(dtype)
        scaled = jnp.asarray(query, dtype) * dtype(scale)
        (value, indexer_mask), gradient = jax.value_and_grad(
            lambda raw, attention_mask=attention_mask, scaled=scaled, dtype=dtype, sparse=sparse:
            published_loss(methods, dtype, raw, attention_mask, scaled, jnp.asarray(key, dtype), sparse),
            has_aux=True)(jnp.asarray(scores, dtype))
        arrays[f"{name}/loss{tail}"] = np.asarray(value)
        arrays[f"{name}/grad{tail}"] = np.asarray(gradient)
        if tail == "":
            arrays[f"{mask}/selected"] = np.asarray(indexer_mask) == 0.0
    return arrays


if __name__ == "__main__":
    main()
    write_published()
