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
Both are fetched at that commit and run as published under JAX
(`published_methods`: EPS and DEFAULT_MASK_VALUE read from the same commit's
globals.py and common_types.py, the config at base.yml's defaults: exact
top-k, no head chunking, default matmul precision; the float64 run reads the
loss's float32 casts as float64), behind `Indexer.__call__`'s masking and
`jax.lax.top_k` (`published_loss`), each query's KL the published loss of
that query alone. Dew's `dew.nn.mla.indexer_kl` and `top_k_keys` are tested
against what lands under tests/fixtures/indexer, so the suite needs no
maxtext install.

    PYTHONPATH=src:tools python tools/indexer_reference.py

What it writes:

- indexer_loss.npz (`write_legacy`): tiny inputs (the main heads' query and
  key, the indexer's scores with exact-zero ties at the boundary, a packed
  row's segments) and, at float32, each mask's exact top-k selection and the
  per-query KL and its mean for the dense and the sparse loss.
- meta.json: the reference release and lines, and the tensor shapes.
- maxtext.npz (`write_published`): on larger tensors, the dense loss over a
  causal and a packed mask and the sparse loss over the causal one, each
  with its gradient in the indexer's scores, in float32 and in float64, and
  the exact top-k masks.
- recorded_<case>.npz (`recording`), one a case: the same published loss,
  its gradient and each query's KL recorded over check 1's seeds, each
  seed's inputs drawn again by the test (`published_inputs`), the float32
  run's evidence from its float64 run kept with that run's sketch
  (tools/recorded_references.py).

    PYTHONPATH=src:tools python tools/indexer_reference.py recording
"""

import json
import sys
import types
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "indexer"
COMMIT = "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
MAXTEXT = f"https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/{COMMIT}/src/maxtext/"
PUBLISHED = {"batch": 2, "length": 24, "heads": 4, "head_dim": 16, "top_k": 6, "seed": 29}
"""maxtext.npz's sizes: 1152 scores per mask, which the float64 rule's RMS
settles over."""

LEGACY = {"batch": 2, "length": 7, "heads": 3, "head_dim": 8, "top_k": 3, "seed": 23}
"""indexer_loss.npz's sizes: tiny tensors, two documents in the first row and one plus padding in the second,
whose exact top-k selections the tests read whole."""
LEGACY_SEGMENTS = np.array([[1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 1, 0, 0]], np.int32)
"""indexer_loss.npz's two rows: the padding queries of the second attend nothing."""


def write_legacy() -> None:
    """indexer_loss.npz and meta.json: the tiny inputs, with ReLU-shaped scores whose exact zeros tie at the
    top-k boundary, so the exact top-k's earliest-key rule decides the mask; and each mask's selection and
    per-query KL, dense and (causal) sparse, from the published methods at float32."""
    import jax.numpy as jnp

    batch, length, heads, dim = (LEGACY[name] for name in ("batch", "length", "heads", "head_dim"))
    rng = np.random.default_rng(LEGACY["seed"])
    scale = np.float32(dim ** -0.5)
    query = rng.normal(size=(batch, length, heads, dim)).astype(np.float32)
    key = rng.normal(size=(batch, length, heads, dim)).astype(np.float32)
    scores = np.maximum(rng.normal(scale=1.5, size=(batch, length, length)), 0).astype(np.float32)
    arrays = {"query": query, "key": key, "scores": scores, "segments": LEGACY_SEGMENTS, "scale": scale}
    methods, mask_value = published_methods(jnp.float32)
    for mask in ("causal", "packed"):
        attention_mask = jnp.where(published_keep(mask, LEGACY_SEGMENTS), 0.0, mask_value).astype(jnp.float32)
        # The sparse loss adds the mask twice, which overflows a fully masked (padding) row to nan in float32;
        # the reference never scores such a row sparsely, so only the causal mask has a sparse case.
        for sparse in (False, True) if mask == "causal" else (False,):
            value, (indexer_mask, per_query) = published_loss(
                methods, jnp.float32, jnp.asarray(scores), attention_mask, jnp.asarray(query) * scale,
                jnp.asarray(key), sparse, LEGACY["top_k"])
            phase = "sparse" if sparse else "dense"
            arrays[f"{mask}_{phase}_kl"] = np.asarray(per_query)
            arrays[f"{mask}_{phase}_loss"] = np.asarray(value)
            if not sparse:
                arrays[f"{mask}_selected"] = np.asarray(indexer_mask) == 0.0
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES / "indexer_loss.npz", **arrays)
    meta = {
        "maxtext": "0.2.4",
        "commit": COMMIT,
        "source": {
            "calculate_indexer_loss": "src/maxtext/layers/attention_mla.py:1117-1229",
            "generate_mask": "src/maxtext/layers/attention_mla.py:228-268",
            "indexer_call_masking": "src/maxtext/layers/attention_mla.py:428-452",
            "query_scale": "src/maxtext/layers/attention_mla.py:955",
        },
        **{name: LEGACY[name] for name in ("batch", "length", "heads", "head_dim", "top_k", "seed")},
    }
    (FIXTURES / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, value in arrays.items():
        print(f"{name}: {value.shape} {value.dtype}"
              + (f" = {float(value):.6f}" if value.ndim == 0 else ""))


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
    from reference_source import defined_in, definitions, fetched

    scope = {"jax": jax, "jnp": Numpy(dtype), "checkpoint_name": checkpoint_name, "np": np,
             "Array": jax.Array, "Optional": __import__("typing").Optional}
    definitions(MAXTEXT + "utils/globals.py", {"EPS"}, scope)
    definitions(MAXTEXT + "common/common_types.py", {"DEFAULT_MASK_VALUE"}, scope)
    text = fetched(MAXTEXT + "layers/attention_mla.py")
    defined_in(text, {"generate_mask"}, scope, within="Indexer", filename="attention_mla.py")
    defined_in(text, {"calculate_indexer_loss"}, scope, within="MLA", filename="attention_mla.py")
    methods = {name: scope[name] for name in ("generate_mask", "calculate_indexer_loss")}
    return methods, scope["DEFAULT_MASK_VALUE"]


def published_loss(methods: dict, dtype, raw, attention_mask, query, key, sparse: bool, top_k: int):
    """The published loss on the indexer's raw scores behind
    `Indexer.__call__`'s tail (:428-452): the attention mask goes on before
    `jax.lax.top_k` and again after `generate_mask`, and the masked score is
    what the loss receives. Returns the loss, and the indexer's mask with each
    query's KL: the published loss, a mean over every query, of the one query
    alone, its row of scores and masks against every key."""
    import jax

    layer = types.SimpleNamespace(config=types.SimpleNamespace(
        indexer_mask_exact_topk=True, mla_qk_head_chunk_size=0, matmul_precision="default"), dtype=dtype)
    indexer_score = raw + attention_mask
    topk_values, _ = jax.lax.top_k(indexer_score, k=top_k)
    indexer_mask = methods["generate_mask"](layer, indexer_score, topk_values) + attention_mask
    value = methods["calculate_indexer_loss"](layer, indexer_score, query, key, attention_mask, indexer_mask,
                                              sparse_loss=sparse, scaling_factor=1.0)

    def alone(score, one, keys, row_mask, row_indexer_mask):
        return methods["calculate_indexer_loss"](layer, score[None, None], one[None, None], keys[None],
                                                 row_mask[None, None], row_indexer_mask[None, None],
                                                 sparse_loss=sparse, scaling_factor=1.0)

    per_query = jax.vmap(jax.vmap(alone, in_axes=(0, 0, None, 0, 0)))(indexer_score, query, key,
                                                                         attention_mask, indexer_mask)
    return value, (indexer_mask, per_query)


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
    arrays = {"query": query, "key": key, "scores": scores, "segments": PUBLISHED_SEGMENTS,
              "scale": np.float64(scale)}
    keeps = {mask: published_keep(mask) for mask in ("causal", "packed")}
    for dtype, tail in ((jnp.float32, ""), (jnp.float64, "_f64")):
        methods, mask_value = published_methods(dtype)
        with jax.enable_x64(new_val=True):
            arrays.update(published_case(methods, mask_value, dtype, tail, query, key, scores, scale,
                                         keeps, PUBLISHED_CASES))
    np.savez(FIXTURES / "maxtext.npz", **arrays)
    print(f"{FIXTURES / 'maxtext.npz'}: " + ", ".join(
        f"{name} {float(arrays[f'{name}/loss_f64']):.6f}" for name in PUBLISHED_CASES))


PUBLISHED_CASES = {"causal_dense": ("causal", False), "packed_dense": ("packed", False),
                   "causal_sparse": ("causal", True)}
"""Each recorded case: its mask and whether its loss is the sparse one."""


PUBLISHED_SEGMENTS = np.repeat(np.array([[1, 2, 3], [1, 2, 0]], np.int32), PUBLISHED["length"] // 3, axis=1)
"""maxtext.npz's two rows: three documents, then two and padding."""


def published_inputs(seed: int) -> dict[str, np.ndarray]:
    """A seed's inputs to the published loss at maxtext.npz's sizes, as `write_published` draws them:
    the main heads' query and key and the indexer's ReLU-shaped scores, with exact zeros at the top-k
    boundary. Check 1's draw for the recorded reference, the same on every run
    (tests/test_indexer_training.py draws them again)."""
    batch, length, heads = PUBLISHED["batch"], PUBLISHED["length"], PUBLISHED["heads"]
    rng = np.random.default_rng([PUBLISHED["seed"], seed])
    query = rng.normal(size=(batch, length, heads, PUBLISHED["head_dim"])).astype(np.float32)
    key = rng.normal(size=(batch, length, heads, PUBLISHED["head_dim"])).astype(np.float32)
    scores = np.maximum(rng.normal(scale=1.5, size=(batch, length, length)), 0).astype(np.float32)
    return {"query": query, "key": key, "scores": scores}


def published_keep(mask: str, segments: np.ndarray = PUBLISHED_SEGMENTS) -> np.ndarray:
    """The causal, or over `segments` (`[B, S]`, 0 a padding query that may read nothing) the packed causal,
    `[B, S, S]` keys a query may read."""
    length = segments.shape[1]
    causal = np.tril(np.ones((length, length), bool))[None]
    if mask == "causal":
        return np.broadcast_to(causal, (segments.shape[0], length, length))
    return (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0) & causal


def write_published_recording(command: str) -> None:
    """recorded_<case>.npz, one a case, each under the size a fixture keeps in git: the case's published
    loss and its gradient in the indexer's scores recorded over check 1's seeds (`record_case`), the
    float32 run against the float64 run, the count planned on Dew's `indexer_kl` and confirmed on the
    reference."""
    import jax
    import jax.numpy as jnp
    from fetched_fixtures import save
    from recorded_references import provenance, record_case

    from dew.nn.mla import indexer_kl
    from dew.nn.sparse_selection import top_k_keys

    scale = PUBLISHED["head_dim"] ** -0.5
    published = {dtype: published_methods(dtype) for dtype in (jnp.float32, jnp.float64)}
    for name, (mask, sparse) in PUBLISHED_CASES.items():
        def evaluate(inputs, mask=mask, sparse=sparse):
            keep = published_keep(mask)
            found = []
            for dtype in (jnp.float32, jnp.float64):
                methods, mask_value = published[dtype]
                attention_mask = jnp.where(keep, 0.0, mask_value).astype(dtype)
                scaled = jnp.asarray(inputs["query"], dtype) * dtype(scale)
                (value, (_, per_query)), gradient = jax.value_and_grad(
                    lambda raw, methods=methods, attention_mask=attention_mask, scaled=scaled, dtype=dtype:
                    published_loss(methods, dtype, raw, attention_mask, scaled,
                                   jnp.asarray(inputs["key"], dtype), sparse, PUBLISHED["top_k"]),
                    has_aux=True)(jnp.asarray(inputs["scores"], dtype))
                found.append({"loss": np.asarray(value), "gradient": np.asarray(gradient),
                              "per query": np.asarray(per_query)})
            return found[0], found[1]

        def proxy(inputs, mask=mask, sparse=sparse):
            keep = jnp.asarray(published_keep(mask))
            region = top_k_keys(inputs["scores"], keep, PUBLISHED["top_k"]) if sparse else keep

            def loss(scores):
                per_query = indexer_kl(scores, inputs["query"], inputs["key"], region, scale)
                return jnp.mean(per_query), per_query

            (value, per_query), gradient = jax.value_and_grad(loss, has_aux=True)(inputs["scores"])
            return {"loss": value, "gradient": gradient, "per query": per_query}

        with jax.enable_x64(new_val=True):
            recording = record_case(name, published_inputs, evaluate, proxy, truth_proxy=proxy,
                                    store_inputs=False, summary=True)
        recording[f"{name}.provenance"] = provenance("MaxText MLA.calculate_indexer_loss", COMMIT, "cpu",
                                                     command, dtype="float32", top_k=PUBLISHED["top_k"])
        written = save(FIXTURES / f"recorded_{name}.npz", recording, reference="MaxText 0.2.4",
                       license="Apache-2.0")
        print(f"{written}: {written.stat().st_size} bytes", flush=True)


def published_case(methods, mask_value, dtype, tail, query, key, scores, scale, keeps, cases) -> dict:
    """Every case's published loss and gradient at `dtype`, and at float32
    its mask."""
    import jax
    import jax.numpy as jnp

    arrays = {}
    for name, (mask, sparse) in cases.items():
        attention_mask = jnp.where(keeps[mask], 0.0, mask_value).astype(dtype)
        scaled = jnp.asarray(query, dtype) * dtype(scale)
        (value, (indexer_mask, _)), gradient = jax.value_and_grad(
            lambda raw, attention_mask=attention_mask, scaled=scaled, dtype=dtype, sparse=sparse:
            published_loss(methods, dtype, raw, attention_mask, scaled, jnp.asarray(key, dtype), sparse,
                           PUBLISHED["top_k"]),
            has_aux=True)(jnp.asarray(scores, dtype))
        arrays[f"{name}/loss{tail}"] = np.asarray(value)
        arrays[f"{name}/grad{tail}"] = np.asarray(gradient)
        if tail == "":
            arrays[f"{mask}/selected"] = np.asarray(indexer_mask) == 0.0
    return arrays


if __name__ == "__main__":
    if sys.argv[1:] == ["recording"]:
        write_published_recording("PYTHONPATH=src:tools python tools/indexer_reference.py recording")
    else:
        write_legacy()
        write_published()
