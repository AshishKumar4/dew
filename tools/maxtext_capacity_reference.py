#!/usr/bin/env python3
"""Write the MoE capacity fixture with MaxText's own token dropping.

MaxText's `RoutedMoE.generate_masks` (src/maxtext/layers/moe.py at commit
733b5a91, checked against its SHA-256) decides which routed slots an
expert's capacity keeps: each sequence queues its slots at each expert in
token order, a token's choices in order, and keeps
`max(ceil(length * top_k / experts) * capacity_factor, capacity_factor)`
of them. The tool runs that method's body unchanged, extracted from the
pinned file and called on a stand-in `self` that carries its three fields
and the identity for its sharding hint, and records each case's routing, the
per-expert routing weights and the dispatch and combine masks it returns.

Run with the Dew test environment's jax, on CPU:

    python tools/maxtext_capacity_reference.py OUTPUT.npz
"""

import ast
import hashlib
import json
import math
import sys
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO, COMMIT = "AI-Hypercomputer/maxtext", "733b5a9147ed87ecf509fbb4779ed187f56c6045"
PATH = "src/maxtext/layers/moe.py"
DIGEST = "1d427dcdd4db0e8b7677d9c57426f69390e20883b676caacae0d6f83fea7a5d1"
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "maxtext" / COMMIT
# case: (batch, length, experts, top_k, capacity factor), each dropping slots
CASES = {
    "whole-sequences": (4, 12, 8, 2, 1.0),
    "short": (3, 5, 8, 1, 1.0),
    "loose": (2, 10, 4, 1, 1.25),
    "tight": (2, 9, 16, 4, 0.5),
}
SEED = 73


def generate_masks():
    """`RoutedMoE.generate_masks` at `COMMIT`, its body unchanged, as a function."""
    local = CACHE / PATH
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{PATH}").read())
    found = hashlib.sha256(local.read_bytes()).hexdigest()
    if found != DIGEST:
        raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {DIGEST}")
    tree = ast.parse(local.read_text())
    [routed] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RoutedMoE"]
    [method] = [node for node in routed.body
                if isinstance(node, ast.FunctionDef) and node.name == "generate_masks"]
    namespace = {"math": math, "jax": jax, "jnp": jnp,
                 "max_logging": types.SimpleNamespace(log=lambda message: None)}
    exec(compile(ast.Module([method], type_ignores=[]), str(local), "exec"), namespace)
    return namespace["generate_masks"]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    masks = generate_masks()
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}
    for case, (batch, length, experts, top_k, factor) in CASES.items():
        scores = rng.normal(size=(batch, length, experts))
        scores[..., :2] += 1.5  # two popular experts, so queues overflow
        indices = np.argsort(-scores, axis=-1)[..., :top_k].astype(np.int32)
        weights = rng.uniform(0.1, 0.9, size=(batch, length, top_k)).astype(np.float32)
        # The per-expert routing weights MaxText's masks scale: each token's
        # weight at the experts it chose, zero elsewhere.
        probs = np.zeros((batch, length, experts), np.float32)
        np.put_along_axis(probs, indices, weights, axis=-1)
        layer = types.SimpleNamespace(num_experts=experts, num_experts_per_tok=top_k,
                                      config=types.SimpleNamespace(capacity_factor=factor),
                                      _maybe_shard_with_logical=lambda value, axes: value)
        dispatch, combine = masks(layer, jnp.asarray(indices), jnp.asarray(probs))
        kept = np.asarray(dispatch).any(-1)  # [batch, length, experts]
        dropped = int(np.count_nonzero(probs) - np.count_nonzero(kept))
        assert dropped > 0, case
        arrays.update({f"{case}/indices": indices, f"{case}/weights": weights,
                       f"{case}/dispatch": np.asarray(dispatch), f"{case}/combine": np.asarray(combine)})
        print(f"{case}: capacity {np.asarray(dispatch).shape[-1]}, {dropped} of {probs.astype(bool).sum()} "
              f"slots dropped")
    meta = {"repo": REPO, "commit": COMMIT, "path": PATH, "sha256": DIGEST,
            "cases": {case: dict(zip(("batch", "length", "experts", "top_k", "factor"), shape, strict=True))
                      for case, shape in CASES.items()}, "jax": jax.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.3f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
