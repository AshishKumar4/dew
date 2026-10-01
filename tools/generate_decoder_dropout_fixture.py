"""Export the zero-dropout decoder's CPU fp32 forward and gradient graph.

The committed export was recorded at 42ddfc14, before the dropout change.
Run this generator with PYTHONPATH naming that checkout's src directory.
"""
import hashlib
import importlib.util
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from dew.nn.backbones.causal_transformer import CausalTransformer

model = CausalTransformer(vocab_size=17, emb_features=8, num_layers=1, num_heads=2,
                         mlp_features=16, max_seq_len=8, attention_impl="reference",
                         precision=jax.lax.Precision.HIGHEST)
with jax.default_device(jax.devices("cpu")[0]):
    ids = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], jnp.int32)
    params = model.init(jax.random.key(2026), ids)
    def forward_and_gradient(params, ids):
        forward = model.apply(params, ids, train=True)
        gradient = jax.grad(lambda p: jnp.sum(model.apply(p, ids, train=True)))(params)
        return forward, gradient

    exported = export.export(jax.jit(forward_and_gradient), platforms=("cpu",))(params, ids)
directory = Path(__file__).resolve().parents[1] / "tests/fixtures"
stem = "decoder-default-dropout"
held = {"ids": np.asarray(ids)}
for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    held["params/" + "/".join(item.key for item in path)] = np.asarray(leaf)
np.savez(directory / f"{stem}.npz", **held)
serialized = exported.serialize()
(directory / f"{stem}.jaxexport").write_bytes(serialized)
checkout = Path(importlib.util.find_spec("dew").origin).parents[2]
(directory / f"{stem}.json").write_text(json.dumps({
    "dew_commit": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
    "jax_version": jax.__version__, "jaxlib_version": version("jaxlib"), "platforms": ["cpu"],
    "calling_convention_version": exported.calling_convention_version,
    "computation": "forward logits and parameter gradient of their sum, with weights and token IDs as arguments",
    "export_sha256": hashlib.sha256(serialized).hexdigest(),
    "dtype": "float32", "precision": "highest",
    "reason": "Execute old and current graphs on the same CPU; cross-CPU fp32 outputs can differ by ULPs.",
}, indent=2) + "\n")
