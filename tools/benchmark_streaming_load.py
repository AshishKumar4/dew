#!/usr/bin/env python3
"""What a published checkpoint costs the host on its way onto a mesh.

`measure` loads Qwen3-0.6B at a pinned revision in one of two ways and prints
one JSON line: wall time, peak RSS (VmHWM), the RSS left after, the placed
bytes, the largest device's share and the first device's allocator peak.

- `host` is the path before streaming: `load_pretrained` builds the whole
  translated tree in host memory, then one `jax.device_put` places it under
  the trainer's `Layout`.
- `stream` is `load_pretrained(mesh=..., layout=...)`: every decoder leaf is
  a `SourceLeaf` over the mapped checkpoint, and each device's shard is read,
  cast and transposed on its own (`dew.interop.streaming`).

`abstract` runs Qwen3.8-27B's real load path onto an 8-device mesh without
its weights. Its safetensors headers come by range reads, each tensor is a
zero-stride view of its shape and dtype, and the placement computes each
leaf's sharding and per-device bytes instead of reading values. It reports
what the recipes cover, what is still built on the host, the largest single
read and the per-device share, and checks the tree against
`jax.eval_shape(model.init)`.

Usage (one process per measurement, so VmHWM is that load's alone):
    flock /tmp/dew-gpu.lock env XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 PYTHONPATH=src \\
        python tools/benchmark_streaming_load.py measure stream float32
    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu PYTHONPATH=src \\
        python tools/benchmark_streaming_load.py measure host bfloat16
    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/benchmark_streaming_load.py abstract
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

QWEN3 = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
QWEN38 = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")


def status() -> dict[str, float]:
    """This process's VmHWM, VmRSS and RssAnon, in GB."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmHWM", "VmRSS", "RssAnon"):
            fields[key] = round(int(value.split()[0]) * 1024 / 1e9, 2)
    return fields


def measure(mode: str, param_dtype: str) -> dict[str, object]:
    import jax

    from dew.interop import load_pretrained
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import build_mesh

    repo, revision = QWEN3
    mesh, layout = MeshSpec(fsdp=jax.device_count()), Layout()
    before = status()
    start = time.perf_counter()
    if mode == "host":
        loaded = load_pretrained(repo, revision=revision, param_dtype=param_dtype)
        variables = jax.device_put(loaded.variables, layout.shardings(build_mesh(mesh), loaded.variables))
    else:
        variables = load_pretrained(repo, revision=revision, param_dtype=param_dtype,
                                    mesh=mesh, layout=layout).variables
    jax.block_until_ready(variables)
    seconds = time.perf_counter() - start
    leaves = jax.tree.leaves(variables)
    per_device = max(sum(shard.data.nbytes for leaf in leaves for shard in leaf.addressable_shards
                         if shard.device == device) for device in jax.devices())
    after = status()
    return {"mode": mode, "param_dtype": param_dtype, "backend": jax.default_backend(),
            "devices": jax.device_count(), "seconds": round(seconds, 1), "rss_before_gb": before["VmRSS"],
            "peak_rss_gb": after["VmHWM"], "rss_after_gb": after["VmRSS"],
            "placed_gb": round(sum(leaf.nbytes for leaf in leaves) / 1e9, 2),
            "max_device_gb": round(per_device / 1e9, 3),
            "device_peak_gb": round((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9, 2)}


def abstract() -> dict[str, object]:
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    import jax
    import ml_dtypes
    import numpy as np
    from huggingface_hub import get_safetensors_metadata

    from dew.interop import hf_decoders, pretrained
    from dew.interop.streaming import SourceLeaf
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import build_mesh

    repo, revision = QWEN38
    stored = {"BF16": ml_dtypes.bfloat16, "F32": np.float32, "F16": np.float16}
    metadata = get_safetensors_metadata(repo, revision=revision)
    table = {name: np.broadcast_to(np.zeros((), stored[info.dtype]), tuple(info.shape))
             for shard in metadata.files_metadata.values() for name, info in shard.tensors.items()}
    report: dict[str, object] = {"tensors": len(table),
                                 "checkpoint_gb": round(sum(t.dtype.itemsize * t.size for t in table.values()) / 1e9, 1)}
    snapshot = hf_decoders._snapshot
    hf_decoders._load_shards = lambda directory: table
    hf_decoders._snapshot = lambda name, revision, weights=True: snapshot(name, revision, weights=False)

    def placement(variables, mesh, layout):
        device_mesh = build_mesh(mesh)
        shardings = layout.shardings(device_mesh, variables)
        layout.check(variables, shardings, device_mesh)
        leaves = jax.tree.leaves(variables)
        lazy = [leaf for leaf in leaves if isinstance(leaf, SourceLeaf)]
        reads = [int(np.prod(sharding.shard_shape(leaf.shape))) * np.dtype(leaf.dtype).itemsize
                 for leaf, sharding in zip(leaves, jax.tree.leaves(shardings), strict=True)]
        report.update({"leaves": len(leaves), "lazy_leaves": len(lazy),
                       "lazy_gb": round(sum(leaf.nbytes for leaf in lazy) / 1e9, 1),
                       "host_built_gb": round(sum(leaf.nbytes for leaf in leaves
                                                  if not isinstance(leaf, SourceLeaf)) / 1e9, 3),
                       "per_device_gb": round(sum(reads) / 1e9, 2),
                       "largest_single_read_gb": round(max(reads) / 1e9, 3)})
        return variables

    pretrained.place = placement
    loaded = pretrained.load_pretrained(repo, revision=revision, dtype="bfloat16", param_dtype="bfloat16",
                                        mesh=MeshSpec(fsdp=8), layout=Layout())
    template = jax.eval_shape(lambda: loaded.model.init(jax.random.key(0), np.zeros((1, 2), np.int32)))
    shapes = {jax.tree_util.keystr(path): tuple(leaf.shape)
              for path, leaf in jax.tree_util.tree_leaves_with_path(template)}
    loaded_shapes = {jax.tree_util.keystr(path): tuple(leaf.shape)
                     for path, leaf in jax.tree_util.tree_leaves_with_path(loaded.variables)}
    report.update({"matches_eval_shape": shapes == loaded_shapes, **status()})
    return report


if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[2], sys.argv[3]) if sys.argv[1] == "measure" else abstract()))
