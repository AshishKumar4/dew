"""Peak host memory of loading a published diffusion pipeline, eager and streamed.

Each mode runs in a fresh process, so one's pages do not count in the
other's, and prints its peak resident set (`ru_maxrss`), the seconds the
load took and a fingerprint of every leaf it bound: path, shape, dtype and
the SHA-256 of its bytes. `compare` loads both ways and holds the two
fingerprints equal, which is the same tree bit for bit.

- `eager`: `load_diffusion_source` as it reads by default, every leaf
  translated on the host, then placed on the device.
- `streamed`: `load_diffusion_source(mesh=MeshSpec())`, every leaf placed
  as it is read (`dew.interop.streaming`).

`--no-text` loads both without the text encoder (`text=False`), whose
prompts a pipeline then takes encoded by its conditioner alone, and
`conditioner` loads that conditioner alone, streamed
(`--conditioner HiddenStatesConditioner` names its class). Run it on a
GPU: on a CPU backend a placed leaf is host memory too. The eager load holds
the translated tree on the host, so measure it only where that fits.

    python tools/pipeline_load_memory.py compare Wan-AI/Wan2.1-T2V-1.3B-Diffusers --no-text
    python tools/pipeline_load_memory.py streamed Tongyi-MAI/Z-Image-Turbo --no-text
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time


def fingerprint(variables) -> dict[str, str]:
    import jax
    import numpy as np

    return {jax.tree_util.keystr(path): f"{leaf.shape} {leaf.dtype} "
            + hashlib.sha256(np.asarray(jax.device_get(leaf)).tobytes()).hexdigest()
            for path, leaf in jax.tree_util.tree_leaves_with_path(variables)}


def load(mode: str, checkpoint: str, revision: str | None, param_dtype: str, text: bool,
         conditioner: str) -> dict:
    import jax

    import dew.inputs.diffusion as conditioners
    from dew.interop.pretrained import load_diffusion_source
    from dew.training import MeshSpec

    start = time.perf_counter()
    if mode == "conditioner":
        encoder = getattr(conditioners, conditioner).from_pretrained(
            checkpoint, revision=revision, dtype=param_dtype, param_dtype=param_dtype, mesh=MeshSpec())
        jax.block_until_ready(encoder.params)
        return {"mode": mode, "checkpoint": checkpoint, "conditioner": conditioner,
                "param_dtype": param_dtype,
                "seconds": round(time.perf_counter() - start, 1),
                "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
                "device": jax.devices()[0].device_kind, "leaves": fingerprint(encoder.params)}
    if mode == "eager":
        loaded = load_diffusion_source(checkpoint, revision=revision, dtype=param_dtype,
                                       param_dtype=param_dtype, text=text)
        variables = jax.device_put(loaded.variables)
    else:
        loaded = load_diffusion_source(checkpoint, revision=revision, dtype=param_dtype,
                                       param_dtype=param_dtype, mesh=MeshSpec(), text=text)
        variables = loaded.variables
    jax.block_until_ready(variables)
    seconds = time.perf_counter() - start
    return {"mode": mode, "checkpoint": checkpoint, "revision": loaded.revision, "param_dtype": param_dtype,
            "text_encoder": text,
            "seconds": round(seconds, 1),
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
            "device": jax.devices()[0].device_kind, "leaves": fingerprint(variables)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("eager", "streamed", "conditioner", "compare"))
    parser.add_argument("checkpoint")
    parser.add_argument("--revision")
    parser.add_argument("--param-dtype", default="bfloat16")
    parser.add_argument("--no-text", action="store_true", help="load without the text encoder")
    parser.add_argument("--conditioner", default="DiffusionConditioner", help="the conditioner's class")
    args = parser.parse_args()
    if args.mode != "compare":
        loaded = load(args.mode, args.checkpoint, args.revision, args.param_dtype, not args.no_text,
                      args.conditioner)
        print(json.dumps({**loaded, "leaves": len(loaded["leaves"]),
                          "tree_sha256": hashlib.sha256(json.dumps(loaded["leaves"], sort_keys=True)
                                                        .encode()).hexdigest()}))
        return
    runs = {}
    for mode in ("eager", "streamed"):
        command = [sys.executable, __file__, mode, args.checkpoint, "--param-dtype", args.param_dtype,
                   *(("--revision", args.revision) if args.revision else ()),
                   *(("--no-text",) if args.no_text else ())]
        runs[mode] = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    same = runs["eager"]["tree_sha256"] == runs["streamed"]["tree_sha256"]
    print(json.dumps({**runs, "same_tree": same}, indent=1))
    if not same:
        raise SystemExit("the eager and the streamed load bound different trees")


if __name__ == "__main__":
    main()
