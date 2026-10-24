"""Size one live Unified reference workload without evaluating Dew's candidate arm.

    PYTHONPATH=src python tools/qualify_gemma4_unified.py forward --output qualification.json

Run in the project's reference environment. The tests own the workloads; reference_error owns
the sizing rule. --start and --most bound one reference-only sizing job.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def workload(site, directory):
    from test_gemma4_unified import (
        FIXTURE,
        cached_functions,
        equal_geometry_source,
        forward_functions,
        gradient_functions,
        optional_source,
        source_pair,
        unified_draw,
    )

    source = source_pair(FIXTURE, "bfloat16" if site == "bfloat16" else "float32")
    if site in ("backward", "cached"):
        return (gradient_functions if site == "backward" else cached_functions)(source)[1:]
    if site in ("text", "vision", "audio", "equal"):
        source, draw = (equal_geometry_source(source, directory) if site == "equal"
                        else optional_source(source, directory, site))
    else:
        def draw(seed):
            return unified_draw(source[1].config, seed, side=4 if site == "long" else 2)
    _, reference, truth = forward_functions(source)
    return reference, truth, lambda seed: (draw(seed),)


def main():
    import numpy as np
    from reference_error import FEWEST_SEEDS, seeds_for
    from reference_source import git_head
    from test_gemma4_unified import FIXTURE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=("forward", "long", "bfloat16", "backward", "cached",
                                         "text", "vision", "audio", "equal"))
    parser.add_argument("--start", type=int, default=FEWEST_SEEDS)
    parser.add_argument("--most", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        reference, truth, draw = workload(args.site, Path(directory))

        @functools.lru_cache(maxsize=1)
        def observed(seed):
            # The reference-vs-itself control asks for the same output twice in immediate succession.
            return reference(*draw(seed))

        count = seeds_for(lambda seed: observed(int(seed)),
                          lambda seed: (np.asarray(seed, np.int64),), args.site,
                          start=args.start, most=args.most, truth_fn=lambda seed: truth(*draw(int(seed))))
    record = {"site": args.site, "seeds": count, "revision": git_head(ROOT),
              "model_sha256": hashlib.sha256((FIXTURE / "model.safetensors").read_bytes()).hexdigest(),
              "method": "reference_vs_reference", "start": args.start, "most": args.most,
              "packages": {name: importlib.metadata.version(name)
                           for name in ("jax", "torch", "transformers", "numpy")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.site}: {count} declared seeds", flush=True)


if __name__ == "__main__":
    main()
