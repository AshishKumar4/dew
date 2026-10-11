"""Stop a GPU session whose ref's Pallas kernels the kernel lab has not verified on its GPU.

    python -m tools.colab.lab_gate --gpu a100 [--tree .] [--store ~/.config/dew/dew-data.json] || exit 1

A session that verifies a ref's kernels and then trains on them passes its preflight before the
record it is to write exists. Its preflight lets it through only where a shell script runs this gate
before anything that trains and stops on the gate's failure (`preflight.runtime_gate`). The gate
reads what the preflight would have (`preflight.lab_verified`): the ref's KERNELS row for the GPU's
generation, the digest of its kernels and pinned JAX, and `lab/verified/<gpu>/<digest>.json`, with
DEW_DATA_READ_TOKEN or the store's read or lab token (`preflight.store_read`). It prints what it
read as JSON and exits 0 where the row names no Pallas kernel or the record is ok, 1 otherwise.
"""

import argparse
import json
from pathlib import Path

from tools.colab.preflight import LAB_GPUS, STORE_CONFIG, lab_verified


def gate(tree: Path, gpu: str, config: Path = STORE_CONFIG) -> tuple[dict, list[str]]:
    """What the preflight's lab check reads of `tree` on the lab's `gpu`, and why it fails, if it does."""
    hardware = {name: label for label, name in LAB_GPUS.items()}[gpu]
    failures: list[str] = []
    return lab_verified(tree.resolve(), hardware, failures, config=config), failures


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--gpu", required=True, choices=sorted(LAB_GPUS.values()))
    parser.add_argument("--tree", type=Path, default=Path("."), help="the Dew checkout the session trains")
    parser.add_argument("--store", type=Path, default=STORE_CONFIG, help="the store's JSON (provision.sh's)")
    args = parser.parse_args()
    found, failures = gate(args.tree, args.gpu, args.store)
    print(json.dumps({**found, "failures": failures}, indent=1, sort_keys=True))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
