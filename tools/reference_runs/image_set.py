#!/usr/bin/env python3
"""Fixed images and their epoch orders, the one batch source both diffusion
frameworks read.

Reads the first `--images` records of `tools/prepare_images.py` ArrayRecord
shards (uint8 HxWx3 already at training size), in shard order, and draws
`--epochs` permutations from one numpy generator seeded with `--seed`.

    python tools/reference_runs/image_set.py --shards 00000.array_record 00001.array_record \\
        --images 4096 --epochs 2 --seed 0 --out flowers-4096.npz
"""

import argparse
import json

import numpy as np
from common import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--images", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from dew.data.images import DecodeImage
    from dew.data.sources.records import ArrayRecords

    # Dew's ArrayRecord reader over the shards in the order given, each record decoded as Dew decodes a
    # prepared image's, a malformed one failing the set.
    records, decode = ArrayRecords(args.shards), DecodeImage(rejects=False)
    try:
        if len(records) < args.images:
            raise ValueError(f"the shards hold {len(records)} records, {args.images} are needed")
        images = [decode.map(records[index])["image"] for index in range(args.images)]
    finally:
        records.close()
    generator = np.random.default_rng(args.seed)
    order = np.stack([generator.permutation(args.images) for _ in range(args.epochs)]).astype(np.int32)
    meta = {"shards": [f"{name} {sha256(name)}" for name in args.shards], "images": args.images,
            "epochs": args.epochs, "seed": args.seed}
    np.savez(args.out, images=np.stack(images), order=order, meta=json.dumps(meta))
    print(json.dumps({**meta, "shape": list(np.stack(images).shape), "out_sha256": sha256(args.out)},
                     indent=1))


if __name__ == "__main__":
    main()
