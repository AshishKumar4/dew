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
from array_record.python.array_record_module import ArrayRecordReader
from common import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--images", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from dew.data.images import unpack_dict_of_byte_arrays

    images = []
    for shard in args.shards:
        reader = ArrayRecordReader(shard)
        for index in range(reader.num_records()):
            record = unpack_dict_of_byte_arrays(reader.read([index])[0])
            height, width = np.frombuffer(record["shape"], "<i4")
            images.append(np.frombuffer(record["image"], np.uint8).reshape(height, width, 3))
            if len(images) == args.images:
                break
        reader.close()
        if len(images) == args.images:
            break
    if len(images) < args.images:
        raise ValueError(f"the shards hold {len(images)} records, {args.images} are needed")
    generator = np.random.default_rng(args.seed)
    order = np.stack([generator.permutation(args.images) for _ in range(args.epochs)]).astype(np.int32)
    meta = {"shards": [f"{name} {sha256(name)}" for name in args.shards], "images": args.images,
            "epochs": args.epochs, "seed": args.seed}
    np.savez(args.out, images=np.stack(images), order=order, meta=json.dumps(meta))
    print(json.dumps({**meta, "shape": list(np.stack(images).shape), "out_sha256": sha256(args.out)},
                     indent=1))


if __name__ == "__main__":
    main()
