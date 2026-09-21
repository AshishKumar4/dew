"""Prepare an image dataset at training resolution as ArrayRecord shards.

    python tools/prepare_images.py --dataset oxford_flowers102 \
        --data-path ~/.cache/dew/datasets/oxford_flowers102/2.1.1 \
        --split all --image-size 64 --out prepared/64

Every record becomes `image`/`shape`/`caption`/`label` packed bytes at
`image_size` square, so a training read is a memcpy instead of a JPEG decode
and resize. The output directory loads with `ArrayRecordImages(path=dir)` —
an empty `shards` reads the files the tool writes. The transform's
deterministic half runs here once: decode_image at the coarsest DCT scale
covering the size, then resize_image; augmentation still happens per batch.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from dew.data.images import (ArrayRecordImages, HFImages, OxfordFlowers,
                             decode_image, pack_dict_of_byte_arrays,
                             resize_image)


def build_spec(args):
    """The dataset spec the recipe would build, minus the runtime knobs."""
    common: dict = {"augmentation": "none", "val_batches": None}
    if args.dataset == "oxford_flowers102":
        return OxfordFlowers(path=args.data_path, split=args.split,
                             image_size=args.image_size, **common)
    if args.dataset == "hf_images":
        if not args.name:
            raise ValueError("--name is the hub repo id hf_images reads")
        return HFImages(name=args.name, split=args.split,
                        image_size=args.image_size, **common)
    if args.dataset == "array_record_images":
        shards = tuple(args.source_shard or [])
        return ArrayRecordImages(path=args.data_path, shards=shards,
                                 image_size=args.image_size, **common)
    raise ValueError(f"--dataset must be oxford_flowers102, hf_images or "
                     f"array_record_images, got {args.dataset!r}")

def prepare(spec, out: str, shards: int | None, *, source: dict) -> dict:
    """Write `spec`'s records, decoded and resized to `spec.image_size`, as
    packed ArrayRecord shards under `out`, and its manifest.json."""
    elements = spec.source()
    records = spec.records(elements)
    shards = shards or math.ceil(records / 2048)
    per_shard = math.ceil(records / shards)
    os.makedirs(out, exist_ok=True)

    from array_record.python.array_record_module import ArrayRecordWriter

    writer = None
    written = 0
    shard_sizes = []
    for index in range(records):
        if index % per_shard == 0:
            if writer is not None:
                writer.close()
            name = f"{index // per_shard:05d}.array_record"
            writer = ArrayRecordWriter(os.path.join(out, name),
                                       options="group_size:1")
            shard_sizes.append(0)
        # The caption's rng draw is keyed by index, the same way grain keys
        # it by record index during training.
        image, caption, label = spec.record(
            elements[index], np.random.default_rng(index))
        if isinstance(image, bytes):
            image = decode_image(image, at_least=spec.image_size)
        image = resize_image(np.ascontiguousarray(image), spec.image_size)
        packed = {"image": image.tobytes(),
                  "shape": np.array(image.shape[:2], np.int32).tobytes(),
                  "caption": caption.encode("utf-8")}
        if label is not None:
            packed["label"] = np.int32(label).tobytes()
        record = pack_dict_of_byte_arrays(packed)
        assert writer is not None
        writer.write(record)
        shard_sizes[-1] += len(record)
        written += 1
    if writer is not None:
        writer.close()

    manifest = {"records": written, "image_size": spec.image_size,
                "source": source, "shard_sizes": shard_sizes}
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="oxford_flowers102")
    parser.add_argument("--data-path", required=True,
                        help="the spec's path= (prepared TFDS dir or shard root)")
    parser.add_argument("--name", default=None, help="hub repo id for hf_images")
    parser.add_argument("--source-shard", action="append", default=None,
                        help="shard dir under --data-path for array_record_images")
    parser.add_argument("--split", default="all")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, default=None,
                        help="output shard count; default ceil(records/2048)")
    args = parser.parse_args()

    spec = build_spec(args)
    started = time.perf_counter()
    manifest = prepare(spec, args.out, args.shards,
                       source={"dataset": args.dataset, "path": args.data_path,
                               "split": args.split})
    elapsed = time.perf_counter() - started
    print(f"{manifest['records']} records -> {len(manifest['shard_sizes'])} shard(s), "
          f"{sum(manifest['shard_sizes']) / 1e6:.1f} MB of records in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
