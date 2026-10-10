#!/usr/bin/env python3
"""Freeze legacy loader batches on armada before the spec migration.

Run from the baseline checkout with this script from the pipeline branch:
git show feat/data-pipeline-core:tools/data_oracle.py | python - --output=ORACLE.json
Only synthetic records are used; real corpus equivalence is a separate gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import grain.python as grain
import jax
import numpy as np

from dew.data import ImageDataset


@dataclasses.dataclass(frozen=True)
class InMemoryImages(ImageDataset):
    pixels: tuple = ()

    def source(self, split=None):
        return self.pixels

    def record(self, element, rng):
        return element["image"], element["caption"], element["label"]


def digest(batch):
    tree = jax.tree.structure(batch)
    leaves = []
    for value in jax.tree.leaves(batch):
        array = np.asarray(value)
        if array.dtype.hasobject:
            encoded = json.dumps(array.tolist(), sort_keys=True).encode()
        else:
            encoded = array.tobytes()
        leaves.append({"shape": list(array.shape), "dtype": array.dtype.str,
                       "sha256": hashlib.sha256(encoded).hexdigest()})
    return {"tree": str(tree), "leaves": leaves}


def _read(reader, count):
    return [digest(batch) for batch in itertools.islice(reader, count)]


def capture(factory: Callable, workers: int, shares: int):
    from dew.data import DataPartition, Loading

    data = factory(Loading(workers=workers, threads=1, read_buffer=4, worker_buffer=1))
    rows = []
    for share in range(shares):
        partition = DataPartition(share, shares)
        reader = data.train(partition)
        try:
            first = _read(reader, 4)
            saved = reader.get_state()
            rest = _read(reader, 4)
        finally:
            reader.close()
        resumed = data.train(partition)
        try:
            resumed.set_state(saved)
            after = _read(resumed, 4)
        finally:
            resumed.close()
        if rest != after:
            raise RuntimeError("baseline loader did not resume exactly")
        held = None
        if data.val is not None:
            reader = data.val(partition)
            try:
                held = _read(reader, 100)
                if next(reader, None) is not None:
                    raise RuntimeError("baseline validation did not terminate")
            finally:
                reader.close()
        rows.append({"first": first, "after_save": rest, "validation": held})
    return {"records": data.records, "held_out": data.held_out, "shares": rows}


def fixtures(root: Path):
    from dew.data import (
        ArrayRecordImages,
        ChatMessages,
        DataPhase,
        HFImages,
        HFOptions,
        HubDataset,
        PreferencePairs,
        Prompts,
        TokenWindows,
    )
    from dew.data.images import pack_dict_of_byte_arrays
    from dew.data.text import ByteTokenizer
    from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout, Weighted
    from dew.nn.backbones.causal_transformer import CausalTransformer

    def token_store(name, shift):
        directory = root / name
        directory.mkdir()
        train = np.asarray([token for index in range(100)
                            for token in [*((np.arange(5 + index % 15) + index + shift) % 200), 255]], np.uint16)
        held = train[:180]
        (directory / "train.bin").write_bytes(train.tobytes())
        (directory / "val.bin").write_bytes(held.tobytes())
        (directory / "meta.json").write_text(json.dumps({"tokenizer": "byte", "vocab_size": 256,
                                                        "dtype": "uint16", "train_tokens": len(train),
                                                        "val_tokens": len(held), "eos_id": 255}))
        return str(directory)

    tokens = token_store("tokens", 0)
    another = token_store("other", 10)
    tokenizer = "tests/fixtures/tokenizers/tiny-chat"
    messages = tuple({"messages": [{"role": "user", "content": f"question {index}"},
                                    {"role": "assistant", "content": f"answer {index % 5}"}]}
                     for index in range(40))
    chats = root / "chat.jsonl"
    chats.write_text("".join(json.dumps(row) + "\n" for row in messages))
    prompts = tuple(json.dumps({"prompt": list(range(1, 3 + index % 6)), "data_source": "rule",
                                "ground_truth": str(index % 3), "extra_info": ""}) for index in range(40))
    pairs = tuple(json.dumps({"chosen": list(range(1, 3 + index % 4)), "rejected": [2, 3, 4],
                              "chosen_mask": [0] + [1] * (1 + index % 4), "rejected_mask": [0, 1, 1]})
                  for index in range(40))
    pixels = tuple({"image": ((np.arange(16 * 16 * 3).reshape(16, 16, 3) + index * 7) % 256).astype(np.uint8),
                    "label": index % 3, "caption": f"picture {index}"} for index in range(40))

    image_root = root / "images"
    image_root.mkdir()
    from array_record.python.array_record_module import ArrayRecordWriter

    writer = ArrayRecordWriter(str(image_root / "00.array_record"), "group_size:1")
    for row in pixels:
        writer.write(pack_dict_of_byte_arrays({"image": row["image"].tobytes(),
                                               "shape": np.asarray([16, 16], "<i4").tobytes(),
                                               "caption": row["caption"].encode(),
                                               "label": np.asarray(row["label"], np.int32).tobytes()}))
    writer.close()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset as HFDataset

    table = root / "records.parquet"
    pq.write_table(pa.table({"id": list(range(40)), "x": [[index, index + 1] for index in range(40)]}), table)
    arrow_images = HFDataset.from_list(pixels)
    image_table = root / "image.parquet"
    arrow_images.to_parquet(str(image_table))
    backbone = CausalTransformer(vocab_size=256, emb_features=8, num_layers=1, num_heads=2, mlp_features=16)
    objective = DecisionObjective(backbone, tokenizer=ByteTokenizer(), specials=Specials(0, 1, 2, 3),
                                  layout=StateFirstLayout(max_len=128), shuffle_options=True)
    examples = [Example(f"state {index}", {"class": Choice("Pick", ["yes", "no"])}, {"class": index % 2})
                for index in range(40)]
    return {
        "text_windows": lambda loading: TokenWindows(path=tokens, seq_len=16, seed=7, loading=loading).load(batch=4),
        "text_packed": lambda loading: TokenWindows(path=tokens, seq_len=16, pack=True, seed=7,
                                                    loading=loading).load(batch=4),
        "text_phases": lambda loading: TokenWindows(phases=(DataPhase(tokens, 3), DataPhase(another)), seq_len=16,
                                                    seed=7, loading=loading).load(batch=4),
        "chat": lambda loading: ChatMessages(tokenizer=tokenizer, path=str(chats), val_path=str(chats),
                                              seq_len=64, seed=7, loading=loading).load(batch=4),
        "prompts": lambda loading: Prompts(tokenizer="byte", records=prompts, max_prompt_len=16,
                                            seed=7, loading=loading).load(batch=4),
        "preferences": lambda loading: PreferencePairs(records=pairs, seq_len=16, seed=7,
                                                        loading=loading).load(batch=4),
        "images_memory": lambda loading: InMemoryImages(pixels=pixels, image_size=16, augmentation_backend="host", seed=7,
                                                        loading=loading).load(batch=4),
        "images_arrayrecord": lambda loading: ArrayRecordImages(path=str(image_root), image_size=16,
                                                                 augmentation_backend="host", seed=7,
                                                                 loading=loading).load(batch=4),
        "images_hf": lambda loading: HFImages(name="parquet", options=HFOptions(data_files={"train": str(image_table)}),
                                               image_size=16, augmentation_backend="host", seed=7,
                                               loading=loading).load(batch=4),
        "hub_records": lambda loading: HubDataset(name="parquet", options=HFOptions(data_files={"train": str(table)}),
                                                   seed=7, loading=loading).load(batch=4),
        "decision_table": lambda loading: objective.dataset(examples, validation=examples[:5], batch=4,
                                                             seed=7, loading=loading),
        "decision_mix": lambda loading: objective.dataset({"a": Weighted(examples, 3), "b": Weighted(examples[:8], 1)},
                                                           validation=examples[:5], batch=4, seed=7, loading=loading),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dew-data-oracle-") as directory:
        configurations = fixtures(Path(directory))
        oracle = {"baseline": "66beb5305", "cases": {}}
        for name, factory in configurations.items():
            if args.case and name not in args.case:
                continue
            oracle["cases"][name] = {}
            for workers, shares in itertools.product((0, 2), (1, 2)):
                oracle["cases"][name][f"workers={workers},shares={shares}"] = capture(factory, workers, shares)
                print(name, workers, shares, flush=True)
    args.output.write_text(json.dumps(oracle, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
