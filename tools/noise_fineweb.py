"""Noise's fixed FineWeb data (qlabs-eng/noise), prepared from the Hub and written as Dew record stores.

Noise's `utils/prepare_data.py` at the pinned commit streams FineWeb `sample-10BT` in stream order, skips
its first 200,000 documents (they trained the BPE-4096 tokenizer), and encodes the rest with that
tokenizer, each document after an <|endoftext|>, into rows of 2,049 ids: the Dust paper's ~20.5M-token
training pool, its validation documents, its test documents (unused) and then the documents past them. Its
training set is the pool followed by those later documents, 14,648 rows, the first 4,880 the Dust paper's
10M-token set; its validation set is 544 rows. This writes the same rows, checked against Noise's pinned
SHA-256 of their int32 bytes, as `train/` and `val/`: Dew ArrayRecord stores of one `text` record a row,
which `dew.data.Files` reads and `.batch(8)` batches for an `LMObjective(seq_len=2048)`.

    python tools/noise_fineweb.py data/noise
    python tools/noise_fineweb.py data/noise --parquet shard-0.parquet shard-1.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

NOISE, COMMIT = "qlabs-eng/noise", "48e9ea73a7d74c6811a52eaa3deb4c2d76977367"
TOKENIZER_SHA256 = "a15168ae52f4a1b1c80b270a5e18a2ce3e038b5e3772a1ad3982044602da0ae7"
SEQ, TRAIN_ROWS, VAL_ROWS = 2049, 14648, 544
SKIP_DOCS, POOL_TOKENS, VAL_POOL_TOKENS = 200_000, 20_500_000, 1_150_000
EXPECTED = {"train": "2a92b8e9c3bc418c39880d337951775b9d29d0389bf1059cc826c412c44def37",
            "val": "5fdb9242ce8ab83e519ab5dd5cf8d64f4d8a99f6e74c137f4ca0f723da5f7437"}
"""Noise's pinned SHA-256 of each split's int32 little-endian token bytes."""


def encoded(documents: Iterator[str], tokenizer, end: int, least: int) -> np.ndarray:
    """Documents' ids, each after `end`, until at least `least` ids, cut to whole rows of SEQ, as Noise's
    `encode_split` makes them; the rest of the last document is dropped."""
    tokens: list[int] = []
    while len(tokens) < least:
        tokens.append(end)
        tokens.extend(tokenizer.encode(next(documents)).ids)
    usable = len(tokens) // SEQ * SEQ
    return np.asarray(tokens[:usable], np.int32).reshape(-1, SEQ)


def digest(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, "<i4").tobytes()).hexdigest()


def prepared(parquet: list[str] | None) -> dict[str, np.ndarray]:
    """The two splits as Noise prepares them, refused unless their SHA-256 are Noise's."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import load_dataset
    from reference_source import github
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(github(NOISE, COMMIT, "tokenizer/tokenizer.json", TOKENIZER_SHA256)))
    end = tokenizer.token_to_id("<|endoftext|>")
    corpus = (load_dataset("parquet", data_files=parquet, split="train", streaming=True) if parquet else
              load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True))
    texts = (record["text"] for record in itertools.islice(corpus, SKIP_DOCS, None))
    pool = encoded(texts, tokenizer, end, POOL_TOKENS)
    val = encoded(texts, tokenizer, end, VAL_POOL_TOKENS)[:VAL_ROWS]
    encoded(texts, tokenizer, end, VAL_POOL_TOKENS)  # the Dust repo's test documents, unused
    later = encoded(texts, tokenizer, end, (TRAIN_ROWS - len(pool)) * SEQ)
    splits = {"train": np.concatenate([pool, later])[:TRAIN_ROWS], "val": val}
    for name, rows in splits.items():
        if digest(rows) != EXPECTED[name]:
            raise SystemExit(f"{name}: {rows.shape} of SHA-256 {digest(rows)}, not Noise's {EXPECTED[name]}")
    return splits


def written(destination: Path, splits: dict[str, np.ndarray]) -> None:
    """Each split as a Dew ArrayRecord store of one `text` record a row."""
    from dew.data import Memory
    from dew.data.sinks import export

    for name, rows in splits.items():
        export(Memory([{"text": row} for row in rows]), str(destination / name), format="array_record",
               metadata={"source": f"{NOISE} {COMMIT} utils/prepare_data.py", "sha256": digest(rows)})
        print(f"{destination / name}: {rows.shape} sha256 {digest(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--parquet", nargs="+", help="local FineWeb sample-10BT shards, in stream order")
    arguments = parser.parse_args()
    written(arguments.destination, prepared(arguments.parquet))


if __name__ == "__main__":
    main()
