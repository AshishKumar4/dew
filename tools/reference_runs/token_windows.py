#!/usr/bin/env python3
"""Fixed token windows and their epoch orders, the one batch source both
frameworks read.

Documents are read in file order from one fineweb-edu parquet shard,
tokenized with the checkpoint's own tokenizer, joined with the tokenizer's
end-of-text id after each document and cut into `--windows` rows of
`--seq + 1` ids (inputs and shifted targets). `--epochs` permutations of the
rows are drawn from one numpy generator seeded with `--seed`. Both are
stored, so neither side re-derives an order.

    python tools/reference_runs/token_windows.py \\
        --parquet <shard>.parquet --tokenizer <snapshot dir> \\
        --windows 1024 --seq 1024 --epochs 2 --seed 0 --out windows.npz
"""

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from common import sha256
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--windows", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    separator = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if separator is None or separator == tokenizer.unk_token_id:
        separator = tokenizer.eos_token_id
    needed = args.windows * (args.seq + 1)
    ids: list[int] = []
    documents = 0
    for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=256, columns=["text"]):
        texts = batch.column("text").to_pylist()
        for encoded in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            ids.extend(encoded)
            ids.append(separator)
            documents += 1
        if len(ids) >= needed:
            break
    if len(ids) < needed:
        raise ValueError(f"the shard holds {len(ids)} tokens, {needed} are needed")
    windows = np.asarray(ids[:needed], np.int32).reshape(args.windows, args.seq + 1)
    generator = np.random.default_rng(args.seed)
    order = np.stack([generator.permutation(args.windows) for _ in range(args.epochs)]).astype(np.int32)
    meta = {"parquet": args.parquet, "parquet_sha256": sha256(args.parquet),
            "tokenizer": args.tokenizer, "separator": int(separator), "documents": documents,
            "windows": args.windows, "seq": args.seq, "epochs": args.epochs, "seed": args.seed}
    np.savez(args.out, windows=windows, order=order, meta=json.dumps(meta))
    print(json.dumps({**meta, "out": args.out, "out_sha256": sha256(args.out)}, indent=1))


if __name__ == "__main__":
    main()
