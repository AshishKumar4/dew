#!/usr/bin/env python3
"""Tokenize a text corpus into train.bin / val.bin / meta.json.

Reads a single file or every *.txt under a directory (recursively), encodes it
with the byte tokenizer or a huggingface one, splits the token stream by
--val-fraction (validation is the head of the stream, in file order) and
writes the nanoGPT-style output that `dew.data.sources.text.TokenFileSource`
and `get_token_dataset_grain` read back. With --pack-seq-len an EOS id is
written between the input files, so the packed loader can treat each file as
one document; meta.json then records the id under `eos_id`.

The corpus is processed one line-bounded chunk at a time: tokens are encoded,
written and dropped, so a corpus larger than memory costs disk, never RAM.
Chunks carry their partial last line forward, so a BPE merge never spans a
chunk boundary and the ids match a whole-corpus encode.

Usage:
    python tools/tokenize_text.py --input data/raw --out data/tokens \
        --tokenizer byte --val-fraction 0.01
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

# Characters read per chunk; small enough that the encoded ids of one chunk
# are a rounding error against memory, large enough to amortize reads.
CHUNK_CHARS = 1 << 20


@dataclasses.dataclass(frozen=True)
class TokenizeArgs:
    """Tokenize a text corpus into a train/val token directory."""

    input: str
    """File, or directory read as every *.txt inside it (recursive)."""
    out: str
    """Directory to write train.bin, val.bin and meta.json into."""
    tokenizer: str = "byte"
    """'byte' for utf-8 bytes, else a huggingface tokenizer name."""
    pack_seq_len: int = 0
    """Write an EOS id between documents (files) when positive, so
    `get_packed_token_dataset_grain` can split the stream back into them;
    meta.json then records the id under `eos_id`. 0 keeps the bare stream."""


def iter_text_chunks(paths, chunk_chars=CHUNK_CHARS):
    """Yield the corpus as chunks of at most a few chunk_chars, line-bounded.

    Every chunk ends at a newline (except a file's final partial line), so a
    tokenizer that merges across its input - BPE pretokenizers do - produces
    the same ids it would on the unbroken text.
    """
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            carry = ""
            while True:
                chunk = handle.read(chunk_chars)
                if not chunk:
                    break
                chunk = carry + chunk
                split = chunk.rfind("\n")
                if split >= 0:
                    yield chunk[: split + 1]
                    carry = chunk[split + 1:]
                else:
                    # A line longer than the chunk grows until it ends; the
                    # stream stays chunked for every realistic corpus.
                    carry = chunk
            if carry:
                yield carry


def iter_input_paths(raw_input: str):
    """Resolve --input to the files to read, in a stable order."""
    root = Path(raw_input)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"--input '{raw_input}' is neither a file nor a directory")
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def dtype_for(vocab_size: int) -> np.dtype:
    """Smallest unsigned dtype that holds every id the tokenizer emits."""
    if vocab_size <= 1 << 8:
        return np.dtype("uint8")
    if vocab_size <= 1 << 16:
        return np.dtype("uint16")
    return np.dtype("uint32")


def main(args: TokenizeArgs):
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be within [0, 1)")

    from dew.data.text import ByteTokenizer, HFTokenizer

    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    else:
        tokenizer = HFTokenizer(args.tokenizer)
        # Every chunk is longer than the model's context, which is the point
        # of packing; without this the tokenizer warns about it per chunk.
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

    paths = iter_input_paths(args.input)
    if not paths:
        raise SystemExit(f"--input '{args.input}' contains no text files")
    print(f"{len(paths)} file(s), "
          f"{sum(p.stat().st_size for p in paths) / 1e6:.1f} MB of text")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = dtype_for(tokenizer.vocab_size)

    # One encode pass writes the whole stream to a scratch file; the split
    # point needs the total count, and slicing a memmap of it costs a linear
    # copy, not a second tokenization.
    scratch = out_dir / "all.bin"
    total = 0
    try:
        with open(scratch, "wb") as handle:
            eos = tokenizer.eos_id if args.pack_seq_len > 0 else None
            first = True
            for chunk in iter_text_chunks(paths):
                ids = tokenizer.encode(chunk)
                if not ids:
                    continue
                if not first and eos is not None:
                    # A document ends where its file ends: the packing loader
                    # splits the stream at this id, and the loss drops the
                    # transition across it.
                    ids = ids + [eos]
                first = False
                handle.write(np.asarray(ids, dtype=dtype).tobytes())
                total += len(ids)
        if total < 2:
            raise SystemExit("the corpus tokenized to fewer than 2 tokens")

        val_len = min(int(round(total * args.val_fraction)), total - 1)
        stream = np.memmap(scratch, dtype=dtype, mode="r")
        stream[:val_len].tofile(out_dir / "val.bin")
        stream[val_len:].tofile(out_dir / "train.bin")
        train_len = total - val_len
    finally:
        scratch.unlink(missing_ok=True)

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype.name,
        "train_tokens": train_len,
        "eos_id": tokenizer.eos_id if args.pack_seq_len > 0 else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {out_dir / 'train.bin'} ({train_len} tokens)")
    print(f"wrote {out_dir / 'val.bin'} ({val_len} tokens)")
    print(f"wrote {out_dir / 'meta.json'}: {json.dumps(meta)}")


if __name__ == "__main__":
    main(tyro.cli(TokenizeArgs))
