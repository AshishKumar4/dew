#!/usr/bin/env python3
"""Write tests/fixtures/nanogpt: a token directory nanoGPT writes, and the
batches nanoGPT reads out of it.

nanoGPT's data/shakespeare_char/prepare.py, fetched at a pinned commit, runs
as published on the first 20,000 characters of char-rnn's tinyshakespeare
(fetched at its own pinned commit and placed where the script looks for
its input, so it downloads nothing). It writes train.bin, val.bin and
meta.pkl. Then train.py's `get_batch`, fetched at the same commit, reads
eight batches of each split as published (CPU, block 32, batch 6, torch's
generator seeded per split), and batch.npz records each batch's offsets and
its x and y.

    python tools/nanogpt_reference.py
"""

from __future__ import annotations

import ast
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

NANOGPT = "https://raw.githubusercontent.com/karpathy/nanoGPT/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/"
SHAKESPEARE = ("https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
               "/data/tinyshakespeare/input.txt")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nanogpt"
CHARACTERS, BLOCK, BATCH, BATCHES = 20_000, 32, 6, 8


def fetched(url: str) -> str:
    return urllib.request.urlopen(url).read().decode()


def get_batch():
    """train.py's `get_batch`, with the names it reads."""
    text = fetched(NANOGPT + "train.py")
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "get_batch")
    scope = {"np": np, "torch": torch, "os": os, "data_dir": str(FIXTURE), "block_size": BLOCK,
             "batch_size": BATCH, "device_type": "cpu", "device": "cpu"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "train.py", "exec"), scope)
    return scope["get_batch"]


def main() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    (FIXTURE / "input.txt").write_text(fetched(SHAKESPEARE)[:CHARACTERS])
    prepare = fetched(NANOGPT + "data/shakespeare_char/prepare.py")
    exec(compile(prepare, "prepare.py", "exec"), {"__file__": str(FIXTURE / "prepare.py"),
                                                  "__name__": "__main__"})
    (FIXTURE / "input.txt").unlink()
    (FIXTURE / "meta.pkl").unlink()
    read = get_batch()
    arrays = {}
    published = torch.randint
    for split in ("train", "val"):
        # The offsets `get_batch` draws, recorded as it draws them.
        torch.manual_seed(0)
        ix = []

        def recording(*args, ix=ix, **kwargs):
            value = published(*args, **kwargs)
            ix.append(value.numpy().copy())
            return value

        torch.randint = recording
        try:
            batches = [read(split) for _ in range(BATCHES)]
        finally:
            torch.randint = published
        arrays[f"{split}/ix"] = np.stack(ix)
        arrays[f"{split}/x"] = np.stack([x.numpy() for x, _ in batches])
        arrays[f"{split}/y"] = np.stack([y.numpy() for _, y in batches])
    np.savez(FIXTURE / "batch.npz", block=np.asarray(BLOCK), **arrays)
    print(f"{FIXTURE}: train.bin, val.bin and {BATCHES} batches of each split")


if __name__ == "__main__":
    main()
