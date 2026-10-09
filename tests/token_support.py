"""The binary schema shared by distinct token-corpus generators."""

import json
from pathlib import Path

import numpy as np


def write_token_corpus(directory: Path, train: np.ndarray, val: np.ndarray | None = None,
                       *, tokenizer: str = "byte", vocab_size: int = 256,
                       eos_id: int | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.bin").write_bytes(train.tobytes())
    if val is not None:
        assert train.dtype == val.dtype
        (directory / "val.bin").write_bytes(val.tobytes())
    meta = {"tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": train.dtype.name,
            "train_tokens": len(train), "val_tokens": 0 if val is None else len(val)}
    if eos_id is not None:
        meta["eos_id"] = eos_id
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory
