"""The token-corpus layout a token store reads (`dew.data.sources.TokenFiles`): its tokens in train.bin
and val.bin, and `meta.json` beside them saying how to read them. The standard library alone, so the
wheel smoke test's harness writes one before any package is installed."""

import json
from pathlib import Path


def write_token_meta(directory: Path, train_tokens: int, val_tokens: int = 0, *, dtype: str = "uint16",
                     tokenizer: str = "byte", vocab_size: int = 256, eos_id: int | None = None) -> None:
    """A token store's `meta.json` in `directory`, which every token store, binary or records,
    carries; no `eos_id` where its documents state no end."""
    meta: dict[str, object] = {"tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": dtype,
                               "train_tokens": train_tokens, "val_tokens": val_tokens}
    if eos_id is not None:
        meta["eos_id"] = eos_id
    (directory / "meta.json").write_text(json.dumps(meta))
