"""Token-file sources: nanoGPT-style memmapped `.bin` files of token ids.

A dataset directory is `train.bin`, `val.bin` and `meta.json` as written by
`tools/tokenize_text.py`. The files hold a flat token stream.

`TokenFileSource` reads a record as a contiguous window of `seq_len + 1` ids
starting at `i * seq_len`, so record i's last token is record i+1's first and
the model sees every transition exactly once. No decoding, no randomness: the
shuffle lives in the sampler.

`TokenDocumentSource` reads a record as one document: the span from after the
previous eos id through its own. It exists for the packed pipeline, which
cares where documents end and lets grain pack several of them into one
window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# meta.json's "dtype" names numpy dtypes; uint16 covers byte tokenizers and
# most HF ones, uint32 the rest.
_DEFAULT_DTYPE = np.dtype("<u2")


class TokenFileSource:
    """Random access over a flat `.bin` of token ids, in `seq_len + 1` windows.

    The dtype comes from the sibling `meta.json` when present (the tokenize
    tool records it there), else uint16, the nanoGPT default. The memmap is
    never loaded into memory: a worker reads only the window it is asked for.
    """

    def __init__(self, path: str, seq_len: int):
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")
        self.path = str(path)
        self.seq_len = seq_len

        self.dtype = _DEFAULT_DTYPE
        self.vocab_size = None
        meta_path = Path(self.path).with_name("meta.json")
        if meta_path.exists():
            with open(meta_path) as meta_file:
                meta = json.load(meta_file)
            self.dtype = np.dtype(meta.get("dtype", _DEFAULT_DTYPE))
            self.vocab_size = meta.get("vocab_size")

        self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self._tokens) < seq_len + 1:
            raise ValueError(
                f"{self.path} holds {len(self._tokens)} tokens, too few for even "
                f"one window of seq_len {seq_len}"
            )

    def __len__(self) -> int:
        return (len(self._tokens) - 1) // self.seq_len

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        # A memmap slice past its end silently yields an empty array, so the
        # bounds are checked here rather than left to numpy.
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.seq_len
        window = self._tokens[start : start + self.seq_len + 1]
        return {"text": window.astype(np.int32)}

    def __getstate__(self):
        # The memmap does not survive grain's pickle round trip to workers;
        # the path and dtype are enough to reopen it there.
        state = dict(self.__dict__)
        state["_tokens"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")


class TokenDocumentSource:
    """Random access over a flat `.bin` of token ids, one document per record.

    A document is the span from after the previous `eos_id` through its own,
    so the eos tokens are the record separators and every document ends with
    one. The dtype and `eos_id` come from the sibling `meta.json` (the
    tokenize tool records them when run with --pack-seq-len); `eos_id` there
    is required, since without it the stream has no boundaries to find.
    The memmap is never loaded into memory: a worker reads only the span it
    is asked for.
    """

    def __init__(self, path: str, eos_id: Optional[int] = None):
        self.path = str(path)
        self.dtype = _DEFAULT_DTYPE
        self.vocab_size = None
        meta_path = Path(self.path).with_name("meta.json")
        if meta_path.exists():
            with open(meta_path) as meta_file:
                meta = json.load(meta_file)
            self.dtype = np.dtype(meta.get("dtype", _DEFAULT_DTYPE))
            self.vocab_size = meta.get("vocab_size")
            if eos_id is None:
                eos_id = meta.get("eos_id")
        if eos_id is None:
            raise ValueError(
                f"{meta_path} records no eos_id: document boundaries are the "
                "eos tokens the tokenize tool writes with --pack-seq-len")
        self.eos_id = int(eos_id)

        self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        ends = np.flatnonzero(self._tokens == self.eos_id)
        if len(ends) == 0:
            raise ValueError(
                f"{self.path} holds no eos id {self.eos_id}, so it has no "
                "document boundaries")
        # Exclusive span ends; record i is tokens[prev_end : ends[i]].
        self._ends = (ends + 1).astype(np.int64)

    def __len__(self) -> int:
        return len(self._ends)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = 0 if index == 0 else self._ends[index - 1]
        end = self._ends[index]
        return {"text": self._tokens[start:end].astype(np.int32)}

    def __getstate__(self):
        # The memmap does not survive grain's pickle round trip to workers;
        # the path, dtype and span ends are enough to reopen it there.
        state = dict(self.__dict__)
        state["_tokens"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
