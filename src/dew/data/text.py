"""Tokenizers for language-model data: utf-8 bytes, or any HF tokenizer.

`import dew.data` stays cheap. Nothing here imports `transformers` at module
scope, and ByteTokenizer needs nothing but numpy. HFTokenizer loads its
tokenizer on first use, so a host without the hub cache still imports
`dew.data.text` (and everything that re-exports it) fine.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jax.typing import ArrayLike
    from transformers import PreTrainedTokenizerBase


_load_lock = threading.Lock()


# The return stays as transformers types it, unannotated, because the
# typed PreTrainedTokenizerBase signatures (list[dict[str, str]] chat
# messages, save_pretrained's tuple) do not satisfy interop's HostProcessor,
# which `interop.pretrained` hands a loaded tokenizer as.
def load_tokenizer(name: str, *, revision: str | None = None, local_files_only: bool = False):
    """The HF tokenizer of a hub repo or local directory, loaded once per process.

    Every Dew path that needs an HF tokenizer loads it here, so one name at
    one revision is one object however many readers share it. Grain workers
    unpickle only the name and load their own copy on their first record.
    """
    return _loaded(name, revision, local_files_only)


# Positional, because functools.cache keys `f(x)` and `f(x, flag=False)` apart.
@functools.cache
def _loaded(name: str, revision: str | None, local_files_only: bool):
    # The lock serializes the first import: transformers swaps in its lazy
    # module object while it initializes, and two threads importing it
    # together can catch it half built.
    with _load_lock:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            name, revision=revision, local_files_only=local_files_only)


@runtime_checkable
class TokenArray(Protocol):
    """Holds ids as an array rather than as a list.

    numpy's and jax's arrays both hand their ids over through `tolist`, and a
    decode takes either beside a plain sequence, because a sampler returns a
    row of a device array while a caller writes a list. A row has a length;
    the scalars `ArrayLike` also covers are one id and have none.
    """

    def __len__(self) -> int: ...

    def tolist(self) -> list[int]: ...


def _token_row(ids: ArrayLike | Sequence[int]) -> list[int]:
    values = ids.tolist() if isinstance(ids, TokenArray) else ids
    if not isinstance(values, Sequence):
        raise TypeError("decode takes a row of token ids, not one id")
    return [int(token) for token in values]


class ByteTokenizer:
    """Encodes text as one id per utf-8 byte, over a vocabulary of 256.

    It trains nothing and downloads nothing, which makes it the default for
    small corpora and for tests. Its decode inverts its encode on any unicode
    input, so a generated sequence rounds back to text byte for byte.
    """

    def __init__(self):
        self.vocab_size = 256
        self.eos_id = 255
        self.bos_id = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: ArrayLike | Sequence[int]) -> str:
        return bytes(_token_row(ids)).decode("utf-8", errors="replace")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class HFTokenizer:
    """A huggingface tokenizer, loaded from its hub name on first use.

    Lazy loading keeps `import dew.data.text` (and `import dew.data`) from
    paying for `transformers` and any hub lookup a caller never asked for.
    """

    def __init__(self, name: str, *, local_files_only: bool = False):
        self.name = name
        self.local_files_only = local_files_only

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return load_tokenizer(self.name, local_files_only=self.local_files_only)

    @property
    def vocab_size(self) -> int:
        # len() counts every vocabulary entry, including added tokens;
        # some fast tokenizers report a smaller .vocab_size than they emit.
        return len(self.tokenizer)

    @property
    def eos_id(self) -> int | None:
        """The eos token's id, or None for a tokenizer that declares none."""
        eos = self.tokenizer.eos_token_id
        if eos is None or isinstance(eos, int):
            return eos
        raise TypeError(f"tokenizer {self.name!r} reports eos_token_id {eos!r}, not one id")

    @property
    def bos_id(self) -> int | None:
        """The id the tokenizer starts a sequence with, or None where it has none."""
        bos = self.tokenizer.bos_token_id
        if bos is None or isinstance(bos, int):
            return bos
        raise TypeError(f"{self.name} names a bos_token_id that is not one id: {bos!r}")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: ArrayLike | Sequence[int]) -> str:
        # batch_decode of one row is decode's text, typed as one str.
        return self.tokenizer.batch_decode([_token_row(ids)])[0]

    def save_pretrained(self, directory) -> None:
        """Writes this tokenizer's own files into an export directory.

        The tokenizer writes tokenizer.json, tokenizer_config.json and its
        vocabulary itself, so the export copies no bytes by hand and the
        result loads in anything that reads the HF layout.
        """
        self.tokenizer.save_pretrained(str(directory))

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


def tokenizer_for(name: str, *, local_files_only: bool = False):
    """Builds the tokenizer `name` asks for.

    `byte` is dew's own utf-8 vocabulary; any other name is the HF tokenizer
    of that repo or local directory. Resolving names here alone keeps a
    training run and an export of what it trained reading one name one way.
    An export passes `local_files_only`, since writing a checkpoint out is no
    reason to reach the hub.
    """
    if name == "byte":
        return ByteTokenizer()
    return HFTokenizer(name, local_files_only=local_files_only)
