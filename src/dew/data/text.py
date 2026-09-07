"""Tokenizers for language-model data: utf-8 bytes, or any HF tokenizer.

`import dew.data` stays cheap. Neither class imports `transformers` at module
scope, and ByteTokenizer needs nothing but numpy. HFTokenizer loads its
tokenizer on first use, so a host without the hub cache still imports
`dew.data.text` (and everything that re-exports it) fine.
"""

from __future__ import annotations

from typing import List


class ByteTokenizer:
    """Vocabulary 256, one id per utf-8 byte of the text.

    It trains nothing and downloads nothing, which makes it the default for
    small corpora and for tests; its decode is the inverse of its encode on
    any unicode input, so a generated sequence rounds back to text byte for
    byte.
    """

    def __init__(self):
        self.vocab_size = 256
        self.eos_id = 255

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

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
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.name, local_files_only=self.local_files_only)
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        # len() counts every vocabulary entry, including added tokens;
        # some fast tokenizers report a smaller .vocab_size than they emit.
        return len(self.tokenizer)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def save_pretrained(self, directory) -> None:
        """Write this tokenizer's own files into an export directory.

        What makes an exported checkpoint loadable by anything that reads the
        HF layout: the tokenizer writes tokenizer.json, tokenizer_config.json
        and its vocabulary itself, so the export copies no bytes by hand.
        """
        self.tokenizer.save_pretrained(str(directory))

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


def tokenizer_for(name: str, *, local_files_only: bool = False):
    """The tokenizer a name asks for: `byte` for Dew's own utf-8 vocabulary,
    any other name for the HF tokenizer of that repo or local directory.

    The one place a tokenizer name is resolved, so a training run and an
    export of what it trained read the same name the same way. An export
    passes local_files_only, since writing a checkpoint out is no reason to
    reach the hub for a tokenizer the host does not already have.
    """
    if name == "byte":
        return ByteTokenizer()
    return HFTokenizer(name, local_files_only=local_files_only)
