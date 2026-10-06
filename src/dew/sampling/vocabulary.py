"""Host tokenizer protocols and the text or bytes each vocabulary piece spells."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from dew.records import JSON


@runtime_checkable
class Vocabulary(Protocol):
    """The tokenizer surface `stop_strings` reads once, on the host.

    These are a Transformers tokenizer's own public vocabulary methods plus
    the piece-name lookup its slow and fast classes both expose. Nothing here
    runs generation code; the tables are built from token strings.
    """

    def get_vocab(self) -> dict[str, int]: ...
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...
    def _convert_id_to_token(self, index: int) -> str: ...
    def __call__(self, text: str, *, add_special_tokens: bool) -> Mapping[str, list[int]]: ...


def _decoder_has(config: JSON, name: str) -> bool:
    match config:
        case dict():
            return config.get("type") == name or any(_decoder_has(entry, name) for entry in config.values())
        case list():
            return any(_decoder_has(entry, name) for entry in config)
    return False


class PieceDecoder(Protocol):
    """A `tokenizers` decoder, which pickles as the JSON that configures it.

    `matching_mode` reads the decoder kinds that JSON names.
    """

    def __getstate__(self) -> bytes | str | None: ...


class Backend(Protocol):
    """The Rust tokenizer that a fast Transformers tokenizer wraps.

    Its decoder says how pieces spell bytes, and is None on a tokenizer
    without one.
    """

    @property
    def decoder(self) -> PieceDecoder | None: ...


@runtime_checkable
class Fast(Protocol):
    """A fast Transformers tokenizer, which holds its Rust backend.

    A slow tokenizer has no backend, so its pieces are read through their text.
    """

    @property
    def backend_tokenizer(self) -> Backend: ...


@runtime_checkable
class Referencing(Protocol):
    """A processor that holds the source's own processor or tokenizer as `reference`.

    `dew.interop.pretrained.Processor` is one.
    """

    @property
    def reference(self) -> Vocabulary | Referencing | Tokenizing: ...


@runtime_checkable
class Tokenizing(Protocol):
    """A processor that holds its tokenizer.

    Examples are a Transformers processor, a run's `RunProcessor`, and
    `dew.data.HFTokenizer` over a Hub tokenizer.
    """

    @property
    def tokenizer(self) -> Vocabulary | Tokenizing: ...


def matching_mode(tokenizer: Vocabulary) -> str | None:
    """Whether a tokenizer's pieces are bytes, and in which spelling.

    `StopStringCriteria._get_stop_string_matching_mode`: a byte-level decoder
    stores pieces in GPT-2's alphabet and a byte-fallback one spells unknown
    bytes `<0xNN>`. Either way the match runs over bytes, so a stop string is
    encoded to UTF-8 and a piece that is half a code point still counts.
    """
    if not isinstance(tokenizer, Fast):
        return None
    decoder = tokenizer.backend_tokenizer.decoder
    if decoder is None:
        return None
    if type(decoder).__name__ == "ByteLevel":
        return "byte_level"
    # A tokenizers decoder pickles as its JSON; any other object's state
    # is not text and reads as no configuration.
    state = decoder.__getstate__()
    if isinstance(state, str):
        state = state.encode()
    config = None
    if isinstance(state, bytes):
        try:
            config = json.loads(state)
        except json.JSONDecodeError:
            config = None
    if config is not None:
        if _decoder_has(config, "ByteFallback"):
            return "byte_fallback"
        if _decoder_has(config, "ByteLevel"):
            return "byte_level"
    return None


def vocabulary_pieces(tokenizer: Vocabulary, mode: str | None,
                      prefix: str = "abcdef") -> tuple[list[str | bytes], list[int]]:
    """What each vocabulary entry contributes to the text, and its id.

    `StopStringCriteria.clean_tokenizer_vocab`: a byte-mode piece is read
    through its byte spelling, and anything else through
    `convert_tokens_to_string` behind an ordinary prefix, because a decoder
    adds or removes a leading space depending on what came before. The prefix
    is tokenized once and its text is cut off the front of every piece.
    """
    alphabet = None
    if mode == "byte_level":
        # A byte-level piece spells each byte as a character of GPT-2's
        # alphabet; reading it back byte by byte keeps a code point two
        # tokens split.
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        alphabet = {char: byte for byte, char in bytes_to_unicode().items()}
    base = [tokenizer._convert_id_to_token(token)
            for token in tokenizer(prefix, add_special_tokens=False)["input_ids"]]
    pieces: list[str | bytes] = []
    ids: list[int] = []
    for token, index in tokenizer.get_vocab().items():
        piece = _piece_bytes(token, mode, alphabet)
        if piece is None:
            text = tokenizer.convert_tokens_to_string([*base, token])
            if prefix not in text:
                raise ValueError(
                    f"the tokenizer cannot spell the probe {prefix!r}, so a piece's own text "
                    "cannot be separated from what precedes it")
            text = text[text.index(prefix) + len(prefix):]
            piece = text.encode("utf-8") if mode is not None else text
        pieces.append(piece)
        ids.append(index)
    return pieces, ids


def _piece_bytes(token: str, mode: str | None, alphabet: dict[str, int] | None) -> bytes | None:
    if mode == "byte_level" and alphabet is not None:
        if all(char in alphabet for char in token):
            return bytes(alphabet[char] for char in token)
        return None
    if (mode == "byte_fallback" and len(token) == 6 and token.startswith("<0x")
            and token.endswith(">")
            and all(char in "0123456789abcdefABCDEF" for char in token[3:5])):
        return bytes([int(token[3:5], 16)])
    return None


def vocabulary_of(tokenizer: Vocabulary | Referencing | Tokenizing, reader: str) -> Vocabulary:
    """The tokenizer as it stands, or beneath a processor: dew's own processor
    holds the source processor as `reference`, and that holds the tokenizer."""
    source = tokenizer
    for _ in range(3):
        if isinstance(source, Vocabulary):
            break
        if isinstance(source, Referencing):
            source = source.reference
        elif isinstance(source, Tokenizing):
            source = source.tokenizer
        else:
            break
    if not isinstance(source, Vocabulary):
        raise TypeError(f"{reader} needs a tokenizer that can list its vocabulary")
    return source
