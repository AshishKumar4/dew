"""Byte vocabularies that merge pieces as a BPE vocabulary does: one string as one token (`Merging`), and
ranked merges that cascade (`RankedPairs`): the tests' one owner of them (rows 10981, 11839)."""

import itertools

from dew.data.text import ByteTokenizer

MERGED_ID = 256
"""The id of the merged string, past every byte's."""


class Merging(ByteTokenizer):
    """Bytes, but for `merged`, one token (`MERGED_ID`) covering its characters: a row's tokens then depend
    on where the merged string falls, its offsets (`offsets`) spanning it whole."""

    def __init__(self, merged: bytes = b"B. y"):
        super().__init__()
        self.merged = merged
        # Its one merge, which nothing cascades from, takes back its string's characters but one
        # (`TakesBack`).
        self.taken_back = len(merged) - 1

    def offsets(self, text: str) -> tuple[list[int], list[int], list[int]]:
        """Its ids, each covering its characters, the merged string one id covering all of its."""
        ids, starts, ends = super().offsets(text)
        merged: tuple[list[int], list[int], list[int]] = ([], [], [])
        width, at = len(self.merged), 0
        while at < len(ids):
            whole = ids[at:at + width] == list(self.merged)
            merged[0].append(MERGED_ID if whole else ids[at])
            merged[1].append(starts[at])
            merged[2].append(ends[at + (width if whole else 1) - 1])
            at += width if whole else 1
        return merged

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.offsets(text)[0]

    def decode(self, ids) -> str:
        return b"".join(self.merged if int(id_) == MERGED_ID else bytes([int(id_)]) for id_ in ids).decode(
            errors="replace")


class RankedPairs(ByteTokenizer):
    """A ranked BPE over characters: its `merges`, pairs of tokens by rank, the best-ranked pair present
    merged wherever it stands left to right, again until none is, each merged token an id of its own past
    every byte's. Its merges cascade, so it states no bound on what a longer cut takes back
    (`dew.data.text.TakesBack`), though it inherits the byte vocabulary."""

    def __init__(self, merges: list[tuple[str, str]]):
        super().__init__()
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.ids = {left + right: MERGED_ID + rank for rank, (left, right) in enumerate(merges)}
        self.texts = {number: text for text, number in self.ids.items()}

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        """`text`'s tokens, each with the character it starts at."""
        pieces = [(character, at) for at, character in enumerate(text)]
        while True:
            ranked = [self.ranks.get((left[0], right[0])) for left, right in itertools.pairwise(pieces)]
            best = min((rank for rank in ranked if rank is not None), default=None)
            if best is None:
                return pieces
            merged, at = [], 0
            while at < len(pieces):
                if at + 1 < len(pieces) and ranked[at] == best:
                    merged.append((pieces[at][0] + pieces[at + 1][0], pieces[at][1]))
                    at += 2
                else:
                    merged.append(pieces[at])
                    at += 1
            pieces = merged

    def offsets(self, text: str) -> tuple[list[int], list[int], list[int]]:
        pieces = self._pieces(text)
        ids = [self.ids[piece] if len(piece) > 1 else ord(piece) for piece, _ in pieces]
        return ids, [at for _, at in pieces], [at + len(piece) for piece, at in pieces]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.offsets(text)[0]

    def decode(self, ids) -> str:
        return "".join(self.texts.get(int(id_), chr(int(id_))) for id_ in ids)
