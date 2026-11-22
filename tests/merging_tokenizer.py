"""A byte vocabulary that encodes one string as one token, as a BPE vocabulary merges its pieces: the
tests' one owner of it (row 10981)."""

from dew.data.text import ByteTokenizer

MERGED_ID = 256
"""The id of the merged string, past every byte's."""


class Merging(ByteTokenizer):
    """Bytes, but for `merged`, one token (`MERGED_ID`) covering its characters: a row's tokens then depend
    on where the merged string falls, its offsets (`offsets`) spanning it whole."""

    def __init__(self, merged: bytes = b"B. y"):
        super().__init__()
        self.merged = merged

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
