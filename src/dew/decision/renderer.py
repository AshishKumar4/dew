"""One renderer of decision rows (docs/design/decision-render.md): a row is a sequence of chunks, text
with character marks or ids (a special token, a request's media) with marks of whole ids, its chunks'
boundaries those of the pieces a release's reference encodes. `render` encodes them by the layout's
unit and gives each mark its tokens:

- `pieces` encodes each text chunk alone, as a release that encodes its prompt piece by piece does
  (Laya's, the joint release's); a mark then covers whole chunks;
- `whole` encodes each run of text chunks with no ids between them once, with the tokenizer's own
  character offsets (`dew.data.text.Offsets`), as a release that encodes its whole prompt does (the
  slot release's): a mark takes every token that overlaps its characters, so a token across two
  marks belongs to both.
"""

from __future__ import annotations

import bisect
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from dew.data.text import Offsets

if TYPE_CHECKING:
    from dew.data.text import Tokenizer

type Unit = Literal["pieces", "whole"]
"""How a layout encodes its text: piece by piece, or each run of it whole."""

type Mark = tuple[Hashable, int, int]
"""A named span of a chunk: its key, and its first and past-last character, or id."""

type Point = tuple[Hashable, int]
"""A named boundary of a chunk: its key, and the character, or id, it stands before."""


@dataclass(frozen=True)
class Chunk:
    """A piece of a row: `text`, or `ids` where it holds ids (a special token, media), with `marks`
    in its own characters or ids, and `points`, boundaries in them."""

    text: str = ""
    ids: tuple[int, ...] | None = None
    marks: tuple[Mark, ...] = ()
    points: tuple[Point, ...] = ()


@dataclass(frozen=True)
class Rendered:
    """A row's tokens, each mark's `(start, end)` tokens in them, and each point's token: the first
    that does not end at or before it, as a release splits its prompt at a character (serve.patch
    `token_layout`)."""

    tokens: tuple[int, ...]
    marks: dict[Hashable, tuple[int, int]]
    points: dict[Hashable, int]


def render(chunks: Sequence[Chunk], encoder: Tokenizer, unit: Unit) -> Rendered:
    """`chunks` encoded by `unit`, each mark's tokens and each point's located in them. A `whole`
    encoding needs a vocabulary that keeps character offsets (TypeError), and a mark that covers no
    token, or under `pieces` a mark of part of a chunk or a point inside one, is refused (ValueError)."""
    tokens: list[int] = []
    marks: dict[Hashable, tuple[int, int]] = {}
    points: dict[Hashable, int] = {}
    run: list[Chunk] = []

    def flush() -> None:
        # Only a whole layout gathers runs.
        if run:
            tokens.extend(_whole(run, encoder, len(tokens), marks, points))
            run.clear()

    for chunk in chunks:
        if chunk.ids is not None:
            flush()
            for key, start, end in chunk.marks:
                marks[key] = (len(tokens) + start, len(tokens) + end)
            points.update({key: len(tokens) + at for key, at in chunk.points})
            tokens.extend(chunk.ids)
        elif unit == "whole":
            run.append(chunk)
        else:
            ids = encoder.encode(chunk.text, add_special_tokens=False) if chunk.text else []
            for key, start, end in chunk.marks:
                if (start, end) != (0, len(chunk.text)):
                    raise ValueError(f"mark {key!r} covers part of a piece, which a layout of pieces encodes "
                                     "whole")
                marks[key] = (len(tokens), len(tokens) + len(ids))
            for key, at in chunk.points:
                if at not in (0, len(chunk.text)):
                    raise ValueError(f"point {key!r} stands inside a piece, which a layout of pieces encodes "
                                     "whole")
                points[key] = len(tokens) + (len(ids) if at else 0)
            tokens.extend(ids)
    flush()
    return Rendered(tuple(tokens), marks, points)


def _whole(run: Sequence[Chunk], encoder: Tokenizer, offset: int, marks: dict[Hashable, tuple[int, int]],
           points: dict[Hashable, int]) -> list[int]:
    """A run of text chunks encoded once, its marks' tokens and its points', after `offset` tokens,
    written to `marks` and `points`."""
    if not isinstance(encoder, Offsets):
        raise TypeError(f"a layout that encodes its text whole locates its spans by the tokenizer's "
                        f"character offsets, which a {type(encoder).__name__} does not keep")
    text = "".join(chunk.text for chunk in run)
    ids, starts, ends = encoder.offsets(text)
    at = 0
    for chunk in run:
        for key, start, end in chunk.marks:
            first = bisect.bisect_right(ends, at + start)
            last = bisect.bisect_left(starts, at + end)
            if last <= first:
                raise ValueError(f"mark {key!r} covers no token")
            marks[key] = (offset + first, offset + last)
        points.update({key: offset + bisect.bisect_right(ends, at + point) for key, point in chunk.points})
        at += len(chunk.text)
    return ids
