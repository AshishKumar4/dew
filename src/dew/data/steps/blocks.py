"""The blocks a record's tokens make: the runs a prefix run reads whole (`prefix_blocks`)."""

from __future__ import annotations

import dataclasses

import grain.python as grain
import numpy as np

from dew.batch import Batch
from dew.data.steps.text import recounted


@dataclasses.dataclass(frozen=True)
class Blocks(grain.MapTransform):
    """A record's `<field>_block_groups` `[S]` int32: each token's block, -1 for none, from one source.

    `prefix=N` makes each document's first N tokens a block, which a prefix run (`LMObjective(prefix=N)`)
    trains. `prompt=True` makes each document's tokens before its first assistant token a block, read from
    the record's `<field>_roles` (a chat row's), which a row without refuses. `given` names a per-token field
    of the record, -1 for none, whose blocks are the runs of consecutive tokens in one document that carry one
    number, each run a block of its own. Padding is -1, and an unpacked record is one document. A prefix's
    and a prompt's blocks follow the one rule a prefix run's loss reads (`prefix_groups`): a document's
    places, where the record gives none, count its own tokens. With `fills`, a record that carries its own
    blocks keeps them, as the loss keeps a batch's (`prefixed_fields`): the stage a prefix run installs
    (`LMObjective.data`) fills the others. A count the record carries is counted again over its blocks
    (`recounted`), so counts agree with the loss.
    """

    prefix: int | None = None
    prompt: bool = False
    given: str | None = None
    field: str = "text"
    fills: bool = False

    def __post_init__(self) -> None:
        if [self.prefix is not None, self.prompt, self.given is not None].count(True) != 1:
            raise ValueError("Blocks takes exactly one source of blocks: prefix, prompt or given")
        if self.prefix is not None and (type(self.prefix) is not int or self.prefix < 1):
            raise ValueError(f"Blocks' prefix is a positive number of tokens, got {self.prefix!r}")

    def map(self, element: Batch) -> Batch:
        from dew.data.chat import Role
        from dew.nn.visibility import prefix_groups

        if self.fills and f"{self.field}_block_groups" in element:
            return element
        length = len(element[self.field])
        segments = element.get(f"{self.field}_segment_ids")
        segments = np.ones(length, np.int32) if segments is None else np.asarray(segments, np.int32)
        if self.given is not None:
            groups = _given_groups(segments, _named(element, self.given))
        else:
            places = element.get(f"{self.field}_positions")
            roles = None if self.prefix is not None else _named(element, f"{self.field}_roles")[None]
            groups = prefix_groups(segments[None], segments[None] != 0,
                                   "prompt" if self.prefix is None else self.prefix,
                                   positions=None if places is None else np.asarray(places, np.int32)[None],
                                   roles=roles, assistant=Role.ASSISTANT, xp=np)[0]
        blocked = {**element, f"{self.field}_block_groups": np.asarray(groups, np.int32)}
        return recounted(element, blocked)


def _named(element: Batch, name: str) -> np.ndarray:
    if name not in element:
        raise ValueError(f"Blocks needs the record's {name!r} field, which this record lacks")
    return np.asarray(element[name])


def _given_groups(segments: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """Each run of consecutive tokens of one document that carry one number, 0 or more, a block of its own.
    A block is a contiguous run (`AttentionMetadata.block_groups`), which keeps its targets exact: a token
    reads the block's tokens and nothing across the run's ends."""
    held = (segments > 0) & (numbers >= 0)
    opens = np.ones(len(segments), bool)
    opens[1:] = ~(held[1:] & held[:-1] & (segments[1:] == segments[:-1]) & (numbers[1:] == numbers[:-1]))
    return np.where(held, np.cumsum(held & opens) - 1, -1).astype(np.int32)
