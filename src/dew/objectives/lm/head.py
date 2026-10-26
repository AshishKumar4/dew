"""How a token objective runs its model's vocabulary head: one policy every objective that scores
tokens through a head takes, and the fit ladder moves.

A head keeps its whole fp32 logits for the backward pass (`head_tile` None), the fastest where they
fit, or computes them a tile at a time (`head_tile`, `chunked.chunked_cross_entropy`), whose
temporaries stay one tile in size whatever the vocabulary. A tiled head scored for a summed loss may
take its gradient in the forward (`head_fused`, `chunked.fused_head_loss`), faster than the tiled
backward but holding a chunk of the whole vocabulary's logits beside it. The trainer's fit ladder
moves a step from the whole logits to the fused tile to the tiled backward before it recomputes
more (`dew.training.memory.recompute_more`), reading the head the objective reports
(`Objective.program_key`), and a resumed run climbs to the rung it recorded.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from flax import linen as nn

from dew.nn.protocols import AffineHead, OutputTable, head_table
from dew.objectives.base import Effects, Loss, Objective, ProgramModule
from dew.objectives.lm.chunked import Terms, chunked_tile, head_cross_entropy, reads_states


class HeadedObjective(Objective[Loss, Effects]):
    """An objective that scores tokens through its model's vocabulary head, by the one head policy
    (this module's): the head's backward tile (`head_tile`), whether a summed loss takes a tiled
    head's gradient in its forward (`head_fused`), the vocabulary slices a tiled head scores in
    (`head_chunks`), and the fit ladder's moves of them (`tile_head`, `unfuse_head`,
    `restore_head`), which the step's program reports (`program_key`)."""

    keeps_whole_logits: ClassVar[bool] = True
    """Whether the head's default (`head_tile` None) keeps the whole fp32 logits for the backward pass.

    A plain step does, and the trainer's fit ladder tiles it when the step does not fit
    (`recompute_more`). An objective whose devices also hold rollouts or a frozen reference, as
    GRPO's and DPO's do, keeps the tiled head as its default, so the loss's temporaries stay one
    tile in size whatever the vocabulary.
    """

    head_chunks: int
    head_tile: tuple[int, int] | None

    head_fused: bool = True
    """Whether a tiled head scored for its weighted sums takes their gradient in its forward
    (`fused_head`), where the objective's loss fuses one (`fuses_head`): the fit ladder turns it
    off where the fused chunk does not fit (`unfuse_head`), and tiling the head turns it on."""

    def use_head(self, head_chunks: int, head_tile: tuple[int, int] | Literal['whole', 'tiled'] | None
                 ) -> None:
        """Set the head's policy from an objective's arguments: `head_tile` None the class's
        default (`keeps_whole_logits`), 'whole' and 'tiled' naming the choice, or a tile."""
        self.head_chunks = head_chunks
        self.head_tile = _head_tile(head_tile, self.keeps_whole_logits)
        # Each decoder a fused loss traced, by identity, and whether its head was one matrix
        # (`chunked.head_table`): past a nonlinear head transform, beside an adapter's factors or
        # in place of a frozen FP8 kernel no matrix is, and no head is fused. A decoder not yet
        # traced is taken to have one.
        self._head_matrices: dict[int, tuple[object, bool]] = {}

    def fuses_head(self) -> bool:
        """Whether this objective's training loss takes a tiled head's gradient in its forward
        (`fused_head`); one whose loss reads per-token terms alone fuses none."""
        return False

    def head_terms(self, model, params, read, targets, **options) -> Terms:
        """The per-token terms of `targets` from `read`, what `model`'s `final_read` gave, through
        its head under this policy (`chunked.head_cross_entropy`); `options` are its own."""
        return head_cross_entropy(model, params, read, targets, self.head_chunks, tile=self.head_tile,
                                  **options)

    def fused_head(self, model, params) -> OutputTable | None:
        """`model`'s head as the one matrix a summed loss takes its gradient through in the forward
        (`chunked.fused_head_loss`), None where it keeps the whole logits, is not fused or is no
        matrix. What the trace finds of the decoder's head is kept, which `head_program` reads."""
        if self.head_tile is None or not reads_states(model):
            return None
        head = head_table(model, params)
        self._head_matrices[id(model)] = (model, head is not None)
        return head if self.head_fused else None

    def head_program(self, model: nn.Module) -> ProgramModule:
        """`model` as a trained module of the step's program with this objective's head: its tile, and
        fused where the loss fuses a tiled head the model holds as one matrix."""
        fused = (self.fuses_head() and self.head_fused and self.head_tile is not None and reads_states(model)
                 and isinstance(model, AffineHead) and self._head_matrices.get(id(model), (None, True))[1])
        return ProgramModule(model, self.head_tile, trained=True, head_fused=fused)

    def program_key(self) -> tuple[ProgramModule, ...]:
        return (self.head_program(self.model),)

    def tile_head(self, tile: tuple[int, int] | None = None) -> str | None:
        """Move a head that keeps its whole logits to `tile`, and report what it moved to.

        `tile` defaults to the generation's tile (`chunked.chunked_tile`).
        """
        if self.head_tile is not None:
            return None
        self.head_tile = chunked_tile() if tile is None else tile
        self.head_fused = True
        return f"the head tiled {self.head_tile}"

    def unfuse_head(self) -> str | None:
        """Move a tiled head to take its gradient in the backward, and report the move; None where
        the step's program does not change, a head no step fuses. A tiled head is unfused either
        way, so a decoder traced later does not fuse a head its rung unfused."""
        moved = any(module.head_fused for module in self.program_key())
        if self.head_tile is not None:
            self.head_fused = False
        return f"the head tiled {self.head_tile}, its gradient taken in the backward" if moved else None

    def restore_head(self, tile: tuple[int, int] | None, *, fused: bool) -> None:
        self.head_tile = tile
        self.head_fused = fused


def _head_tile(tile, keeps_whole_logits: bool) -> tuple[int, int] | None:
    """The head's tile: None keeps the whole logits; 'whole' and 'tiled' name
    the choice; None on an objective that does not keep them by default is
    the generation's tile (`chunked_tile`). A config's pair is a list, and
    the tile is a static argument, so it becomes a tuple."""
    if tile == 'whole' or (tile is None and keeps_whole_logits):
        return None
    if tile is None or tile == 'tiled':
        return chunked_tile()
    return (int(tile[0]), int(tile[1]))
