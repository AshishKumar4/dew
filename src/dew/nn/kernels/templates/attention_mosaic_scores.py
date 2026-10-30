"""What the attention template's Mosaic GPU backward launches share (`attention_mosaic_backward`): a launch's
static choices (`_Backward`), whether a layout's consumers share a key block (`split_queries`), and a step's
hook parameters and its weights and score cotangents under the forward's statistics (`_scores`)."""

import dataclasses

from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels.templates import attention_algebra as algebra, mosaic
from dew.nn.kernels.templates.attention import DQ_GRADS, Kind
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _param,
    _Program,
    _streamed,
    _views_of,
)


def split_queries(family: mosaic.Family) -> bool:
    """Whether the dK/dV program's consumers share one key block and split each slot's queries between them:
    on a layout whose consumers feed their own rings (`mosaic.ConsumerFed`), each consumer's step its share
    of the queries' tiles and per-query statistics (`a05-twoconsumer`); else each consumer its own keys."""
    return isinstance(family, mosaic.ConsumerFed)

@dataclasses.dataclass(frozen=True)
class _Backward:
    """A backward launch's static choices: its program's geometry, its forward's (`spec`: the call's form, the
    logits' scale and units, its score hook and initial state), its form, its consumers, whether it is
    interpreted, the score hook's parameter kinds (`kinds`), and whether the dK/dV program adds each tile's
    dQ too (`fused`: the hookless fast form's one launch, dQ by atomic adds, D the forward's anchored one
    unrefined)."""

    program: _Program[BackwardTiles]
    spec: algebra.Forward
    accurate: bool
    consumers: int
    interpret: bool
    fused: bool
    kinds: tuple[Kind, ...]
    approx_math: bool
    """Whether its launches take the hardware's approximate transcendentals (`approximate`): never in the
    accurate form."""

    @property
    def how(self) -> algebra.Recompute:
        """How the backward recomputes the forward's weights."""
        return algebra.recompute(self.spec)

    @property
    def scale(self) -> float:
        """The logits' scale."""
        return self.spec.scale

    @property
    def grad_scale(self) -> float:
        """The scale the gradients' cotangents take: the call's whole scale, the logits' times the power its
        query took before its products (`algebra.Forward.prescale`), against the call's own keys for dQ and
        its own queries for dK, so their products stay within the call's own range."""
        return self.spec.scale * self.spec.prescale

    @property
    def scaled(self) -> bool:
        """Whether the call's queries take a power of two for their logits (`by_power`): the fused launch's
        in registers, the keys-first launch's a stream or a stage of its own (`_dkv_step`), dK the call's
        own queries'."""
        return self.spec.prescale != 1.0

    @property
    def summed(self) -> tuple[bool, ...]:
        """Which score parameters the dQ program sums per query (`DQ_GRADS`); the others' gradients the dK/dV
        program writes."""
        return tuple(kind in DQ_GRADS for kind in self.kinds)

def _score_params(family: mosaic.Family, refs, kinds: tuple[Kind, ...], row, head, first, like,
                  k_first=None, *, keys_first: bool = False) -> tuple:
    """A tile's score hook parameters (`_param`), one per kind of `kinds`, read from their refs."""
    return tuple(_param(family, ref, kind, row, head, first, like, k_first, keys_first=keys_first)
                 for ref, kind in zip(refs, kinds, strict=True))

def _scores(walk: _Backward, coords: algebra.Coords, params: tuple, s, dp, peak, total, anchor, delta,
            chosen, *, masked: bool):
    """A backward tile's weights unnormalized and over `total`, its score cotangents (`algebra.grad_scores`)
    and the score hook's slopes (`algebra.tile_logits`), from its products `s` and `dp`, its coordinates, its
    hook parameters `params`, its mask (`algebra.seen` where `masked`) and its queries' statistics."""
    seen = algebra.seen(walk.spec, coords, s) if masked else None
    how = walk.how
    logits, slopes = algebra.tile_logits(coords.hooked(walk.spec.score_mod), s, seen, coords.q_pos,
                                         coords.k_pos, params,
                                         scale=walk.spec.scale if how.factor is None else 1.0, slopes=True)
    # Each query's statistics along a keys-first tile's query axis (`algebra.Coords.queries`); a queries-first
    # tile's the algebra broadcasts at each use, so no broadcast tile stays live across the step (a fused
    # launch at 128 x 64 and width 64 spilled 60 bytes holding them).
    def per_query(x):
        return coords.queries(x, s) if coords.keys_first else x

    inverse_total = per_query(algebra.inverse(total))
    exponentials, weights = algebra.normalized(logits, per_query(peak), inverse_total, how)
    at_anchor = coords.k_pos == coords.queries(chosen, coords.k_pos)
    multiplier = algebra.drawn(coords, coords.q_pos, coords.k_pos)
    cotangents = algebra.grad_scores(exponentials, dp, per_query(anchor), per_query(delta), seen, at_anchor,
                                     inverse_total, multiplier=multiplier)
    return exponentials, weights, cotangents, slopes, seen

def _dq_added(walk: _Backward, dq_ref, out_smem, place: tuple, index, q_first, dq) -> None:
    """A fused dK/dV consumer's tile `dq` added into dQ at its queries from `q_first` (`place` its row and
    head), through its output stage, free until its dK and dV are stored; interpreted, into a copy of the
    consumer's own (`Family.added`)."""
    row, head = place
    rows = dq.shape[0]
    view = dq_ref.at[index if walk.interpret else 0, row, pl.ds(q_first, rows), head]
    walk.program.family.added(dq, view, walk.interpret, smem=out_smem.at[index, pl.ds(0, rows)])


def _query_program(program: _Program, streams, k_ref, v_ref, spans, *, heads: int, kv_heads: int,
                   queries: int, consumers: int) -> tuple:
    """A program over query tiles' place and key streams (the dQ launch's and dropout's first pass's): its
    batch row and query head, its key and value views (`_views_of`) and their streams opened, its
    consumers' first query (`_Program.query_tile`, a tile each) and the key tiles any of them reads
    (`_Program.reach`, `_streamed`)."""
    block_q = program.tiles.block_q
    row, head = lax.axis_index("batch"), lax.axis_index("heads")
    views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads))
    streams = program.family.opened(streams, views)
    pair = consumers * block_q
    pair_first = program.query_tile(queries // pair) * pair
    streamed = _streamed([program.reach(pair_first + index * block_q, spans, row)[:2]
                          for index in range(consumers)])
    return row, head, views, streams, pair_first, streamed


def _query_tile_loaded(family: mosaic.Family, refs: tuple, smem: tuple, loaded: tuple, place: tuple, first,
                       index) -> None:
    """Consumer `index`'s query tile from `first` and its output cotangents, each `(ref, smem)` of `refs`
    and `smem` loaded whole against its `loaded` barrier at the program's batch row and head (`place`)."""
    row, head = place
    rows = (row, pl.ds(first, smem[0].shape[1]), head)
    family.loaded(tuple((ref.at[rows], stage.at[index], barrier.at[index])
                        for ref, stage, barrier in zip(refs, smem, loaded, strict=True)))
