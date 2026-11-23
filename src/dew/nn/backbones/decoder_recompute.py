"""A `CausalTransformer`'s recompute ladder: the rung above its remat policy that keeps less of what its
layers mark (`more`, `marked_residuals`), the host-offloaded sibling of the ladder's top, and a recorded
rung restored. The decoder's methods of these names read them here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dew._records import JSON

from ..mixers import AttentionMixer
from ..protocols import RecomputeStatement
from ..residuals import MLP_RESIDUALS, RESIDUALS
from .decoder_block import DECODER_REMAT, REMAT_POLICIES, RematPolicy, recomputed_inside, remat_record

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer

def more[D: CausalTransformer](decoder: D) -> D | None:
    """One rung up `DECODER_REMAT`, past any rung that keeps nothing more of the names
    these layers mark than the rung above it (`MixerBase.marks`), whose step would
    compile the same; None at the top, off the ladder, or where the top keeps what
    this rung keeps, so no rung above holds less."""
    if decoder.remat not in DECODER_REMAT[:-1]:
        return None
    marked = marked_residuals(decoder)

    def kept(policy: RematPolicy | None) -> tuple[frozenset[str], ...] | None:
        """What `policy` does with the names these layers mark: None where it recomputes none of
        them and keeps the rest, as no remat does; else what it saves, offloads, and has the layers
        recompute in their own backwards (`recomputed_inside`, the block's own statement of it: a
        recompute rung's names, `full`'s in-context memory chunks; train 933 F2, 936 F3)."""
        inside = frozenset(recomputed_inside(policy)) & marked
        if policy is None or (policy.recompute and not inside):
            return None
        return frozenset(policy.save) & marked, frozenset(policy.offload) & marked, inside

    index = DECODER_REMAT.index(decoder.remat) + 1
    while (index < len(DECODER_REMAT) - 1
           and kept(DECODER_REMAT[index]) in (kept(decoder.remat), kept(DECODER_REMAT[index + 1]))):
        index += 1
    if kept(DECODER_REMAT[index]) == kept(decoder.remat):
        return None
    return decoder.clone(remat=DECODER_REMAT[index])


def offloaded[D: CausalTransformer](decoder: D) -> D | None:
    """At the ladder's top ('full'), the decoder that keeps what 'minimal' saves in pinned host memory
    ('minimal_offloaded'): as many device bytes as 'full' (tests/test_remat_rungs.py's `_saved`), the
    recompute of 'minimal' and its residuals over the host link (row 1932); else None."""
    if decoder.remat != DECODER_REMAT[-1]:
        return None
    return decoder.clone(remat=REMAT_POLICIES['minimal_offloaded'])


def statement(decoder: CausalTransformer) -> RecomputeStatement:
    """What the decoder's rung recomputes (`RecomputeStatement`): its layers' forward values, by the names
    its policy keeps or recomputes; none where it runs no remat."""
    policy = decoder.remat
    scope = f"{type(decoder).__name__}.stack/"
    if policy is None:
        return RecomputeStatement(scope, None)
    if policy.recompute:
        return RecomputeStatement(scope, None, frozenset(policy.recompute))
    return RecomputeStatement(scope, frozenset(policy.save))


def marked_residuals(decoder: CausalTransformer) -> frozenset[str]:
    """The `RESIDUALS` this decoder's layers can mark: each layer's mixer kind's
    (`MixerBase.marks`, every name where a kind does not say) and, where a layer, its
    gated cross-attention, a prediction depth or the drafter has a feed-forward, its
    (`MLP_RESIDUALS`, the dense MLP's and the experts' alike)."""
    types = decoder.per_layer_types
    kinds = decoder.layer_kinds(types)
    mixers = [kind.mixer or decoder.mixer for kind in kinds.values()]
    if any(mixer is not None and mixer.marks() is None for mixer in mixers):
        return frozenset(RESIDUALS)
    default = AttentionMixer.residuals or ()
    feeds = (any(spec.routed or spec.width or spec.cross is not None
                 for spec in decoder.layer_specs(types, kinds, decoder.layer_mixer))
             or decoder.num_nextn_predict_layers or decoder.dspark is not None)
    marked = (default if mixer is None else mixer.marks() or () for mixer in mixers)
    return frozenset(MLP_RESIDUALS if feeds else ()).union(*marked)


def restored[D: CausalTransformer](decoder: D, record: JSON) -> D:
    """The decoder at the rung `record` names (`remat_record`) where it lies above the decoder's own on the
    ladder, as a checkpoint's recorded climb resumes; else the decoder as it is."""
    records = [remat_record(remat) for remat in DECODER_REMAT]
    here = remat_record(decoder.remat)
    if here in records and record in records and records.index(record) > records.index(here):
        return decoder.clone(remat=DECODER_REMAT[records.index(record)])
    return decoder
