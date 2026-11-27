"""A tight start's second compile at the place its whole schedule reads as fitting (`FitSearch.jumped`), on a
compile-only H100, whose compile states the live intervals of its buffers
(`dew.training.live_buffers.live_intervals`; a CPU's states none)."""

import functools

import jax
import numpy as np
import optax
import pytest
from test_compile_ahead import gpus

from dew._records import record
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec, Trainer

SEQ, ROWS, VOCAB, ROOMY = 512, 32, 32768, 2**36
TEXT = {"text": (np.arange(ROWS * (SEQ + 1)).reshape(ROWS, SEQ + 1) * 7 % VOCAB).astype(np.int32)}
"""A batch whose whole logits (16384 tokens by 32768 columns, 2 GiB) and whose four layers' activations each
weigh far more than the decoder's weights: a tile of the logits (a fused head's 4096 tokens at a time) frees
most of them, and the remat its layers' share."""


def decoder(limit: int, remat: str | None = None, tiled: bool = False) -> Trainer:
    """The decoder on a compile-only H100 whose allocator holds `limit` bytes, its step whole, recomputing
    as its model's `remat` states, its head tiled where `tiled`."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=128, num_layers=4, num_heads=2, mlp_features=512,
                              max_seq_len=SEQ, **({} if remat is None else {"remat": remat}))
    trainer = Trainer(LMObjective(model, seq_len=SEQ), optax.sgd(1e-2), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus("sm90", limit=limit)), split=1)
    if tiled:
        trainer.objective.tile_head()
    return trainer


def searched(trainer: Trainer) -> dict:
    """The plan `trainer`'s step compiled ahead settles (`compile_ahead`)."""
    from dew.training.ahead import compile_ahead

    return record(compile_ahead(trainer, TEXT).plan, "plan")


@functools.cache
def need(remat: str | None, tiled: bool = False) -> int:
    """What the step needs at `remat`, its head tiled where `tiled`: a room past all less the headroom its
    first compile left there."""
    return ROOMY - searched(decoder(ROOMY, remat, tiled))["probes"][0]["headroom"]


@pytest.mark.parametrize(("remat", "tiled"), [("minimal", False), ("minimal", True)],
                         ids=["remat", "remat-and-tile"])
def test_a_tight_start_compiles_second_a_place_that_fits(tmp_path, monkeypatch, remat, tiled):
    """The decoder started without recomputation, its head whole, on an H100 whose allocator holds what
    its step needs at `remat` (its head tiled where `tiled`): its second compile is the place the start's
    whole schedule reads as fitting, and that place fits (the owner's 2026-11-26 ruling: the fitting place
    in the second compile; a GH200's 4B start asked 13). Where only the remat and the tile together fit,
    neither the remat alone at the whole head nor the tile alone does: the place fits only where their
    savings are priced together, each buffer dropped once."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    room = need(remat, tiled)
    assert need(None) > room
    if tiled:
        assert need(remat) > room and need(None, tiled=True) > room, (need(remat), need(None, tiled=True))
    plan = searched(decoder(room))
    probes = plan["probes"]
    assert [probe["fits"] for probe in probes] == [False, True], (probes, plan.get("jump"))
