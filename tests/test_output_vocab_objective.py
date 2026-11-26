"""The objective's and sampler's side of a head scoring fewer ids than its table holds rows
(`CausalTransformer.output_vocab`, `dew.nn.protocols.logits_width`; theme M71, Inkling 2's 200058 of 201024):
the loss, the stops a draw may end at and a distillation's shared vocabulary read the head's width."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.distillation import DistillationObjective
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import _refuse_unlike_logits
from dew.sampling.text import Sampling, check_inputs

ROWS, SCORED, SEQ = 80, 56, 8
"""A table of 80 rows, none of the tiny decoder's widths (32, 64), whose head scores its first 56."""


def _decoder(output_vocab: int | None = None) -> CausalTransformer:
    return CausalTransformer(**{**TINY_DECODER, "vocab_size": ROWS, "dtype": "float32", "max_seq_len": 16,
                                "output_vocab": output_vocab})


def test_a_narrow_heads_loss_reads_none_of_the_rows_past_the_ids_it_scores():
    """An LM over a head scoring its table's first 56 of 80 rows reads none past them: moving every parameter
    row from the 57th on leaves its loss, over targets it scores, the same to the bit, where it moves the
    full head's, whose softmax spreads each target's probability over every row."""
    narrow, full = _decoder(SCORED), _decoder()
    tokens = jnp.asarray(np.random.default_rng(0x48).integers(1, SCORED, (2, SEQ + 1)), jnp.int32)
    variables = nn.unbox(full.init(jax.random.key(0), tokens[:, :-1]))

    def moved(leaf):
        # Each axis of the table's rows, its rows past the scored ones moved.
        for axis, size in enumerate(jnp.shape(leaf)):
            if size == ROWS:
                leaf = leaf.at[(slice(None),) * axis + (slice(SCORED, None),)].add(1.0)
        return leaf

    def loss(model, held):
        objective, step = LMObjective(model, SEQ), Step(jnp.asarray(0), jax.random.key(1), None)
        return np.asarray(jax.jit(lambda: objective.scalar_loss(held, {"text": tokens}, step)[0])())

    shifted = jax.tree.map(moved, variables)
    assert loss(narrow, shifted) == loss(narrow, variables)
    assert loss(full, shifted) != loss(full, variables)


def test_a_forward_read_as_logits_is_held_to_the_ids_its_head_scores():
    """A forward read as its logits (`final_read`) is held to the width its head scores: the narrow head's
    56 columns pass, and 80, its table's rows, are refused."""
    narrow = _decoder(SCORED)
    _refuse_unlike_logits(narrow, jnp.zeros((1, 2, SCORED)))
    with pytest.raises(ValueError):
        _refuse_unlike_logits(narrow, jnp.zeros((1, 2, ROWS)))


def test_a_draw_stops_only_at_an_id_its_head_scores_and_pads_with_a_row_of_its_table():
    """A draw may stop at an id its head scores, never one past them that its table holds a row for, which it
    could never draw; its pad, an input, may be any row of the table."""
    narrow = _decoder(SCORED)
    ids = np.ones((1, 4), np.int32)

    def checked(sampling: Sampling) -> np.ndarray:
        return check_inputs(narrow, ids, {}, 2, sampling, 1)

    assert checked(Sampling(eos_token_ids=SCORED - 1, pad_token_id=ROWS - 1)).all()
    with pytest.raises(ValueError):
        checked(Sampling(eos_token_ids=SCORED, pad_token_id=0))


def test_selected_distillation_shares_the_ids_both_heads_score():
    """Selected distillation reads one id's likelihood on both sides, so the student and teacher score the
    same ids: a narrow head beside a full one of the same table is refused, two narrow ones taken."""
    teacher, student = LMObjective(_decoder(SCORED), SEQ), LMObjective(_decoder(SCORED), SEQ)
    DistillationObjective(student, teacher, divergence="selected")
    with pytest.raises(ValueError):
        DistillationObjective(student, LMObjective(_decoder(), SEQ), divergence="selected")
