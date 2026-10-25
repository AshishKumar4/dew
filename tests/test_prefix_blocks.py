"""Prefix runs: a run trains and serves one mask.

The data stage `Blocks` makes a packed row's blocks, each document's
first tokens, its prompt, or a user's runs. The objective trains on
them, scoring no target in its source's block, and a served prompt
takes the same blocks (`prompt_blocks`). The trained loss and a greedy
cached rollout agree with float64 under the mask `block_causal` writes
apart from Dew (check 1); the counted targets are exact; with no prefix
the loss is the causal one, bitwise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from block_support import VOCAB, block_causal, model
from reference_error import assert_as_exact, seeds_for

from dew.data.chat import Role
from dew.data.steps.blocks import Blocks
from dew.data.steps.text import CountTargets
from dew.nn.inputs import coordinates
from dew.nn.mixers import AttentionMixer
from dew.nn.visibility import prompt_blocks
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.objectives.lm.objective import batch_text

SEQ = 8
ROW = SEQ + 1
PREFIX = 3
BATCH = 2
PROMPT = 5
NEW = 3

SEGMENTS = np.asarray([[1, 1, 1, 1, 1, 2, 2, 2, 0], [1, 1, 1, 1, 1, 1, 2, 2, 2]], np.int32)
POSITIONS = np.asarray([[0, 1, 2, 3, 4, 0, 1, 2, 0], [0, 1, 2, 3, 4, 5, 0, 1, 2]], np.int32)
BLOCKS = np.asarray([[1, 1, 1, -1, -1, 2, 2, 2, -1], [1, 1, 1, -1, -1, -1, 2, 2, 2]], np.int32)
"""Each packed row's blocks under `PREFIX`: each document's first tokens, numbered by the document."""


def blocky():
    """A decoder whose one kind reads its blocks whole, so a prefix run trains and serves on it."""
    return model(full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)})


def packed_batch(tokens):
    """The packed rows `tokens` as the data pipeline lays them out: their coordinates, and their blocks."""
    laid = coordinates(segments=SEGMENTS, positions=POSITIONS, max_segments=ROW, right_padded=True, xp=np)
    return {TEXT_KEY: tokens, "text_coordinates": laid, "text_block_groups": BLOCKS}


def counted_targets() -> np.ndarray:
    """Which targets count, written apart from Dew: a target of its source's document, unless both
    lie in one block, which the source reads whole."""
    same_document = (SEGMENTS[:, :-1] == SEGMENTS[:, 1:]) & (SEGMENTS[:, 1:] != 0)
    same_block = (BLOCKS[:, :-1] == BLOCKS[:, 1:]) & (BLOCKS[:, :-1] >= 0)
    return same_document & ~same_block


def trained(objective, params, tokens):
    """The objective's per-token losses and the targets it scores, over the packed rows."""
    scores = objective.token_scores(params, batch_text(packed_batch(tokens)))
    return scores.losses, scores.weights


def masked(decoder, params, tokens):
    """The shifted losses under `block_causal`'s mask over the packed rows, and the targets that count."""
    keep = block_causal(BLOCKS, segments=SEGMENTS)[:, :-1, :-1]
    logits = decoder.apply(params, tokens[:, :-1], segment_ids=SEGMENTS[:, :-1], positions=POSITIONS[:, :-1],
                           attention_pairwise_mask=jnp.asarray(keep))
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])
    return losses, jnp.asarray(counted_targets(), losses.dtype)


def test_the_stage_makes_each_documents_first_tokens_the_blocks_of_its_packed_rows():
    for segments, positions, expected in zip(SEGMENTS, POSITIONS, BLOCKS, strict=True):
        record = {"text": np.zeros(ROW, np.int32), "text_segment_ids": segments, "text_positions": positions}
        np.testing.assert_array_equal(Blocks(prefix=PREFIX).map(record)["text_block_groups"], expected)


def test_a_prefix_run_trains_the_block_mask_on_par_with_float64():
    decoder = blocky()
    objective = LMObjective(decoder, SEQ, prefix=PREFIX)
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")

    init = jax.jit(decoder.init)

    def draw(seed):
        tokens = jax.random.randint(jax.random.fold_in(jax.random.key(1), seed), (BATCH, ROW), 0, VOCAB)
        return init(jax.random.fold_in(jax.random.key(0), seed), tokens[:, :-1]), tokens

    reference = jax.jit(lambda params, tokens: masked(decoder, params, tokens))
    truth = jax.jit(lambda params, tokens: masked(wide, params, tokens))
    dew = jax.jit(lambda params, tokens: trained(objective, params, tokens))
    seeds = seeds_for(reference, draw, "prefix run loss", truth_fn=truth)
    assert_as_exact(dew, reference, draw, "prefix run loss", seeds=seeds, truth_fn=truth)


def test_the_counted_targets_leave_out_exactly_the_in_block_ones():
    objective = LMObjective(blocky(), SEQ, prefix=PREFIX)
    counts = objective.scored_tokens(packed_batch(np.zeros((BATCH, ROW), np.int32)))
    np.testing.assert_array_equal(counts, counted_targets().sum(axis=1))
    # Row one's first document scores its last two transitions, its second none; row two's first scores three.
    np.testing.assert_array_equal(counts, [2, 3])


def test_a_greedy_cached_rollout_serves_the_prefix_mask_on_par_with_float64():
    decoder = blocky()
    objective = LMObjective(decoder, SEQ, prefix=PREFIX)
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")

    init = jax.jit(decoder.init)

    def draw(seed):
        prompt = jax.random.randint(jax.random.fold_in(jax.random.key(1), seed), (BATCH, PROMPT), 0, VOCAB)
        params = init(jax.random.fold_in(jax.random.key(0), seed), prompt)
        task = objective.build_task(params, processor=None)
        return params, prompt, task(prompt, NEW, key=0).tokens[:, PROMPT:]

    def rollout(params, prompt, continuation):
        return objective.build_task(params, processor=None)(prompt, NEW, key=0).raw_log_probs

    def full_pass(model_, params, prompt, continuation):
        """The continuation's log-probabilities under one full pass over prompt and continuation: the
        prompt's first `PREFIX` tokens are one block, and the continuation's tokens are causal."""
        sequence = jnp.concatenate([prompt, continuation], axis=1)
        blocks = np.full(sequence.shape, -1, np.int32)
        blocks[:, :PREFIX] = 1
        keep = block_causal(blocks)[:, :-1, :-1]
        logits = model_.apply(params, sequence[:, :-1], attention_pairwise_mask=jnp.asarray(keep))
        log_probs = jax.nn.log_softmax(logits[:, PROMPT - 1:], axis=-1)
        return jnp.take_along_axis(log_probs, continuation[..., None], axis=-1)[..., 0]

    reference = jax.jit(lambda *inputs: full_pass(decoder, *inputs))
    truth = jax.jit(lambda *inputs: full_pass(wide, *inputs))
    seeds = seeds_for(reference, draw, "prefix rollout", truth_fn=truth)
    assert_as_exact(rollout, reference, draw, "prefix rollout", seeds=seeds, truth_fn=truth)


def test_with_no_prefix_the_loss_is_the_causal_one_bitwise():
    tokens = jax.random.randint(jax.random.key(1), (BATCH, ROW), 0, VOCAB)
    causal = model(full_attention={})
    params = causal.init(jax.random.key(0), tokens[:, :-1])
    batch = {name: value for name, value in packed_batch(tokens).items() if name != "text_block_groups"}
    off = LMObjective(blocky(), SEQ)
    assert not any(isinstance(step, Blocks) for step in off.data().train)
    assert "block_groups" not in batch_text(batch).token_fields
    scored = off.token_scores(params, batch_text(batch))
    plain = LMObjective(causal, SEQ).token_scores(params, batch_text(batch))
    np.testing.assert_array_equal(np.asarray(scored.losses), np.asarray(plain.losses))
    np.testing.assert_array_equal(np.asarray(scored.weights), np.asarray(plain.weights))


def test_a_prefix_run_is_refused_where_its_model_or_its_length_cannot_take_it():
    causal = model(full_attention={})
    with pytest.raises(ValueError):
        LMObjective(causal, SEQ, prefix=PREFIX)
    with pytest.raises(ValueError):
        LMObjective(blocky(), SEQ, prefix=ROW)
    with pytest.raises(ValueError):
        LMObjective(blocky(), SEQ, prefix=0)
    with pytest.raises(ValueError):
        LMObjective(blocky().clone(num_nextn_predict_layers=1), SEQ, prefix=PREFIX, mtp_weight=0.1)


def test_a_served_prompt_shorter_than_its_prefix_is_refused():
    with pytest.raises(ValueError):
        prompt_blocks(np.ones((1, PREFIX - 1), bool), PREFIX, xp=np)


def test_the_block_stage_refuses_a_source_it_cannot_read():
    row = {"text": np.zeros(ROW, np.int32)}
    with pytest.raises(ValueError):
        Blocks()
    with pytest.raises(ValueError):
        Blocks(prefix=PREFIX, prompt=True)
    with pytest.raises(ValueError):
        Blocks(prefix=0)
    with pytest.raises(ValueError):
        Blocks(prompt=True).map(row)
    with pytest.raises(ValueError):
        Blocks(given="blocks").map(row)


def test_a_prompt_block_is_each_documents_tokens_before_its_first_assistant_token():
    roles = np.asarray([Role.USER, Role.USER, Role.ASSISTANT, Role.ASSISTANT, Role.USER, Role.ASSISTANT,
                        Role.ASSISTANT, Role.PAD], np.int8)
    segments = np.asarray([1, 1, 1, 1, 2, 2, 2, 0], np.int32)
    record = {"text": np.zeros(8, np.int32), "text_segment_ids": segments, "text_roles": roles}
    np.testing.assert_array_equal(Blocks(prompt=True).map(record)["text_block_groups"],
                                  [1, 1, -1, -1, 2, -1, -1, -1])


def test_a_given_block_is_each_run_of_one_number_within_one_document():
    segments = np.asarray([1, 1, 1, 1, 1, 2, 2, 0], np.int32)
    record = {"text": np.zeros(8, np.int32), "text_segment_ids": segments,
              "blocks": np.asarray([0, 0, -1, 0, 0, 1, 1, 5], np.int32)}
    np.testing.assert_array_equal(Blocks(given="blocks").map(record)["text_block_groups"],
                                  [0, 0, -1, 1, 1, 2, 2, -1])
    # The same number on two documents' tokens is two blocks, never one.
    split = {"text": np.zeros(2, np.int32), "text_segment_ids": np.asarray([1, 2], np.int32),
             "blocks": np.asarray([0, 0], np.int32)}
    np.testing.assert_array_equal(Blocks(given="blocks").map(split)["text_block_groups"], [0, 1])


def test_a_count_the_record_carries_is_counted_again_over_its_blocks():
    record = {"text": np.arange(1, 7), "text_segment_ids": np.ones(6, np.int32),
              "text_positions": np.arange(6, dtype=np.int32), "_dew_valid_targets": 5}
    assert Blocks(prefix=PREFIX).map(record)["_dew_valid_targets"] == 3


def test_the_prefix_option_puts_the_block_stage_before_the_counts():
    data = LMObjective(blocky(), SEQ, prefix="prompt").data()
    assert data.train[0] == Blocks(prompt=True)
    assert isinstance(data.train[1], CountTargets)
