"""Prefix runs: a run trains and serves one mask.

The data stage `Blocks` makes a packed row's blocks, each document's
first tokens, its prompt, or a user's runs. The objective trains on
them, scoring no target in its source's block, and a served prompt
takes the same blocks (`prompt_blocks`). The trained loss agrees with
float64 under the mask `block_causal` writes apart from Dew (check 1); a
served rollout is bitwise the one a request carrying the trained blocks
gets; the counted targets are exact; with no prefix the loss is the
causal one, bitwise.
"""

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from block_support import VOCAB, block_causal, model, wrapped
from model_support import TINY_DECODER, model_draw
from reference_error import assert_as_exact, seeds_for, widened, without_ynn

from dew.batch import BLOCKS_FIELD, coordinates
from dew.data.chat import Role
from dew.data.steps.blocks import Blocks
from dew.inference.serving import Server
from dew.inference.tasks import TextGeneration
from dew.nn.inputs import ModelInputs
from dew.nn.kv_cache import KVCache
from dew.nn.mixers import AttentionMixer
from dew.nn.visibility import prefix_blocks, prompt_blocks, served_blocks, with_served_blocks
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.objectives.lm.objective import batch_text
from dew.sampling import Sampling

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


def blocky(images: bool = False):
    """A decoder whose one kind reads its blocks whole, so a prefix run trains and serves on it, and with
    `images` its images too."""
    return model(full_attention={"mixer": AttentionMixer(bidirectional_blocks=True,
                                                         bidirectional_images=images)})


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
    """Each document's first tokens make its block, by the places the record gives or, where it gives
    none, by each document's own count of its tokens, as the loss counts them (`prefix_groups`)."""
    for segments, positions, expected in zip(SEGMENTS, POSITIONS, BLOCKS, strict=True):
        record = {"text": np.zeros(ROW, np.int32), "text_segment_ids": segments, "text_positions": positions}
        np.testing.assert_array_equal(Blocks(prefix=PREFIX).map(record)["text_block_groups"], expected)
        unplaced = {name: value for name, value in record.items() if name != "text_positions"}
        np.testing.assert_array_equal(Blocks(prefix=PREFIX).map(unplaced)["text_block_groups"], expected)
    two = {"text": np.zeros(6, np.int32), "text_segment_ids": np.asarray([1, 1, 1, 2, 2, 2], np.int32)}
    np.testing.assert_array_equal(Blocks(prefix=2).map(two)["text_block_groups"], [1, 1, -1, 2, 2, -1])


def test_a_prefix_run_trains_the_block_mask_on_par_with_float64():
    decoder = blocky()
    objective = LMObjective(decoder, SEQ, prefix=PREFIX)
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")

    init = jax.jit(decoder.init)

    def draw(seed):
        tokens = jax.random.randint(jax.random.fold_in(jax.random.key(1), seed), (BATCH, ROW), 0, VOCAB)
        return init(jax.random.fold_in(jax.random.key(0), seed), tokens[:, :-1]), tokens

    reference = without_ynn(lambda params, tokens: masked(decoder, params, tokens))
    truth = without_ynn(lambda params, tokens: masked(wide, params, tokens))
    dew = without_ynn(lambda params, tokens: trained(objective, params, tokens))
    seeds = seeds_for(reference, draw, "prefix run loss", truth_fn=truth)
    assert_as_exact(dew, reference, draw, "prefix run loss", seeds=seeds, truth_fn=truth)


def test_the_counted_targets_leave_out_exactly_the_in_block_ones():
    objective = LMObjective(blocky(), SEQ, prefix=PREFIX)
    counts = objective.scored_tokens(packed_batch(np.zeros((BATCH, ROW), np.int32)))
    np.testing.assert_array_equal(counts, counted_targets().sum(axis=1))
    # Row one's first document scores its last two transitions, its second none; row two's first scores three.
    np.testing.assert_array_equal(counts, [2, 3])


PROMPT_ROLES = np.asarray(
    [[Role.USER] * 3 + [Role.ASSISTANT] * 2 + [Role.USER, Role.ASSISTANT, Role.ASSISTANT, Role.PAD],
     [Role.USER] * 4 + [Role.ASSISTANT] * 2 + [Role.USER, Role.ASSISTANT, Role.ASSISTANT]], np.int8)
"""Each packed row's roles: each document's user tokens, then its assistant's."""


@pytest.mark.parametrize("prefix", [PREFIX, "prompt"])
def test_a_prefix_runs_batch_without_blocks_trains_on_the_blocks_it_serves(prefix):
    """A prefix run's batch that carries no blocks, as a precompiled loader or a raw stream gives it,
    trains on the prefix's blocks all the same: its loss, its gradient and the host's count of scored
    targets are bit for bit those of the same rows given the data stage's blocks (`Blocks`), which the run
    serves. A prompt run reads its blocks off the rows' roles, and a batch without them is refused."""
    from dew.data.chat import ROLES_KEY
    from dew.objectives.base import Step

    decoder = blocky()
    objective = LMObjective(decoder, SEQ, prefix=prefix)
    tokens = jax.random.randint(jax.random.key(1), (BATCH, ROW), 0, VOCAB)
    params = decoder.init(jax.random.key(0), tokens[:, :-1])
    stage = Blocks(prefix=PREFIX) if prefix == PREFIX else Blocks(prompt=True)
    staged = np.stack([stage.map({"text": np.zeros(ROW, np.int32), "text_segment_ids": segments,
                                  "text_positions": positions, "text_roles": roles})["text_block_groups"]
                       for segments, positions, roles in zip(SEGMENTS, POSITIONS, PROMPT_ROLES, strict=True)])
    given = {**packed_batch(tokens), ROLES_KEY: PROMPT_ROLES, "text_block_groups": staged}
    bare = {name: value for name, value in given.items() if name != "text_block_groups"}
    step = Step(jnp.asarray(0), jax.random.key(2), None)

    def trained(batch):
        return jax.value_and_grad(objective.scalar_loss, has_aux=True)(params, batch, step)

    for got, want in zip(jax.tree.leaves(trained(bare)), jax.tree.leaves(trained(given)), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(objective.scored_tokens(bare), objective.scored_tokens(given))
    if prefix == "prompt":
        unroled = {name: value for name, value in bare.items() if name != ROLES_KEY}
        with pytest.raises(ValueError):
            objective.scored_tokens(unroled)


HOLED = np.asarray([[1, 1, 0, 0, 1, 1, 1, 1, 1]], bool)
"""A row's validity with a hole inside its one document."""
HOLED_ROLES = np.asarray([[Role.USER, Role.ASSISTANT] + [Role.USER] * 3 + [Role.ASSISTANT] * 4], np.int8)
"""The row's roles: a user token, the assistant's, and past the hole another user turn and the assistant's."""
HOLED_BLOCKS = {2: [[1, 1, -1, -1, -1, -1, -1, -1, -1]], "prompt": [[1, -1, -1, -1, -1, -1, -1, -1, -1]]}
"""The row's blocks under each prefix: its first two tokens, or its user turn before its first assistant
token; past the hole no token is a block's."""


@pytest.mark.parametrize("prefix", [2, "prompt"])
def test_a_document_runs_on_across_a_hole_in_its_validity(prefix):
    """Masked slots inside a row's one document neither restart its places nor open another document, as
    the decoder's coordinates count them (`coordinates`, `document_starts`): under a prefix of 2 its block
    is its first two tokens, under "prompt" its first, the user's turn before its first assistant token.
    The row without blocks trains bit for bit as with those blocks given, and the host counts the same
    targets."""
    from dew.data.chat import ROLES_KEY
    from dew.objectives.base import Step

    decoder = blocky()
    objective = LMObjective(decoder, SEQ, prefix=prefix)
    tokens = jax.random.randint(jax.random.key(1), (1, ROW), 0, VOCAB)
    params = decoder.init(jax.random.key(0), tokens[:, :-1])
    bare = {TEXT_KEY: ModelInputs(tokens, token_fields={"attention_mask": jnp.asarray(HOLED)}),
            ROLES_KEY: HOLED_ROLES}
    given = {**bare, "text_block_groups": np.asarray(HOLED_BLOCKS[prefix], np.int32)}
    step = Step(jnp.asarray(0), jax.random.key(2), None)

    def trained(batch):
        return jax.value_and_grad(objective.scalar_loss, has_aux=True)(params, batch, step)

    for got, want in zip(jax.tree.leaves(trained(bare)), jax.tree.leaves(trained(given)), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(objective.scored_tokens(bare), objective.scored_tokens(given))


def test_a_prefix_runs_task_serves_the_blocks_the_run_trained_on():
    """A prefix run's task serves the trained mask: its greedy cached rollout is bit for bit the rollout
    the same model's causal task gives a request that carries the run's blocks itself, each prompt's
    first `PREFIX` real tokens one block (`prefix_blocks`, the rule the run's data stage writes): the same
    program on the same mask. That a cached rollout reads its blocks as a full pass does is
    tests/test_block_visibility.py's check 1."""
    decoder = blocky()
    params = jax.jit(decoder.init)(jax.random.key(0), jnp.zeros((1, PROMPT), jnp.int32))
    prompt = jax.random.randint(jax.random.key(1), (BATCH, PROMPT), 0, VOCAB)
    served = LMObjective(decoder, SEQ, prefix=PREFIX).build_task(params, processor=None)(prompt, NEW, key=0)
    places = jnp.broadcast_to(jnp.arange(PROMPT), prompt.shape)
    blocks = prefix_blocks(jnp.ones(prompt.shape, jnp.int32), places, PREFIX)
    causal = LMObjective(decoder, SEQ).build_task(params, processor=None)
    given = causal(ModelInputs(prompt, {BLOCKS_FIELD: blocks}), NEW, key=0)
    np.testing.assert_array_equal(served.tokens, given.tokens)
    np.testing.assert_array_equal(served.raw_log_probs, given.raw_log_probs)


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


def test_generate_refuses_a_prefix_its_model_or_its_value_cannot_take():
    """`generate(..., prefix=)` refuses as a prefix run's objective and task do (`refuse_prefix`): a causal
    decoder, which would serve the prompt causally, and a prefix of no tokens."""
    from dew.sampling import generate

    sampling = Sampling(temperature=0, pad_token_id=0)
    prompt = ModelInputs(jnp.asarray([[1, 2, 3, 4]], jnp.int32))
    for decoder, prefix in ((model(full_attention={}), PREFIX), (blocky(), 0)):
        decoder = decoder.clone(max_seq_len=128)
        variables = decoder.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
        with pytest.raises(ValueError):
            generate(decoder, variables, prompt, NEW, key=1, sampling=sampling, prefix=prefix)


def test_a_served_prompt_shorter_than_its_prefix_is_refused():
    """A prompt shorter than its prefix is refused where it is served, by the count of its real tokens the
    caller holds: its trained block would cover tokens it does not hold."""
    decoder = blocky().clone(max_seq_len=128)
    variables = decoder.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    task = TextGeneration(decoder, variables, sampling=Sampling(temperature=0, pad_token_id=0), prefix=PREFIX)
    with pytest.raises(ValueError):
        task(np.asarray([[1, 2]]), NEW, key=0)
    with pytest.raises(ValueError):
        Server.from_task(task, slots=2, capacity=64).submit([1, 2], NEW, key=0)


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


@pytest.mark.parametrize("prefix", [PREFIX, "prompt"])
def test_a_prefix_runs_stage_keeps_the_blocks_a_record_carries(prefix):
    """The rows a prefix run's compiled chain delivers (`fitted` with its objective's stages), to training
    and to evaluation alike: a record that carries its own blocks keeps them, as the loss keeps a batch's
    (`prefixed_fields`), and one without takes the prefix's, each document's first tokens or its prompt;
    a prompt run's record that carries blocks needs no roles of its own, and each row's targets the loss
    scores are counted over those blocks. A stage a pipeline names itself replaces them."""
    from dew.data import Memory
    from dew.data.compile import fitted
    from dew.data.dataset import DataPartition

    own = np.asarray([0, 0, -1, -1, 1, 1, 1, -1, -1], np.int32)
    roles = np.asarray([Role.USER] * 3 + [Role.ASSISTANT] * 3 + [Role.USER] + [Role.ASSISTANT] * 2, np.int8)
    bare = {"text": np.arange(1, ROW + 1, dtype=np.int32), "text_segment_ids": SEGMENTS[1],
            "text_roles": roles}
    records = [{**bare, "text_block_groups": own}, bare]
    filled = BLOCKS[1] if prefix == PREFIX else np.asarray([1, 1, 1, -1, -1, -1, 2, -1, -1], np.int32)
    chain = Memory(records).batch(2).read("memory").workers(0)
    objective = LMObjective(blocky(), SEQ, prefix=prefix)
    loader, readers = fitted(chain, chain, objective=objective.data())
    for stream in (loader.train(DataPartition()), readers["val"](DataPartition())):
        try:
            delivered = next(stream)
        finally:
            stream.close()
        np.testing.assert_array_equal(delivered["text_block_groups"], np.stack([own, filled]))
        # Each row's targets its loss scores over those blocks: a document's next tokens, those inside a
        # block dropped (own: 3 of the first document's 5 and both of the second's; a prefix of 3: 3
        # and none; the prompts: 3 and both).
        laid = coordinates(segments=delivered["text_segment_ids"], max_segments=ROW, right_padded=True, xp=np)
        scored = {TEXT_KEY: delivered["text"], "text_coordinates": laid,
                  "text_block_groups": delivered["text_block_groups"]}
        np.testing.assert_array_equal(objective.scored_tokens(scored), [5, 3] if prefix == PREFIX else [5, 5])
    named = Blocks(prefix=PREFIX).map({**records[0], "text_positions": POSITIONS[1]})
    np.testing.assert_array_equal(named["text_block_groups"], BLOCKS[1])


SERVED_BUDGET = 3
"""The tokens each served request draws."""

WHOLE = {"slots": 2, "capacity": 64, "admission": 2}
"""A dense server prefilling each prompt whole, its three requests queued past its two slots."""
PIECES = {"slots": 2, "capacity": 64, "kv_cache": KVCache(page_size=16, pages=8), "chunk": 2}
"""A paged server prefilling in pieces of 2 tokens, shorter than the prompts' blocks."""
SHARED = {"slots": 2, "capacity": 64, "kv_cache": KVCache(page_size=4, pages=40), "prefix_cache": True}
"""A paged server sharing its full pages of 4 prompt tokens with a later prompt that agrees with them."""


@dataclasses.dataclass(frozen=True)
class Served:
    """One served request: its prompt's first `length` tokens of the draw's row, with `flipped` places
    changed, carrying the groups `groups` names (a field's group of each token, -1 for none)."""

    length: int
    groups: Mapping[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    flipped: tuple[int, ...] = ()


THREE = (Served(5), Served(6), Served(7))
"""Three prompts of several lengths carrying no groups: a prefix run's server makes their blocks."""
CARRIED = tuple(Served(len(blocks), {BLOCKS_FIELD: blocks})
                for blocks in [(-1, 0, 0, 0, -1), (-1, -1, 4, 4, 4, 4), (2, 2, -1, 5, 5, 5, -1)])
"""Three prompts carrying their own blocks, as a decision row lays its blocks out: runs of 2 to 4 tokens,
longer than `PIECES`' pieces, two in the last prompt."""
SCATTERED = (Served(7, {BLOCKS_FIELD: (-1, 7, -1, 7, -1, -1, -1)}),
             Served(6, {BLOCKS_FIELD: (3, -1, -1, -1, 3, -1)}))
"""Blocks whose members are not adjacent, which a layer reads as one block wherever they lie."""
MEDIA = (Served(7, {BLOCKS_FIELD: (-1, 0, 0, -1, -1, -1, -1), "image_groups": (-1, -1, 1, 1, 1, -1, -1)}),
         Served(6, {"image_groups": (-1, 4, 4, 4, -1, -1)}))
"""Prompts carrying an image beside a block, overlapping it, and an image alone."""


def served_inputs(row: np.ndarray, request: Served) -> ModelInputs:
    """`request`'s prompt of the draw's `row`, carrying its groups."""
    prompt = row[:request.length].copy()
    prompt[list(request.flipped)] = (prompt[list(request.flipped)] + 1) % VOCAB
    return ModelInputs(jnp.asarray(prompt[None], jnp.int32),
                       {name: jnp.asarray([held], jnp.int32) for name, held in request.groups.items()})


def assert_served_as_exact(requests: Sequence[Served], label: str, *, serving: dict, prefix=None,
                           images: bool = False, hits: Sequence[int] | None = None) -> None:
    """Check 1 of a server's draws: greedy requests through a server with `serving`, all queued at once, or
    with `hits` one after another (`prefix_hits` after each, its shared prefix pages), draw what the lone
    task draws, and score each drawn token's raw log probability on par with a full forward of the decoder
    over the prompt, its groups and the tokens drawn before it, from the float64 twin's."""
    decoder = blocky(images).clone(max_seq_len=128)
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")
    variables = decoder.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    task = TextGeneration(decoder, variables, sampling=Sampling(temperature=0, pad_token_id=0), prefix=prefix)
    width = max(request.length for request in requests)

    def draw(seed):
        row = np.random.default_rng([0x1536, seed]).integers(1, VOCAB, width).astype(np.int32)
        drawn = tuple(np.asarray(task(served_inputs(row, request), SERVED_BUDGET, key=index).host().tokens)
                      [0, -SERVED_BUDGET:] for index, request in enumerate(requests))
        return row, drawn

    def served(row, drawn):
        server = Server.from_task(task, **serving)
        tickets, shared = [], []
        for index, request in enumerate(requests):
            tickets.append(server.submit(served_inputs(row, request), SERVED_BUDGET, key=index))
            if hits is not None:
                server.run()
                shared.append(server.prefix_hits)
        server.run()
        assert hits is None or shared == list(hits), shared
        generations = [ticket.result().host() for ticket in tickets]
        for generation, tokens in zip(generations, drawn, strict=True):
            np.testing.assert_array_equal(generation.tokens[0, -SERVED_BUDGET:], tokens)
        return tuple(np.asarray(generation.raw_log_probs[0], np.float32) for generation in generations)

    def explicit(model, wide_weights: bool = False):
        forward = without_ynn(lambda weights, tokens, groups: jax.nn.log_softmax(
            model.apply(weights, tokens, **groups)))

        def scored(row, drawn):
            weights = widened(variables) if wide_weights else variables
            scores = []
            for request, tokens in zip(requests, drawn, strict=True):
                inputs = served_inputs(np.asarray(row), request)
                prompt, tokens = np.asarray(inputs.tokens), np.asarray(tokens)
                carried = {name: np.asarray(value) for name, value in inputs.token_fields.items()}
                held = with_served_blocks(carried, np.ones(prompt.shape, bool), prefix)
                tail = np.full((1, SERVED_BUDGET - 1), -1, np.int32)
                logits = forward(weights, jnp.asarray(np.concatenate([prompt, tokens[None, :-1]], 1)),
                                 {name: jnp.asarray(np.concatenate([value, tail], 1))
                                  for name, value in held.items()})
                read = logits[0, prompt.shape[1] - 1:]
                scores.append(np.asarray(jnp.take_along_axis(read, jnp.asarray(tokens)[:, None], -1)[:, 0]))
            return tuple(scores)
        return scored

    seeds = seeds_for(explicit(decoder), draw, label, truth_fn=explicit(wide, wide_weights=True))
    assert_as_exact(served, explicit(decoder), draw, label, seeds=seeds,
                    truth_fn=explicit(wide, wide_weights=True))


@pytest.mark.parametrize("prefix, serving", [(PREFIX, WHOLE), (PREFIX, PIECES), ("prompt", PIECES)],
                         ids=["first-tokens", "first-tokens-in-pieces", "prompt-in-pieces"])
def test_a_served_prefix_run_scores_what_each_request_scores_alone(prefix, serving):
    """A server prefills each prompt with the blocks the run trained (`with_served_blocks`), so requests of
    several lengths, queued past the slots, draw what the lone task draws and score it as a full forward
    over those blocks does (check 1). A server prefilling in pieces shorter than a block prefills that block
    in one longer piece (`Server._piece_end`), as a layer reads it whole, and the rest of the prompt in
    pieces."""
    assert_served_as_exact(THREE, f"served prefix {prefix} run", serving=serving, prefix=prefix)


@pytest.mark.parametrize("requests, serving, prefix, images", [
    (CARRIED, WHOLE, None, False), (CARRIED, PIECES, None, False), (CARRIED, PIECES, PREFIX, False),
    (SCATTERED, PIECES, None, False), (MEDIA, PIECES, None, True)],
    ids=["blocks", "blocks-in-pieces", "blocks-over-a-prefix-run", "scattered-in-pieces",
         "images-and-blocks"])
def test_a_served_prompt_carrying_its_groups_scores_what_it_scores_alone(requests, serving, prefix, images,
                                                                          request):
    """A prompt that carries its own groups, blocks or images, has them slotted beside its tokens and kept
    in its row's cache with its keys: requests queued past the slots draw what the lone task draws and score
    it as a full forward over those groups does (check 1). A piece never splits a group, its members
    adjacent or not, and a prompt's own blocks stand over a prefix run's, as `generate` keeps them."""
    assert_served_as_exact(requests, f"served {request.node.callspec.id}", serving=serving, prefix=prefix,
                           images=images)


LONG_BLOCK, SHORT_BLOCK = (-1, 0, 0, 0, 0, 0, -1, -1), (-1, 0, 0, 0, -1, -1, -1, -1)
"""Two layouts of one eight-token prompt whose block reaches a different last token."""
REACHING = (-1, 2, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1)
"""A block of the second and ninth tokens, so its first member reads what the ninth reads, the sixth too."""


@pytest.mark.parametrize("requests, prefix, hits", [
    ((Served(8, {BLOCKS_FIELD: LONG_BLOCK}), Served(8, {BLOCKS_FIELD: SHORT_BLOCK}),
      Served(8, {BLOCKS_FIELD: LONG_BLOCK})), None, (0, 0, 4)),
    ((Served(12, {BLOCKS_FIELD: REACHING}), Served(12, {BLOCKS_FIELD: REACHING}, flipped=(6,)),
      Served(12, {BLOCKS_FIELD: REACHING})), None, (0, 0, 8)),
    ((Served(9), Served(10)), 8, (0, 8))],
    ids=["blocks-alike", "scattered-reach", "prefix-run-lengths"])
def test_a_prefix_page_is_shared_only_between_prompts_whose_groups_agree(requests, prefix, hits):
    """A prompt's prefix pages are named under its groups and its tokens up to its last grouped one
    (`_Row.salt`), as a token's keys read its whole group and what each member reads: a prompt of the same
    tokens whose block ends sooner prefills its own pages, as does one changed only at an ungrouped token
    a later member reads, while one that agrees through its last grouped token reads the first's full pages,
    a prefix run's prompts of several lengths alike, its resumed piece starting inside a group. Each scores
    what it scores alone (check 1)."""
    assert_served_as_exact(requests, f"shared pages {hits}", serving=SHARED, prefix=prefix, hits=hits)


def chunked(size: int, *, unchunked: bool = False):
    """A decoder whose block-reading kind cuts its attention into chunks of `size` places, with a second
    block-reading kind that has no chunk beside it when `unchunked`."""
    kinds = {"chunked": {"mixer": AttentionMixer(bidirectional_blocks=True), "chunk": size}}
    if unchunked:
        kinds["whole"] = {"mixer": AttentionMixer(bidirectional_blocks=True)}
    return model(TINY_DECODER["num_layers"], **kinds)


def test_a_prefix_run_needs_a_trunk_layer_that_reads_blocks():
    """A block-reading kind only the prediction depth uses leaves the trunk causal, so a prefix run would drop
    in-block targets the trunk never saw: refused."""
    depth_only = model(full_attention={}).clone(
        kinds={"full_attention": {}, "depth": {"mixer": AttentionMixer(bidirectional_blocks=True)}},
        mtp_layer_type="depth", num_nextn_predict_layers=1)
    with pytest.raises(ValueError):
        LMObjective(depth_only, SEQ, prefix=PREFIX)


def test_a_prefix_run_refuses_a_block_that_crosses_an_attention_chunk():
    """A block-reading layer with an attention chunk reads a block in part wherever the block crosses one
    of the chunk's multiples, so a target across the cut is invisible to its source while the loss drops
    it. A block at a shifted origin crosses one whatever the chunk's length: a row that continues its
    document at place 6 under a chunk of 8 holds its block over places 6-8, which the chunk cuts at 8. So a
    prefix run needs a block-reading layer with no chunk, and serving refuses the same."""
    for chunk in (2, PREFIX, SEQ):
        with pytest.raises(ValueError):
            LMObjective(chunked(chunk), SEQ, prefix=PREFIX)
    with pytest.raises(ValueError):
        LMObjective(chunked(SEQ), SEQ, prefix="prompt")
    decoder = chunked(SEQ).clone(max_seq_len=128)
    with pytest.raises(ValueError):
        TextGeneration(decoder, decoder.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)), prefix=PREFIX)
    assert LMObjective(chunked(2, unchunked=True), SEQ, prefix=PREFIX).prefix == PREFIX
    assert LMObjective(chunked(2, unchunked=True), SEQ, prefix="prompt").prefix == "prompt"


UNPACKED_BLOCKS = np.broadcast_to(np.where(np.arange(ROW) < PREFIX, 1, -1).astype(np.int32), (BATCH, ROW))
"""Each unpacked row's blocks under `PREFIX`: its first tokens, one block."""


def test_a_batch_of_blocks_is_refused_where_every_layer_that_reads_them_is_chunked():
    """A batch's blocks are runs of any places, which a chunk cuts wherever they cross its multiples: a
    model whose only block-reading layer is chunked refuses them, and one with an unchunked layer beside it
    scores them."""
    tokens = jax.random.randint(jax.random.key(1), (BATCH, ROW), 0, VOCAB)
    batch = batch_text({TEXT_KEY: tokens, "text_block_groups": UNPACKED_BLOCKS})
    cut = chunked(2)
    with pytest.raises(ValueError):
        LMObjective(cut, SEQ).token_scores(cut.init(jax.random.key(0), tokens[:, :-1]), batch)
    whole = chunked(2, unchunked=True)
    scored = LMObjective(whole, SEQ).token_scores(whole.init(jax.random.key(0), tokens[:, :-1]), batch)
    assert np.asarray(scored.weights).sum() > 0


def test_a_prefix_run_and_a_batch_of_blocks_are_admitted_by_one_rule():
    """A prefix run and a batch that carries its own blocks admit the same models (`refuse_block_reading`):
    a decoder with no block-reading layer and one whose every block-reading layer is chunked refuse both,
    and one with an unchunked block-reading layer takes both."""
    tokens = jax.random.randint(jax.random.key(1), (BATCH, ROW), 0, VOCAB)
    batch = batch_text({TEXT_KEY: tokens, "text_block_groups": UNPACKED_BLOCKS})

    def admits(decoder) -> tuple[bool, bool]:
        verdicts = []
        for admit in (lambda: LMObjective(decoder, SEQ, prefix=PREFIX),
                      lambda: LMObjective(decoder, SEQ).token_scores(
                          decoder.init(jax.random.key(0), tokens[:, :-1]), batch)):
            try:
                admit()
            except ValueError:
                verdicts.append(False)
            else:
                verdicts.append(True)
        return verdicts[0], verdicts[1]

    assert admits(model(full_attention={})) == (False, False)
    assert admits(chunked(2)) == (False, False)
    assert admits(chunked(2, unchunked=True)) == (True, True)


def test_a_prefix_run_trains_and_serves_through_a_wrapper_of_a_block_reading_decoder():
    """A wrapper answers for its decoder's block reading (`DecoderWrapper`): its prefix run is built, scored
    as the bare decoder's objective scores it (check 1, float64 from the widened decoder), and served the
    draws the bare decoder's task serves."""
    decoder = blocky()
    wrapper = wrapped(decoder)
    objective = LMObjective(wrapper, SEQ, prefix=PREFIX)
    bare = LMObjective(decoder, SEQ, prefix=PREFIX)
    wide = LMObjective(decoder.clone(dtype=jnp.float64, attention_impl="reference"), SEQ, prefix=PREFIX)
    draw = model_draw(wrapper, (BATCH, ROW), VOCAB)

    def language_model(variables):
        return {"params": variables["params"][wrapper.wraps]}

    def scored(scorer, variables, tokens):
        batch = batch_text({TEXT_KEY: tokens, "text_block_groups": UNPACKED_BLOCKS})
        scores = scorer.token_scores(variables, batch)
        return scores.losses, scores.weights

    reference = without_ynn(lambda variables, tokens: scored(bare, language_model(variables), tokens))
    truth = without_ynn(lambda variables, tokens: scored(wide, language_model(variables), tokens))
    dew = without_ynn(lambda variables, tokens: scored(objective, variables, tokens))
    seeds = seeds_for(reference, draw, "wrapped prefix run loss", truth_fn=truth)
    assert_as_exact(dew, reference, draw, "wrapped prefix run loss", seeds=seeds, truth_fn=truth)
    variables = wrapper.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    prompt = np.asarray([[3, 1, 4, 1, 5]])
    sampling = Sampling(temperature=0, pad_token_id=0)
    served = TextGeneration(wrapper, variables, sampling=sampling, prefix=PREFIX)
    alone = TextGeneration(decoder, language_model(variables), sampling=sampling, prefix=PREFIX)
    expected = alone(prompt, NEW, key=0).host().tokens
    np.testing.assert_array_equal(served(prompt, NEW, key=0).host().tokens, expected)


def test_a_served_prompt_block_is_made_under_a_trace_as_on_the_host():
    """`prompt_blocks` reads no host value, so a trace makes the blocks the host makes of the same mask."""
    valid = np.asarray([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], bool)
    traced = jax.jit(lambda mask: prompt_blocks(mask, PREFIX))(jnp.asarray(valid))
    np.testing.assert_array_equal(np.asarray(traced), prompt_blocks(valid, PREFIX, xp=np))


def test_a_shared_prefix_page_inside_a_block_is_reused_only_with_a_block_of_the_same_tokens():
    """A paged server reuses a prefix page only where the page's keys read the same block: a later request
    whose block holds the same tokens shares its pages, and one whose first page holds the same tokens under
    another block shares none. Each draws what it draws alone."""
    decoder = blocky().clone(max_seq_len=128)
    variables = decoder.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    task = TextGeneration(decoder, variables, sampling=Sampling(temperature=0, pad_token_id=0), prefix=8)
    server = Server.from_task(task, slots=2, capacity=64, kv_cache=KVCache(page_size=4, pages=40),
                              prefix_cache=True)
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14],
               [1, 2, 3, 4, 20, 21, 22, 23, 24]]
    reused, tickets = [], []
    for prompt in prompts:
        before = server.prefix_hits
        tickets.append(server.submit(prompt, NEW, key=0))
        server.run()
        reused.append(server.prefix_hits - before)
    assert reused[0] == 0 and reused[1] > 0 and reused[2] == 0
    for ticket, prompt in zip(tickets, prompts, strict=True):
        lone = task(np.asarray([prompt]), NEW, key=0).host().tokens
        np.testing.assert_array_equal(ticket.result().host().tokens, lone)


CONTEXT = 5
"""The context ids of each harness row, its continuation the rest."""


@pytest.mark.parametrize("prefix", [PREFIX, "prompt"])
def test_a_prefix_runs_harness_scores_a_continuation_as_its_generation_reads_it(prefix):
    """lm-eval's likelihood of a prefix run's continuation is read through the blocks a server makes of
    its context (`served_blocks`), each continuation token in none: on par from float64 with an explicit
    forward of the decoder over those blocks (check 1), where a causal read of the context, which the
    run's generation never makes, is far off."""
    from dew.data.text import ByteTokenizer
    from dew.eval.harness import DewLM
    from dew.inference.loading import RunProcessor

    decoder = blocky()
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")
    draw = model_draw(decoder, (BATCH, ROW), VOCAB)
    context = np.broadcast_to(np.arange(ROW) < CONTEXT, (BATCH, ROW))
    blocks = jnp.asarray(np.where(context, served_blocks(context, prefix), -1), jnp.int32)

    def explicit(model):
        def scored(variables, tokens):
            logits = model.apply(variables, tokens[:, :-1], block_groups=blocks[:, :-1])
            picked = jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[:, 1:, None], axis=-1)[..., 0]
            return picked[:, CONTEXT - 1:].sum(axis=-1)
        return without_ynn(scored)

    def harness(variables, tokens):
        task = TextGeneration(decoder, variables, RunProcessor(ByteTokenizer()), prefix=prefix)
        rows = [(None, row[:CONTEXT].tolist(), row[CONTEXT:].tolist()) for row in np.asarray(tokens)]
        return np.asarray([value for value, _ in DewLM(task, batch_size=BATCH)._loglikelihood_tokens(rows)],
                          np.float32)

    seeds = seeds_for(explicit(decoder), draw, f"prefix {prefix} harness likelihood", truth_fn=explicit(wide))
    assert_as_exact(harness, explicit(decoder), draw, f"prefix {prefix} harness likelihood", seeds=seeds,
                    truth_fn=explicit(wide))
