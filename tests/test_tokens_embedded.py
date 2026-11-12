"""No token id reaches a model's embedding table without a row of it silently: an id outside the table read
no row, and its gather filled NaN, so a run trained, or a task served, NaN with finite logits beside it. The
ids a run's rows or a request hold are checked where the host holds them, every training batch before it is
placed and every request as it is prepared (`InputSpec.bounded`, the token field declaring the table it
indexes); a run that holds none is never refused, whatever ids its tokenizer could emit."""

from pathlib import Path

import jax
import numpy as np
import optax
import pytest

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.data.text import HFTokenizer
from dew.decision import DecisionObjective
from dew.diffusion.discrete import MDLM
from dew.inference.tasks import RunProcessor, TextGeneration
from dew.inputs import OutsideTable
from dew.objectives.diffusion import MaskedDiffusionObjective
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Trainer

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "qwen35-native-tiny"
"""A tiny tokenizer of 256 ids, `token0` to `token255` and its specials among them."""
ROWS = 200
"""A model's table short of it: ids 200 to 255 have no row."""
SEQ_LEN = 16


def short(**fields):
    return models.build("causal_transformer", vocab_size=ROWS, emb_features=32, num_layers=1, num_heads=2,
                        mlp_features=64, max_seq_len=64, **fields)


def outside(raised: pytest.ExceptionInfo[OutsideTable]) -> tuple[int, int]:
    """The id a refusal names and the rows of the table it names."""
    return raised.value.token, raised.value.rows


def token_rows(*past: int):
    """Sixteen rows of `SEQ_LEN + 1` token ids in the table, the last row's last ids `past`, eight a batch,
    which the suite's eight devices divide."""
    rows = [np.arange(SEQ_LEN + 1, dtype=np.int32) + index for index in range(16)]
    rows[-1][-len(past):] = past
    return Pipeline(Memory([{"text": row} for row in rows])).repeat().batch(8)


@pytest.mark.parametrize("past", [ROWS, ROWS + 37, -1])
def test_an_lm_run_whose_rows_hold_an_id_outside_its_models_table_is_refused_at_that_batch(past):
    """An LM run reading token rows, one of whose ids is outside its model's table (past its rows, or
    below zero), is refused at the batch that holds it, naming the id and the table's rows; the batch
    before it trains."""
    trainer = Trainer(LMObjective(short(), seq_len=SEQ_LEN), optax.sgd(0.1), key=jax.random.key(0))
    with pytest.raises(OutsideTable) as raised:
        trainer.fit(token_rows(past), steps=2)
    assert outside(raised) == (past, ROWS)
    finite = trainer.fit(token_rows(ROWS - 1), steps=2)
    assert int(finite.step) == 2


def test_an_lm_runs_validation_split_holding_an_id_outside_its_models_table_is_refused():
    """A validation split whose rows hold an id outside the model's table is refused as its rows are read,
    on the host before they are placed, as a training batch is; a split within the table scores."""
    def split(*past: int):
        rows = [np.arange(SEQ_LEN + 1, dtype=np.int32) + index for index in range(8)]
        if past:
            rows[-1][-len(past):] = past
        return Pipeline(Memory([{"text": row} for row in rows])).batch(8)

    trainer = Trainer(LMObjective(short(), seq_len=SEQ_LEN), optax.sgd(0.1), key=jax.random.key(0))
    with pytest.raises(OutsideTable) as raised:
        trainer.fit(token_rows(ROWS - 1), steps=2, validation=split(ROWS + 3), eval_every=1)
    assert outside(raised) == (ROWS + 3, ROWS)
    scored = trainer.fit(token_rows(ROWS - 1), steps=2, validation=split(), eval_every=1)
    assert int(scored.step) == 2


def test_a_decision_run_whose_rows_place_a_token_outside_its_backbones_table_is_refused_naming_it(tmp_path):
    """A decision run, whose rows its layout makes with its own tokenizer (its text and its specials), is
    refused at the batch holding a token its backbone's table has no row for, naming the id and the table's
    rows; the same rows over a backbone whose table holds the tokenizer's every id train."""
    from dew.decision import decisions

    tokenizer = HFTokenizer(str(FIXTURE), local_files_only=True)
    rows = Pipeline(Memory([{"text": "token7 token250", "label": label} for label in ("a", "b")] * 8))
    data = rows.pipe(decisions, question="label", options=("a", "b")).repeat().batch(8)

    def fitted(backbone):
        objective = DecisionObjective(backbone, tokenizer=tokenizer)
        return Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(data, steps=1)

    with pytest.raises(OutsideTable) as raised:
        fitted(short())
    token, rows_held = outside(raised)
    assert rows_held == ROWS <= token
    # Twice the fixture's ids, its added specials among them.
    wide = models.build("causal_transformer", vocab_size=512, emb_features=32, num_layers=1, num_heads=2,
                        mlp_features=64, max_seq_len=64)
    assert int(fitted(wide).step) == 1


def test_a_request_holding_an_id_outside_its_models_table_is_refused_by_a_task_and_a_server():
    """A text generation task built over a model whose table is shorter than its processor's vocabulary
    serves requests within the table; a request that holds an id outside it is refused as it is prepared,
    by the task and by a server over it alike, every request's ids passing one check (`check_inputs`),
    naming the id, the table's rows, and its token where the server's processor knows it."""
    from dew.inference.serving import Server

    model = short()
    variables = model.init(jax.random.key(0), np.zeros((1, 4), np.int32))
    task = TextGeneration(model, variables, RunProcessor(HFTokenizer(str(FIXTURE), local_files_only=True)))
    task([[4, 5, 6]], 2, key=0)
    with pytest.raises(OutsideTable) as raised:
        task([[4, 5, 250]], 2, key=0)
    assert outside(raised) == (250, ROWS)
    server = Server.from_task(task, slots=1, capacity=32)
    with pytest.raises(OutsideTable) as raised:
        server.submit([4, 5, 250], max_new_tokens=2, key=0)
    assert outside(raised) == (250, ROWS)


PLACEHOLDER = ROWS + 50
"""A multimodal model's image placeholder that its table holds no row of: as `_fused` reads it, only the
slots a call's media fill replace it, and the table reads it at any other."""


def placeholder_model():
    """A Gemma 3 wrapper over `short()`'s table, a tiny vision tower reading one image a slot, its image
    placeholder `PLACEHOLDER`."""
    import jax.numpy as jnp

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    decoder = CausalTransformer(vocab_size=ROWS, emb_features=32, num_layers=1, num_heads=2, mlp_features=64,
                                max_seq_len=64)
    return MultimodalTransformer(
        decoder, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, image_size=8,
                              patch_size=4),
        GemmaProjector(text_width=32, patches_per_side=2, tokens_per_side=1), family="gemma3",
        image_token_id=PLACEHOLDER, dtype=jnp.float32)


def media_rows(ids, images):
    """Token rows `ids` whose slots `images` marks (>= 0) read an image of their own."""
    import jax.numpy as jnp

    from dew.nn.inputs import ModelInputs

    ids, images = jnp.asarray(ids, jnp.int32), jnp.asarray(images, jnp.int32)
    count = ids.shape[0]
    pixels = jnp.linspace(-.5, .5, count * 3 * 8 * 8, dtype=jnp.float32).reshape(count, 1, 3, 8, 8)
    return ModelInputs(ids, {"image_indices": images}, {"pixel_values": pixels})


def test_a_request_holding_a_placeholder_no_media_fill_is_refused_and_one_its_image_fills_is_served():
    """A multimodal model replaces its image placeholder only at the slots a call's media fill (its
    `_fused`), so a text request holding it, which its table would read, is refused as it is prepared,
    naming it; the same ids with an image at that slot are served, the slot reading the tower, and a
    second media slot holding an id past the table (a placeholder the processor wrote) is served too."""
    model = placeholder_model()
    request = media_rows([[4, PLACEHOLDER, 5, 6]], [[-1, 0, -1, -1]])
    variables = model.init(jax.random.key(0), request.tokens, conditioning=request.conditioning,
                           image_indices=request.token_fields["image_indices"])
    task = TextGeneration(model, variables)
    with pytest.raises(OutsideTable) as raised:
        task([[4, PLACEHOLDER, 5, 6]], 2, key=0)
    assert outside(raised) == (PLACEHOLDER, ROWS)
    served = task(request, 2, key=0).host()
    assert np.all(np.isfinite(served.raw_log_probs[0, :int(served.lengths[0])]))
    task(media_rows([[4, PLACEHOLDER + 1, 5, 6]], [[-1, 0, -1, -1]]), 2, key=0)


def test_a_masked_request_whose_image_fills_a_placeholder_past_the_table_is_filled():
    """A masked generation over a bidirectional multimodal model is bounded as a text request is: an id
    past the table at a slot its image fills (the tower's, never the table's) is served, and a response
    filled; at a slot no image fills, refused, naming it."""
    import jax.numpy as jnp

    from dew.diffusion.discrete import DiscreteProcess, LogLinear
    from dew.inference import MaskedGeneration
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    text = CausalTransformer(vocab_size=32, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                             mlp_features=32, max_seq_len=16, causal=False, mask_token_id=31)
    model = MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=40, dtype=jnp.float32)
    inputs = media_rows([[2, 40, 3, 4]], [[-1, 0, -1, -1]])
    variables = model.init(jax.random.key(0), inputs.tokens, conditioning=inputs.conditioning,
                           image_indices=inputs.token_fields["image_indices"])
    task = MaskedGeneration(model, variables, DiscreteProcess(LogLinear(), mask_id=31), steps=4)
    generated = task(inputs, 5, key=0).host()
    assert generated.tokens.shape == (1, 9) and not np.any(generated.tokens[:, 4:] == 31)
    with pytest.raises(OutsideTable) as raised:
        task(media_rows([[2, 40, 3, 4]], [[-1, -1, -1, -1]]), 5, key=0)
    assert outside(raised) == (40, 32)


def test_a_block_request_whose_image_fills_an_id_past_the_table_is_served():
    """A block generation over DiffusionGemma, which replaces the id at each slot its image fills before its
    table reads it (`DiffusionGemma.replaces_media`), is bounded as a text request is: an id past the table
    at a slot its image fills, the conditioner's and never the table's, is served, and at a slot no image
    fills, which the table would read, refused, naming it (train 768 F4: DiffusionGemma stated no such
    replacement, so its block requests refused that id)."""
    import jax.numpy as jnp
    from bank_support import fixture

    from dew.diffusion.block import BlockProcess
    from dew.inference import BlockGeneration
    from dew.nn.inputs import ModelInputs

    model, variables, tokens, indices, media = fixture("diffusion")
    past = tokens.at[0, 1].set(32 + 5)
    task = BlockGeneration(model, variables, BlockProcess(canvas_length=2, vocab_size=32))
    generated = task(ModelInputs(past, {"image_indices": indices}, media), 2, key=0).host()
    assert generated.tokens.shape == (1, 6)
    with pytest.raises(OutsideTable) as raised:
        task(ModelInputs(past, {"image_indices": jnp.full_like(indices, -1)}, media), 2, key=0)
    assert outside(raised) == (37, 32)


def test_a_media_slot_a_model_reads_from_its_table_bounds_its_id():
    """A multimodal model whose decoder hashes n-grams of the ids media fill (`Engram.media_stops` False,
    Qwen4-Exp's) keeps a media slot's id and reads it from a token map of its table's rows, so an id past
    the table there is refused, naming it, as at any slot its table reads; the same model hashing none
    replaces that id with its tower's embedding, and the request passes (train 768 F4: every multimodal
    model was taken to replace it, and the hashed id read past the map with no error)."""
    from dew.inputs import Field
    from dew.nn.engram import Engram

    model = placeholder_model()
    hashing = model.clone(language_model=model.language_model.clone(engram=Engram(
        layer_ids=(0,), num_embeddings=(88,), max_ngram_size=3, vocab_size=ROWS, n_heads=2, head_dim=4,
        compressed_vocab_size=ROWS, media_stops=False)))
    request = media_rows([[4, PLACEHOLDER + 9, 5, 6]], [[-1, 0, -1, -1]])
    Field.indexing("tokens", (), model).bounded(request)
    with pytest.raises(OutsideTable) as raised:
        Field.indexing("tokens", (), hashing).bounded(request)
    assert outside(raised) == (PLACEHOLDER + 9, ROWS)


def test_a_host_batch_is_bounded_on_the_host():
    """A run's host batch, numpy rows with media beside them (a `ModelInputs` of host arrays), is bounded
    where it lies: no array moves to or from a device, the media slots read on the host, an id past the
    table at a slot its image fills passing and one its table reads refused (train 768 F5: the slots were
    read on a device for every such batch, on the prefetch worker)."""
    from dew.nn.inputs import ModelInputs

    inputs = LMObjective(placeholder_model(), 8, head_chunks=1).inputs
    assert inputs is not None
    ids = np.tile(np.asarray([[2, 3, PLACEHOLDER + 9, 4, 5, 6, 7, 8, 9]], np.int32), (2, 1))
    images = np.tile(np.asarray([[-1, -1, 0, -1, -1, -1, -1, -1, -1]], np.int32), (2, 1))
    pixels = np.zeros((2, 1, 3, 8, 8), np.float32)
    with jax.transfer_guard("disallow"):
        inputs.bounded({"text": ModelInputs(ids, {"image_indices": images}, {"pixel_values": pixels})})
        with pytest.raises(OutsideTable) as raised:
            inputs.bounded({"text": ModelInputs(ids, {"image_indices": np.full_like(images, -1)},
                                                {"pixel_values": pixels})})
    assert outside(raised) == (PLACEHOLDER + 9, ROWS)


def test_a_training_batch_whose_media_fill_ids_past_the_table_trains_and_one_its_table_reads_is_refused():
    """A multimodal run's rows whose media slots hold ids past its model's table, which the towers fill and
    the table never reads, pass the bound a run's every batch passes (`InputSpec.bounded`) and train to a
    finite loss; the same rows with such an id at a slot no image fills, which the table would read, are
    refused, naming it."""
    import optax

    model = placeholder_model()
    rows = jax.device_count()
    ids = np.tile([[2, 3, PLACEHOLDER + 9, 4, 5, 6, 7, 8, 9]], (rows, 1))
    images = np.tile([[-1, -1, 0, -1, -1, -1, -1, -1, -1]], (rows, 1))
    objective = LMObjective(model, 8, head_chunks=1)
    inputs = objective.inputs
    assert inputs is not None
    batch = {"text": media_rows(ids, images)}
    inputs.bounded(batch)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5))
    start, _, _ = trainer.place()
    _, loss, *_ = trainer.compile(start, batch)(start, batch)
    assert np.isfinite(float(loss))
    with pytest.raises(OutsideTable) as raised:
        inputs.bounded({"text": media_rows(ids, np.full_like(images, -1))})
    assert outside(raised) == (PLACEHOLDER + 9, ROWS)


def test_a_masked_run_whose_mask_id_is_outside_its_models_table_is_refused_as_its_objective_is_made():
    """A masked diffusion objective places its mask id inside the compiled step, where no host check of a
    batch sees it, so an objective whose mask id its model's table holds no row for is refused as it is
    made, naming the id; one masking to the table's last row is made."""
    with pytest.raises(OutsideTable) as raised:
        MaskedDiffusionObjective(short(causal=False), MDLM(mask_id=ROWS)(), SEQ_LEN)
    assert outside(raised) == (ROWS, ROWS)
    MaskedDiffusionObjective(short(causal=False), MDLM(mask_id=ROWS - 1)(), SEQ_LEN)


def test_rows_given_already_placed_are_bounded_where_they_lie():
    """A batch a caller places itself (device arrays, an in-memory run's) is bounded as a host batch is
    (`Field.bounded`): an id outside the table refused, naming it, every id within it passing, and an id
    the model replaces before its table reads it passing too, the valid batch read back by its two bounds
    alone."""
    import jax.numpy as jnp

    from dew.inputs import Field
    from dew.nn.blocks import TokenTable

    field = Field("tokens", (4,), table=TokenTable(ROWS, frozenset({ROWS + 7}), media=False))
    valid = jnp.asarray([[1, 2, ROWS - 1, ROWS + 7]], jnp.int32)
    # A valid batch, a replaced id past the table among it, reads back its two bounds alone (explicitly),
    # never the batch whole (an implicit read-back, which the guard refuses).
    with jax.transfer_guard_device_to_host("disallow"):
        field.bounded(valid)
    with pytest.raises(OutsideTable) as refused:
        field.bounded(jnp.asarray([[1, ROWS + 3, 2, 3]], jnp.int32))
    assert (refused.value.token, refused.value.rows) == (ROWS + 3, ROWS)
