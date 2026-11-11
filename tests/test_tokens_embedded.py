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
    the model replaces before its table reads it passing too."""
    import jax.numpy as jnp

    from dew.inputs import Field

    field = Field("tokens", (4,), rows=ROWS, replaced=frozenset({ROWS + 7}))
    field.bounded(jnp.asarray([[1, 2, ROWS - 1, ROWS + 7]], jnp.int32))
    with pytest.raises(OutsideTable) as refused:
        field.bounded(jnp.asarray([[1, ROWS + 3, 2, 3]], jnp.int32))
    assert (refused.value.token, refused.value.rows) == (ROWS + 3, ROWS)
