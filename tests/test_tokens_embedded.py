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
from dew.inference.tasks import RunProcessor, TextGeneration
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Trainer

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "qwen35-native-tiny"
"""A tiny tokenizer of 256 ids, `token0` to `token255` and its specials among them."""
ROWS = 200
"""A model's table short of it: ids 200 to 255 have no row."""
SEQ_LEN = 16


def short():
    return models.build("causal_transformer", vocab_size=ROWS, emb_features=32, num_layers=1, num_heads=2,
                        mlp_features=64, max_seq_len=64)


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
    with pytest.raises(ValueError, match=rf"holds the id {past}, outside the {ROWS} rows"):
        trainer.fit(token_rows(past), steps=2)
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
    with pytest.raises(ValueError, match=rf"holds the id {ROWS + 3}, outside the {ROWS} rows"):
        trainer.fit(token_rows(ROWS - 1), steps=2, validation=split(ROWS + 3), eval_every=1)
    scored = trainer.fit(token_rows(ROWS - 1), steps=2, validation=split(), eval_every=1)
    assert int(scored.step) == 2


def test_a_decision_run_whose_rows_place_a_token_outside_its_backbones_table_is_refused_naming_it(tmp_path):
    """A decision run, whose rows its layout makes with its own tokenizer, is refused at the batch holding a
    token its backbone's table has no row for, naming the token."""
    from dew.decision import decisions

    objective = DecisionObjective(short(), tokenizer=HFTokenizer(str(FIXTURE), local_files_only=True))
    rows = Pipeline(Memory([{"text": "token7 token250", "label": label} for label in ("a", "b")] * 8))
    data = rows.pipe(decisions, question="label", options=("a", "b")).repeat().batch(8)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0))
    with pytest.raises(ValueError, match=rf"\(.token250.\), outside the {ROWS} rows"):
        trainer.fit(data, steps=1)


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
    with pytest.raises(ValueError, match=rf"holds the id 250\b.*outside the {ROWS} rows"):
        task([[4, 5, 250]], 2, key=0)
    server = Server.from_task(task, slots=1, capacity=32)
    with pytest.raises(ValueError, match=rf"holds the id 250 \('token250'\), outside the {ROWS} rows"):
        server.submit([4, 5, 250], max_new_tokens=2, key=0)
