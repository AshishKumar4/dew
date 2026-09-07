"""Prompts for online RL: fixed-width rows and their reward context.

A `Prompts` source encodes verl-layout rows into a left-padded `prompt` of
`max_prompt_len` ids plus `prompt_length`, with `data_source`,
`ground_truth` and `extra_info` riding along as fixed-width UTF-8 bytes, so
every leaf survives the device transfer. These tests pin the encoding: the
tail kept, the pad on the left, the reward columns round-tripping byte for
byte, and the refusals for rows without a prompt, unknown fields and empty
files.
"""

import json

import numpy as np
import pytest

from dew.data import Loading, Prompts
from dew.data.prompts import (INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY,
                              TRUTH_KEY, PromptSource)

TOKENIZER = "tests/fixtures/tokenizers/tiny-chat"
WINDOW = 8

IDS = [1, 2, 3, 4, 5]
LONG = list(range(1, 20))


def records(*rows):
    return tuple(json.dumps(row) for row in rows)


def read(row):
    """A fixed-width int32 UTF-8 row back to its string, one byte per value."""
    return bytes(row[row != 0].astype(np.uint8)).decode("utf-8")


def source(*rows, width=WINDOW):
    return PromptSource.from_records(records(*rows), TOKENIZER, width, 0)


def test_ids_prompts_left_pad_with_their_length():
    rows = source({"prompt": IDS, "data_source": "rule", "ground_truth": "4"})

    batch = rows[0]

    np.testing.assert_array_equal(batch[PROMPT_KEY], [0, 0, 0, 1, 2, 3, 4, 5])
    assert int(batch[LENGTH_KEY]) == 5
    assert batch[PROMPT_KEY].dtype == np.int32


def test_a_long_prompt_keeps_its_tail():
    rows = source({"prompt": LONG})

    batch = rows[0]

    np.testing.assert_array_equal(batch[PROMPT_KEY], LONG[-WINDOW:])
    assert int(batch[LENGTH_KEY]) == WINDOW


def test_the_reward_columns_round_trip_as_bytes():
    rows = source({"prompt": IDS, "data_source": "rule", "ground_truth": "4",
                   "extra_info": '{"k": 1}'})

    batch = rows[0]

    assert read(batch[SOURCE_KEY]) == "rule"
    assert read(batch[TRUTH_KEY]) == "4"
    assert read(batch[INFO_KEY]) == '{"k": 1}'

    assert batch[SOURCE_KEY].shape == batch[TRUTH_KEY].shape == batch[INFO_KEY].shape


def test_missing_reward_columns_default_to_empty():
    rows = source({"prompt": IDS})

    batch = rows[0]

    assert all(int(batch[key].sum()) == 0 for key in (SOURCE_KEY, TRUTH_KEY, INFO_KEY))


def test_non_string_reward_columns_travel_as_json():
    rows = source({"prompt": IDS, "ground_truth": {"answer": 4}})

    batch = rows[0]

    assert read(batch[TRUTH_KEY]) == '{"answer": 4}'


def test_a_string_prompt_encodes_on_its_own():
    """Plain text is tokenized as it is: no chat template, no special tokens."""
    from dew.data.chat import load_tokenizer

    rows = source({"prompt": "hi"})

    batch = rows[0]

    ids = load_tokenizer(TOKENIZER).encode("hi", add_special_tokens=False)
    assert int(batch[LENGTH_KEY]) == len(ids)
    np.testing.assert_array_equal(batch[PROMPT_KEY][WINDOW - len(ids):], ids)
    np.testing.assert_array_equal(batch[PROMPT_KEY][:WINDOW - len(ids)], 0)


def test_messages_render_with_the_generation_prompt():
    from dew.data.chat import load_tokenizer

    conversation = [{"role": "user", "content": "hi"}]
    rows = source({"prompt": conversation}, width=64)
    load = load_tokenizer(TOKENIZER)
    plain = list(load.apply_chat_template(
        conversation, tokenize=True, return_dict=False, add_generation_prompt=False))
    prompted = list(load.apply_chat_template(
        conversation, tokenize=True, return_dict=False, add_generation_prompt=True))
    assert len(prompted) > len(plain), "the flag matters for this template"

    batch = rows[0]

    ids = [int(token) for token in batch[PROMPT_KEY][64 - int(batch[LENGTH_KEY]):]]
    assert ids == prompted


def test_a_row_without_a_prompt_is_refused():
    with pytest.raises(ValueError, match="without a prompt"):
        source({"data_source": "rule"})


def test_unknown_fields_are_refused():
    with pytest.raises(ValueError, match="unknown fields"):
        source({"prompt": IDS, "reward": 1.0})

def test_an_empty_prompt_is_refused():
    with pytest.raises(ValueError, match="no tokens"):
        source({"prompt": []})[0]
    with pytest.raises(ValueError, match="blank"):
        source({"prompt": ""})[0]


def test_a_malformed_message_is_refused():
    with pytest.raises(ValueError, match="role as a string"):
        source({"prompt": [{"content": "hi"}]})[0]


def test_a_non_object_row_is_refused():
    with pytest.raises(ValueError, match="not JSON"):
        PromptSource.from_records(("{oops",), TOKENIZER, WINDOW, 0)
    with pytest.raises(ValueError, match="an object"):
        source([1, 2])[0]


def test_a_window_below_one_is_refused():
    with pytest.raises(ValueError, match="max_prompt_len"):
        PromptSource.from_records(records({"prompt": IDS}), TOKENIZER, 0, 0)


@pytest.mark.parametrize("prompt", ["hi", IDS])
def test_tool_schemas_need_a_chat_prompt(prompt):
    """Plain text and pretokenized ids have no template to render schemas."""
    spec = Prompts(tokenizer=TOKENIZER,
                   records=records({"prompt": prompt, "tools": []}),
                   max_prompt_len=WINDOW, loading=Loading(workers=0))
    with pytest.raises(ValueError, match="tools require chat messages"):
        next(spec.load(batch=1).train())


def test_parquet_carries_the_verl_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "prompts.parquet"
    pq.write_table(pa.table({
        "prompt": [IDS, LONG],
        "data_source": ["rule", "other"],
        "ground_truth": ["4", "19"],
        "extra_info": ["", "x"],
    }), path)

    rows = PromptSource.from_parquet(str(path), TOKENIZER, WINDOW, 0)

    assert len(rows) == 2
    assert read(rows[1][SOURCE_KEY]) == "other"
    np.testing.assert_array_equal(rows[1][PROMPT_KEY], LONG[-WINDOW:])


def test_parquet_without_a_prompt_column_is_refused(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "noprompt.parquet"
    pq.write_table(pa.table({"data_source": ["rule"]}), path)
    with pytest.raises(ValueError, match="prompt column is required"):
        PromptSource.from_parquet(str(path), TOKENIZER, WINDOW, 0)


def test_the_dataset_batches_fixed_width_rows():
    data = Prompts(tokenizer=TOKENIZER, records=records(
        {"prompt": IDS}, {"prompt": LONG}, {"prompt": "hi"}),
        max_prompt_len=WINDOW, loading=Loading(workers=0)).load(batch=2)

    batch = next(data.train())

    assert data.records == 3
    assert batch[PROMPT_KEY].shape == (2, WINDOW)
    assert batch[LENGTH_KEY].shape == (2,)
    assert set(batch) == {PROMPT_KEY, LENGTH_KEY, SOURCE_KEY, TRUTH_KEY, INFO_KEY}


def test_the_dataset_reads_one_source():
    with pytest.raises(ValueError, match="one source"):
        Prompts(tokenizer=TOKENIZER).load(batch=2)
    with pytest.raises(ValueError, match="one source"):
        Prompts(tokenizer=TOKENIZER, path="x.parquet",
                records=records({"prompt": IDS})).load(batch=2)


def test_validation_is_one_pass_over_a_second_file(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    train, val = tmp_path / "train.parquet", tmp_path / "val.parquet"
    pq.write_table(pa.table({"prompt": [IDS, LONG]}), train)
    pq.write_table(pa.table({"prompt": [IDS]}), val)
    data = Prompts(tokenizer=TOKENIZER, path=str(train), val_path=str(val),
                   max_prompt_len=WINDOW, loading=Loading(workers=0)).load(batch=1)

    assert data.val is not None
    batches = list(data.val())

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0][PROMPT_KEY][0], [0, 0, 0, 1, 2, 3, 4, 5])


@pytest.mark.slow
def test_the_train_stream_encodes_inside_workers():
    """The same rows through real worker processes: the source pickles, the
    workers load their own tokenizer copy, the batches match."""
    data = Prompts(tokenizer=TOKENIZER, records=records({"prompt": IDS}),
                   max_prompt_len=WINDOW, loading=Loading(workers=1)).load(batch=1)

    batch = next(data.train())

    np.testing.assert_array_equal(batch[PROMPT_KEY][0], [0, 0, 0, 1, 2, 3, 4, 5])
