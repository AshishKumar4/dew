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

from dew.data import Files, Loading, Memory, presets
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition
from dew.data.prompts import LENGTH_KEY, PROMPT_KEY

TOKENIZER = "tests/fixtures/tokenizers/tiny-chat"
WINDOW = 8

IDS = [1, 2, 3, 4, 5]
LONG = list(range(1, 20))


def records(*rows):
    return tuple(json.dumps(row) for row in rows)


def rendered(dataset, *, width=WINDOW, tokenizer=TOKENIZER, **options):
    """`dataset`'s prompts rendered by the prompts preset, unbatched."""
    return dataset.pipe(presets.prompts, tokenizer, max_prompt_len=width, batch=None, **options)


def source(*rows, width=WINDOW):
    """`rows`, each through JSON as a file holds it, rendered."""
    chain = rendered(Memory([json.loads(row) for row in records(*rows)]), width=width)
    return list(iterate(chain, read="memory", loading=Loading(workers=0)))


def trained(dataset, batch, *, loading=None, **options):
    """`dataset`'s prompts rendered, then in a run's order in batches of `batch`."""
    chain = rendered(dataset, **options).shuffle(0).repeat().batch(batch)
    return compile(chain, read="memory", loading=loading or Loading(workers=0))


def test_an_overlong_prompt_keeps_its_tail_by_default_and_drop_counts_it_rejected():
    """A prompt longer than the window keeps its last `max_prompt_len` ids (P62), not dropped; asked
    to drop it (`long="drop"`), the read leaves it out and counts it rejected as over length."""
    from dew.data import prompts

    rows = Memory([{"prompt": LONG}, {"prompt": IDS}])
    kept, short = list(iterate(rendered(rows), read="memory", loading=Loading(workers=0)))
    assert int(kept[LENGTH_KEY]) == WINDOW
    np.testing.assert_array_equal(kept[PROMPT_KEY], LONG[-WINDOW:])
    assert int(short[LENGTH_KEY]) == len(IDS)
    # The chain's own prompt step keeps the tail as the preset does.
    step = prompts.encode(TOKENIZER, max_prompt_len=WINDOW)
    (chained,) = list(Memory([{"prompt": LONG}]).filter_map(step))
    np.testing.assert_array_equal(chained[PROMPT_KEY], LONG[-WINDOW:])
    data = compile(rendered(rows, long="drop").batch(1), read="memory", loading=Loading(workers=0))
    assert data.records == 1
    assert [count.rejected["over_length"] for count in data.counts.values()] == [1]


def test_a_string_prompt_encodes_on_its_own():
    """Plain text is tokenized as it is: no chat template, no special tokens."""
    from dew.data.text import load_tokenizer

    rows = source({"prompt": "hi"})

    batch = rows[0]

    ids = load_tokenizer(TOKENIZER).encode("hi", add_special_tokens=False)
    assert int(batch[LENGTH_KEY]) == len(ids)
    np.testing.assert_array_equal(batch[PROMPT_KEY][WINDOW - len(ids):], ids)
    np.testing.assert_array_equal(batch[PROMPT_KEY][:WINDOW - len(ids)], 0)


@pytest.mark.parametrize("prompt", ["dew", "caf\u00e9", "a longer prompt"])
def test_byte_string_prompts_load_as_the_native_tokenizers_ids(prompt):
    from dew.data.text import tokenizer_for

    data = trained(Memory([json.loads(row) for row in records({"prompt": prompt}) * 8]), 8, tokenizer="byte")
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    ids = tokenizer_for("byte").encode(prompt)[-WINDOW:]
    expected = [0] * (WINDOW - len(ids)) + ids
    np.testing.assert_array_equal(batch[PROMPT_KEY], np.tile(expected, (8, 1)))
    np.testing.assert_array_equal(batch[LENGTH_KEY], np.full(8, len(ids), np.int32))


def test_a_string_prompt_opens_as_training_text_does_and_never_closes(tmp_path):
    """A prompt is a sequence the model reads opens: a tokenizer that starts every encoding with a BOS
    (Llama 3's, Gemma's) opens a prompt with it, as `Tokenize` opens training text, and the EOS its
    encoding ends with is left off, since the completion follows the prompt."""
    from tokenizers import Tokenizer, models, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, "dew": 3},
                                         unk_token="<unk>"))
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                        bos_token="<bos>", eos_token="<eos>")
    tokenizer.save_pretrained(tmp_path)
    ids = tokenizer.encode("dew", add_special_tokens=False)
    assert tokenizer.encode("dew") != ids
    data = trained(Memory([json.loads(row) for row in records({"prompt": "dew"})]), 1,
                   tokenizer=str(tmp_path))
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    opened = [1, *ids]
    np.testing.assert_array_equal(batch[PROMPT_KEY][0, WINDOW - len(opened):], opened)
    np.testing.assert_array_equal(batch[PROMPT_KEY][0, :WINDOW - len(opened)], 0)
    assert int(batch[LENGTH_KEY][0]) == len(opened)


def test_messages_render_with_the_generation_prompt():
    from dew.data.text import load_tokenizer

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


def test_template_variables_reach_the_chat_template():
    """Qwen3's `enable_thinking=False` closes an empty think block after the generation prompt."""
    from dew.data.text import load_tokenizer

    conversation = [{"role": "user", "content": "hi"}]
    render = load_tokenizer(TOKENIZER).apply_chat_template
    thinking = list(render(conversation, tokenize=True, return_dict=False, add_generation_prompt=True))
    plain = list(render(conversation, tokenize=True, return_dict=False, add_generation_prompt=True,
                        enable_thinking=False))
    assert plain != thinking, "the variable matters for this template"
    data = trained(Memory([json.loads(row) for row in records({"prompt": conversation})]), 1, width=64,
                   thinking=False)
    batch = next(data.train(DataPartition()))
    assert [int(token) for token in batch[PROMPT_KEY][0, 64 - int(batch[LENGTH_KEY][0]):]] == plain


def test_validation_is_one_pass_over_a_second_file(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    train, val = tmp_path / "train.parquet", tmp_path / "val.parquet"
    pq.write_table(pa.table({"prompt": [IDS, LONG]}), train)
    pq.write_table(pa.table({"prompt": [IDS]}), val)
    data = compile(rendered(Files(str(train))).shuffle(0).repeat().batch(1),
                   validation=rendered(Files(str(val))).batch(1), read="memory", loading=Loading(workers=0))

    assert data.val is not None
    batches = list(data.val(DataPartition()))

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0][PROMPT_KEY][0], [0, 0, 0, 1, 2, 3, 4, 5])


@pytest.mark.slow
def test_the_train_stream_encodes_inside_workers():
    """The same rows through real worker processes: the source pickles, the
    workers load their own tokenizer copy, the batches match."""
    data = trained(Memory([json.loads(row) for row in records({"prompt": IDS})]), 1,
                   loading=Loading(workers=1))

    batch = next(data.train(DataPartition()))

    np.testing.assert_array_equal(batch[PROMPT_KEY][0], [0, 0, 0, 1, 2, 3, 4, 5])


def test_prompts_validate_on_a_hub_split_of_their_own(tmp_path, monkeypatch):
    """Prompts from a Hub dataset validate on its own validation split, which the source names,
    where the reward facts wrapped around the source once hid its splits and refused."""
    import datasets
    from hub_support import native_hub

    from dew.data.sources import Hub

    native_hub(monkeypatch, tmp_path, "acme/prompts",
               {"train": datasets.Dataset.from_dict({"prompt": ["a", "b", "c", "d"]}),
                "validation": datasets.Dataset.from_dict({"prompt": ["x", "y"]})})
    train, held = (rendered(part) for part in Hub("acme/prompts").holdout(split="validation"))
    rows = {name: list(iterate(pipeline, read="memory", provision=True, loading=Loading(workers=0)))
            for name, pipeline in (("train", train), ("held", held))}
    assert len(rows["train"]) == 4 and len(rows["held"]) == 2


@pytest.mark.parametrize("inserted", [True, False])
def test_a_prompt_that_starts_with_its_bos_keeps_one(tmp_path, inserted):
    """A prompt whose text already starts with its tokenizer's BOS opens with that one BOS, whether the
    tokenizer inserts a BOS (Llama's) or only declares one (GPT-2's)."""
    from tokenizers import Tokenizer, models, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, "dew": 3}, unk_token="<unk>"))
    if inserted:
        backend.post_processor = processors.TemplateProcessing(single="<bos> $A",
                                                               special_tokens=[("<bos>", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>",
                                        eos_token="<eos>")
    tokenizer.save_pretrained(tmp_path)
    data = trained(Memory([json.loads(row) for row in records({"prompt": "<bos>dew"})]), 1,
                   tokenizer=str(tmp_path))
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    length = int(batch[LENGTH_KEY][0])
    assert list(batch[PROMPT_KEY][0, WINDOW - length:]) == [1, 3]
