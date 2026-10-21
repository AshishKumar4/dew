"""Text read from each store a run takes it from, through the one tokenized chain: a list in memory, a
Hugging Face dataset, a Grain dataset and a TFDS split make the same training batches and resume
alike; and a tokenizer trained on a corpus reads it back."""

from pathlib import Path

import grain.python as grain
import numpy as np
import pytest

from dew.data import Loading, Memory, from_grain, presets, text
from dew.data.compile import compile
from dew.data.dataset import DataPartition

LOADING = Loading(workers=0, threads=1, read_buffer=1)
CHAT = Path(__file__).parent / "fixtures" / "tokenizers" / "tiny-chat"
FIXTURE = Path(__file__).parent / "fixtures" / "tfds" / "dew_passages" / "1.0.0"
PASSAGES = [f"passage {index}: " + "the quick brown fox jumps over the lazy dog " * (1 + index % 4)
            for index in range(24)]


def stored(backend: str, tmp_path: Path):
    """The passages as `backend` holds them, each a record with its text under "text"."""
    rows = [{"text": passage} for passage in PASSAGES]
    if backend == "memory":
        return Memory(rows)
    if backend == "hugging face":
        import datasets

        from dew.data import from_hf

        return from_hf(datasets.Dataset.from_list(rows), name="passages")
    if backend == "grain":
        return from_grain(grain.MapDataset.source(rows), name="passages")
    from dew.data import TFDS

    # Prepared by tools/datasets/tfds_reference.py from the same passages.
    return TFDS("dew_passages", split="train", path=str(FIXTURE))


def batches(data, count: int, state=None):
    reader = data.train(DataPartition())
    try:
        if state is not None:
            reader.set_state(state)
        taken = [next(reader) for _ in range(count)]
        return [np.asarray(batch["text"]) for batch in taken], reader.get_state()
    finally:
        reader.close()


@pytest.mark.parametrize("backend", ["hugging face", "grain", "tfds"])
def test_each_store_of_text_trains_on_the_batches_a_list_in_memory_makes_and_resumes_alike(tmp_path, backend):
    """The passages held in memory, by Hugging Face, by Grain or by TFDS, shuffled and packed into
    rows of 32 bytes: the same batches, and a position saved after three reads on to the same
    batches as the uninterrupted read."""

    def built(store):
        chain = store.shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=32, batch=4)
        return compile(chain, read="indexed", loading=LOADING)

    expected, _ = batches(built(stored("memory", tmp_path)), 6)
    data = built(stored(backend, tmp_path))
    made, _ = batches(data, 6)
    for wanted, found in zip(expected, made, strict=True):
        np.testing.assert_array_equal(found, wanted)
    _, saved = batches(data, 3)
    resumed, _ = batches(data, 3, saved)
    for wanted, found in zip(made[3:], resumed, strict=True):
        np.testing.assert_array_equal(found, wanted)


def test_a_tokenizer_trained_on_a_corpus_reads_it_back(tmp_path):
    """A tokenizer of 300 tokens trained on the passages by the chat tokenizer's algorithm: every
    passage decodes from its tokens as it was, in fewer tokens than the base tokenizer takes, and
    the lm preset packs them with it."""
    from transformers import AutoTokenizer

    path = text.train_tokenizer(Memory([{"text": passage} for passage in PASSAGES]), str(CHAT),
                                str(tmp_path / "trained"), vocab_size=300)
    trained, base = AutoTokenizer.from_pretrained(path), AutoTokenizer.from_pretrained(str(CHAT))
    assert len(trained) <= 300
    for passage in PASSAGES:
        ids = trained(passage, add_special_tokens=False)["input_ids"]
        assert trained.decode(ids) == passage
    assert sum(len(trained(passage)["input_ids"]) for passage in PASSAGES) < sum(
        len(base(passage)["input_ids"]) for passage in PASSAGES)
    packed = Memory([{"text": passage} for passage in PASSAGES]).pipe(presets.lm, path, seq_len=16, batch=2)
    rows = list(packed)
    assert np.asarray(rows[0]["text"]).shape == (2, 17)


def test_a_tokenizer_smaller_than_what_its_algorithm_keeps_is_refused(tmp_path):
    """The chat tokenizer's algorithm keeps its byte alphabet and special tokens, more than a
    hundred: a vocabulary of a hundred is refused, and nothing is saved."""
    with pytest.raises(ValueError):
        text.train_tokenizer(Memory([{"text": passage} for passage in PASSAGES]), str(CHAT),
                             str(tmp_path / "trained"), vocab_size=100)
    assert not (tmp_path / "trained").exists()
