"""The data the objectives' recipes train on: the image fixture, byte-token windows and a chain's
first batch."""

from pathlib import Path

import pytest

from dew.data import Loading, Memory, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition

IMAGES = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"
SEQ_LEN = 15


def first(chain, objective=None):
    """The first training batch `chain` makes, compiled with `objective`'s own stages as a run's is."""
    compiled = compile(chain.read("memory").loading(Loading(workers=0)),
                       objective=None if objective is None else objective.data())
    return next(compiled.train(DataPartition()))


def corpus_windows(corpus: Path):
    """Windows of SEQ_LEN byte tokens over a short text, each document ended by the tokenizer's EOS and
    the stream cut into windows, in pairs (`presets.lm`), what the LM recipe trains on."""
    documents = [{"text": "the quick brown fox jumps over the lazy dog " * 8} for _ in range(4)]
    return Memory(documents).pipe(presets.lm, "byte", seq_len=SEQ_LEN, batch=2, eos=True, pack=None)


@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    return corpus_windows(tmp_path_factory.mktemp("corpus"))
