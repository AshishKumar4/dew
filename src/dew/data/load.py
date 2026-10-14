"""Load a named dataset through the data chain's text or record steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dew.data import presets
from dew.data.compile import compile
from dew.data.dataset import Loader, Loading
from dew.data.holdout import HoldOut
from dew.data.pipeline import Dataset
from dew.data.sources import TFDS, Hub
from dew.data.sources.base import Read

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TEXT_HOLDOUT = HoldOut(fraction=0.01)
RECORD_HOLDOUT = HoldOut(records=256)


def load(
    name: str | Dataset,
    *,
    batch: int,
    tokenizer: str | PreTrainedTokenizerBase | None = None,
    seq_len: int | None = None,
    read: Read | None = None,
    provision: bool = True,
    seed: int = 0,
    loading: Loading | None = None,
) -> Loader:
    """Read `hf/<repo>[@<revision>]`, `tfds/<builder>[:<version>]`, or a dataset.

    `tokenizer` or `seq_len` selects packed language-model rows, with the
    byte tokenizer and 256-token sequences where either is unstated. Text
    holds out 1% of document identities; records hold out the first 256.
    The sources choose their read unless `read` is stated, and `provision`
    permits preparation of its missing artifacts.
    """
    source: Dataset
    if isinstance(name, str):
        provider, _, dataset = name.partition("/")
        if provider not in ("hf", "tfds") or not dataset or dataset.startswith("/"):
            raise ValueError(f"load reads hf/<repo> or tfds/<builder>, not {name!r}; "
                             "use Hub(), TFDS() or Files() for other sources")
        repo, separator, revision = dataset.partition("@" if provider == "hf" else ":")
        if not repo or (separator and not revision):
            raise ValueError(f"load needs a dataset name in {name!r}; "
                             "use Hub(), TFDS() or Files() for other sources")
        source = Hub(repo, revision=revision if separator else None) if provider == "hf" else TFDS(
            repo, version=revision if separator else None)
    else:
        source = name
    if tokenizer is not None or seq_len is not None:
        tokenizer = "byte" if tokenizer is None else tokenizer
        seq_len = 256 if seq_len is None else seq_len
        train, held = source.holdout(TEXT_HOLDOUT.fraction)
        train = presets.lm(train, tokenizer, seq_len=seq_len, batch=batch)
        held = presets.lm(held, tokenizer, seq_len=seq_len, batch=batch)
    else:
        train, held = source.holdout(records=RECORD_HOLDOUT.records)
        train, held = train.batch(batch), held.batch(batch)
    return compile(train, validation=held, read=read, provision=provision, seed=seed, loading=loading)
