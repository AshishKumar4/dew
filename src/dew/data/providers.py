"""Datasets a provider already holds: `datasets["tfds"]`, `datasets["hf"]`.

Each provider is a registered spec like every other dataset, so a run config
can name one, `to_dict` can write it and `from_dict` can load it back. Both
return the same `Dataset` every spec returns, over the same grain plumbing:
`train_stream` for the order, the sharding and the position,
`validation_pass` for an ordered pass, `bounded` for its length, and grain's
own iterator transformations where the rows only stream. There is no second
pipeline here.

What each provider does:

- `tfds/<builder>` reads the ArrayRecords a preparation run wrote under
  `TFDSOptions.path`. Nothing is prepared, downloaded or generated: TFDS's
  read-only builder is used, so the training process needs no TensorFlow.
- `hf/<name>` reads one Arrow-backed split through `datasets.load_dataset`,
  which downloads the dataset and writes its Arrow cache if the local cache
  has neither. That is the library's own behaviour on its own terms.
- `hf/<name>` with `streaming=True` reads an `IterableDataset` as it goes:
  no length, so `records=None` unless a caller supplies one, and a position
  only where an unshuffled stream can be restored exactly.

Each provider's own options are one frozen value of its own type,
`HFOptions` or `TFDSOptions`, which is the field the spec holds and the
argument `load` takes. An option of the other provider is then a type its
spec has no field for rather than a name checked at run time.

`preprocess(record, rng)` is where a record becomes batch fields. It has no
default, because a provider's rows are its own shape and a loader that
guessed would decode images meant to stay bytes or drop a column a run needs.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np

from dew.registry import datasets

from .dataset import (
    Batch,
    Corpus,
    Dataset,
    DatasetSpec,
    Loading,
    Records,
    Tokenize,
    local_batch,
    mixed_records,
    mixed_stream,
    mixture,
    train_stream,
    validation_pass,
)
from .sources.hf import HFOptions, HubOptions
from .sources.tfds import PreparedOptions, TFDSOptions
from .tokens import bounded

if TYPE_CHECKING:
    from datasets import Dataset as ArrowDataset, IterableDataset

PROVIDERS = ("tfds", "hf")

Row = Mapping[str, object]
Preprocess = Callable[[Row, np.random.Generator], Row]


@runtime_checkable
class Counted(Protocol):
    """Reports how many records a source holds."""

    def __len__(self) -> int: ...



def name_ordered(named: str | Mapping[str, float] | None) -> dict[str, float]:
    """One name at weight one, or each name of a mapping at its weight, in
    name order; empty when nothing is named.

    Which corpus record k comes from depends on the order the corpora are
    mixed in, so sorting by name keeps one written mapping to one run.
    """
    if not named:
        return {}
    weighted = {named: 1.0} if isinstance(named, str) else dict(named)
    return {} if not all(weighted) else {name: weighted[name] for name in sorted(weighted)}


def corpora_dataset(train: Sequence[Corpus], held: Sequence[Corpus] | None,
                    operations: Sequence[pygrain.Transformation], *, batch: int, seed: int,
                    loading: Loading, val_batches: int | None,
                    records: int | None = None) -> Dataset:
    """The batches of one corpus, or of several mixed at their weights.

    One corpus streams reshuffled every epoch, and a pass is its records
    (`records` for a source that cannot count itself). Several mix
    (`mixed_stream`), and a pass is the records in which every corpus has
    been read at least once (`mixed_records`), so a mixture takes no
    `records`. `held` holds the same corpora's validation splits: one
    ordered pass, a mixture's mixed at the same weights with each split in
    its own order, bounded by `val_batches`.
    """
    rows = local_batch(batch)
    if len(train) == 1:
        stream = train_stream(train[0].source, operations, batch=rows, seed=seed,
                              loading=loading)
        pass_records = counted(train[0].source, records, train[0].name)
    else:
        if records is not None:
            raise ValueError(
                "a mixture's pass is the records in which every corpus has been "
                "read at least once, which its corpora's lengths and weights give, "
                "so it takes no records=")
        stream = mixed_stream(train, operations, batch=rows, seed=seed, loading=loading)
        pass_records = mixed_records(train)
    validation = None
    if held is not None:
        ordered = held[0].source if len(held) == 1 else mixture(held, None)
        validation = bounded(validation_pass(ordered, operations, batch=rows, seed=seed,
                                             loading=loading), val_batches)
    return Dataset(train=stream, val=validation, records=pass_records, batch=batch)

class Preprocessing(pygrain.RandomMapTransform):
    """`preprocess` as the grain transformation that runs inside the workers."""

    def __init__(self, preprocess: Preprocess):
        self.preprocess = preprocess

    def random_map(self, element: object, rng: np.random.Generator) -> Batch:
        if not isinstance(element, Mapping):
            raise TypeError(
                f"a provider record is a mapping of fields; this one is "
                f"{type(element).__name__}")
        return dict(self.preprocess(element, rng))


def provider_of(source: str) -> tuple[str, str]:
    """`("tfds", "dew_images")`, `("hf", "owner/name")`; the name keeps its slashes."""
    provider, _, name = source.partition("/")
    if provider not in PROVIDERS:
        raise ValueError(
            f"{source!r} names no provider dew reads; write "
            f"{' or '.join(f'{known}/<name>' for known in PROVIDERS)}")
    if not name:
        raise ValueError(
            f"{source!r} names the {provider} provider and no dataset; write "
            f"{provider}/<name>")
    return provider, name


def counted(source: object, given: int | None, name: str) -> int:
    """The records the run reports over `source`.

    `given` is for a source that cannot count itself. One that can is not
    overridden, because the stream reads every record it holds whatever the
    number says, and a smaller one would report an epoch the run never
    trains.
    """
    if not isinstance(source, Counted):
        if given is None:
            raise ValueError(
                f"{name} reports no record count, so load() needs records= set "
                f"to the records one pass over it holds")
        return given
    held = len(source)
    if given is not None and given != held:
        raise ValueError(
            f"records={given} disagrees with the {held} records of {name}, and "
            f"the stream reads all of them; drop records=, or name a split of "
            f"the size you want")
    return held


type Named = str | Mapping[str, float]
"""Which dataset a provider spec reads: one name, or several with the share
of a step each one fills."""


@dataclasses.dataclass(frozen=True)
class ProviderDataset(DatasetSpec):
    """Holds what both providers need: which dataset, which splits, and how a
    row becomes batch fields.

    `name` is the provider's own name for the dataset, or a mapping of
    several to the share of a step each fills, whose semantics are
    `mixture`'s. The corpora are read in name order, so the mixture is the
    same whichever order the mapping was written in. Every corpus is read
    through the same options, so a mixture of two providers, or of two
    datasets needing different options, is two specs whose sources a caller
    mixes with `dew.data.dataset.mixed_stream`.

    `split` is the provider's own split expression and `val_split` a second
    one read as an ordered validation pass, bounded by `val_batches`. A
    mixture's validation pass mixes the same corpora at the same weights,
    each split in its own order, and stops before any of them would come
    round again. A pass therefore scores each held-out record at most once,
    and scores the same records every time.

    `records` is the record count for a source that cannot report its own. A
    mixture computes its own and takes none, since one pass over it is the
    records in which every corpus has been read at least once.
    """

    name: Named = ""
    split: str = "train"
    val_split: str | None = None
    val_batches: int | None = None
    records: int | None = None
    preprocess: Preprocess | None = None

    @property
    def weighted(self) -> dict[str, float]:
        """Each dataset this reads and the share of a step it fills, in name
        order (`name_ordered`)."""
        weighted = name_ordered(self.name)
        if not weighted:
            raise ValueError(f"{type(self).__name__} needs name= set to a dataset")
        return weighted

    @property
    def transforms(self) -> list[pygrain.Transformation]:
        """`preprocess` as the transformation the workers run, or none."""
        return [] if self.preprocess is None else [Preprocessing(self.preprocess)]

    def read(self, name: str, split: str) -> Records:
        """One split of one of this spec's datasets, read by index."""
        raise NotImplementedError

    def random_access(self, *, batch: int) -> Dataset:
        """The batches of a provider whose splits are read by index.

        The mixture, the ordered validation pass and the record count are the
        same for both providers; only `read` differs.
        """
        held = None if self.val_split is None else [
            Corpus(name, self.read(name, self.val_split), weight)
            for name, weight in self.weighted.items()]
        return corpora_dataset(
            [Corpus(name, self.read(name, self.split), weight)
             for name, weight in self.weighted.items()],
            held, self.transforms, batch=batch, seed=self.seed, loading=self.loading,
            val_batches=self.val_batches, records=self.records)


@datasets("tfds")
@dataclasses.dataclass(frozen=True)
class PreparedTFDS(ProviderDataset):
    """Reads splits of a prepared TFDS builder where preparation left them."""

    options: PreparedOptions = TFDSOptions()

    def read(self, name: str, split: str) -> Records:
        return self.options.source(name, split)

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        self.uncaptioned(tokenize)
        return self.random_access(batch=batch)


@datasets("hf")
@dataclasses.dataclass(frozen=True)
class HubDataset(ProviderDataset):
    """Reads one Hugging Face split, Arrow-backed or streamed.

    `streaming` reads the split as it comes instead of by index.
    `shuffle_buffer` is then how many rows the shuffle holds, zero being file
    order, and a pass has no length, so `records` is whatever a caller knows.
    A validation pass is never shuffled either way.
    """

    streaming: bool = False
    shuffle_buffer: int = 0
    options: HubOptions = HFOptions()

    def read(self, name: str, split: str) -> Records:
        from .sources.hf import HFDatasetSource

        return HFDatasetSource(name=name, split=split, options=self.options)

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        self.uncaptioned(tokenize)
        return self.rows(batch=batch, dataset=None)

    def rows(self, *, batch: int,
             dataset: ArrowDataset | IterableDataset | None) -> Dataset:
        """This spec's batches, over `dataset` when a caller already holds
        the split.

        A table in memory has no JSON form, so it is an argument here rather
        than a spec field. `dew.data.load(dataset=)` is the one caller that
        passes one.
        """
        if self.streaming:
            return self._streamed(batch=batch, dataset=dataset)
        if self.shuffle_buffer:
            raise TypeError(
                "shuffle_buffer is the streamed shuffle; an Arrow split is read at "
                "random and shuffled whole from seed=")
        if dataset is None:
            return self.random_access(batch=batch)
        return dataclasses.replace(self, name=_GIVEN)._given(batch=batch, dataset=dataset)

    def _given(self, *, batch: int, dataset: ArrowDataset | IterableDataset) -> Dataset:
        """The batches of an Arrow table the caller built, read by index."""
        from .sources.hf import HFDatasetSource

        source = HFDatasetSource(split=self.split, dataset=dataset)
        rows = local_batch(batch)
        return Dataset(
            train=train_stream(source, self.transforms, batch=rows, seed=self.seed,
                               loading=self.loading),
            val=None if self.val_split is None else bounded(
                validation_pass(HFDatasetSource(split=self.val_split, dataset=dataset),
                                self.transforms, batch=rows, seed=self.seed,
                                loading=self.loading), self.val_batches),
            records=counted(source, self.records, _GIVEN),
            batch=batch)

    def _streamed(self, *, batch: int,
                  dataset: ArrowDataset | IterableDataset | None) -> Dataset:
        """Endless training over the streamed split, one ordered pass for val."""
        named = self.weighted
        if len(named) > 1:
            raise TypeError(
                "a mixture reads its corpora at random, so it can hold their "
                "proportions and report one record count as its position; a "
                "streamed split is read as it comes and has neither. Mix "
                "splits read at random, or train on one stream")
        if self.records is not None and self.records < 1:
            raise ValueError("records over a stream is a positive count or None")
        name, rows = next(iter(named)), local_batch(batch)
        return Dataset(
            train=_stream(name, self.split, options=self.options, dataset=dataset,
                          batch=rows, seed=self.seed, shuffle_buffer=self.shuffle_buffer,
                          loading=self.loading, epochs=None, preprocess=self.preprocess),
            # A validation pass is the split in its own order and is never
            # shuffled; a score over other rows every time is not a score.
            val=None if self.val_split is None else bounded(
                _stream(name, self.val_split, options=self.options, dataset=dataset,
                        batch=rows, seed=self.seed, shuffle_buffer=0,
                        loading=self.loading, epochs=1, preprocess=self.preprocess),
                self.val_batches),
            records=self.records,
            batch=batch,
        )


_GIVEN = "the dataset load() was given"
"""What a table a caller built is called, in a refusal and in a record count."""


def load(source: Named, *, batch: int,
         options: HFOptions | TFDSOptions | None = None, split: str = "train",
         val_split: str | None = None, val_batches: int | None = None,
         records: int | None = None, preprocess: Preprocess | None = None,
         seed: int = 0, shuffle_buffer: int = 0, streaming: bool = False,
         loading: Loading = Loading(),
         dataset: ArrowDataset | IterableDataset | None = None) -> Dataset:
    """The `Dataset` behind `source`, read where the provider already holds it.

    This builds the registered spec, so `load("hf/wiki", batch=32,
    options=HFOptions(config="20231101.en"))` and
    `datasets["hf"](name="wiki", options=HFOptions(config="20231101.en"))
    .load(batch=32)` are the same dataset. A run that wants the second in its
    config writes it there.

    `source` is `"tfds/<builder>"` or `"hf/<name>"`, or several of them with
    the share of a step each one fills. `options` is the provider's own
    value, `TFDSOptions` for tfds and `HFOptions` for hf, so an option of the
    other provider is a type error rather than a name. Everything else is
    what both providers take. `dataset=` is a split the caller already holds,
    an argument rather than a spec field because a table in memory has no
    record in a config.
    """
    provider, names = _sources(source)
    if provider == "tfds":
        if options is not None and not isinstance(options, TFDSOptions):
            raise TypeError(
                f"the tfds provider reads TFDSOptions; {type(options).__name__} is "
                f"the other provider's")
        if streaming or shuffle_buffer or dataset is not None:
            raise TypeError(
                "streaming, shuffle_buffer and dataset= are the hf provider's; a "
                "prepared tfds split is read at random and shuffled whole from seed=")
        return PreparedTFDS(
            name=names, split=split, val_split=val_split, val_batches=val_batches,
            records=records, preprocess=preprocess, options=options or TFDSOptions(),
            seed=seed, loading=loading).load(batch=batch)
    if options is not None and not isinstance(options, HFOptions):
        raise TypeError(
            f"the hf provider reads HFOptions; {type(options).__name__} is the "
            f"other provider's")
    return HubDataset(
        name=names, split=split, val_split=val_split, val_batches=val_batches,
        records=records, preprocess=preprocess, streaming=streaming,
        shuffle_buffer=shuffle_buffer, options=options or HFOptions(),
        seed=seed, loading=loading).rows(batch=batch, dataset=dataset)


def _sources(source: Named) -> tuple[str, Named]:
    """`source` split into the one provider it names and the dataset names
    without it.

    A mixture reads one provider through one set of its options, so two
    providers in one mapping is refused here.
    """
    weighted = {source: 1.0} if isinstance(source, str) else dict(source)
    providers = {provider_of(name)[0] for name in weighted}
    if len(providers) != 1:
        raise ValueError(
            f"a mixture reads one provider through one set of its options, and "
            f"{sorted(weighted)} names {sorted(providers)}; load each provider on "
            f"its own and mix what they read")
    named = {provider_of(name)[1]: weight for name, weight in weighted.items()}
    return providers.pop(), next(iter(named)) if len(named) == 1 else named


def _stream(name: str, split: str, *, options: HFOptions,
            dataset: ArrowDataset | IterableDataset | None, batch: int, seed: int,
            shuffle_buffer: int, loading: Loading, epochs: int | None,
            preprocess: Preprocess | None) -> Callable[[], Iterator[Batch]]:
    """Opens one process's share of a streamed split, once per call.

    The rows are grain's from the first stage on. The per-record transform is
    `random_map`, the batch is `batch`, and the buffer ahead of the step is
    grain's thread prefetch, bounded by `Loading.worker_buffer` batches. The
    position is handed on only where the rows can be put back exactly.
    """
    from grain.experimental import ThreadPrefetchIterDataset

    from .sources.hf_stream import HFRows, Unresumable

    # This factory's own copy of a dataset a caller handed over, taken before
    # anything reads it: the library sets state on the object it iterates, so
    # a stream sharing the caller's object could leave it somewhere other
    # than its beginning. Copied once rather than per pass, because copying
    # an object another thread is iterating reads its attributes as they
    # change.
    own = None if dataset is None else copy.deepcopy(_iterable(dataset, _GIVEN))

    def open_split() -> IterableDataset:
        if own is None:
            return _iterable(options.load(name, split, streaming=True), f"{name}/{split}")
        return own

    where = f"{name!r} split {split!r}" if dataset is None else "the given dataset"

    def stream() -> Iterator[Batch]:
        source = HFRows(open_split, what=where, seed=seed, rank=jax.process_index(),
                        world_size=jax.process_count(),
                        shuffle_buffer=shuffle_buffer, epochs=epochs,
                        given=dataset is not None)
        piped: pygrain.IterDataset = source
        if preprocess is not None:
            piped = piped.random_map(Preprocessing(preprocess), seed=seed)
        batches = ThreadPrefetchIterDataset(
            piped.batch(batch, drop_remainder=True),
            prefetch_buffer_size=max(1, loading.worker_buffer))
        reads = iter(batches)
        return reads if source.resumable else Unresumable(reads)

    return stream


def _iterable(rows: object, where: str) -> IterableDataset:
    """`rows` as a streamed split, or the refusal that it is not one."""
    import datasets

    if isinstance(rows, datasets.IterableDataset):
        return rows
    raise TypeError(
        f"{where} is {type(rows).__name__}; a streamed split is an IterableDataset, "
        f"so name one split rather than a whole dataset, or drop streaming=True to "
        f"read it at random")
