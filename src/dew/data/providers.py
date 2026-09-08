"""Datasets a provider already holds: `dew.data.load("tfds/...")`, `"hf/..."`.

One function routes a `"<provider>/<name>"` string to the source that reads
it and returns the same `Dataset` every spec returns, over the same grain
plumbing: `train_stream` for the order, the sharding and the position,
`validation_pass` for an ordered pass, `bounded` for its length, and grain's
own iterator transformations where the rows only stream. There is no second
pipeline here.

What each provider does:

- `tfds/<builder>` reads the ArrayRecords a preparation run wrote under
  `path=`. Nothing is prepared, downloaded or generated: TFDS's read-only
  builder is used, so the training process needs no TensorFlow.
- `hf/<name>` reads one Arrow-backed split through `datasets.load_dataset`,
  which downloads the dataset and writes its Arrow cache if the local cache
  has neither. That is the library's own behaviour on its own terms.
- `hf/<name>` with `streaming=True` reads an `IterableDataset` as it goes:
  no length, so `records=None` unless a caller supplies one, and a position
  only where an unshuffled stream can be restored exactly.

Every provider option is a named argument with the type the library gives
it, so a name neither provider knows is a `TypeError` from Python and an
option belonging to the other provider is named here. `preprocess(record,
rng)` is where a record becomes batch fields; there is no default, because a
provider's rows are its own shape and a loader that guessed would decode
images meant to stay bytes or drop a column a run needs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Optional, Protocol, TypeAlias, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np

from .dataset import (Batch, Corpus, Dataset, Loading, local_batch, mixed_records,
                      mixed_stream, mixture, train_stream, validation_pass)
from .sources.hf import HFOptions
from .tokens import bounded

if TYPE_CHECKING:
    from datasets import (Dataset as ArrowDataset, DownloadConfig, DownloadMode, Features,
                          IterableDataset, VerificationMode, Version)
    from tensorflow_datasets import DecoderTree

PROVIDERS = ("tfds", "hf")

Row = Mapping[str, object]
Preprocess = Callable[[Row, np.random.Generator], Row]


@runtime_checkable
class Counted(Protocol):
    """A source that knows how many records it holds."""

    def __len__(self) -> int: ...


class Preprocessing(pygrain.RandomMapTransform):
    """`preprocess` as the grain transformation that runs inside the workers."""

    def __init__(self, preprocess: Preprocess):
        self.preprocess = preprocess

    def random_map(self, element: object, rng: np.random.Generator) -> dict[str, object]:
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


def counted(source: object, given: Optional[int], name: str) -> int:
    """The records the run reports over `source`.

    `records` is for a source that cannot count itself. One that can is not
    overridden: the stream reads every record it holds whatever the number
    says, so a smaller one would report an epoch the run never trains.
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


def _unwanted(provider: str, given: Mapping[str, object]) -> None:
    """Refuse the options that belong to the other provider."""
    named = sorted(name for name, value in given.items() if value is not None)
    if named:
        raise TypeError(
            f"the {provider} provider does not take {named}; those belong to "
            f"{'hf' if provider == 'tfds' else 'tfds'}")


def load(source: str | Mapping[str, float], *, batch: int, split: str = "train",
         val_split: Optional[str] = None,
         val_batches: Optional[int] = None, records: Optional[int] = None,
         preprocess: Optional[Preprocess] = None, seed: int = 0,
         shuffle_buffer: int = 0, loading: Loading = Loading(),
         # which configuration of the dataset, for either provider
         config: Optional[str] = None,
         # what the tfds provider reads
         path: Optional[str] = None, version: Optional[str] = None,
         decoders: Optional["DecoderTree"] = None,
         # what the hf provider reads
         streaming: bool = False,
         dataset: Optional["ArrowDataset | IterableDataset"] = None,
         data_dir: Optional[str] = None,
         data_files: Optional[str | Sequence[str] | Mapping[str, str | Sequence[str]]] = None,
         cache_dir: Optional[str] = None, features: Optional["Features"] = None,
         download_config: Optional["DownloadConfig"] = None,
         download_mode: Optional["DownloadMode | str"] = None,
         verification_mode: Optional["VerificationMode | str"] = None,
         keep_in_memory: Optional[bool] = None, save_infos: bool = False,
         revision: Optional["str | Version"] = None,
         token: Optional[str | bool] = None, num_proc: Optional[int] = None,
         storage_options: Optional[Mapping[str, object]] = None) -> Dataset:
    """The `Dataset` behind `source`, read where the provider already holds it.

    `source` is `"tfds/<builder>"` or `"hf/<name>"`, or several of them with
    the share of a step each one fills: `load({"hf/wiki": 0.7, "hf/code":
    0.3}, ...)` reads a weighted mixture, whose semantics are `mixture`'s.
    The corpora are read in name order, so the mixture is the same whichever
    order the mapping was written in, and every source in it is the same
    provider read with the same options: a mixture of two providers, or of
    two datasets that need different `data_files`, is two `load` calls whose
    sources a caller mixes with `dew.data.dataset.mixed_stream`.

    `batch` is the global batch, `split` the provider's own split expression
    and `val_split` a second one read as an ordered validation pass, bounded
    by `val_batches`. A mixture's validation pass mixes the same corpora at
    the same weights, each split in its own order, and stops before any of
    them would come round again, so a pass scores each held-out record at
    most once and scores the same records every time.
    `preprocess(record, rng)` turns one of the provider's records into the
    batch fields a run reads. `records` is the record count for a source that
    cannot report its own; a mixture computes its own and takes none, since
    one pass over it is the records in which every corpus has been read at
    least once.

    `seed` and `shuffle_buffer` decide which records a step trains on:
    `seed` keys the order and the per-record rng, and `shuffle_buffer` is how
    many rows a streamed split shuffles through, zero being file order. A
    validation pass is never shuffled. `loading` is performance only, as
    everywhere else.

    `config` names which configuration of the dataset to read, for either
    provider. `path`, `version` and `decoders` are the tfds provider's, and
    everything from `streaming` down is the hf provider's: `dataset` reads a split the caller
    already has, and the rest are `datasets.load_dataset`'s own arguments
    with its own types. `config` is that function's `name`. Naming an option
    of the other provider is refused, and a name neither knows is a
    `TypeError` from this signature.
    """
    weighted = {source: 1.0} if isinstance(source, str) else dict(source)
    # Name order, not the mapping's: which corpus a record of the mixture
    # comes from depends on the order they are mixed in, and a run is not two
    # runs because its weights were written the other way round.
    named = sorted(weighted)
    dataset_of = {name: provider_of(name)[1] for name in named}
    providers = {provider_of(name)[0] for name in named}
    if len(providers) != 1:
        raise ValueError(
            f"a mixture reads one provider through one set of its options, and "
            f"{named} names {sorted(providers)}; load each provider on its own "
            f"and mix what they read")
    provider = providers.pop()
    rows = local_batch(batch)
    transforms = [] if preprocess is None else [Preprocessing(preprocess)]
    if provider == "tfds":
        _unwanted("tfds", {
            "streaming": streaming or None, "dataset": dataset,
            "data_dir": data_dir, "data_files": data_files, "cache_dir": cache_dir,
            "features": features, "download_config": download_config,
            "download_mode": download_mode, "verification_mode": verification_mode,
            "keep_in_memory": keep_in_memory, "save_infos": save_infos or None,
            "revision": revision, "token": token, "num_proc": num_proc,
            "storage_options": storage_options})
        if shuffle_buffer:
            raise TypeError(
                "shuffle_buffer is the streamed shuffle; a prepared tfds split is "
                "read at random and shuffled whole from seed=")
        read = _tfds(path=path, config=config, version=version, decoders=decoders)
    else:
        _unwanted("hf", {"path": path, "version": version, "decoders": decoders})
        options = HFOptions(
            config=config, data_dir=data_dir, data_files=data_files,
            cache_dir=cache_dir, features=features, download_config=download_config,
            download_mode=download_mode, verification_mode=verification_mode,
            keep_in_memory=keep_in_memory, save_infos=save_infos, revision=revision,
            token=token, num_proc=num_proc, storage_options=storage_options)
        if streaming:
            if len(named) > 1:
                raise TypeError(
                    "a mixture reads its corpora at random, so it can hold their "
                    "proportions and report one record count as its position; a "
                    "streamed split is read as it comes and has neither. Mix "
                    "splits read at random, or train on one stream")
            return _streamed(dataset_of[named[0]], split, val_split,
                             options=options, dataset=dataset,
                             batch=batch, rows=rows, seed=seed,
                             shuffle_buffer=shuffle_buffer, loading=loading,
                             records=records, val_batches=val_batches,
                             preprocess=preprocess)
        read = _arrow(options=options, dataset=dataset)
    if shuffle_buffer and provider == "hf":
        raise TypeError(
            "shuffle_buffer is the streamed shuffle; an Arrow split is read at "
            "random and shuffled whole from seed=")

    def corpora_over(which: str) -> list[Corpus]:
        """Every corpus of the mixture, over one of the provider's splits."""
        return [Corpus(name, read(dataset_of[name], which), weighted[name])
                for name in named]

    corpora = corpora_over(split)
    if len(corpora) == 1:
        train = train_stream(corpora[0].source, transforms, batch=rows, seed=seed,
                             loading=loading)
        pass_records = counted(corpora[0].source, records, named[0])
    else:
        if records is not None:
            raise ValueError(
                "a mixture's pass is the records in which every corpus has been "
                "read at least once, which its corpora's lengths and weights give, "
                "so it takes no records=")
        train = mixed_stream(corpora, transforms, batch=rows, seed=seed, loading=loading)
        pass_records = mixed_records(corpora)
    val = None
    if val_split is not None:
        held = corpora_over(val_split)
        ordered = held[0].source if len(held) == 1 else mixture(held, None)
        val = bounded(validation_pass(ordered, transforms, batch=rows, seed=seed,
                                      loading=loading), val_batches)
    return Dataset(train=train, val=val, records=pass_records, batch=batch)


Reader: TypeAlias = Callable[[str, str], pygrain.RandomAccessDataSource[object]]
"""A reader of one dataset's one split at a time: the provider's own options
are bound, the name and the split are not, because a mixture reads several
names through the same options."""


def _tfds(*, path: Optional[str], config: Optional[str],
          version: Optional[str], decoders: Optional["DecoderTree"]) -> Reader:
    """A reader of prepared splits."""
    from .sources.tfds import prepared_source

    if not path:
        raise ValueError(
            "a tfds source needs path= naming what a preparation run wrote: its "
            "builder.data_dir, or the version directory under it. Training "
            "never prepares its own data.")
    return lambda name, split: prepared_source(path, split, builder=name, config=config,
                                               version=version, decoders=decoders)


def _arrow(*, options: HFOptions,
           dataset: Optional["ArrowDataset | IterableDataset"]) -> Reader:
    """A reader of Arrow-backed splits, through grain's random access."""
    from .sources.hf import HFDatasetSource

    if dataset is None:
        return lambda name, split: HFDatasetSource(name=name, split=split, options=options)
    return lambda name, split: HFDatasetSource(split=split, dataset=dataset)


def _streamed(name: str, split: str, val_split: Optional[str], *, options: HFOptions,
              dataset: Optional["ArrowDataset | IterableDataset"], batch: int, rows: int,
              seed: int, shuffle_buffer: int, loading: Loading, records: Optional[int],
              val_batches: Optional[int], preprocess: Optional[Preprocess]) -> Dataset:
    """The streamed split's `Dataset`: endless training, one ordered pass for val."""
    if records is not None and records < 1:
        raise ValueError("records over a stream is a positive count or None")
    return Dataset(
        train=_stream(name, split, options=options, dataset=dataset, batch=rows,
                      seed=seed, shuffle_buffer=shuffle_buffer, loading=loading,
                      epochs=None, preprocess=preprocess),
        # A validation pass is the split in its own order, so it is never
        # shuffled: a score over other rows every time is not a score.
        val=None if val_split is None else bounded(
            _stream(name, val_split, options=options, dataset=dataset, batch=rows,
                    seed=seed, shuffle_buffer=0, loading=loading, epochs=1,
                    preprocess=preprocess), val_batches),
        records=records,
        batch=batch,
    )


def _stream(name: str, split: str, *, options: HFOptions,
            dataset: Optional["ArrowDataset | IterableDataset"], batch: int, seed: int,
            shuffle_buffer: int, loading: Loading, epochs: Optional[int],
            preprocess: Optional[Preprocess]) -> Callable[[], Iterator[Batch]]:
    """A factory over one process's share of a streamed split.

    The rows are grain's from the first stage on: the per-record transform is
    `random_map`, the batch is `batch`, and the buffer ahead of the step is
    grain's thread prefetch, bounded by `Loading.worker_buffer` batches. The
    position is handed on only where the rows can be put back exactly.
    """
    from grain.experimental import ThreadPrefetchIterDataset

    from .sources.hf_stream import HFRows, Unresumable

    # This factory's own copy of a dataset a caller handed over, taken here,
    # before anything reads it: the library sets state on the object it is
    # iterating and restored into, so a stream that shared the caller's
    # object could leave it somewhere other than its beginning. Copied once
    # rather than per pass, because copying an object another thread is
    # iterating reads its attributes as they change.
    own = None if dataset is None else copy.deepcopy(
        _iterable(dataset, "the dataset load() was given"))

    def open_split() -> "IterableDataset":
        if own is None:
            return _iterable(options.load(name, split, streaming=True), f"{name}/{split}")
        return own

    what = f"{name!r} split {split!r}" if dataset is None else "the given dataset"

    def stream() -> Iterator[Batch]:
        source = HFRows(open_split, what=what, seed=seed, rank=jax.process_index(),
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


def _iterable(rows: object, what: str) -> "IterableDataset":
    """`rows` as a streamed split, or the refusal that it is not one."""
    import datasets

    if isinstance(rows, datasets.IterableDataset):
        return rows
    raise TypeError(
        f"{what} is {type(rows).__name__}; a streamed split is an IterableDataset, "
        f"so name one split rather than a whole dataset, or drop streaming=True to "
        f"read it at random")
