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

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np

from .dataset import Batch, Dataset, Loading, local_batch, train_stream, validation_pass
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


def load(source: str, *, batch: int, split: str = "train", val_split: Optional[str] = None,
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

    `source` is `"tfds/<builder>"` or `"hf/<name>"`. `batch` is the global
    batch, `split` the provider's own split expression and `val_split` a
    second one read as an ordered validation pass, bounded by `val_batches`.
    `preprocess(record, rng)` turns one of the provider's records into the
    batch fields a run reads. `records` is the record count for a source that
    cannot report its own.

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
    provider, name = provider_of(source)
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
        read = _tfds(name, path=path, config=config, version=version, decoders=decoders)
    else:
        _unwanted("hf", {"path": path, "version": version, "decoders": decoders})
        options = HFOptions(
            config=config, data_dir=data_dir, data_files=data_files,
            cache_dir=cache_dir, features=features, download_config=download_config,
            download_mode=download_mode, verification_mode=verification_mode,
            keep_in_memory=keep_in_memory, save_infos=save_infos, revision=revision,
            token=token, num_proc=num_proc, storage_options=storage_options)
        if streaming:
            return _streamed(name, split, val_split, options=options, dataset=dataset,
                             batch=batch, rows=rows, seed=seed,
                             shuffle_buffer=shuffle_buffer, loading=loading,
                             records=records, val_batches=val_batches,
                             preprocess=preprocess)
        read = _arrow(name, options=options, dataset=dataset)
    if shuffle_buffer and provider == "hf":
        raise TypeError(
            "shuffle_buffer is the streamed shuffle; an Arrow split is read at "
            "random and shuffled whole from seed=")
    train = read(split)
    return Dataset(
        train=train_stream(train, transforms, batch=rows, seed=seed, loading=loading),
        val=None if val_split is None else bounded(
            validation_pass(read(val_split), transforms, batch=rows, seed=seed,
                            loading=loading), val_batches),
        records=counted(train, records, source),
        batch=batch,
    )


def _tfds(name: str, *, path: Optional[str], config: Optional[str],
          version: Optional[str], decoders: Optional["DecoderTree"]
          ) -> Callable[[str], pygrain.RandomAccessDataSource[object]]:
    """A reader of one prepared split at a time."""
    from .sources.tfds import prepared_source

    if not path:
        raise ValueError(
            f"tfds/{name} needs path= naming what a preparation run wrote: its "
            f"builder.data_dir, or the version directory under it. Training "
            f"never prepares its own data.")
    return lambda split: prepared_source(path, split, builder=name, config=config,
                                         version=version, decoders=decoders)


def _arrow(name: str, *, options: HFOptions,
           dataset: Optional["ArrowDataset | IterableDataset"]
           ) -> Callable[[str], pygrain.RandomAccessDataSource[object]]:
    """A reader of one Arrow-backed split at a time, through grain's random access."""
    from .sources.hf import HFDatasetSource

    if dataset is None:
        return lambda split: HFDatasetSource(name=name, split=split, options=options)
    return lambda split: HFDatasetSource(split=split, dataset=dataset)


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

    def open_split() -> "IterableDataset":
        if dataset is None:
            return _iterable(options.load(name, split, streaming=True), f"{name}/{split}")
        return _iterable(dataset, "the dataset load() was given")

    what = f"{name!r} split {split!r}" if dataset is None else "the given dataset"

    def stream() -> Iterator[Batch]:
        source = HFRows(open_split, what=what, seed=seed, rank=jax.process_index(),
                        world_size=jax.process_count(),
                        shuffle_buffer=shuffle_buffer, epochs=epochs)
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
