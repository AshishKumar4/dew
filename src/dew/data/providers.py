"""Datasets a provider already holds: `dew.data.load("tfds/...")`, `"hf/..."`.

One function routes a `"<provider>/<name>"` string to the source that reads
it and hands back the same `Dataset` every spec hands back, over the same
grain plumbing: `train_stream` for the order, the sharding and the global
position, `validation_pass` for an ordered pass, `bounded` for its length,
and grain's own iterator transformations where the rows only stream.
Nothing here is a second pipeline, and nothing here prepares, downloads or
converts anything.

The providers differ in what they can honestly promise, and the routing
keeps the differences visible instead of papering over them:

- `tfds/<builder>` reads prepared ArrayRecords, with `path=` naming what a
  preparation run wrote. Random access, a known record count, and the
  resumable global position every `train_stream` dataset has.
- `hf/<owner>/<name>` reads one Arrow-backed split the same way.
- `hf/<owner>/<name>` with `streaming=True` reads an `IterableDataset`: no
  length, `records=None` unless a caller supplies one, a bounded shuffle, a
  bounded read, and no position at all, so a run over it is not
  checkpointable and `Trainer.fit` says so.

`preprocess(record, rng)` is where a record becomes batch fields. There is
no default: a provider's rows are its own shape, and a loader that guessed
would decode images meant to stay bytes or drop a column a run needs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Optional, Protocol, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np

from .dataset import Dataset, Loading, local_batch, train_stream, validation_pass
from .tokens import bounded

PROVIDERS = ("tfds", "hf")

Row = Mapping[str, object]
Options = Mapping[str, object]
Preprocess = Callable[[Row, np.random.Generator], Row]

TFDS_OPTIONS = ("path", "config", "version", "decoders")
"""What the tfds provider takes beside the shared arguments."""


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


def _text(option: object, name: str) -> Optional[str]:
    """A provider option that has to be a name, or None when it is unset."""
    if option is None:
        return None
    if not isinstance(option, str):
        raise TypeError(f"{name} is a name, and this one is {type(option).__name__}")
    return option


def tfds_source(name: str, split: str, options: Options) -> Sequence[Row]:
    """One split of a prepared TFDS dataset, checked against its own metadata."""
    from .sources.tfds import prepared_source

    unknown = sorted(set(options) - set(TFDS_OPTIONS))
    if unknown:
        raise TypeError(
            f"the tfds provider does not take {unknown}; it takes "
            f"{', '.join(TFDS_OPTIONS)}, where path is the prepared data_dir or "
            f"version directory")
    path = _text(options.get("path"), "tfds path=")
    if not path:
        raise ValueError(
            f"tfds/{name} needs path= naming what a preparation run wrote: its "
            f"builder.data_dir, or the version directory under it. Training "
            f"never prepares its own data.")
    return prepared_source(path, split, builder=name,
                           config=_text(options.get("config"), "tfds config="),
                           version=_text(options.get("version"), "tfds version="),
                           decoders=options.get("decoders"))


def hf_source(name: str, split: str, options: Options) -> Sequence[Row]:
    """One Arrow-backed split of a hub dataset, through grain's random access."""
    from .sources.hf import HFDatasetSource

    return HFDatasetSource(name=name, split=split, options=_load_options(options))


def _load_options(options: Options) -> dict[str, object]:
    """The `load_dataset` arguments in `options`.

    `streaming` is dew's own switch between two source kinds, so it does not
    travel; everything else goes to `datasets.load_dataset` unchanged, which
    is what makes a misspelled option raise there instead of vanishing here.
    """
    return {key: value for key, value in options.items() if key != "streaming"}


def hf_stream(name: str, split: str, options: Options, *, batch: int, seed: int,
              loading: Loading, epochs: Optional[int],
              preprocess: Optional[Preprocess]) -> Callable[[], Iterator[Row]]:
    """A factory over one process's share of a streamed split.

    The rows are grain's from the first stage on: the per-record transform is
    `random_map`, the batch is `batch`, and the buffer ahead of the step is
    grain's thread prefetch, so what bounds this path is `Loading` and not a
    reader written here. The position is withheld, because there is none.
    """
    from grain.experimental import ThreadPrefetchIterDataset

    from .sources.hf_stream import HFRows, Unresumable

    read = _load_options(options)

    def stream() -> Unresumable:
        rows: pygrain.IterDataset = HFRows(
            name, split, options=read, seed=seed, rank=jax.process_index(),
            world_size=jax.process_count(), buffer=loading.read_buffer, epochs=epochs)
        if preprocess is not None:
            rows = rows.random_map(Preprocessing(preprocess), seed=seed)
        batches = rows.batch(batch, drop_remainder=True)
        return Unresumable(iter(ThreadPrefetchIterDataset(
            batches, prefetch_buffer_size=max(1, loading.worker_buffer))))

    return stream


def load(source: str, *, batch: int, split: str = "train", val_split: Optional[str] = None,
         val_batches: Optional[int] = None, records: Optional[int] = None,
         preprocess: Optional[Preprocess] = None, seed: int = 0,
         loading: Loading = Loading(), **options: object) -> Dataset:
    """The `Dataset` behind `source`, read where the provider already holds it.

    `source` is `"tfds/<builder>"` or `"hf/<owner>/<name>"`. `batch` is the
    global batch, `split` the provider's own split expression and `val_split`
    a second one read as an ordered validation pass, bounded by `val_batches`.
    `preprocess(record, rng)` turns one of the provider's records into the
    batch fields a run reads; `records` is the record count for a source that
    cannot report its own. `seed` and `loading` mean what they mean
    everywhere else.

    The remaining keyword arguments belong to the provider: `path`, `config`,
    `version` and `decoders` for tfds, `streaming` and every `load_dataset`
    argument for hf. A provider names any option it does not take rather
    than reading a dataset the caller did not ask for.
    """
    provider, name = provider_of(source)
    rows = local_batch(batch)
    if provider == "hf" and options.get("streaming"):
        if records is not None and records < 1:
            raise ValueError("records over a stream is a positive count or None")
        return Dataset(
            train=hf_stream(name, split, options, batch=rows, seed=seed, loading=loading,
                            epochs=None, preprocess=preprocess),
            val=None if val_split is None else bounded(
                hf_stream(name, val_split, options, batch=rows, seed=seed,
                          loading=loading, epochs=1, preprocess=preprocess), val_batches),
            records=records,
            batch=batch,
        )
    read = tfds_source if provider == "tfds" else hf_source
    transforms = [] if preprocess is None else [Preprocessing(preprocess)]
    train = read(name, split, options)
    return Dataset(
        train=train_stream(train, transforms, batch=rows, seed=seed, loading=loading),
        val=None if val_split is None else bounded(
            validation_pass(read(name, val_split, options), transforms, batch=rows,
                            seed=seed, loading=loading), val_batches),
        records=counted(train, records, source),
        batch=batch,
    )
