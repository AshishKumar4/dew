"""What the data suites share: Grain's flags parsed for its worker processes, a caption reader that
hands the words back, the caption templates and host augmentation the image suites draw from, a
random step that draws a wide integer, objectives' expansions of a record into rows of it, a step that
refuses every n-th record and a draw configured for the run."""

import dataclasses
import itertools

import grain.python as grain
import numpy as np
from absl import flags

from dew.data import Memory, images


@dataclasses.dataclass(frozen=True)
class Logged(Memory):
    """Records in memory that log each read a run takes of them."""

    reads: list = dataclasses.field(default_factory=list, compare=False)

    def records(self, read):
        self.reads.append(read)
        return super().records(read)


def parse_grain_flags() -> None:
    """Mark absl's flags parsed, as Grain's worker processes read them: a suite that never ran
    absl.app would trip UnparsedFlagAccessError at any worker_count > 0. A suite whose loading
    workers import it calls this as it is imported, in its own process and in each worker."""
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()

CAPTION_TEMPLATES = ("a photo of a {}", "a photo of a {} flower", "This is a photo of a {}")
"""More than one, so a test can see which the record's rng drew."""

FLIP_JITTER = (images.Flip(), images.ColorJitter())
"""A flip and a colour jitter on the host, as `presets.images(augment=)` takes them."""


def keep_captions(captions):
    """A caption reader that hands the words back, so a test reads what the dataset wrote before a
    run's encoder tokenizes it."""
    return {"caption": np.asarray(captions)}


class Questioned:
    """An objective's expansion of each record into `id % 3 + 1` rows, one a question, as a decision
    layout of a question a row makes of an example (`dew.decision.rows.QuestionRows`), its identity
    stable, so a resumed run knows it for the step it saved."""

    identity = "questioned@1"

    def flat_map(self, element):
        return tuple({"id": element["id"], "question": np.int32(question)}
                     for question in range(int(element["id"]) % 3 + 1))


class Fanned:
    """An objective's expansion of each record `id` into `id % 4` rows, none for some, each row's two
    tokens naming it (`id * 10 + question`)."""

    identity = "fanned@1"

    def flat_map(self, element):
        record = int(element["id"])
        return tuple({"text": np.full(2, record * 10 + question, np.int32)} for question in range(record % 4))


class Twice:
    """An objective's expansion of each record, or packed window, into two rows of it (`FlatMap.examples`)."""

    identity = "twice@1"

    def flat_map(self, element):
        return (element, element)


@dataclasses.dataclass(frozen=True)
class Keeps:
    """Rejects a record whose id is a multiple of `every`, permanently: a record steps cannot make."""

    every: int
    identity = "keeps@1"
    rejects = True

    def __call__(self, row):
        from dew.data.pipeline import Rejected

        return Rejected("unkept") if int(row["id"]) % self.every == 0 else row


class _Passing(grain.RandomMapTransform):
    """A random step that keeps each record as it is."""

    def random_map(self, element, rng):
        return element


def jittered(metadata):
    """A random step configured for the run (`Configured(jittered, ..., random=True)`), which binds to a
    random map (`RandomMap`) that keeps each record."""
    return _Passing()


@dataclasses.dataclass(frozen=True)
class WideDraw(grain.RandomMapTransform):
    """A record's draw of an integer below 2**32, which its generator makes."""

    def random_map(self, element, rng):
        return {**element, "draw": np.int64(rng.integers(0, 2**32))}


def listed(reader, count: int) -> list[dict[str, list]]:
    """The next `count` batches of `reader` as plain lists, field by field, so a comparison covers
    every field of every record rather than one of them."""
    return [{name: np.asarray(value).tolist() for name, value in batch.items()}
            for batch in itertools.islice(reader, count)]


def record_batches(records, batch: int, validation=None, *, loading=None):
    """`records` (rows or columns) as a run's batches of `batch`, read in memory, with `validation`'s
    records as its held-out split, its last batch filled out and marked (`dew.data.compile.compile`)."""
    from dew.data import Memory
    from dew.data.compile import compile

    held = None if validation is None else Memory(validation).batch(batch)
    return compile(Memory(records).batch(batch), validation=held, read="memory", loading=loading)
