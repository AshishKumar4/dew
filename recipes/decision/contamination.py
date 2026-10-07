"""Keep evaluation items out of training: an example that repeats an evaluation item is dropped.

The evaluation side is reduced to hashes, so the index can travel to a
training job without the text of suites that may not be republished, such as
Decision Index's rows:

    python recipes/decision/contamination.py --out eval-index.npz \\
        --jsonl suite-0.2.1/selected-rows.jsonl.gz suite-0.2.1/added-rows.jsonl.gz \\
        --hub LocalLLaMA/typed-decisions@d0e2f0c4:all/test-00000-of-00001.parquet

Two things identify an evaluation item. One is a decision: a state with one
question's type, instructions and options, compared exactly after
normalisation, at any length, with the options in any order. A training
decision that is an evaluation decision is leakage, however short its state.
The other is the item's content: its state, or, where the state is empty
because the request carries its content in the question, as Decision
Index's rows and half of the recipes' framed rows do, each question's
instructions. A question and its options alone are never compared: they are
a task's schema, which every row of a table shares (`DecisionTable` asks one
question of every row) and a training split shares with its test split by
design, so rows of different states asked the same question are all kept.

Text is compared after Unicode normalisation, lowercasing, and every run of
characters other than letters and digits read as one space, and content is
compared whole and line by line, since a request's content often follows a
line of instructions. An example of natural text is dropped when its
content, or a line of it, of at least four words equals an evaluation item's
content or a line of it, or when it shares a run of 13 words with one. A
synthetic set's items are built from shared templates, so lines and runs of
words recur across its splits by construction; its content is compared only
whole (a structured state as its JSON), and the set's own split separation
is relied on beyond that, as Open-Jev's and typed-decisions' authors check
theirs.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sources import Value

from dew.decision import Example

GRAM = 13
WHOLE = 4


def words(text: str) -> list[str]:
    """`text` as the words compared: normalised, lowercased, punctuation read as space."""
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).lower()).split()


def _hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def texts(value: Value) -> Iterator[str]:
    """Every text a JSON value holds: a string, or each string inside a structure, and the
    structure itself rendered as JSON with sorted keys."""
    match value:
        case str():
            yield value
        case dict() | list():
            yield json.dumps(value, sort_keys=True, ensure_ascii=False)
            for inner in (value.values() if isinstance(value, dict) else value):
                yield from texts(inner)
        case _:
            return


def _fingerprints(text: str, *, natural: bool = True) -> tuple[list[int], list[int]]:
    """A text's hashes compared whole, of at least four words, and its 13-word runs'.

    Natural text is also compared line by line, and by runs; a synthetic set's
    text is compared whole alone.
    """
    wholes, grams = [], []
    units = [text, *text.splitlines()] if natural else [text]
    for unit in dict.fromkeys(units):
        tokens = words(unit)
        if len(tokens) >= WHOLE:
            wholes.append(_hash(" ".join(tokens)))
        if natural:
            grams.extend(_hash(" ".join(tokens[start:start + GRAM]))
                         for start in range(len(tokens) - GRAM + 1))
    return wholes, grams


def _canonical(value: Value) -> Value:
    """`value` as a decision's identity compares it: every string as its words, an empty one as nothing."""
    match value:
        case str():
            return " ".join(words(value)) or None
        case dict():
            return {" ".join(words(key)): _canonical(inner) for key, inner in value.items()}
        case list():
            return [_canonical(inner) for inner in value]
        case _:
            return value


def decision(state: Value, question: Value) -> int:
    """The identity of one decision: `state` with one question, in Jev's wire format.

    A choice's options are compared as a set, whether listed or described.
    """
    asked = question if isinstance(question, dict) else {}
    criteria = asked.get("criteria") or {}
    if asked.get("type") == "choice" and isinstance(criteria, list):
        criteria = dict.fromkeys(criteria)
    parts = [_canonical(state), asked.get("type"), _canonical(asked.get("instructions")),
             _canonical(criteria)]
    return _hash(json.dumps(parts, sort_keys=True, ensure_ascii=False))


@dataclass(frozen=True)
class Evaluated:
    """One evaluation item: its content's texts, and its decisions where it asks questions."""

    texts: tuple[str, ...]
    decisions: tuple[int, ...] = ()

    @classmethod
    def request(cls, state: Value, questions: Value) -> "Evaluated":
        """A Jev request: its content (`request_texts`) and one decision per question."""
        asked = questions.values() if isinstance(questions, dict) else ()
        decisions = tuple(decision(state, question) for question in asked)
        return cls(tuple(request_texts(state, questions)), decisions)


def request_texts(state: Value, questions: Value) -> Iterator[str]:
    """A Jev request's content: its state, or, with an empty state, each question's instructions."""
    if state not in (None, "", {}, []):
        yield from texts(state)
    elif isinstance(questions, dict):
        for question in questions.values():
            if isinstance(question, dict):
                yield from texts(question.get("instructions"))


def example_texts(example: Example, *, natural: bool) -> Iterator[str]:
    """A training example's content, read as a request's is (`request_texts`); a
    synthetic set's structured state is read as its JSON alone."""
    if example.state in (None, "", {}, []):
        for question in example.questions.values():
            yield from texts(question.instructions)
    elif isinstance(example.state, str) or natural:
        yield from texts(example.state)
    else:
        yield json.dumps(example.state, sort_keys=True, ensure_ascii=False)


@dataclass
class Overlaps:
    """The hashed evaluation items, and what `keep` dropped from each training set."""

    wholes: np.ndarray
    grams: np.ndarray
    decisions: np.ndarray
    report: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def of(cls, evaluated: Iterable[Evaluated]) -> "Overlaps":
        wholes, grams, decisions = set(), set(), set()
        for item in evaluated:
            decisions.update(item.decisions)
            for text in item.texts:
                whole, runs = _fingerprints(text)
                wholes.update(whole)
                grams.update(runs)
        return cls(*(np.array(sorted(hashes), np.uint64) for hashes in (wholes, grams, decisions)))

    @classmethod
    def load(cls, path: str | Path) -> "Overlaps":
        with np.load(path) as saved:
            return cls(saved["wholes"], saved["grams"], saved["decisions"])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, wholes=self.wholes, grams=self.grams, decisions=self.decisions)

    def hits(self, example: Example, *, natural: bool = True) -> bool:
        """Whether one of `example`'s decisions is an evaluation decision, or its content
        equals an evaluation item's, or, for natural text, shares a 13-word run with one."""
        asked = [decision(example.state, question.wire()) for question in example.questions.values()]
        if _member(self.decisions, asked):
            return True
        for text in example_texts(example, natural=natural):
            whole, runs = _fingerprints(text, natural=natural)
            if _member(self.wholes, whole) or _member(self.grams, runs):
                return True
        return False

    def keep(self, name: str, examples: Sequence[Example], natural: bool) -> list[Example]:
        """`examples` without those sharing content with the evaluation, the count recorded under `name`."""
        kept = [example for example in examples if not self.hits(example, natural=natural)]
        self.report[name] = {"checked": len(examples), "dropped": len(examples) - len(kept)}
        return kept


def _member(sorted_hashes: np.ndarray, hashes: list[int]) -> bool:
    if not len(sorted_hashes) or not hashes:
        return False
    probe = np.array(hashes, np.uint64)
    at = np.minimum(np.searchsorted(sorted_hashes, probe), len(sorted_hashes) - 1)
    return bool(np.any(sorted_hashes[at] == probe))


def jsonl_requests(path: str | Path) -> Iterator[Evaluated]:
    """Each request in a JSON-lines file, gzipped or not."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                yield Evaluated.request(row.get("state"), row.get("questions"))


def hub_rows(reference: str, columns: Sequence[str]) -> Iterator[Evaluated]:
    """The items of a pinned Hub dataset file, `repo@revision:path`: the texts of its `columns`.

    A row with `state` and `questions` columns is read as a request (`Evaluated.request`).
    """
    from huggingface_hub import hf_hub_download

    repo, rest = reference.split("@", 1)
    revision, path = rest.split(":", 1)
    local = hf_hub_download(repo, path, repo_type="dataset", revision=revision)
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = pq.read_table(local).to_pylist()
    elif path.endswith(".csv"):
        csv.field_size_limit(2**31 - 1)  # an email body can pass csv's default of 131,072 characters
        with open(local, newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        with open(local) as file:
            rows = [json.loads(line) for line in file if line.strip()]
    for row in rows:
        if {"state", "questions"} <= set(row):
            yield Evaluated.request(_decoded(row["state"]), _decoded(row["questions"]))
        else:
            yield Evaluated(tuple(text for column in columns for text in texts(_decoded(row[column]))))


def _decoded(value: Value) -> Value:
    """A cell that holds JSON text, as the value it encodes; any other cell as it is."""
    if isinstance(value, str) and value[:1] in ('{', '[', '"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jsonl", nargs="*", default=[], help="request files with state and questions")
    parser.add_argument("--hub", nargs="*", default=[],
                        help="repo@revision:path=column,column files and the columns that hold their content")
    options = parser.parse_args()
    sources = [jsonl_requests(path) for path in options.jsonl]
    for given in options.hub:
        reference, _, columns = given.partition("=")
        sources.append(hub_rows(reference, columns.split(",") if columns else []))
    evaluated = (item for source in sources for item in source)
    overlaps = Overlaps.of(evaluated)
    overlaps.save(options.out)
    counts = {"whole_texts": len(overlaps.wholes), "runs": len(overlaps.grams),
              "decisions": len(overlaps.decisions)}
    print(json.dumps({"out": str(options.out), **counts}))


if __name__ == "__main__":
    main()
