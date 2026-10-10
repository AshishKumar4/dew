"""The labelled sets the decision recipes train on, each read at a pinned commit and written as JSON lines.

    python recipes/decision/sources.py --out data/mixture --decontaminate eval-index.npz

Every source turns its rows into `dew.decision.Example`s: a state, typed
questions, and each question's gold, an answer or a distribution. `Mixture`
holds one of each at its weight, and this writes each set's training
examples to `<out>/<name>.jsonl` and the ones it holds back to
`<out>/<name>.held.jsonl`, in the rows `Example.of` reads, and records the
weights, every source's settings and what decontamination dropped in
`<out>/mixture.json`. A run reads the directory as a
`dew.decision.DecisionMixture` (`encoder.py`, `clef.py`). `--decontaminate`
names an evaluation index `contamination.py` built: every example whose
content an evaluation item shares is dropped before it is written.

Sources whose rows are benchmark-style items (a passage and a question, a
request and its intents) are framed two ways, half of the rows each, chosen
by a hash of the row: the content as the state and the question as the
instructions, or no state and the content inside the instructions; and the
options under their own names, or under letters (`option_i` past 26) with
their text as the description. Requests reach a decision model in both
shapes, so the model learns both.

Licences: Open-Jev's generated rows are CC0; typed-decisions, gliclass and
BoolQ's card state Apache-2.0, Apache-2.0 and CC BY-SA 3.0; BANKING77 CC BY
4.0; CLINC150 CC BY 3.0; WANLI CC BY 4.0; HellaSwag and GSM8K MIT; ARC CC
BY-SA 4.0. Nothing non-commercial is read.
"""

import csv
import dataclasses
import gzip
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from dew.decision import Choice, Example, Noul, Question, Score

if TYPE_CHECKING:
    from contamination import Overlaps

type Value = str | int | float | bool | None | list["Value"] | dict[str, "Value"]
"""A JSON value, as `json.loads` returns one."""


def _download(repo: str, revision: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset", revision=revision)


def _rows(repo: str, revision: str, path: str, indices: set[int] | None = None) -> Iterator[dict]:
    """A pinned file's rows, materialising only the selected indices when supplied."""
    local = _download(repo, revision, path)
    if path.endswith((".jsonl", ".jsonl.gz")):
        with (gzip.open(local, "rt") if path.endswith(".gz") else open(local)) as file:
            for index, line in enumerate(file):
                if line.strip() and (indices is None or index in indices):
                    yield json.loads(line)
        return
    if path.endswith(".csv"):
        with open(local, newline="") as file:
            for index, row in enumerate(csv.DictReader(file)):
                if indices is None or index in indices:
                    yield row
        return
    import pyarrow.parquet as pq

    file = pq.ParquetFile(local)
    offset = 0
    for group in range(file.num_row_groups):
        size = file.metadata.row_group(group).num_rows
        selected = None if indices is None else {index - offset for index in indices
                                                 if offset <= index < offset + size}
        offset += size
        if selected == set():
            continue
        start = 0
        for batch in file.iter_batches(batch_size=4096, row_groups=[group]):
            positions = (None if selected is None else
                         [index - start for index in sorted(selected) if start <= index < start + len(batch)])
            if positions is None:
                yield from batch.to_pylist()
            elif positions:
                yield from batch.take(positions).to_pylist()
            start += len(batch)


def _count(repo: str, revision: str, path: str) -> int:
    """The pinned train file's row count, without loading its rows into memory."""
    local = _download(repo, revision, path)
    if path.endswith((".jsonl", ".jsonl.gz", ".csv")):
        with (gzip.open(local, "rt") if path.endswith(".gz") else open(local, newline="")) as file:
            return sum(1 for _ in (csv.DictReader(file) if path.endswith(".csv") else file))
    import pyarrow.parquet as pq

    return pq.ParquetFile(local).metadata.num_rows


def _held(count: int, held: int | None) -> int:
    return held if held is not None else min(2000, max(50, count // 50), count // 2)


def _draw(count: int, limit: int | None, held: int | None, seed: int) -> set[int] | None:
    """Choose the capped train and hold-out rows before decoding or converting them."""
    needed = count if limit is None else min(count, limit + _held(count, held))
    return None if needed == count else set(random.Random(seed).sample(range(count), needed))


def _github(repo: str, revision: str, path: str, *, media: bool = False) -> Path:
    """Cache a pinned GitHub file, reading LFS bytes from the media host when requested."""
    import shutil
    import urllib.request

    from huggingface_hub import cached_assets_path

    cached = cached_assets_path(library_name="dew", namespace=repo, subfolder=revision) / path
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        host = "media.githubusercontent.com/media" if media else "raw.githubusercontent.com"
        partial = cached.with_suffix(cached.suffix + ".partial")
        with (urllib.request.urlopen(f"https://{host}/{repo}/{revision}/{path}", timeout=300) as response,
              partial.open("wb") as target):
            shutil.copyfileobj(response, target)
        partial.replace(cached)
    return cached


def _unit(text: str, salt: str) -> float:
    """A number in [0, 1) that `text` fixes, so a row is framed the same way on every read."""
    digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class Source:
    """One labelled set a mixture reads, at `weight`, a share of every step; weight 0 leaves it out.

    `limit` caps the training examples, drawn with `seed`, and `held` holds that
    many back for validation and calibration. With no count, hold out 2% of
    the examples (rounded down), at least 50 and at most 2,000, and never
    more than half of them, so a small set still trains. An explicit zero
    keeps every example in train.
    """

    weight: float = 0.0
    limit: int | None = 200_000
    held: int | None = None
    seed: int = 0
    revision: str = ""
    natural: ClassVar[bool] = True
    """Whether the set is natural text rather than built from templates (`contamination.Overlaps`)."""
    repo: ClassVar[str] = ""
    files: ClassVar[tuple[str, ...]] = ()
    licence: ClassVar[str] = ""
    licence_url: ClassVar[str] = ""
    label_column: ClassVar[str] = "label"

    def examples(self) -> list[Example]:
        """The seeded, capped training candidates, with room for the held-out split."""
        return [example for row in self.rows() if (example := self.convert(row)) is not None]

    def convert(self, row: dict) -> Example | None:
        """One labelled training row as a decision, or None for a row without gold."""
        raise NotImplementedError

    def count(self) -> int:
        """The pinned source's train rows, before sampling or holding any out."""
        return sum(_count(self.repo, self.revision, path) for path in self.files)

    def rows(self) -> Iterator[dict]:
        """Seeded rows from training files only, capped before they become Python objects."""
        counts = [_count(self.repo, self.revision, path) for path in self.files]
        selected = _draw(sum(counts), self.limit, self.held, self.seed)
        offset = 0
        for path, count in zip(self.files, counts, strict=True):
            indices = None if selected is None else {index - offset for index in selected
                                                     if offset <= index < offset + count}
            yield from _rows(self.repo, self.revision, path, indices)
            offset += count

    def labels(self, column: str) -> list[str]:
        """A ClassLabel column's option names in the pinned parquet's metadata order."""
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(_download(self.repo, self.revision, self.files[0])).schema_arrow
        feature = json.loads(schema.metadata[b"huggingface"])["info"]["features"][column]
        return list((feature.get("feature") or feature)["names"])

    @cached_property
    def classes(self) -> list[str]:
        return self.labels(self.label_column)

    def split(self) -> tuple[list[Example], list[Example]]:
        """The training examples and the held-out ones."""
        examples = self.examples()
        random.Random(self.seed).shuffle(examples)
        count = _held(self.count() if self.files else len(examples), self.held)
        count = min(count, len(examples))
        held, train = examples[:count], examples[count:]
        return (train if self.limit is None else train[:self.limit]), held


@dataclass(frozen=True)
class Framed(Source):
    """A set of benchmark-style items, framed both ways (the module's description).

    With `most`, an item of more options asks the right one and `most - 1`
    others drawn by a hash of the item, so every option keeps its whole text
    within a row's budget; which others varies from item to item.
    """

    most: int | None = None

    def example(self, content: str, instructions: str, options: Sequence[str], answer: int,
                descriptions: Sequence[str] | None = None) -> Example:
        """`content` asked `instructions` over `options`, whose right one is `answer`.

        Without `descriptions` the options' text is what they say; with them,
        each option is named by its own text and described by its description.
        """
        if self.most is not None and len(options) > self.most:
            others = [index for index in range(len(options)) if index != answer]
            drawn = random.Random(int(_unit(content, "options") * 2**53)).sample(others, self.most - 1)
            kept = sorted([answer, *drawn])
            options, answer = [options[index] for index in kept], kept.index(answer)
            descriptions = None if descriptions is None else [descriptions[index] for index in kept]
        lettered = _unit(content, "keys") < 0.5 or len(set(options)) < len(options)
        if lettered:
            keys = [chr(65 + index) if len(options) <= 26 else f"option_{index}"
                    for index in range(len(options))]
            texts = (list(options) if descriptions is None
                     else [f"{option}: {text}" for option, text in zip(options, descriptions, strict=True)])
            criteria: dict[str, Value] = dict(zip(keys, texts, strict=True))
        else:
            criteria = dict(zip(options, descriptions or [None] * len(options), strict=True))
        if _unit(content, "state") < 0.5:
            state, asked = content, instructions
        else:
            state, asked = "", f"{instructions}\n{content}"
        question = Choice(asked, criteria)
        return Example(state, {"answer": question}, {"answer": question.options[answer]})


@dataclass(frozen=True)
class OpenJev(Source):
    """ZefanCai/Open-Jev's typed decisions (CC0), grouped by state into examples of several questions.

    Rows that share a group and a state become one example, at most `questions`
    of them each, so a joint layout reads them in one row: two fit Clef's
    layout in 4,096 tokens in every example of a sample of 2,000, where a
    fifth of the examples of three or more do not. A row's `target` is its gold
    distribution in its options' order; a choice's options read
    `key: description`. The calibration split is held out, `calibration` of
    its examples or all of them.
    """

    natural: ClassVar[bool] = False
    weight: float = 0.30
    calibration: int | None = None
    config: str = "release-v2-redistributable"
    revision: str = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
    questions: int = 2

    def read(self, split: str) -> list[Example]:
        rows = _rows("ZefanCai/Open-Jev", self.revision, f"data/{self.config}/{split}-00000-of-00001.parquet")
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            groups[(row["group_id"], row["state_json"])].append(row)
        examples = []
        for (_, state), members in groups.items():
            for start in range(0, len(members), self.questions):
                asked, targets = {}, {}
                for index, row in enumerate(members[start:start + self.questions]):
                    name = f"q{index + 1}"
                    asked[name], targets[name] = _open_jev_question(row)
                examples.append(Example(json.loads(state), asked, targets=targets))
        return examples

    def examples(self) -> list[Example]:
        return self.read("train")

    def split(self) -> tuple[list[Example], list[Example]]:
        train = self.examples()
        random.Random(self.seed).shuffle(train)
        held = self.read("calibration")[:self.calibration]
        return (train if self.limit is None else train[:self.limit]), held


def _open_jev_question(row: dict) -> tuple[Question, tuple[float, ...]]:
    """An Open-Jev row's question, and its target in that question's option order."""
    options, target = list(row["options"]), tuple(float(value) for value in row["target"])
    match row["kind"]:
        case "noul" if options == ["no", "yes"]:
            return Noul(row["question"]), target
        case "score":
            return Score(row["question"], options), target
        case "choice" | "noul":
            pairs = [option.split(": ", 1) if ": " in option else [option, ""] for option in options]
            keys = [key for key, _ in pairs]
            if len(set(keys)) < len(keys):
                return Choice(row["question"], options), target
            return Choice(row["question"], {key: text or None for key, text in pairs}), target
        case kind:
            raise ValueError(f"Open-Jev row {row['id']} has kind {kind!r}, which no question type reads")


@dataclass(frozen=True)
class TypedDecisions(Source):
    """LocalLLaMA/typed-decisions' train cases (Apache-2.0): five questions each, the gold a teacher's mean.

    A case's `state` and `questions` are a Jev request body, and its gold gives
    every option's probability. `held` cases are kept back for model selection,
    as OpenDecider-nano kept 100; its test split is never read here.
    """

    natural: ClassVar[bool] = False
    weight: float = 0.05
    held: int = 100
    revision: str = "d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18"

    def examples(self) -> list[Example]:
        examples = []
        for row in _rows("LocalLLaMA/typed-decisions", self.revision, "all/train-00000-of-00001.parquet"):
            wires = json.loads(row["questions"])
            questions = {name: Question.from_wire(wire) for name, wire in wires.items()}
            gold = json.loads(row["gold"])
            targets = {name: tuple(float(gold[name]["probabilities"][option]) for option in question.options)
                       for name, question in questions.items()}
            examples.append(Example(_state(row["state"]), questions, targets=targets))
        return examples


def _state(text: str) -> Value:
    """A state stored as JSON text, or as plain text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


@dataclass(frozen=True)
class GliClass(Source):
    """knowledgator/gliclass-v2.0 (Apache-2.0): texts with candidate labels and the true ones among them.

    A text with exactly one true label asks a choice among its candidates; any
    other asks, for each candidate, whether it applies. Each text already says
    what it is classified for.
    """

    weight: float = 0.20
    limit: int | None = 200_000
    most: int = 24
    """A text of more candidates asks its true ones and others drawn up to this many."""
    revision: str = "93d3cdc82257a9e821f4b21d08e3dc81979bb653"
    shard: str = "data/train-00000-of-00003.parquet"

    def examples(self) -> list[Example]:
        examples = []
        for row in _rows("knowledgator/gliclass-v2.0", self.revision, self.shard):
            labels, true = list(dict.fromkeys(row["all_labels"])), set(row["true_labels"])
            if len(labels) < 2 or not true <= set(labels):
                continue
            if len(labels) > self.most:
                others = [label for label in labels if label not in true]
                drawn = random.Random(int(_unit(row["text"], "labels") * 2**53)).sample(
                    others, max(0, self.most - len(true)))
                labels = [label for label in labels if label in true or label in drawn]
            if len(true) == 1:
                question = Choice("Which label fits the text?", labels)
                examples.append(Example(row["text"], {"label": question}, {"label": next(iter(true))}))
                continue
            asked = {f"label_{index}": Noul(f"Does this label apply to the text? {label}")
                     for index, label in enumerate(labels[:8])}
            answers = {name: str(label in true).lower() for name, label in zip(asked, labels, strict=False)}
            examples.append(Example(row["text"], asked, answers))
        return examples


@dataclass(frozen=True)
class Intents(Framed):
    """BANKING77's training requests (mteb/banking77, CC BY 4.0), each asking its intent among all 77."""

    most: int | None = 24
    weight: float = 0.04
    revision: str = "18072d2685ea682290f7b8924d94c62acc19c0b2"

    def examples(self) -> list[Example]:
        rows = list(_rows("mteb/banking77", self.revision, "data/train-00000-of-00001.parquet"))
        intents = list(dict.fromkeys(row["label_text"] for row in rows))
        return [self.example(row["text"], "Which banking request is this?", intents,
                             intents.index(row["label_text"])) for row in rows]


@dataclass(frozen=True)
class Clinc(Framed):
    """CLINC150's training queries (clinc/clinc_oos `plus`, CC BY 3.0): 150 intents and out-of-scope."""

    most: int | None = 24
    weight: float = 0.04
    revision: str = "155b9c710419136e17307b80d0a13e68cd46b4ec"

    def examples(self) -> list[Example]:
        import pyarrow.parquet as pq

        path = _download("clinc/clinc_oos", self.revision, "plus/train-00000-of-00001.parquet")
        table = pq.read_table(path)
        features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
        intents = list(features["intent"]["names"])
        return [self.example(row["text"], "Which intent does this query express?", intents, row["intent"])
                for row in table.to_pylist()]


@dataclass(frozen=True)
class Wanli(Framed):
    """WANLI's training pairs (alisawuffles/WANLI, CC BY 4.0): entailment, neutral or contradiction."""

    weight: float = 0.08
    limit: int | None = 60_000
    revision: str = "61c95318fd71c55b6ba355d76253254615f387ec"

    def examples(self) -> list[Example]:
        relations = ["entailment", "neutral", "contradiction"]
        return [self.example(f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
                             "How does the hypothesis relate to the premise?", relations,
                             relations.index(row["gold"]))
                for row in _rows("alisawuffles/WANLI", self.revision, "train.jsonl")
                if row["gold"] in relations]


@dataclass(frozen=True)
class HellaSwag(Framed):
    """HellaSwag's training contexts (Rowan/hellaswag, MIT), each with four endings."""

    weight: float = 0.05
    revision: str = "218ec52e09a7e7462a5400043bb9a69a41d06b76"

    def examples(self) -> list[Example]:
        return [self.example(row["ctx"], "Which continuation is most plausible?", list(row["endings"]),
                             int(row["label"]))
                for row in _rows("Rowan/hellaswag", self.revision, "data/train-00000-of-00001.parquet")
                if str(row["label"]).isdigit()]


@dataclass(frozen=True)
class Arc(Framed):
    """ARC's Easy and Challenge training questions (allenai/ai2_arc, CC BY-SA 4.0)."""

    weight: float = 0.03
    revision: str = "210d026faf9955653af8916fad021475a3f00453"

    def examples(self) -> list[Example]:
        examples = []
        for part in ("ARC-Easy", "ARC-Challenge"):
            for row in _rows("allenai/ai2_arc", self.revision, f"{part}/train-00000-of-00001.parquet"):
                labels = list(row["choices"]["label"])
                if row["answerKey"] in labels:
                    texts, right = list(row["choices"]["text"]), labels.index(row["answerKey"])
                    examples.append(self.example(row["question"], "Which answer is correct?", texts, right))
        return examples


@dataclass(frozen=True)
class BoolQ(Source):
    """BoolQ's training questions (google/boolq, CC BY-SA 3.0): a passage and a yes-or-no question."""

    weight: float = 0.05
    revision: str = "35b264d03638db9f4ce671b711558bf7ff0f80d5"

    def examples(self) -> list[Example]:
        examples = []
        for row in _rows("google/boolq", self.revision, "data/train-00000-of-00001.parquet"):
            question = Noul(f"{row['question'].rstrip('?')}?")
            state: Value = row["passage"]
            if _unit(row["passage"], "state") >= 0.5:
                state, question = "", Noul(f"{question.instructions}\nPassage: {row['passage']}")
            answer = "true" if row["answer"] else "false"
            examples.append(Example(state, {"answer": question}, {"answer": answer}))
        return examples


@dataclass(frozen=True)
class Gsm8k(Framed):
    """GSM8K's training problems (openai/gsm8k, MIT) as numeric choices, four or ten options.

    Each wrong option is the answer to another training problem of similar
    size, as Decision Index 0.3 builds its GSM8K rows, so the options alone
    do not give the answer away.
    """

    weight: float = 0.03
    revision: str = "740312add88f781978c0658806c59bc2815b9866"

    def examples(self) -> list[Example]:
        rows = list(_rows("openai/gsm8k", self.revision, "main/train-00000-of-00001.parquet"))
        golds = [row["answer"].rsplit("####", 1)[-1].strip().replace(",", "") for row in rows]
        values = sorted(set(golds), key=float)
        place = {value: index for index, value in enumerate(values)}
        rng = random.Random(self.seed)
        examples = []
        for index, row in enumerate(rows):
            count = 4 if index % 2 else 10
            # The answers nearest this one in size, none of them its own.
            at = place[golds[index]]
            near = [value for value in values[max(0, at - 2 * count):at + 2 * count] if value != golds[index]]
            options = [golds[index], *rng.sample(near, count - 1)]
            rng.shuffle(options)
            asked = "Choose the numeric answer to the problem."
            examples.append(self.example(row["question"], asked, options, options.index(golds[index])))
        return examples


@dataclass(frozen=True)
class Rows(Source):
    """Your own labelled examples: a JSON-lines file of rows `Example.of` reads.

    Each row has a `state`, `questions` (in Jev's wire format) and `answers`
    or `targets`, so a Jev request body plus its gold is a row.
    """

    path: str | None = None

    def examples(self) -> list[Example]:
        if self.path is None:
            raise ValueError("a Rows source with a weight needs path= set to its JSON-lines file")
        with open(self.path) as file:
            return [Example.of(json.loads(line)) for line in file if line.strip()]


@dataclass(frozen=True)
class Esci(Source):
    """Amazon's ESCI training pairs (Apache-2.0), asking Decision Index's native four-way relevance choice.

    Only `small_version` training rows are read. All three product locales
    are kept because DI 0.2.1 evaluates US, ES and JP, and products join on
    both their id and locale. The option names, descriptions and state are
    the kit's, without the alternate framing used by other sources. Of the
    781,638 pairs, `limit` keeps 15,000 drawn with the seed, about what one
    pass of the `--headroom` mix reads at ESCI's weight, so neither the
    decontamination nor a run's layout check reads the rest.
    """

    limit: int | None = 15_000
    revision: str = "7916cdf6ab75a462e77f20ab40428a10923998d5"

    def rows(self) -> Iterator[dict]:
        """The pinned small training split, joined to its product text in example-id order."""
        import pyarrow.parquet as pq

        root = "shopping_queries_dataset/shopping_queries_dataset_"
        examples = pq.read_table(_github("amazon-science/esci-data", self.revision,
                                         root + "examples.parquet", media=True),
                                  filters=[("split", "=", "train"), ("small_version", "=", 1)])
        products = pq.read_table(_github("amazon-science/esci-data", self.revision,
                                         root + "products.parquet", media=True))
        joined = examples.join(products, keys=["product_id", "product_locale"], join_type="left outer")
        if joined.num_rows != examples.num_rows or joined["product_title"].null_count:
            raise ValueError("ESCI needs exactly one product with a title for every training pair")
        for batch in joined.sort_by([("example_id", "ascending")]).to_batches(max_chunksize=4096):
            yield from batch.to_pylist()

    @staticmethod
    def example(row: dict) -> Example:
        """A joined query-product pair in the kit's ESCI request shape."""
        criteria = {"E": "Exact: the product satisfies the search query.",
                    "S": "Substitute: a product that could substitute for the requested product.",
                    "C": "Complement: a product that complements the requested product.",
                    "I": "Irrelevant: the product does not address the requested product need."}
        product = {key.removeprefix("product_"): row[key] for key in
                   ("product_title", "product_description", "product_bullet_point",
                    "product_brand", "product_color")
                   if row.get(key) is not None}
        question = Choice("Classify the relevance of this product to the search query "
                          "using the ESCI categories.", criteria)
        return Example({"search_query": row["query"], "product": product}, {"answer": question},
                       {"answer": row["esci_label"]})

    def examples(self) -> list[Example]:
        return [self.example(row) for row in self.rows()]


@dataclass(frozen=True)
class ISarcasm(Source):
    """iSarcasmEval's English training tweets (MIT), in DI's task-A request shape.

    The binary gold is `sarcastic`, not the `sarcasm` subtype. A CSV reader
    preserves commas, quotes and newlines inside tweets; rephrases and
    subtype annotations are not part of the model's input.
    """

    revision: str = "dfc708b53bde1bb571abfb5692f63231c2232195"

    @staticmethod
    def example(row: dict) -> Example:
        """One training tweet asked the kit's binary sarcasm question."""
        if row["sarcastic"] not in ("0", "1"):
            raise ValueError(f"sarcastic must be 0 or 1, got {row['sarcastic']!r}")
        question = Choice("Is this text intended to be sarcastic?", {"no": "No", "yes": "Yes"})
        return Example(row["tweet"], {"sarcastic": question},
                       {"sarcastic": "yes" if row["sarcastic"] == "1" else "no"})

    def examples(self) -> list[Example]:
        path = _github("iabufarha/iSarcasmEval", self.revision, "train/train.En.csv")
        with path.open(encoding="utf-8-sig", newline="") as file:
            return [self.example(row) for row in csv.DictReader(file)]


@dataclass(frozen=True)
class Sgd(Source):
    """SGD (google-research-datasets/dstc8-schema-guided-dialogue, CC-BY-SA-4.0), DI user frames."""

    weight: float = 1.0
    revision: str = "e852981ae34990f4358979625854259302feaa78"
    repo: ClassVar[str] = "google-research-datasets/dstc8-schema-guided-dialogue"
    licence: ClassVar[str] = "CC-BY-SA-4.0"

    def dialogues(self) -> Iterator[dict]:
        """The 127 pinned training shards, never the dev or test dialogues."""
        for index in range(1, 128):
            yield from json.loads(_github(self.repo, self.revision, f"train/dialogues_{index:03}.json").read_text())

    @staticmethod
    def frames(dialogue: dict, schemas: dict) -> Iterator[dict]:
        """DI's current-user-frame state; later turns and all state labels stay out of it."""
        history = []
        for turn in dialogue["turns"]:
            history.append({"speaker": turn["speaker"], "utterance": turn["utterance"]})
            if turn["speaker"] == "USER":
                for frame in turn["frames"]:
                    schema = schemas[frame["service"]]
                    yield {"history": list(history), "service": frame["service"], "schema": schema,
                           "gold": frame["state"].get("active_intent", "NONE")}

    def convert(self, row: dict) -> Example:
        intents = [item["name"] for item in row["schema"]["intents"]] + ["NONE"]
        question = Choice("Using the dialogue history and service schema in state, choose the active intent "
                          "for this service. Choose NONE when no service intent is active. Do not use future "
                          "turns or hidden labels.", {name: name for name in intents})
        state = {key: row[key] for key in ("history", "service", "schema")}
        return Example({**state, "task": "SGD current-service intent"}, {"intent": question}, {"intent": row["gold"]})

    def count(self) -> int:
        return sum(len(turn["frames"]) for dialogue in self.dialogues() for turn in dialogue["turns"]
                   if turn["speaker"] == "USER")

    def examples(self) -> list[Example]:
        schemas = {item["service_name"]: item for item in
                   json.loads(_github(self.repo, self.revision, "train/schema.json").read_text())}
        selected = _draw(self.count(), self.limit, self.held, self.seed)
        rows = (row for dialogue in self.dialogues() for row in self.frames(dialogue, schemas))
        return [self.convert(row) for index, row in enumerate(rows) if selected is None or index in selected]


def _evidence_database() -> Path:
    """DI's HotpotQA Wikipedia database, accepted only at its pinned SHA-256."""
    import shutil
    import urllib.request

    from huggingface_hub import cached_assets_path

    digest = "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f"
    path = cached_assets_path(library_name="dew", namespace="hover", subfolder=digest) / "wiki_wo_links.db"
    if not path.is_file():
        partial = path.with_suffix(".partial")
        with (urllib.request.urlopen("https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db", timeout=300) as source,
              partial.open("wb") as target):
            shutil.copyfileobj(source, target)
        with partial.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                raise ValueError("HoVer evidence database does not match DI's pinned SHA-256")
        partial.replace(path)
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != digest:
            raise ValueError("HoVer evidence database does not match DI's pinned SHA-256")
    return path


@dataclass(frozen=True)
class Hover(Source):
    """HoVer (hover-nlp/hover, MIT; Wikipedia evidence CC-BY-SA-4.0), in DI's evidence-backed shape."""

    weight: float = 1.0
    revision: str = "39b84697f196308f398a251a7aea9b82ae0f0562"
    repo: ClassVar[str] = "hover-nlp/hover"
    licence: ClassVar[str] = "MIT AND CC-BY-SA-4.0"

    def claims(self) -> list[dict]:
        return json.loads(_github(self.repo, self.revision, "data/hover/hover_train_release_v1.1.json").read_text())

    def convert(self, row: dict) -> Example:
        question = Choice("Is the claim supported by the evidence?",
                          {"SUPPORTED": "The evidence supports the claim.",
                           "NOT_SUPPORTED": "The evidence does not support the claim."})
        return Example({"claim": row["claim"], "evidence": row["evidence"]}, {"q": question}, {"q": row["label"]})

    def examples(self) -> list[Example]:
        import sqlite3
        import unicodedata

        claims = self.claims()
        selected = _draw(len(claims), self.limit, self.held, self.seed)
        db = sqlite3.connect(f"file:{_evidence_database()}?mode=ro", uri=True)
        examples = []
        try:
            for index, row in enumerate(claims):
                if selected is not None and index not in selected:
                    continue
                evidence = []
                for title in dict.fromkeys(title for title, _ in row["supporting_facts"]):
                    found = db.execute("SELECT text FROM documents WHERE id=?",
                                       (unicodedata.normalize("NFD", title),)).fetchall()
                    if len(found) != 1:
                        raise ValueError(f"HoVer needs exactly one article for {title!r}")
                    evidence.append({"title": title, "text": found[0][0]})
                examples.append(self.convert({**row, "evidence": evidence}))
        finally:
            db.close()
        return examples


@dataclass(frozen=True)
class ContractNli(Source):
    """ContractNLI (stanfordnlp/contract-nli, CC-BY-4.0), DI's 17 hypothesis questions per contract."""

    weight: float = 1.0
    revision: str = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
    repo: ClassVar[str] = "stanfordnlp/contract-nli"
    licence: ClassVar[str] = "CC-BY-4.0"

    def data(self) -> dict:
        import zipfile

        with zipfile.ZipFile(_github(self.repo, self.revision, "resources/contract-nli.zip")) as archive:
            return json.loads(archive.read("contract-nli/train.json"))

    def convert(self, row: dict) -> Example:
        criteria = {"Entailment": "The contract entails the hypothesis.",
                    "Contradiction": "The contract contradicts the hypothesis.",
                    "NotMentioned": "The hypothesis is neither entailed nor contradicted by the contract."}
        questions = {key: Choice("Classify the relationship between the contract and this hypothesis:\n" +
                                 label["hypothesis"], criteria) for key, label in row["labels"].items()}
        answers = {key: value["choice"] for key, value in row["annotation_sets"][0]["annotations"].items()}
        return Example(row["text"], questions, answers)

    def examples(self) -> list[Example]:
        data = self.data()
        return [self.convert({**doc, "labels": data["labels"]}) for doc in data["documents"]]


@dataclass(frozen=True)
class WinoGrande(Source):
    """WinoGrande XL (allenai/winogrande, Apache-2.0), DI's native A/B blank-filling question."""

    weight: float = 1.0
    revision: str = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"
    repo: ClassVar[str] = "allenai/winogrande"
    files: ClassVar[tuple[str, ...]] = ("winogrande_xl/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "Apache-2.0"

    def convert(self, row: dict) -> Example:
        question = Choice("Which option correctly fills the blank?\n" + row["sentence"],
                          {"A": row["option1"], "B": row["option2"]})
        return Example({}, {"q1": question}, {"q1": "AB"[int(row["answer"]) - 1]})


@dataclass(frozen=True)
class Fever(Framed):
    """FEVER gold evidence (copenlu/fever_gold_evidence, CC-BY-SA-3.0 AND GPL-3.0-only)."""

    weight: float = 1.0
    revision: str = "a6b8d891d393e97a4efac791afffb2d7de5e57c6"
    repo: ClassVar[str] = "copenlu/fever_gold_evidence"
    files: ClassVar[tuple[str, ...]] = ("train.jsonl",)
    licence: ClassVar[str] = "CC-BY-SA-3.0 AND GPL-3.0-only"

    def convert(self, row: dict) -> Example:
        evidence = "\n".join(f"{item[0]}: {item[2]}" for item in row["evidence"])
        options = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        return self.example(f"Claim: {row['claim']}\nEvidence: {evidence}",
                            "Does the evidence support or refute the claim?", options, options.index(row["label"]))


@dataclass(frozen=True)
class Snli(Framed):
    """SNLI training pairs (stanfordnlp/snli, CC-BY-SA-4.0), excluding unlabelled pairs."""

    weight: float = 1.0
    revision: str = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
    repo: ClassVar[str] = "stanfordnlp/snli"
    files: ClassVar[tuple[str, ...]] = ("plain_text/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "CC-BY-SA-4.0"

    def convert(self, row: dict) -> Example | None:
        if row["label"] == -1:
            return None
        return self.example(f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
                            "How does the hypothesis relate to the premise?",
                            ["entailment", "neutral", "contradiction"], row["label"])


@dataclass(frozen=True)
class MultiNli(Framed):
    """MultiNLI (nyu-mll/multi_nli, OANC, CC-BY-3.0, CC-BY-SA-3.0, MIT and public-domain text)."""

    weight: float = 1.0
    revision: str = "da70db2af9d09693783c3320c4249840212ee221"
    repo: ClassVar[str] = "nyu-mll/multi_nli"
    files: ClassVar[tuple[str, ...]] = ("data/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "LicenseRef-OANC AND CC-BY-3.0 AND CC-BY-SA-3.0 AND MIT AND LicenseRef-Public-Domain"

    def convert(self, row: dict) -> Example | None:
        if row["label"] == -1:
            return None
        return self.example(f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
                            "How does the hypothesis relate to the premise?",
                            ["entailment", "neutral", "contradiction"], row["label"])


@dataclass(frozen=True)
class Paws(Source):
    """PAWS labeled_final (google-research-datasets/paws, Google's free-use grant), gold paraphrases."""

    weight: float = 1.0
    revision: str = "161ece9501cf0a11f3e48bd356eaa82de46d6a09"
    repo: ClassVar[str] = "google-research-datasets/paws"
    files: ClassVar[tuple[str, ...]] = ("labeled_final/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "LicenseRef-Google-PAWS"

    def convert(self, row: dict) -> Example:
        state = {key: row[key] for key in ("sentence1", "sentence2")}
        return Example(state, {"paraphrase": Noul("Do these sentences express the same meaning?")},
                       {"paraphrase": "true" if row["label"] == 1 else "false"})


def _multilabel(text: str, labels: Sequence[str], true: Sequence[int], instructions: str) -> Example:
    """One true label is a Choice; otherwise every candidate gets its own independent Noul."""
    if len(true) == 1:
        return Example(text, {"label": Choice(instructions, labels)}, {"label": labels[true[0]]})
    questions = {f"label_{index}": Noul(f"{instructions} Does this label apply? {label}")
                 for index, label in enumerate(labels)}
    answers = {f"label_{index}": "true" if index in true else "false" for index in range(len(labels))}
    return Example(text, questions, answers)


@dataclass(frozen=True)
class GoEmotions(Source):
    """GoEmotions simplified (google-research-datasets/go_emotions, Apache-2.0), all 28 emotions."""

    weight: float = 1.0
    revision: str = "add492243ff905527e67aeb8b80c082af02207c3"
    repo: ClassVar[str] = "google-research-datasets/go_emotions"
    files: ClassVar[tuple[str, ...]] = ("simplified/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "Apache-2.0"
    label_column: ClassVar[str] = "labels"

    def convert(self, row: dict) -> Example:
        return _multilabel(row["text"], self.classes, row["labels"], "Which emotion is expressed in the text?")


@dataclass(frozen=True)
class DBpedia(Framed):
    """DBpedia-14 (fancyzhx/dbpedia_14, CC-BY-SA-3.0), fourteen ontology classes."""

    weight: float = 1.0
    revision: str = "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
    repo: ClassVar[str] = "fancyzhx/dbpedia_14"
    files: ClassVar[tuple[str, ...]] = ("dbpedia_14/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "CC-BY-SA-3.0"

    def convert(self, row: dict) -> Example:
        return self.example(f"{row['title']}\n{row['content']}", "Which category describes this entity?",
                            self.classes, row["label"])


@dataclass(frozen=True)
class CivilComments(Source):
    """Civil Comments (google/civil_comments, CC0-1.0), annotator fractions for toxicity and subtypes."""

    weight: float = 1.0
    revision: str = "f2970eb3a55777454c94069077cc8d9b5866312d"
    repo: ClassVar[str] = "google/civil_comments"
    files: ClassVar[tuple[str, ...]] = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
    licence: ClassVar[str] = "CC0-1.0"

    def convert(self, row: dict) -> Example:
        names = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit")
        questions = {name: Noul(f"Does this comment contain {name.replace('_', ' ')}?") for name in names}
        return Example(row["text"], questions, targets={name: (1 - row[name], row[name]) for name in names})


@dataclass(frozen=True)
class SmsSpam(Source):
    """SMS Spam Collection (ucirvine/sms_spam, CC-BY-4.0 from UCI), ham versus spam."""

    weight: float = 1.0
    revision: str = "cae486f927c250fe1d4a5b55f11357964ed1646c"
    repo: ClassVar[str] = "ucirvine/sms_spam"
    files: ClassVar[tuple[str, ...]] = ("plain_text/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "CC-BY-4.0"

    def convert(self, row: dict) -> Example:
        return Example(row["sms"], {"spam": Noul("Is this message spam?")},
                       {"spam": "true" if row["label"] == 1 else "false"})


@dataclass(frozen=True)
class BiasInBios(Framed):
    """Bias in Bios (LabHC/bias_in_bios, MIT), occupations without the protected attribute as input."""

    weight: float = 1.0
    revision: str = "052f01de644dba841176e0449528b41f27d94a61"
    repo: ClassVar[str] = "LabHC/bias_in_bios"
    files: ClassVar[tuple[str, ...]] = ("data/train-00000-of-00001-0ab65b32c47407e8.parquet",)
    licence: ClassVar[str] = "MIT"

    def convert(self, row: dict) -> Example:
        occupations = ("accountant architect attorney chiropractor comedian composer dentist dietitian dj filmmaker "
                       "interior_designer journalist model nurse painter paralegal pastor personal_trainer "
                       "photographer physician poet professor psychologist rapper software_engineer surgeon teacher "
                       "yoga_teacher").split()
        return self.example(row["hard_text"], "What is this person's occupation?", occupations, row["profession"])


@dataclass(frozen=True)
class MassiveIntent(Framed):
    """MASSIVE English intents (SetFit/amazon_massive_intent_en-US, CC-BY-4.0), sixty user intents."""

    weight: float = 1.0
    revision: str = "f7672a018e8ceb37fc0184dcfbb7e665155ffea6"
    repo: ClassVar[str] = "SetFit/amazon_massive_intent_en-US"
    files: ClassVar[tuple[str, ...]] = ("train.jsonl",)
    licence: ClassVar[str] = "CC-BY-4.0"

    @cached_property
    def classes(self) -> list[str]:
        labels = {row["label"]: row["label_text"] for row in _rows(self.repo, self.revision, self.files[0])}
        return [labels[index] for index in sorted(labels)]

    def convert(self, row: dict) -> Example:
        return self.example(row["text"], "Which intent does this request express?", self.classes, row["label"])


@dataclass(frozen=True)
class Ledgar(Framed):
    """LEDGAR (coastalcph/lex_glue, CC-BY-4.0), one of a hundred contract clause types."""

    weight: float = 1.0
    revision: str = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
    repo: ClassVar[str] = "coastalcph/lex_glue"
    files: ClassVar[tuple[str, ...]] = ("ledgar/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "CC-BY-4.0"

    def convert(self, row: dict) -> Example:
        return self.example(row["text"], "Which type of contract clause is this?", self.classes, row["label"])


@dataclass(frozen=True)
class UnfairTos(Source):
    """Unfair-ToS (coastalcph/lex_glue, CC-BY-4.0), all applicable unfair clause types."""

    weight: float = 1.0
    revision: str = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
    repo: ClassVar[str] = "coastalcph/lex_glue"
    files: ClassVar[tuple[str, ...]] = ("unfair_tos/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "CC-BY-4.0"
    label_column: ClassVar[str] = "labels"

    def convert(self, row: dict) -> Example:
        return _multilabel(row["text"], self.classes, row["labels"], "Which unfair clause type applies to this text?")


@dataclass(frozen=True)
class Snips(Framed):
    """SNIPS (benayas/snips, Apache-2.0), seven labelled personal-assistant intents."""

    weight: float = 1.0
    revision: str = "16915b895754dd4028068abe52b6851a904977d7"
    repo: ClassVar[str] = "benayas/snips"
    files: ClassVar[tuple[str, ...]] = ("data/train-00000-of-00001.parquet",)
    licence: ClassVar[str] = "Apache-2.0"

    def convert(self, row: dict) -> Example:
        labels = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook",
                  "SearchCreativeWork", "SearchScreeningEvent"]
        return self.example(row["text"], "Which intent does this request express?", labels, labels.index(row["category"]))


@dataclass(frozen=True)
class PhishingEmail(Source):
    """Phishing Email Detection (zefang-liu/phishing-email-dataset, LGPL-3.0-only), labelled emails."""

    weight: float = 1.0
    revision: str = "34085a032c123ca237f314a01a67909cdea35e34"
    repo: ClassVar[str] = "zefang-liu/phishing-email-dataset"
    files: ClassVar[tuple[str, ...]] = ("Phishing_Email.csv",)
    licence: ClassVar[str] = "LGPL-3.0-only"

    def convert(self, row: dict) -> Example | None:
        if not row["Email Text"].strip():
            return None
        return Example(row["Email Text"], {"phishing": Noul("Is this email a phishing attempt?")},
                       {"phishing": "true" if row["Email Type"] == "Phishing Email" else "false"})


@dataclass(frozen=True)
class Mixture:
    """Every source a decision recipe reads, each at its weight; a weight of 0 leaves one out.

    The weights are shares of each step, so they need not sum to one.
    Laya's application battery and every evaluation split stay out: AG News,
    DAIR Emotion, Enron spam, the phishing set, toxic-chat, MS MARCO and the
    customer-support tickets are read by no source here.
    """

    open_jev: OpenJev = field(default_factory=OpenJev)
    typed_decisions: TypedDecisions = field(default_factory=TypedDecisions)
    gliclass: GliClass = field(default_factory=GliClass)
    banking77: Intents = field(default_factory=Intents)
    clinc150: Clinc = field(default_factory=Clinc)
    wanli: Wanli = field(default_factory=Wanli)
    hellaswag: HellaSwag = field(default_factory=HellaSwag)
    arc: Arc = field(default_factory=Arc)
    boolq: BoolQ = field(default_factory=BoolQ)
    gsm8k: Gsm8k = field(default_factory=Gsm8k)
    esci: Esci = field(default_factory=Esci)
    isarcasm: ISarcasm = field(default_factory=ISarcasm)
    rows: Rows = field(default_factory=Rows)

    def sources(self) -> Iterator[tuple[str, Source]]:
        """Each source that fills a share of the step, by name."""
        for entry in fields(self):
            source = getattr(self, entry.name)
            if source.weight > 0:
                yield entry.name, source



def row(example: Example) -> dict[str, Value]:
    """`example` as the JSON row `Example.of` reads back."""
    return {"state": example.state,
            "questions": {name: question.wire() for name, question in example.questions.items()},
            "answers": dict(example.answers),
            "targets": {name: list(target) for name, target in example.targets.items()}}


def write(mixture: Mixture, out: Path, overlaps: "Overlaps | None" = None) -> dict[str, Value]:
    """Write `mixture`'s sets under `out`, each decontaminated against `overlaps`, and return what
    `<out>/mixture.json` records."""
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Value] = {}
    for name, source in mixture.sources():
        train, held = source.split()
        if overlaps is not None:
            train = overlaps.keep(name, train, source.natural)
            held = overlaps.keep(f"{name} (held out)", held, source.natural)
        for suffix, examples in ((".jsonl", train), (".held.jsonl", held)):
            if examples:
                lines = (json.dumps(row(example), ensure_ascii=False) + "\n" for example in examples)
                (out / f"{name}{suffix}").write_text("".join(lines))
        counts[name] = {"train": len(train), "held": len(held)}
    made: dict[str, Value] = {
        "weights": {name: source.weight for name, source in mixture.sources()},
        "rows": counts,
        "sources": {name: {"class": type(source).__name__, **dataclasses.asdict(source)}
                    for name, source in mixture.sources()},
        "decontamination": None if overlaps is None else dict(overlaps.report),
    }
    (out / "mixture.json").write_text(json.dumps(made, indent=1) + "\n")
    return made


@dataclass(frozen=True)
class Conversion:
    """The module's command: the sets, where they go, and the index they are decontaminated against."""

    out: str = "data/mixture"
    decontaminate: str | None = None
    headroom: bool = False
    """Add ESCI and iSarcasmEval at weights 0.02 and 0.01; the default mixture does not read either."""
    mixture: Mixture = field(default_factory=Mixture)

    def selected(self) -> Mixture:
        """The mixture, with the two permissively licensed headroom sources when requested."""
        if not self.headroom:
            return self.mixture
        return dataclasses.replace(self.mixture, esci=dataclasses.replace(self.mixture.esci, weight=0.02),
                                   isarcasm=dataclasses.replace(self.mixture.isarcasm, weight=0.01))


if __name__ == "__main__":
    import tyro
    from contamination import Overlaps

    conversion = tyro.cli(Conversion)
    overlaps = None if conversion.decontaminate is None else Overlaps.load(conversion.decontaminate)
    print(json.dumps(write(conversion.selected(), Path(conversion.out), overlaps)["rows"]))
