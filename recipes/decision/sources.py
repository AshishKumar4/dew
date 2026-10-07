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

import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
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


def _rows(repo: str, revision: str, path: str) -> list[dict]:
    """A pinned parquet or JSON-lines file's rows."""
    local = _download(repo, revision, path)
    if path.endswith(".jsonl"):
        with open(local) as file:
            return [json.loads(line) for line in file if line.strip()]
    import pyarrow.parquet as pq

    return pq.read_table(local).to_pylist()


def _unit(text: str, salt: str) -> float:
    """A number in [0, 1) that `text` fixes, so a row is framed the same way on every read."""
    digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class Source:
    """One labelled set a mixture reads, at `weight`, a share of every step; weight 0 leaves it out.

    `limit` caps the training examples, drawn with `seed`, and `held` holds that
    many back for validation and calibration.
    """

    weight: float = 0.0
    limit: int | None = None
    held: int = 0
    seed: int = 0
    natural: ClassVar[bool] = True
    """Whether the set is natural text rather than built from templates (`contamination.Overlaps`)."""

    def examples(self) -> list[Example]:
        """Every training example the set gives."""
        raise NotImplementedError

    def split(self) -> tuple[list[Example], list[Example]]:
        """The training examples and the held-out ones."""
        examples = self.examples()
        random.Random(self.seed).shuffle(examples)
        held, train = examples[:self.held], examples[self.held:]
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
        rows = _rows("mteb/banking77", self.revision, "data/train-00000-of-00001.parquet")
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
        rows = _rows("openai/gsm8k", self.revision, "main/train-00000-of-00001.parquet")
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
    mixture: Mixture = field(default_factory=Mixture)


if __name__ == "__main__":
    import tyro
    from contamination import Overlaps

    conversion = tyro.cli(Conversion)
    overlaps = None if conversion.decontaminate is None else Overlaps.load(conversion.decontaminate)
    print(json.dumps(write(conversion.mixture, Path(conversion.out), overlaps)["rows"]))
