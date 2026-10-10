"""Pinned decision sources, their held-out splits, and their training requests."""

import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import sqlite3
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from dew.decision import Choice, Example, Noul, Score

RECIPES = Path(__file__).parents[1] / "recipes" / "decision"


@pytest.fixture(scope="module")
def sources():
    sys.path.insert(0, str(RECIPES))
    spec = importlib.util.spec_from_file_location("decision_sources", RECIPES / "sources.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sources_hold_out_a_seeded_two_percent_with_bounds(sources, monkeypatch, tmp_path):
    question = Choice("Which?", ["a", "b"])

    def examples(source):
        count = 120 if isinstance(source, sources.Intents) else 3000
        return [Example(str(index), {"q": question}, {"q": "a"}) for index in range(count)]

    monkeypatch.setattr(sources.Intents, "examples", examples)
    monkeypatch.setattr(sources.Clinc, "examples", examples)
    default = sources.Mixture()
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
           for entry in dataclasses.fields(default)}
    mixture = sources.Mixture(**{**off, "banking77": sources.Intents(seed=17), "clinc150": sources.Clinc()})
    made = sources.write(mixture, tmp_path)
    assert made["rows"] == {"banking77": {"train": 70, "held": 50},
                            "clinc150": {"train": 2940, "held": 60}}
    for name, source in mixture.sources():
        train, held = source.split()
        assert (train, held) == source.split()
        assert {row.state for row in train}.isdisjoint(row.state for row in held)
        assert held != dataclasses.replace(source, seed=source.seed + 1).split()[1]
        written = [json.loads(line) for line in (tmp_path / f"{name}.held.jsonl").read_text().splitlines()]
        assert written == [sources.row(row) for row in held]
    monkeypatch.setattr(sources.Rows, "examples", lambda self: [Example("x", {"q": question})] * 110_000)
    assert len(sources.Rows().split()[1]) == 2000
    monkeypatch.setattr(sources.Rows, "examples", lambda self: [Example("x", {"q": question})] * 30)
    assert [len(part) for part in sources.Rows().split()] == [15, 15]
    assert len(sources.Rows(held=0, limit=3).split()[0]) == 3
    assert sources.Rows(held=0).split()[1] == []
    monkeypatch.setattr(sources.TypedDecisions, "examples", lambda self: examples(sources.Intents()))
    assert len(sources.TypedDecisions().split()[1]) == 100
    calibration = [Example("calibration", {"q": question})]
    monkeypatch.setattr(sources.OpenJev, "read", lambda self, split: calibration if split == "calibration"
                        else examples(sources.Intents()))
    assert sources.OpenJev().split()[1] == calibration


@pytest.mark.parametrize("label", ["E", "S", "C", "I"])
def test_esci_rows_reproduce_the_kits_request(sources, label):
    example = sources.Esci.example({"query": "red shoes", "esci_label": label, "product_title": "blue shoes",
                                    "product_description": "Running shoes", "product_bullet_point": None,
                                    "product_brand": "Example", "product_color": "blue"})
    assert example.state == {"search_query": "red shoes", "product": {
        "title": "blue shoes", "description": "Running shoes", "brand": "Example", "color": "blue"}}
    assert example.questions["answer"].wire() == {
        "type": "choice",
        "instructions": ("Classify the relevance of this product to the search query "
                         "using the ESCI categories."),
        "criteria": {"E": "Exact: the product satisfies the search query.",
                     "S": "Substitute: a product that could substitute for the requested product.",
                     "C": "Complement: a product that complements the requested product.",
                     "I": "Irrelevant: the product does not address the requested product need."}}
    assert example.answers == {"answer": label}


def test_esci_small_cap_skips_batches_with_no_selected_pairs(sources, monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    examples_path, products_path = tmp_path / "examples.parquet", tmp_path / "products.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"product_id": str(index), "product_locale": "us", "split": "train", "small_version": 1,
         "query": f"query {index}", "esci_label": "E"} for index in range(8200)]), examples_path)
    pq.write_table(pa.Table.from_pylist([
        {"product_id": "8199", "product_locale": "us", "product_title": "Last product"}]), products_path)
    monkeypatch.setattr(sources, "_github", lambda repo, rev, path, **kwargs:
                        examples_path if path.endswith("examples.parquet") else products_path)
    monkeypatch.setattr(sources, "_draw", lambda count, limit, held, seed: {8199})
    examples = sources.Esci(limit=1, held=0).examples()
    assert len(examples) == 1
    assert examples[0].state == {"search_query": "query 8199", "product": {"title": "Last product"}}
    assert examples[0].answers == {"answer": "E"}


def test_isarcasm_reads_csv_and_uses_the_binary_target(sources, monkeypatch, tmp_path):
    path = tmp_path / "train.En.csv"
    tweets = ['Wonderful, another "perfect" day.\nReally.', "An ordinary afternoon."]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["tweet", "sarcastic", "sarcasm"])
        writer.writeheader()
        writer.writerows([{"tweet": tweets[0], "sarcastic": "1", "sarcasm": "0"},
                          {"tweet": tweets[1], "sarcastic": "0", "sarcasm": "1"}])
    monkeypatch.setattr(sources, "_github", lambda *args: path)
    examples = sources.ISarcasm().examples()
    assert [row.state for row in examples] == tweets
    assert [row.answers for row in examples] == [{"sarcastic": "yes"}, {"sarcastic": "no"}]
    assert examples[0].questions["sarcastic"].wire() == {
        "type": "choice", "instructions": "Is this text intended to be sarcastic?",
        "criteria": {"no": "No", "yes": "Yes"}}


def test_default_sources_are_decontaminated_like_other_sources(sources, monkeypatch, tmp_path):
    from contamination import Evaluated, Overlaps

    default = sources.Conversion().mixture
    assert {"esci", "isarcasm"} <= dict(default.sources()).keys()
    assert not hasattr(sources.Conversion(), "selected")
    assert not hasattr(sources.Conversion(), "headroom")
    assert not hasattr(sources, "HellaSwag")
    assert "hellaswag" not in dict(default.sources())
    examples = [sources.ISarcasm.example({"tweet": text, "sarcastic": "1"})
                for text in ("What a wonderful day!", "Just what I needed.", "How surprising.")]
    monkeypatch.setattr(sources.Esci, "examples", lambda self: list(examples))
    monkeypatch.setattr(sources.ISarcasm, "examples", lambda self: list(examples))
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
           for entry in dataclasses.fields(default)}
    mixture = sources.Mixture(**{**off, "esci": dataclasses.replace(default.esci, held=1),
                                  "isarcasm": dataclasses.replace(default.isarcasm, held=1)})
    overlaps = Overlaps.of(Evaluated.request(row.state, {name: q.wire() for name, q in row.questions.items()})
                           for row in examples)
    made = sources.write(mixture, tmp_path, overlaps)
    assert made["rows"] == {"esci": {"train": 0, "held": 0}, "isarcasm": {"train": 0, "held": 0}}
    assert sum(entry["dropped"] for entry in made["decontamination"].values()) == 6


@pytest.mark.network
@pytest.mark.parametrize("name", ["esci", "isarcasm"])
def test_pinned_esci_and_isarcasm_have_the_released_train_counts(sources, name):
    if name == "isarcasm":
        examples = sources.ISarcasm().examples()
        assert len(examples) == 3468
        assert Counter(row.answers["sarcastic"] for row in examples) == {"yes": 867, "no": 2601}
    else:
        import pyarrow.parquet as pq

        source = sources.Esci(limit=15_000)
        path = sources._github(source.repo, source.revision,
                               "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
                               media=True)
        table = pq.read_table(path, columns=["product_locale"],
                              filters=[("split", "=", "train"), ("small_version", "=", 1)])
        locales = Counter(table["product_locale"].to_pylist())
        assert locales == {"us": 419653, "es": 152891, "jp": 209094}
        assert sum(locales.values()) == 781638
        train, held = source.split()
        assert (len(train), len(held)) == (15_000, 2_000)


def test_temperature_shares_follow_cube_roots(sources):
    assert sources.shares({"a": 1000, "b": 8000, "c": 27000}) == pytest.approx(
        {"a": 1 / 6, "b": 2 / 6, "c": 3 / 6})
    assert sources.shares({"empty": 0}) == {"empty": 0}
    assert sources.shares({}) == {}
    with pytest.raises(ValueError):
        sources.shares({"a": 1}, cap=0)


def test_temperature_shares_redistribute_after_ten_epochs(sources):
    counts = {"tiny": 1, "small": 10, "medium": 1000, "big": 8000}
    weights = sources.shares(counts)
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["tiny"] == pytest.approx(10 / 9011)
    assert weights["small"] == pytest.approx(100 / 9011)
    assert weights["big"] / weights["medium"] == pytest.approx(2)
    assert all(weights[name] * 9011 / count <= 10 + 1e-12 for name, count in counts.items())


def test_temperature_shares_apply_the_fifty_thousand_cap(sources):
    weights = sources.shares({"a": 400000, "b": 50000, "c": 6250}, cap=50000)
    assert weights == pytest.approx({"a": .4, "b": .4, "c": .2})
    assert sources.Conversion(cap=50000).cap == 50000


def test_written_weights_use_post_split_post_decontamination_counts_and_licences(
        sources, monkeypatch, tmp_path):
    from contamination import Evaluated, Overlaps

    default = sources.Mixture()
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0)
           for entry in dataclasses.fields(default)}
    seen = []

    def split(source):
        seen.append(source.limit)
        count = 8 if isinstance(source, sources.Intents) else 2
        prefix = type(source).__name__
        return [Example(f"{prefix} message number {index}", {"q": Noul("Spam?")}, {"q": "false"})
                for index in range(count)], []

    monkeypatch.setattr(sources.Intents, "split", split)
    monkeypatch.setattr(sources.SmsSpam, "split", split)
    mixture = sources.Mixture(**{**off, "banking77": sources.Intents(weight=900),
                                  "sms_spam": sources.SmsSpam(weight=.001)})
    overlaps = Overlaps.of([Evaluated(("SmsSpam message number 0",))])
    made = sources.write(mixture, tmp_path, overlaps, cap=50000)
    assert seen == [50000, 50000]
    assert made["rows"] == {"banking77": {"train": 8, "held": 0}, "sms_spam": {"train": 1, "held": 0}}
    assert made["weights"] == pytest.approx({"banking77": 2 / 3, "sms_spam": 1 / 3})
    assert (made["temperature"], made["cap"], made["max_epochs"]) == (3, 50000, 10)
    assert made["licences"]["sms_spam"] == {
        "name": "CC-BY-4.0", "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "revision": "cae486f927c250fe1d4a5b55f11357964ed1646c"}
    assert json.loads((tmp_path / "mixture.json").read_text()) == made
    assert default.phishing_email.weight == 0
    assert "phishing_email" not in dict(default.sources())
    assert made["licences"]["phishing_email"] == {
        "name": "LGPL-3.0-only",
        "url": ("https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/blob/"
                "34085a032c123ca237f314a01a67909cdea35e34/README.md"),
        "revision": "34085a032c123ca237f314a01a67909cdea35e34"}
    enabled = dataclasses.replace(default, phishing_email=sources.PhishingEmail(weight=1))
    assert "phishing_email" in dict(enabled.sources())
    for _, source in default.sources():
        licence = source.licence_record()
        assert licence["name"] and licence["url"].startswith("https://")
        assert len(licence["revision"]) == (64 if isinstance(source, sources.Fever) else 40)
    assert made["licences"]["fever"] == {
        "name": "CC-BY-SA-3.0", "url": "https://fever.ai/dataset/fever.html",
        "revision": "eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6",
        "wiki_revision": "4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2"}


def test_seeded_rows_are_capped_before_conversion(sources, monkeypatch):
    seen = []
    monkeypatch.setattr(sources, "_count", lambda *args: 200000)

    def rows(repo, revision, path, indices):
        seen.append(indices)
        return ({"question": f"Question {index}", "answer": bool(index % 2)} for index in sorted(indices))

    monkeypatch.setattr(sources, "_rows", rows)
    source = sources.StrategyQA(limit=3, held=2, seed=17)
    train, held = source.split()
    assert (len(train), len(held)) == (3, 2)
    assert len(seen[0]) == 5 and max(seen[0]) > 100000
    assert (train, held) == source.split()
    assert seen[0] == seen[1]
    assert {row.questions["answer"].instructions for row in train}.isdisjoint(
        row.questions["answer"].instructions for row in held)
    monkeypatch.setattr(sources.StrategyQA, "examples", lambda self: [
        Example(str(index), {"answer": Noul("Ready?")}, {"answer": "true"}) for index in range(3000)])
    assert [len(part) for part in sources.StrategyQA(limit=50000).split()] == [1000, 2000]


def test_open_jev_drops_customer_control_before_grouping(sources, monkeypatch):
    base = {"kind": "noul", "question": "Continue?", "options": ["no", "yes"], "target": [0., 1.],
            "state_json": '{"ready": true}'}
    rows = [{**base, "group_id": f"{group}:42:1"}
            for group in ("customer-control-v1", "workflow-controls-v1")]
    monkeypatch.setattr(sources, "_rows", lambda *args: iter(rows))
    assert len(sources.OpenJev().read("train")) == 1
    assert len(sources.OpenJev().read("calibration")) == 1


def test_phishing_csv_reads_long_fields_and_embedded_newlines(sources, monkeypatch, tmp_path):
    path = tmp_path / "emails.csv"
    message = 'A long "quoted" email\n' + "body " * 30000
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Email Text", "Email Type"])
        writer.writeheader()
        writer.writerows([{"Email Text": message, "Email Type": "Phishing Email"},
                          {"Email Text": "Plain", "Email Type": "Safe Email"}])
    monkeypatch.setattr(sources, "_download", lambda *args: str(path))
    csv.field_size_limit(131072)
    assert sources.PhishingEmail().count() == 2
    csv.field_size_limit(131072)
    examples = sources.PhishingEmail().examples()
    assert [row.state for row in examples] == [message, "Plain"]
    assert [row.answers for row in examples] == [{"phishing": "true"}, {"phishing": "false"}]


@pytest.mark.parametrize("name", ["Snli", "MultiNli"])
def test_inference_converters_keep_the_gold_relation(sources, monkeypatch, name):
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    source = getattr(sources, name)()
    for label, expected in enumerate(("entailment", "neutral", "contradiction")):
        example = source.convert({"premise": "Birds fly.", "hypothesis": "A bird flies.", "label": label})
        assert example.state == "Premise: Birds fly.\nHypothesis: A bird flies."
        assert example.questions["answer"].wire() == {
            "type": "choice", "instructions": "How does the hypothesis relate to the premise?",
            "criteria": {"entailment": None, "neutral": None, "contradiction": None}}
        assert example.answers == {"answer": expected}
    assert source.convert({"label": -1}) is None


def test_fever_includes_evidence_and_binary_gold_labels(sources, monkeypatch):
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    for label in ("SUPPORTS", "REFUTES"):
        example = sources.Fever().convert({"claim": "The city is old.", "label": label,
                                           "evidence": [{"page": "City", "text": "Founded in 1200."}]})
        assert example.state == "Claim: The city is old.\nEvidence: City: Founded in 1200."
        assert example.questions["answer"].wire() == {
            "type": "choice", "instructions": "Does the evidence support or refute the claim?",
            "criteria": {"SUPPORTS": None, "REFUTES": None}}
        assert example.answers == {"answer": label}
    assert sources.Fever().convert({"label": "NOT ENOUGH INFO"}) is None


def test_original_fever_samples_labelled_claims_and_joins_only_gold_sentences(
        sources, monkeypatch, tmp_path):
    claims_path, wiki_path = tmp_path / "train.jsonl", tmp_path / "wiki-pages.zip"
    claims = [
        {"id": 1, "label": "SUPPORTS", "claim": "First claim",
         "evidence": [[[1, 2, "A", 0], [1, 3, "B", 2]], [[4, 5, "A", 0]]]},
        {"id": 2, "label": "NOT ENOUGH INFO", "claim": "No evidence", "evidence": [[[None] * 4]]},
        {"id": 3, "label": "REFUTES", "claim": "Second claim", "evidence": [[[5, 6, "B", 1]]]},
        {"id": 4, "label": "SUPPORTS", "claim": "Third claim", "evidence": [[[7, 8, "Café", 0]]]},
    ]
    claims_path.write_text("".join(json.dumps(row) + "\n" for row in claims))
    pages = [{"id": "A", "lines": "0\tA zero\tignored link\n1\tA unused"},
             {"id": "B", "lines": "2\tB two\n0\tB unused\n1\tB one\tignored link"},
             {"id": unicodedata.normalize("NFD", "Café"), "lines": "0\tC zero"},
             {"id": "Unused", "lines": "0\tDo not retain this page"}]
    with zipfile.ZipFile(wiki_path, "w") as archive:
        archive.writestr("license.html", "Not a JSONL member")
        archive.writestr("__MACOSX/._wiki-001.jsonl", "Not a JSONL member")
        archive.writestr("wiki-pages/wiki-001.jsonl", "".join(json.dumps(page) + "\n" for page in pages))
    monkeypatch.setattr(sources, "_http", lambda url, digest, namespace:
                        wiki_path if url.endswith(".zip") else claims_path)
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    source = sources.Fever()
    assert source.label_counts == {"SUPPORTS": 2, "REFUTES": 1, "NOT ENOUGH INFO": 1}
    assert source.count() == 3
    assert source.sentences({"B": {1}}) == {("B", 1): "B one"}
    full = source.examples()
    assert len(full) == 3
    assert full[0].state == "Claim: First claim\nEvidence: A: A zero\nB: B two"
    assert full[2].state == "Claim: Third claim\nEvidence: Café: C zero"
    assert [row.answers for row in full] == [
        {"answer": "SUPPORTS"}, {"answer": "REFUTES"}, {"answer": "SUPPORTS"}]
    monkeypatch.setattr(sources, "_draw", lambda count, limit, held, seed: {1})
    sampled = sources.Fever(limit=1, held=0).examples()
    assert len(sampled) == 1
    assert sampled[0].state == "Claim: Second claim\nEvidence: B: B one"
    assert sampled[0].answers == {"answer": "REFUTES"}
    with pytest.raises(ValueError, match="missing gold evidence"):
        source.sentences({"B": {9}})


def test_fever_rejects_modified_cached_training_and_wiki_files(sources, monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "cached_assets_path", lambda **kwargs: tmp_path)
    (tmp_path / "train.jsonl").write_bytes(b"modified training claims")
    (tmp_path / "wiki-pages.zip").write_bytes(b"modified Wikipedia archive")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        list(sources.Fever().claims())
    with pytest.raises(ValueError, match="pinned SHA-256"):
        sources.Fever().sentences({"Page": {0}})


def test_release_download_verifies_bytes_before_caching(sources, monkeypatch, tmp_path):
    import urllib.request

    import huggingface_hub

    content = b"a pinned release"
    monkeypatch.setattr(huggingface_hub, "cached_assets_path", lambda **kwargs: tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    digest = hashlib.sha256(content).hexdigest()
    path = sources._http("https://example.com/release.zip", digest, "release")
    assert path.read_bytes() == content
    assert not (tmp_path / "release.zip.partial").exists()
    with pytest.raises(ValueError, match="pinned SHA-256"):
        sources._http("https://example.com/changed.zip", "0" * 64, "release")
    assert not (tmp_path / "changed.zip").exists()


@pytest.mark.parametrize(("name", "text_key", "label_key", "question", "answer_key", "yes", "no"), [
    ("SmsSpam", "sms", "label", "Is this message spam?", "spam", 1, 0),
    ("PhishingEmail", "Email Text", "Email Type", "Is this email a phishing attempt?", "phishing",
     "Phishing Email", "Safe Email"),
])
def test_binary_text_converters(sources, name, text_key, label_key, question, answer_key, yes, no):
    for label, gold in ((yes, "true"), (no, "false")):
        row = {text_key: "A message.\nWith two lines.", label_key: label}
        example = getattr(sources, name)().convert(row)
        assert example.state == "A message.\nWith two lines."
        assert example.questions[answer_key].wire() == {"type": "noul", "instructions": question}
        assert example.answers == {answer_key: gold}


def test_paws_uses_both_sentences_and_binary_gold(sources):
    for label, gold in ((1, "true"), (0, "false")):
        example = sources.Paws().convert({"sentence1": "A follows B.", "sentence2": "B follows A.",
                                         "label": label})
        assert example.state == {"sentence1": "A follows B.", "sentence2": "B follows A."}
        assert example.questions["paraphrase"].wire() == {
            "type": "noul", "instructions": "Do these sentences express the same meaning?"}
        assert example.answers == {"paraphrase": gold}


@pytest.mark.parametrize(("name", "labels", "instructions"), [
    ("GoEmotions", ["admiration", "amusement", "neutral"], "Which emotion is expressed in the text?"),
    ("UnfairTos", ["Limitation of liability", "Unilateral termination", "Unilateral change"],
     "Which unfair clause type applies to this text?"),
])
def test_multilabel_converters_ask_every_candidate(sources, monkeypatch, name, labels, instructions):
    monkeypatch.setattr(sources.Source, "labels", lambda self, column: labels)
    source = getattr(sources, name)()
    single = source.convert({"text": "Text", "labels": [1]})
    assert single.state == "Text"
    assert single.questions["label"].wire() == {
        "type": "choice", "instructions": instructions, "criteria": dict.fromkeys(labels)}
    assert single.answers == {"label": labels[1]}
    for true in ([], [0, 2]):
        example = source.convert({"text": "Text", "labels": true})
        assert len(example.questions) == 3
        for index, label in enumerate(labels):
            question = example.questions[f"label_{index}"]
            assert question.wire() == {"type": "noul",
                                       "instructions": f"{instructions} Does this label apply? {label}"}
            assert example.answers[f"label_{index}"] == ("true" if index in true else "false")


def test_civil_comments_preserves_every_annotator_fraction(sources):
    values = {"toxicity": .3, "severe_toxicity": .1, "obscene": .2, "threat": .4,
              "insult": .7, "identity_attack": 0., "sexual_explicit": 1.}
    example = sources.CivilComments().convert({"text": "Comment", **values})
    assert example.state == "Comment" and not example.answers
    assert set(example.questions) == set(values)
    for name, value in values.items():
        assert example.questions[name].wire() == {
            "type": "noul", "instructions": f"Does this comment contain {name.replace('_', ' ')}?"}
        assert example.targets[name] == pytest.approx((1 - value, value))


@pytest.mark.parametrize(("name", "input_row", "state", "instructions", "labels", "gold"), [
    ("DBpedia", {"title": "Acme", "content": "A company.", "label": 0}, "Acme\nA company.",
     "Which category describes this entity?", ["Company", "EducationalInstitution", "Artist"], "Company"),
    ("Ledgar", {"text": "The parties agree.", "label": 1}, "The parties agree.",
     "Which type of contract clause is this?", ["Adjustments", "Agreements", "Amendments"], "Agreements"),
    ("MassiveIntent", {"text": "Wake me.", "label": 2}, "Wake me.",
     "Which intent does this request express?", ["datetime_query", "iot_hue_lightchange", "alarm_set"],
     "alarm_set"),
    ("Snips", {"text": "Play jazz.", "category": "PlayMusic"}, "Play jazz.",
     "Which intent does this request express?", ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                                                "RateBook", "SearchCreativeWork", "SearchScreeningEvent"],
     "PlayMusic"),
    ("BiasInBios", {"hard_text": "He writes software.", "profession": 24, "gender": 1}, "He writes software.",
     "What is this person's occupation?", ["accountant", "architect", "attorney", "chiropractor", "comedian",
         "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist", "model",
         "nurse", "painter", "paralegal", "pastor", "personal_trainer", "photographer", "physician", "poet",
         "professor", "psychologist", "rapper", "software_engineer", "surgeon", "teacher", "yoga_teacher"],
     "software_engineer"),
])
def test_framed_classifiers(sources, monkeypatch, name, input_row, state, instructions, labels, gold):
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    source = getattr(sources, name)()
    source.__dict__["classes"] = labels
    example = source.convert(input_row)
    assert example.state == state
    assert example.questions["answer"].wire() == {
        "type": "choice", "instructions": instructions, "criteria": dict.fromkeys(labels)}
    assert example.answers == {"answer": gold}
    column = "profession" if name == "BiasInBios" else "category" if name == "Snips" else "label"
    second = source.convert({**input_row, column: labels[-1] if name == "Snips" else len(labels) - 1})
    assert second.state == state
    assert second.questions == example.questions
    assert second.answers == {"answer": labels[-1]}


@pytest.mark.parametrize(("name", "count"), [("Qasc", 8), ("CommonsenseQA", 5)])
def test_choice_qa_converters_use_answer_keys_not_positions(sources, monkeypatch, name, count):
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    texts = [f"Answer {index}" for index in range(count)]
    labels = list(reversed([chr(65 + index) for index in range(count)]))
    for index in (0, count - 1):
        example = getattr(sources, name)().convert({"question": "Why?", "answerKey": labels[index],
            "choices": {"label": labels, "text": texts}, "fact1": "Hidden gold", "combinedfact": "Hidden"})
        assert example.state == "Why?"
        assert example.questions["answer"].wire() == {
            "type": "choice", "instructions": "Which answer is correct?", "criteria": dict.fromkeys(texts)}
        assert example.answers == {"answer": texts[index]}


def test_medmcqa_uses_zero_based_gold_without_explanation(sources, monkeypatch):
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    for index in (0, 3):
        example = sources.MedMCQA().convert({"question": "Diagnosis?", "opa": "A", "opb": "B",
                                            "opc": "C", "opd": "D", "cop": index, "exp": "Secret"})
        assert example.state == "Diagnosis?"
        assert example.questions["answer"].wire() == {
            "type": "choice", "instructions": "Which answer is correct?", "criteria": dict.fromkeys("ABCD")}
        assert example.answers == {"answer": "ABCD"[index]}


def test_pubmedqa_uses_labeled_context_not_long_answer(sources, monkeypatch):
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1. if salt == "keys" else 0.)
    for label in ("yes", "no", "maybe"):
        example = sources.PubMedQA().convert({"question": "Does it work?",
                                              "context": {"contexts": ["A", "B"]},
                                              "long_answer": "Hidden answer", "final_decision": label})
        assert example.state == "A\nB"
        assert example.questions["answer"].wire() == {
            "type": "choice", "instructions": "Does it work?",
            "criteria": dict.fromkeys(["yes", "no", "maybe"])}
        assert example.answers == {"answer": label}


def test_strategyqa_omits_gold_facts(sources):
    for label in (False, True):
        example = sources.StrategyQA().convert({"question": "Is it older?", "answer": label,
                                                "facts": "Secret"})
        assert example.state == ""
        assert example.questions["answer"].wire() == {"type": "noul", "instructions": "Is it older?"}
        assert example.answers == {"answer": str(label).lower()}


def test_scientsbank_keeps_the_five_categorical_grades(sources):
    labels = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]
    for index, label in enumerate(labels):
        example = sources.SciEntsBank().convert({"question": "Why?", "reference_answer": "A",
                                                 "student_answer": "B",
                                                 "label": index})
        assert example.state == {"question": "Why?", "reference_answer": "A", "student_answer": "B"}
        question = example.questions["grade"]
        assert isinstance(question, Choice) and question.options == tuple(labels)
        assert question.instructions == (
            "How should the student's answer be graded against the reference answer?")
        assert question.descriptions == ("Correct", "Contradicts the reference answer",
                                          "Partially correct but incomplete",
                                          "Irrelevant", "Not in the question's domain")
        assert example.answers == {"grade": label}


def test_helpsteer2_asks_five_independent_ordinal_ratings(sources):
    ratings = {"helpfulness": 0, "correctness": 1, "coherence": 2, "complexity": 3, "verbosity": 4}
    example = sources.HelpSteer2().convert({"prompt": "Explain", "response": "An explanation", **ratings})
    assert example.state == {"prompt": "Explain", "response": "An explanation"}
    assert example.answers == ratings
    assert set(example.questions) == set(ratings)
    for name, question in example.questions.items():
        assert isinstance(question, Score)
        assert question.instructions == f"Rate the response's {name} on a scale from 0 to 4."
        assert question.options == ("0", "1", "2", "3", "4")
    assert example.questions["verbosity"].descriptions == (
        "Very brief", "Brief", "Moderate length", "Detailed", "Very detailed")


def test_hate_speech_groups_annotator_votes_by_comment(sources, monkeypatch):
    rows = [{"comment_id": 1, "text": "First", "hatespeech": label} for label in (0., 1., 2., 2.)]
    rows.append({"comment_id": 2, "text": "Second", "hatespeech": 0.})
    monkeypatch.setattr(sources, "_rows", lambda *args: iter(rows))
    examples = sources.HateSpeech().examples()
    assert sources.HateSpeech().count() == 2
    assert len(examples) == 2
    assert [row.state for row in examples] == ["First", "Second"]
    assert examples[0].targets == {"hatespeech": (.25, .25, .5)}
    assert examples[1].targets == {"hatespeech": (1., 0., 0.)}
    question = examples[0].questions["hatespeech"]
    assert isinstance(question, Score)
    assert question.descriptions == ("No", "Unclear", "Yes")
    assert question.instructions == (
        "Does this comment contain hate speech, defined as bias-motivated, hostile and malicious language "
        "targeted at a person/group because of their actual or perceived innate characteristics, "
        "especially when the group is unnecessarily labeled?")


def test_lavoir_keeps_exact_posteriors_in_option_order_and_hides_profiles(sources):
    question = {"type": "choice", "instructions": "Which team?", "criteria": {"a": "Team A", "b": "Team B"}}
    state = [{"role": "user", "text": "Help with a refund."}]
    example = sources.Lavoir().convert({"state": state, "question": question, "target": {"b": .75, "a": .25},
                                       "gold": "a", "profile": {"hidden": True}, "oracle_voi": {"x": .2}})
    assert example.state == state
    assert example.questions["route"].wire() == question
    assert example.targets == {"route": (.25, .75)} and not example.answers
    certain = sources.Lavoir().convert({"state": state, "question": question, "target": {"a": 1., "b": 0.}})
    assert certain.state == state and certain.questions["route"].wire() == question
    assert certain.targets == {"route": (1., 0.)}


def test_winogrande_reproduces_the_kits_empty_state_and_letter_options(sources):
    for answer, expected in (("1", "A"), ("2", "B")):
        example = sources.WinoGrande().convert({"sentence": "_ is taller.", "option1": "Ava", "option2": "Bo",
                                                "answer": answer})
        assert example.state == {}
        assert example.questions["q1"].wire() == {"type": "choice",
            "instructions": "Which option correctly fills the blank?\n_ is taller.",
            "criteria": {"A": "Ava", "B": "Bo"}}
        assert example.answers == {"q1": expected}


def test_sgd_emits_only_user_frames_and_no_future_or_hidden_state(sources, monkeypatch, tmp_path):
    schema = {"service_name": "hotel", "intents": [{"name": "Book"}], "slots": []}
    path = tmp_path / "schema.json"
    path.write_text(json.dumps([schema]))
    dialogue = {"turns": [
        {"speaker": "USER", "utterance": "Book a room", "frames": [
            {"service": "hotel", "state": {"active_intent": "Book", "slot_values": {"secret": ["hidden"]}}}]},
        {"speaker": "SYSTEM", "utterance": "Where?", "frames": []},
        {"speaker": "USER", "utterance": "Never mind", "frames": [
            {"service": "hotel", "state": {"active_intent": "NONE"}}]},
    ]}
    monkeypatch.setattr(sources, "_github", lambda *args: path)
    monkeypatch.setattr(sources.Sgd, "dialogues", lambda self: iter([dialogue]))
    examples = sources.Sgd().examples()
    assert len(examples) == 2
    assert examples[0].state == {"history": [{"speaker": "USER", "utterance": "Book a room"}],
                                  "service": "hotel", "schema": schema, "task": "SGD current-service intent"}
    assert len(examples[1].state["history"]) == 3
    assert [row.answers for row in examples] == [{"intent": "Book"}, {"intent": "NONE"}]
    assert examples[0].questions["intent"].wire() == {"type": "choice", "instructions":
        "Using the dialogue history and service schema in state, choose the active intent for this service. "
        "Choose NONE when no service intent is active. Do not use future turns or hidden labels.",
        "criteria": {"Book": "Book", "NONE": "NONE"}}


def test_contractnli_reads_only_train_and_groups_hypotheses(sources, monkeypatch, tmp_path):
    path = tmp_path / "contracts.zip"
    labels = {"nda": {"hypothesis": "Keep it secret."}, "sale": {"hypothesis": "You may sell it."}}
    data = {"labels": labels, "documents": [{"text": "Contract text", "annotation_sets": [
        {"annotations": {"nda": {"choice": "Entailment"}, "sale": {"choice": "NotMentioned"}}}]}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("contract-nli/train.json", json.dumps(data))
        archive.writestr("contract-nli/test.json", "Do not read evaluation data")
    monkeypatch.setattr(sources, "_github", lambda *args: path)
    example, = sources.ContractNli().examples()
    assert example.state == "Contract text"
    assert example.answers == {"nda": "Entailment", "sale": "NotMentioned"}
    for name, label in labels.items():
        assert example.questions[name].wire() == {"type": "choice",
            "instructions": ("Classify the relationship between the contract and this hypothesis:\n" +
                             label["hypothesis"]),
            "criteria": {"Entailment": "The contract entails the hypothesis.",
                         "Contradiction": "The contract contradicts the hypothesis.",
                         "NotMentioned": ("The hypothesis is neither entailed nor contradicted "
                                          "by the contract.")}}


def test_hover_joins_complete_articles_in_nfd_without_duplicate_titles(sources, monkeypatch, tmp_path):
    path = tmp_path / "wiki.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE documents (id TEXT, text TEXT)")
        db.execute("INSERT INTO documents VALUES (?, ?)",
                   (unicodedata.normalize("NFD", "Café"), "Full article."))
    claims = [{"claim": "There is a cafe.", "label": label, "supporting_facts": [["Café", 0], ["Café", 1]]}
              for label in ("SUPPORTED", "NOT_SUPPORTED")]
    monkeypatch.setattr(sources, "_evidence_database", lambda: path)
    monkeypatch.setattr(sources.Hover, "claims", lambda self: claims)
    examples = sources.Hover().examples()
    for example, claim in zip(examples, claims, strict=True):
        assert example.state == {"claim": "There is a cafe.",
                                  "evidence": [{"title": "Café", "text": "Full article."}]}
        assert example.questions["q"].wire() == {"type": "choice",
            "instructions": "Is the claim supported by the evidence?",
            "criteria": {"SUPPORTED": "The evidence supports the claim.",
                         "NOT_SUPPORTED": "The evidence does not support the claim."}}
        assert example.answers == {"q": claim["label"]}


def test_hover_rejects_a_database_with_a_wrong_hash(sources, monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "cached_assets_path", lambda **kwargs: tmp_path)
    (tmp_path / "wiki_wo_links.db").write_bytes(b"wrong database")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        sources._evidence_database()


TRAIN_COUNTS = {
    "winogrande": 40398, "go_emotions": 43410, "dbpedia": 560000, "civil_comments": 1804874,
    "sms_spam": 5574, "paws": 49401, "snli": 550152, "mnli": 392702,
    "pubmedqa": 1000, "medmcqa": 182822, "qasc": 8134, "bias_in_bios": 257478,
    "massive_intent": 11514, "helpsteer2": 20324, "hate_speech_scales": 135556,
    "commonsense_qa": 9741, "sci_ents_bank": 4969, "ledgar": 60000, "unfair_tos": 5532,
    "strategyqa": 1603, "snips": 13084, "phishing_email": 18650, "lavoir_dialogues": 21601,
}


@pytest.mark.network
@pytest.mark.parametrize(("name", "count"), TRAIN_COUNTS.items())
def test_new_pinned_train_row_counts(sources, name, count):
    source = getattr(sources.Mixture(), name)
    if name == "hate_speech_scales":
        assert sources.Source.count(source) == count
        assert source.count() == 39565
    else:
        assert source.count() == count
    sample = dataclasses.replace(source, limit=3, held=0).examples()
    assert 1 <= len(sample) <= 3
    assert all(row.answers or row.targets for row in sample)


@pytest.mark.network
def test_sgd_pinned_train_dialogue_count(sources):
    assert sum(1 for _ in sources.Sgd().dialogues()) == 16142


@pytest.mark.network
def test_contractnli_pinned_train_document_and_decision_counts(sources):
    examples = sources.ContractNli().examples()
    assert len(examples) == 423
    assert sum(len(row.questions) for row in examples) == 7191


@pytest.mark.network
def test_hover_pinned_train_claim_count_and_evidence(sources):
    assert len(sources.Hover().claims()) == 18171
    examples = sources.Hover(limit=2, held=0).examples()
    assert len(examples) == 2
    assert all(row.state["evidence"] for row in examples)


@pytest.mark.network
def test_original_fever_pinned_training_and_support_refute_counts(sources):
    source = sources.Fever(limit=3, held=0)
    assert source.label_counts == {"SUPPORTS": 80035, "REFUTES": 29775, "NOT ENOUGH INFO": 35639}
    assert sum(source.label_counts.values()) == 145449
    assert source.count() == 109810
    examples = source.examples()
    assert len(examples) == 3
    assert all("Evidence:" in (row.state or row.questions["answer"].instructions) for row in examples)
    assert all(set(row.questions["answer"].options) <= {"SUPPORTS", "REFUTES", "A", "B"}
               for row in examples)
