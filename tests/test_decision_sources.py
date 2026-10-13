"""Pinned decision sources, their held-out splits, and their training requests."""

import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

import pytest

from dew.decision import Choice, Example, Noul

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


def test_url_producer_shards_without_changing_training_or_held_bytes(sources, monkeypatch):
    import numpy as np

    from dew.data.storage import Store

    question = Choice("Pick", ["a", "b"])
    examples = [Example(str(index), {"q": question}, {"q": "a"}) for index in range(20)]
    monkeypatch.setattr(sources.Intents, "examples", lambda self: list(examples))
    default = sources.Mixture()
    off = {
        entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
        for entry in dataclasses.fields(default)
    }
    mixture = sources.Mixture(**{**off, "banking77": sources.Intents(held=4)})
    root = Store("memory://decision-producer/shards")
    made = sources.write(mixture, root.url, shard_bytes=400)
    for split, expected in zip(("train", "held"), mixture.banking77.split(), strict=True):
        part = made["files"]["banking77"][split]
        encoded = b"".join(root.child(shard["path"]).read() for shard in part["shards"])
        golden = "".join(
            json.dumps(sources.row(example), ensure_ascii=False) + "\n" for example in expected
        ).encode()
        assert encoded == golden
        assert part["sha256"] == hashlib.sha256(golden).hexdigest()
        index = np.load(io.BytesIO(root.child(part["index"]["path"]).read()), allow_pickle=False)
        for position, entry in enumerate(index):
            data = root.child(part["shards"][int(entry["shard"])]["path"]).read()
            start = int(entry["offset"])
            row = json.loads(data[start : start + int(entry["bytes"])])
            assert row == sources.row(expected[position])
        assert len(part["shards"]) > 1
    assert made["slots"] == {"questions": 1, "options": 2, "total_options": 2, "choice_questions": 1}
    manifest = root.child("manifest.json").read()
    assert root.child("manifest.sha256").read().strip().decode() == hashlib.sha256(manifest).hexdigest()


def test_sparse_slot_descriptor_references_unchanged_raw_data(sources, monkeypatch, tmp_path):
    examples = [
        Example("one", {"a": Choice("Pick", ["x", "y", "z"]), "b": Noul("Yes?")}, {"a": "x", "b": "true"}),
        Example(
            "two",
            {"a": Choice("Pick", ["x", "y"]), "b": Choice("Pick", ["a", "b", "c", "d"])},
            {"a": "x", "b": "a"},
        ),
    ]
    monkeypatch.setattr(sources.Intents, "examples", lambda self: list(examples))
    default = sources.Mixture()
    off = {
        entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
        for entry in dataclasses.fields(default)
    }
    mixture = sources.Mixture(**{**off, "banking77": sources.Intents(held=0)})
    sources.write(mixture, tmp_path)
    original = (tmp_path / "manifest.json").read_bytes()
    raw = (tmp_path / "banking77.jsonl").read_bytes()
    command = [sys.executable, "tools/measurements/data-pipeline/decision_metadata.py", "--source",
               str(tmp_path)]
    subprocess.run([*command, "--set", "banking77"], check=True, capture_output=True, text=True)
    published = subprocess.run([*command, "--publish"], check=True, capture_output=True, text=True)
    metadata = json.loads(published.stdout)
    descriptor = json.loads(Path(metadata["url"]).read_bytes())
    assert descriptor["slots"] == {"questions": 2, "options": 4, "total_options": 6, "choice_questions": 2}
    assert descriptor["provenance"]["original_manifest_sha"] == hashlib.sha256(original).hexdigest()
    assert descriptor["provenance"]["raw_shards"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / "manifest.json").read_bytes() == original
    assert (tmp_path / "banking77.jsonl").read_bytes() == raw


def test_isarcasm_reads_csv_and_uses_the_binary_target(sources, monkeypatch, tmp_path):
    path = tmp_path / "train.En.csv"
    tweets = ['Wonderful, another "perfect" day.\nReally.', "An ordinary afternoon."]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["tweet", "sarcastic", "sarcasm"])
        writer.writeheader()
        writer.writerows(
            [
                {"tweet": tweets[0], "sarcastic": "1", "sarcasm": "0"},
                {"tweet": tweets[1], "sarcastic": "0", "sarcasm": "1"},
            ]
        )
    monkeypatch.setattr(sources, "_github", lambda *args: path)
    examples = sources.ISarcasm().examples()
    assert [row.state for row in examples] == tweets
    assert [row.answers for row in examples] == [{"sarcastic": "yes"}, {"sarcastic": "no"}]


def test_phishing_csv_reads_long_fields_and_embedded_newlines(sources, monkeypatch, tmp_path):
    path = tmp_path / "emails.csv"
    message = 'A long "quoted" email\n' + "body " * 30000
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Email Text", "Email Type"])
        writer.writeheader()
        writer.writerows(
            [
                {"Email Text": message, "Email Type": "Phishing Email"},
                {"Email Text": "Plain", "Email Type": "Safe Email"},
            ]
        )
    monkeypatch.setattr(sources, "_download", lambda *args: str(path))
    csv.field_size_limit(131072)
    assert sources.PhishingEmail().count() == 2
    csv.field_size_limit(131072)
    examples = sources.PhishingEmail().examples()
    assert [row.state for row in examples] == [message, "Plain"]
    assert [row.answers for row in examples] == [{"phishing": "true"}, {"phishing": "false"}]


def test_original_fever_samples_labelled_claims_and_joins_only_gold_sentences(
        sources, monkeypatch, tmp_path):
    claims_path, wiki_path = tmp_path / "train.jsonl", tmp_path / "wiki-pages.zip"
    claims = [
        {
            "id": 1,
            "label": "SUPPORTS",
            "claim": "First claim",
            "evidence": [[[1, 2, "A", 0], [1, 3, "B", 2]], [[4, 5, "A", 0]]],
        },
        {"id": 2, "label": "NOT ENOUGH INFO", "claim": "No evidence", "evidence": [[[None] * 4]]},
        {"id": 3, "label": "REFUTES", "claim": "Second claim", "evidence": [[[5, 6, "B", 1]]]},
        {"id": 4, "label": "SUPPORTS", "claim": "Third claim", "evidence": [[[7, 8, "Café", 0]]]},
    ]
    claims_path.write_text("".join(json.dumps(row) + "\n" for row in claims))
    pages = [
        {"id": "A", "lines": "0\tA zero\tignored link\n1\tA unused"},
        {"id": "B", "lines": "2\tB two\n0\tB unused\n1\tB one\tignored link"},
        {"id": unicodedata.normalize("NFD", "Café"), "lines": "0\tC zero"},
        {"id": "Unused", "lines": "0\tDo not retain this page"},
    ]
    with zipfile.ZipFile(wiki_path, "w") as archive:
        archive.writestr("license.html", "Not a JSONL member")
        archive.writestr("__MACOSX/._wiki-001.jsonl", "Not a JSONL member")
        archive.writestr("wiki-pages/wiki-001.jsonl", "".join(json.dumps(page) + "\n" for page in pages))

    def whole_member(*args, **kwargs):
        raise AssertionError("Wiki JSONL members must be streamed, not read whole")

    monkeypatch.setattr(zipfile.ZipFile, "read", whole_member)
    monkeypatch.setattr(
        sources, "_http", lambda url, digest, namespace: wiki_path if url.endswith(".zip") else claims_path
    )
    monkeypatch.setattr(sources, "_unit", lambda text, salt: 1.0 if salt == "keys" else 0.0)
    source = sources.Fever()
    assert source.label_counts == {"SUPPORTS": 2, "REFUTES": 1, "NOT ENOUGH INFO": 1}
    assert source.count() == 3
    assert source.sentences({"B": {1}}) == {("B", 1): "B one"}
    full = source.examples()
    assert len(full) == 3
    assert full[0].state == "Claim: First claim\nEvidence: A: A zero\nB: B two"
    assert full[2].state == "Claim: Third claim\nEvidence: Café: C zero"
    assert [row.answers for row in full] == [
        {"answer": "SUPPORTS"},
        {"answer": "REFUTES"},
        {"answer": "SUPPORTS"},
    ]
    monkeypatch.setattr(sources, "_draw", lambda count, limit, held, seed: {1})
    sampled = sources.Fever(limit=1, held=0).examples()
    assert len(sampled) == 1
    assert sampled[0].state == "Claim: Second claim\nEvidence: B: B one"
    assert sampled[0].answers == {"answer": "REFUTES"}
    with pytest.raises(ValueError):
        source.sentences({"B": {9}})


def test_fever_rejects_modified_cached_training_and_wiki_files(sources, monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "cached_assets_path", lambda **kwargs: tmp_path)
    (tmp_path / "train.jsonl").write_bytes(b"modified training claims")
    (tmp_path / "wiki-pages.zip").write_bytes(b"modified Wikipedia archive")
    with pytest.raises(ValueError):
        list(sources.Fever().claims())
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        sources._http("https://example.com/changed.zip", "0" * 64, "release")
    assert not (tmp_path / "changed.zip").exists()


def test_contractnli_reads_only_train_and_groups_hypotheses(sources, monkeypatch, tmp_path):
    path = tmp_path / "contracts.zip"
    labels = {"nda": {"hypothesis": "Keep it secret."}, "sale": {"hypothesis": "You may sell it."}}
    data = {
        "labels": labels,
        "documents": [
            {
                "text": "Contract text",
                "annotation_sets": [
                    {"annotations": {"nda": {"choice": "Entailment"}, "sale": {"choice": "NotMentioned"}}}
                ],
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("contract-nli/train.json", json.dumps(data))
        archive.writestr("contract-nli/test.json", "Do not read evaluation data")
    monkeypatch.setattr(sources, "_github", lambda *args: path)
    (example,) = sources.ContractNli().examples()
    assert example.state == "Contract text"
    assert example.answers == {"nda": "Entailment", "sale": "NotMentioned"}


def test_hover_rejects_a_database_with_a_wrong_hash(sources, monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "cached_assets_path", lambda **kwargs: tmp_path)
    (tmp_path / "wiki_wo_links.db").write_bytes(b"wrong database")
    with pytest.raises(ValueError):
        sources._evidence_database()
