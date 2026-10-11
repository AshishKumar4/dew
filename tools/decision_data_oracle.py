#!/usr/bin/env python3
"""Compare the published gold mix with the pinned eager fit on remote compute."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASELINE = "66beb53059c42f5226a413e26c600655f084f4fa"
TOKENIZER = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def _worker(options):
    from dew.data.text import HFTokenizer
    from dew.decision import DecisionObjective, Example, JointLayout, Specials
    from dew.nn.backbones.causal_transformer import CausalTransformer

    tokenizer = HFTokenizer(str(options.tokenizer), local_files_only=True)
    layout = JointLayout(max_len=3072, max_state_tokens=1024)
    specials = Specials.of(tokenizer)
    objective = DecisionObjective(
        CausalTransformer(vocab_size=tokenizer.vocab_size, emb_features=8, num_layers=1, num_heads=1,
                          mlp_features=16), tokenizer=tokenizer, layout=layout, specials=specials)
    reports = {}
    for split in ("train", "held"):
        path = options.work / f"{split}.jsonl"
        kept_hash, scoring_hash = hashlib.sha256(), hashlib.sha256()
        attempted = accepted = 0
        dropped = []
        with path.open() as source:
            while lines := list(itertools.islice(source, 2048)):
                examples = [Example.of(json.loads(line)) for line in lines]
                if options.worker == "legacy":
                    from dew.decision.config import _fitting
                    from dew.decision.data import Weighted

                    try:
                        weighted, _, _ = _fitting(objective, {options.corpus: Weighted(examples, 1.)}, [])
                        kept = {id(example) for example in weighted[options.corpus].examples if example.labels()}
                    except ValueError as error:
                        if "no row of" not in str(error):
                            raise
                        kept = set()
                else:
                    from dew.data.pipeline import Rejected
                    from dew.decision.objective import Fit

                    fitting = Fit(layout, tokenizer, specials)
                    kept = {id(example) for example in examples
                            if not isinstance(fitting.map({"example": example}), Rejected)}
                for offset, example in enumerate(examples, start=attempted):
                    identity = f"{options.corpus}/{split}/{offset}\n".encode()
                    if id(example) in kept:
                        kept_hash.update(identity)
                        if split == "held" and accepted < 200:
                            scoring_hash.update(identity)
                        accepted += 1
                    else:
                        dropped.append(offset)
                attempted += len(examples)
        reports[split] = {"attempted": attempted, "accepted": accepted, "dropped": dropped,
                          "accepted_ids": kept_hash.hexdigest(), "scoring_ids": scoring_hash.hexdigest()}
    options.output.write_text(json.dumps(reports, sort_keys=True) + "\n")


def _download(root, corpus, work):
    from dew.data.storage import Store

    published = Store(root)
    manifest = published.child("manifest.json").read()
    expected = published.child("manifest.sha256").read().decode().strip()
    if hashlib.sha256(manifest).hexdigest() != expected:
        raise ValueError("decision manifest SHA256 differs from manifest.sha256")
    stated = json.loads(published.child("mixture.json").read())
    files = stated["files"][corpus]
    for split in ("train", "held"):
        digest = hashlib.sha256()
        entry = files.get(split)
        with (work / f"{split}.jsonl").open("wb") as output:
            if entry is None:
                continue
            for shard in entry["shards"]:
                payload = published.child(shard["path"]).read()
                if len(payload) != shard["bytes"] or hashlib.sha256(payload).hexdigest() != shard["sha256"]:
                    raise ValueError(f"corrupt decision shard {shard['path']}")
                digest.update(payload)
                output.write(payload)
        if entry is not None and digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"corpus {corpus}/{split} concatenated checksum differs")
    return expected, stated["weights"][corpus]


def _compare(options):
    from huggingface_hub import snapshot_download

    tokenizer = Path(snapshot_download("Qwen/Qwen3.5-2B", revision=TOKENIZER,
                     allow_patterns=["*.json", "*.jinja", "*.model", "*.txt"]))
    script = Path(__file__).resolve()
    checkout = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="decision-oracle-", dir=checkout) as directory:
        work = Path(directory)
        manifest, weight = _download(options.root, options.corpus, work)
        archive = work / "baseline.tar.gz"
        urllib.request.urlretrieve(f"https://codeload.github.com/AshishKumar4/dew/tar.gz/{BASELINE}", archive)
        with tarfile.open(archive) as source:
            source.extractall(work, filter="data")
        baseline = work / f"dew-{BASELINE}"
        reports = []
        for mode, root in (("legacy", baseline), ("pipeline", checkout)):
            report = work / f"{mode}.json"
            subprocess.run([sys.executable, str(script), "--worker", mode, "--work", str(work),
                            "--tokenizer", str(tokenizer), "--corpus", options.corpus, "--output", str(report)],
                           cwd=root, check=True, env={**os.environ, "PYTHONPATH": str(root / "src"),
                                                     "JAX_PLATFORMS": "cpu"})
            reports.append(json.loads(report.read_text()))
        if reports[0] != reports[1]:
            raise ValueError(f"decision corpus {options.corpus} differs: {reports}")
        options.output.write_text(json.dumps({"corpus": options.corpus, "weight": weight,
                                              "manifest": manifest, "baseline": BASELINE,
                                              "tokenizer": TOKENIZER, "equal": True, **reports[1]},
                                             sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="r2://dew-data/decision/gold-v3")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("legacy", "pipeline"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    options = parser.parse_args()
    if options.worker:
        _worker(options)
    else:
        _compare(options)


if __name__ == "__main__":
    main()
