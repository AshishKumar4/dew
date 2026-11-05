"""Scratch: what a trivial task's record holds, entirely."""

import json

from armada_support import cached, interpreter, loaded, written


def test_probe(tmp_path, monkeypatch, cache_worker):
    ci = loaded(tmp_path, monkeypatch)
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
    written(tmp_path, {"test_a.py": "def test_a():\n    pass\n"})
    interpreter(tmp_path, "3.12")
    ci.task("3.12", ["tests/test_a.py"], "", tmp_path / "verdict.json", 600.0)
    for key, body in stored.items():
        if body and "-green/" not in key and "inputs" not in key and key.endswith("/whole"):
            record = json.loads(body)
            for item in record["depends"] or []:
                print("DEPENDS", item)
