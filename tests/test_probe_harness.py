"""Scratch: what a task's record holds of the harness (tools/armada) it executed."""

import json

from armada_support import cached, copied_tools, interpreter, written


def test_probe(tmp_path, monkeypatch, cache_worker):
    ci = copied_tools(tmp_path, monkeypatch)
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
    written(tmp_path, {"test_a.py": "def test_a():\n    pass\n"})
    interpreter(tmp_path, "3.12")
    ci.task("3.12", ["tests/test_a.py"], "", tmp_path / "verdict.json", 600.0)
    found = [json.loads(body) for key, body in stored.items() if body and "-green/" not in key and "inputs" not in key
             and key.endswith("/whole")]
    for record in found:
        for item in record["depends"] or []:
            if "tools" in item or "armada" in item or "session" in item or "followed" in item or "lifetime" in item:
                print("DEPENDS", item)
        print("DEPENDS-COUNT", len(record["depends"] or []))
    assert found
