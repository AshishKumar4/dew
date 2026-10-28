"""The Hub census's loads without weights (tools/hf_coverage.py `loads`, ledger 1891), over a Hub on the
loopback (tests/local_hub.py), which keeps every request it answers."""

import json
import os
from pathlib import Path

from local_hub import LocalHub
from tool_support import load

FIXTURES = Path(__file__).parent / "fixtures" / "hf"


def test_the_census_loads_each_model_without_its_weights_and_rows_each_refusal(tmp_path):
    """Over a census of a local Hub's models, `loads` loads a published decoder (llama-tiny's files)
    without its weights, its row its parameter count, as a load of the same files on disk counts it;
    refuses a model of a type no family maps, its row naming the refusal; and rows a repo the Hub does
    not hold, and one whose config the census could not read, as the Hub's answer. Each load read its
    weight files' safetensors headers alone: what the Hub sent of the decoder's model.safetensors was
    ranges of it, less than the file, and no weight file was cached (the boundary 1888's network test
    checks on the Hub itself)."""
    import jax

    from dew.interop import Pretrained

    coverage = load("hf_coverage")
    source = FIXTURES / "llama-tiny"
    decoder = {name: source / name for name in ("config.json", "generation_config.json", "model.safetensors")}
    unknown = tmp_path / "unknown" / "config.json"
    unknown.parent.mkdir()
    unknown.write_text(json.dumps({**json.loads((source / "config.json").read_text()),
                                   "model_type": "census_unknown",
                                   "architectures": ["CensusUnknownForCausalLM"]}))
    refused = {"config.json": unknown, "model.safetensors": source / "model.safetensors"}
    with LocalHub({"local/decoder": decoder, "local/unknown": refused}) as hub:
        (tmp_path / "census.json").write_text(json.dumps([
            {"id": "local/decoder", "commit": hub.commit("local/decoder")},
            {"id": "local/unknown", "commit": hub.commit("local/unknown")},
            {"id": "local/absent", "commit": "0" * 40},
            {"id": "local/gated", "commit": "1" * 40, "unread": "gated"}]))
        env = {**{name: value for name, value in os.environ.items()
                  if name not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
               "HF_ENDPOINT": hub.endpoint, "HF_HUB_CACHE": str(tmp_path / "cache"),
               "DEW_HUB_ANSWERS": str(tmp_path / "answers"),
               "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        rows = {row.id: row for row in coverage.loads(tmp_path, jobs=3, env=env)}
    report = (tmp_path / "loads.md").read_text()
    held = Pretrained.load(source, weights=False, dtype="bfloat16")
    assert rows["local/decoder"].outcome == "loads", report
    assert rows["local/decoder"].parameters == sum(leaf.size for leaf in jax.tree.leaves(held.variables))
    assert rows["local/unknown"].outcome == "refused", report
    assert "census_unknown" in (rows["local/unknown"].reason or ""), report
    assert rows["local/absent"].outcome == "hub", report
    assert (rows["local/gated"].outcome, rows["local/gated"].reason) == ("hub", "gated"), report
    reads = hub.sent("local/decoder", "model.safetensors")
    assert reads and all(read.asked is not None for read in reads), hub.served
    assert sum(read.sent for read in reads) < decoder["model.safetensors"].stat().st_size, hub.served
    assert not [path for path in (tmp_path / "cache").rglob("*") if path.suffix in (".safetensors", ".bin")]
