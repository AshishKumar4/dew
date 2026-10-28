"""The Hub census's loads without weights (tools/hf_coverage.py `census` then `loads`, ledger 1891), over a
Hub on the loopback (tests/local_hub.py), which keeps every request it answers."""

import json
import os
import subprocess
import sys
from pathlib import Path

from local_hub import LocalHub
from tool_support import REPO_ROOT, load

FIXTURES = Path(__file__).parent / "fixtures" / "hf"


def test_the_census_loads_each_model_without_its_weights_and_rows_each_refusal(tmp_path):
    """The census lists a local Hub's models, each at the commit the listing states, and `loads` loads
    each without its weights: a published decoder (llama-tiny's files), its row its parameter count, as
    a load of the same files on disk counts it; a model of a type no family maps, refused, its row naming
    the refusal; and a repo with no config, the Hub's row, as are a repo the Hub does not hold and a
    model whose files the Hub, gone, cannot serve. No load
    fetched a weight file: what the Hub sent of the decoder's model.safetensors was ranges of its first
    bytes, where its safetensors header is read, less than the file, and no weight file was cached (the
    boundary 1888's network test checks on the Hub itself)."""
    import jax

    from dew.interop import Pretrained

    source = FIXTURES / "llama-tiny"
    decoder = {name: source / name for name in ("config.json", "generation_config.json", "model.safetensors")}
    unknown = tmp_path / "unknown" / "config.json"
    unknown.parent.mkdir()
    unknown.write_text(json.dumps({**json.loads((source / "config.json").read_text()),
                                   "model_type": "census_unknown",
                                   "architectures": ["CensusUnknownForCausalLM"]}))
    repos = {"local/decoder": decoder,
             "local/unknown": {"config.json": unknown, "model.safetensors": source / "model.safetensors"},
             "local/bare": {"model.safetensors": source / "model.safetensors"}}
    out = tmp_path / "census"
    with LocalHub(repos) as hub:
        env = {**{name: value for name, value in os.environ.items()
                  if name not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
               "HF_ENDPOINT": hub.endpoint, "HF_HUB_CACHE": str(tmp_path / "cache"),
               "DEW_HUB_ANSWERS": str(tmp_path / "answers"), "PYTHONPATH": str(REPO_ROOT / "src")}
        for command in (["census", str(out), "10"], ["loads", str(out), "--jobs", "3"]):
            done = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "hf_coverage.py"), *command],
                                  env=env, capture_output=True, text=True, check=False)
            assert done.returncode == 0, done.stderr[-3000:]
        coverage = load("hf_coverage")
        absent = coverage.loaded("local/absent", "0" * 40, env)
    # The Hub gone, an uncached file's metadata cannot be fetched: the Hub's row, not a refusal.
    unreachable = coverage.loaded("local/decoder", hub.commit("local/decoder"),
                                  {**env, "HF_HUB_CACHE": str(tmp_path / "fresh")})
    rows = {row["id"]: row for row in json.loads((out / "loads.json").read_text())}
    report = (out / "loads.md").read_text()
    held = Pretrained.load(source, weights=False, dtype="bfloat16")
    assert rows["local/decoder"]["commit"] == hub.commit("local/decoder"), report
    assert rows["local/decoder"]["outcome"] == "loads", report
    assert rows["local/decoder"]["parameters"] == sum(leaf.size for leaf in jax.tree.leaves(held.variables))
    assert rows["local/unknown"]["outcome"] == "refused", report
    assert "census_unknown" in (rows["local/unknown"]["reason"] or ""), report
    assert (rows["local/bare"]["outcome"], rows["local/bare"]["reason"]) == ("hub", "no config.json"), report
    assert absent.outcome == "hub" and "RepositoryNotFoundError" in (absent.reason or ""), absent
    assert unreachable.outcome == "hub", unreachable
    reads = hub.sent("local/decoder", "model.safetensors")
    assert reads and all(read.asked is not None and read.asked[0] in (0, 8) for read in reads), hub.served
    assert sum(read.sent for read in reads) < decoder["model.safetensors"].stat().st_size, hub.served
    assert not [path for path in (tmp_path / "cache").rglob("*") if path.suffix in (".safetensors", ".bin")]
