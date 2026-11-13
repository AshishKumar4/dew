"""Peak host memory of an export: one shard at a time against the whole model.

    PYTHONPATH=src python tools/benchmark_streaming_export.py <repo> <max_shard_size> <out_dir> [revision]

Loads `repo` in bfloat16 onto the default device (the mesh load streams, so
the host holds no model afterwards), then saves it with `max_shard_size`.
Prints one JSON line: RSS before the save, the highest RSS a 2 ms sampler of
the exact page counts saw during it (`dew.telemetry.benchmark.resident_peak`)
and its rise, the same of anonymous memory alone (`anon_*`: without the pages
of files the process maps, which the page cache and the host's load move), the
rise of the bytes its allocator holds allocated (`allocated_rise_gb`,
`allocated_memory`: what the save holds home, which resident pages undercount
where the allocator serves it from pages freed earlier, rows 5266 and 5562),
the export's weight bytes and the shard count. A save that gathers the model to
the host first rises one model above the baseline; a streamed one, a few
shards. `main` as the shard size calls `save` without one, for the writer
before sharding.
"""

import json
import sys
import time
from pathlib import Path

from dew.distributed.placement import MeshSpec
from dew.interop import Pretrained
from dew.telemetry.benchmark import resident_peak


def main() -> None:
    repo, shard_size, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    revision = sys.argv[4] if len(sys.argv) > 4 else None
    loaded = Pretrained.load(repo, revision=revision, param_dtype="bfloat16", mesh=MeshSpec())
    with resident_peak() as peak:
        start = time.perf_counter()
        if shard_size == "main":
            loaded.save(out)
        else:
            loaded.save(out, max_shard_size=int(shard_size) if shard_size.isdigit() else shard_size)
        seconds = time.perf_counter() - start
    before = (peak.before["Rss"] / 2**30, peak.before["Anonymous"] / 2**30)
    highest = (peak.most["Rss"] / 2**30, peak.most["Anonymous"] / 2**30)
    files = sorted(out.rglob("*.safetensors"))
    print(json.dumps({"repo": repo, "max_shard_size": shard_size, "rss_before_gb": round(before[0], 2),
                      "rss_peak_during_save_gb": round(highest[0], 2),
                      "rise_gb": round(highest[0] - before[0], 3),
                      "anon_before_gb": round(before[1], 2), "anon_rise_gb": round(highest[1] - before[1], 3),
                      "allocated_rise_gb": round(peak.rise("Allocated") / 2**30, 3),
                      "seconds": round(seconds, 1), "shards": len(files),
                      "weights_gb": round(sum(f.stat().st_size for f in files) / 2**30, 2)}))


if __name__ == "__main__":
    main()
