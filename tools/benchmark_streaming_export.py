"""Peak host memory of an export: one shard at a time against the whole model.

    PYTHONPATH=src python tools/benchmark_streaming_export.py <repo> <max_shard_size> <out_dir>

Loads `repo` in bfloat16 onto the default device (the mesh load streams, so
the host holds no model afterwards), then saves it with `max_shard_size`.
Prints one JSON line: RSS before the save, the highest RSS a 2 ms sampler
saw during it, the export's weight bytes and the shard count. A save that
gathers the model to the host first rises one model above the baseline; a
streamed one, one shard. `main` as the shard size calls `save` without one,
for the writer before sharding.
"""

import json
import sys
import threading
import time
from pathlib import Path

from dew.interop import load_pretrained
from dew.training.distributed import MeshSpec


def _rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("/proc/self/status has no VmRSS")


def main() -> None:
    repo, shard_size, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    loaded = load_pretrained(repo, param_dtype="bfloat16", mesh=MeshSpec())
    before = _rss_gb()
    highest = [before]
    done = threading.Event()

    def sample() -> None:
        while not done.wait(0.002):
            highest[0] = max(highest[0], _rss_gb())

    sampler = threading.Thread(target=sample)
    sampler.start()
    start = time.perf_counter()
    if shard_size == "main":
        loaded.save(out)
    else:
        loaded.save(out, max_shard_size=int(shard_size) if shard_size.isdigit() else shard_size)
    seconds = time.perf_counter() - start
    done.set()
    sampler.join()
    files = sorted(out.glob("model*.safetensors"))
    print(json.dumps({"repo": repo, "max_shard_size": shard_size, "rss_before_gb": round(before, 2),
                      "rss_peak_during_save_gb": round(highest[0], 2),
                      "rise_gb": round(highest[0] - before, 2), "seconds": round(seconds, 1),
                      "shards": len(files), "weights_gb": round(sum(f.stat().st_size for f in files) / 2**30, 2)}))


if __name__ == "__main__":
    main()
