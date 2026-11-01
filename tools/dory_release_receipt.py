"""Dory 1.0 4B's release receipt: Dew's decoding of the release held to vLLM's recorded engine run by
check 1, at every loop count, cache and dtype the recorder ran (tools/dory_vllm_reference.py `RUNS`),
each run's verdict and seconds written to OUT/receipt.json.

    python tools/dory_release_receipt.py OUT [RUN ...]

A RUN is `loops-mode-dtype` (`2-per_loop-bfloat16`); none means every run. It runs on a GPU, the
release's weights fetched from the Hub, at about three minutes a recorded seed: a receipt for a GPU job,
not a suite test, whose gate the suite holds on the tiny model.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    import test_dory

    runs = {"-".join(map(str, run)): run for run in test_dory.RELEASE_RUNS}
    chosen = sys.argv[2:] or list(runs)
    unknown = sorted(set(chosen) - set(runs))
    if unknown:
        raise SystemExit(f"no recorded run is named {unknown}; the runs are {sorted(runs)}")
    receipt = {}
    for name in chosen:
        began = time.perf_counter()
        try:
            test_dory.the_release_computes_vllms_log_probabilities_through_each_cache_and_loop_count(runs[name])
            receipt[name] = {"verdict": "passed"}
        except Exception as error:
            message = "".join(traceback.format_exception_only(error))
            receipt[name] = {"verdict": "failed", "error": message[-2000:]}
        receipt[name]["seconds"] = round(time.perf_counter() - began, 1)
        print(name, receipt[name]["verdict"], receipt[name]["seconds"], flush=True)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    if any(entry["verdict"] != "passed" for entry in receipt.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
