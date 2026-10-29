"""Write a saved run of each run kind at this checkout, for tests/test_saved_runs.py to read at every
later main (row 2282).

Each kind's run (tests/test_run_options.py's `KINDS`, its data as there) trains two steps from Python and
keeps both checkpoints; its run directory, the run.json and the checkpoints, is packed as `<kind>.tar.xz`
in `--out`, and `source.json` names the commit that wrote them. A later main reads each back as a user's
process would: the record (`RunConfig.load`), the checkpoint's inference declaration (`SavedRun.pinned`),
the kind's saved task (`from_run`), and a resume a step past the run's last.

    python tools/saved_runs_reference.py --out tests/fixtures/saved_runs

Run it on CPU from a clean checkout of the main it records, then commit the files: the corpus only grows,
one directory per recorded main, so a later main is read against every earlier one. A kind whose run
packs past a mebibyte (a decision run's pretrained backbone, 16 MB) goes to the fetched store
(tools/fetched_fixtures.py) instead, `--kinds` naming the rest.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="the corpus directory, tests/fixtures/saved_runs")
    parser.add_argument("--kinds", nargs="+", help="the kinds to write, each of `KINDS`; unset, every one")
    args = parser.parse_args()

    from test_run_options import KINDS

    from dew.config import TrainerConfig

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    written = args.out / commit[:12]
    written.mkdir(parents=True)
    kinds = args.kinds or sorted(KINDS)
    for kind in kinds:
        build = KINDS[kind]
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            trainer = TrainerConfig(checkpoint_dir=str(root / "runs"), steps=2, log_every=1, eval_every=None,
                                    checkpoint_every=1, multi_host=False, compilation_cache_dir=None)
            dataclasses.replace(build(root, drawn=False), trainer=trainer).prepare().train(kind)
            with tarfile.open(written / f"{kind}.tar.xz", "w:xz") as packed:
                packed.add(root / "runs" / kind, arcname=kind)
    source = {"commit": commit, "kinds": kinds}
    (written / "source.json").write_text(json.dumps(source, indent=1) + "\n")


if __name__ == "__main__":
    main()
