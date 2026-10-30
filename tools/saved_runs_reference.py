"""Write a saved run of each run kind at this checkout, for tests/test_saved_runs.py to read at every
later main (row 2282).

Each kind's run (tests/saved_run_kinds.py's `SAVED_KINDS`: tests/test_run_options.py's `KINDS`, its data
as there, synthetic tickets, the repo's tiny image records and token files, the tiny Laya fixture, an LM
run over a tiny multimodal source, and each serialized variant beside them) trains two steps from Python,
or to its examples clock's bound, its own trainer otherwise, and keeps its last checkpoint; what
the run serialized is recorded beside it (`evidence`: its parameters' and compute dtypes, its loss scale,
the examples it consumed and the rung its step compiled at), which a later main's load and resume
assert; its run directory, the run.json and the checkpoint, is packed as an xz tarball, held as the
bytes of one recording (`run`) at `<commit>/<kind>.npz` under tests/fixtures/saved_runs
(`fetched_fixtures.save`: committed where it is small, in the fetched store pinned by its digest where it
is not, as a decision run's pretrained backbone and its optimizer state make it), and `source.json` names
the commit that wrote them. A later main reads each back as a user's process would: the record
(`RunConfig.load`), the checkpoint's inference declaration (`SavedRun.pinned`), the kind's saved task
(`from_run`), and a resume a step past the run's last.

    python tools/saved_runs_reference.py

Run it on CPU from a clean checkout of the main it records, then commit the files (and `upload` a pinned
one): the corpus only grows, one directory per recorded main, so a later main is read against every
earlier one.
"""

import argparse
import dataclasses
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
CORPUS = "tests/fixtures/saved_runs"


def anonymous(run: Path, name: str, places: dict[str, str]) -> bytes:
    """The directory `run` as the corpus holds it: an xz tarball of it under `name`, its members in
    name order, none with an owner, every one at time 0 and of one mode a kind, and each of `places`,
    a directory of the machine that wrote it (its scratch directory, its checkout), read in every text
    file as the place a later checkout reads it from, so the archive names no user, host or directory
    of the writer's; a binary file, which a later main reads as it was written, is kept as it is."""
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w:xz") as archive:
        for path in sorted([run, *run.rglob("*")]):
            info = tarfile.TarInfo((Path(name) / path.relative_to(run)).as_posix())
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            if path.is_dir():
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                archive.addfile(info)
                continue
            data = path.read_bytes()
            try:
                text = data.decode()
            except UnicodeDecodeError:
                text = None
            if text is not None:
                # The longest place first, so a checkout within a scratch directory is read whole.
                for place in sorted(places, key=len, reverse=True):
                    text = text.replace(place, places[place])
                data = text.encode()
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    return packed.getvalue()


EVIDENCE = 1
"""The version of what a recorded generation's runs record they serialized (`evidence`), which its
`source.json` states; a generation that states one holds every kind's."""


def evidence(run, state, directory: Path) -> dict:
    """What a saved run serialized, as a later main's load and resume read it back
    (tests/test_saved_runs.py): its parameters' dtypes, its compute's as its run states it, the dtypes
    one request of the task loaded from it multiplies in (`computed`), its dynamic loss scale and the
    finite steps it counted, the examples it consumed, its last step, and the rung its step compiled at,
    the remat and the plan it was admitted under (`StepCompiled`) but the plan's probes and kernels."""
    import jax
    from saved_run_kinds import computed

    import dew
    from dew.registry import objectives

    lines = (directory / "tracking" / "records.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    compiled = [row["value"] for row in records if row["type"] == "StepCompiled"][-1]
    plan = compiled["plan"] or {}
    task = None if objectives[run.objective.name].saved_task is None else dew.pipeline(str(directory))
    return {"step": int(state.step), "compute": run.compute_dtype(),
            "products": None if task is None else computed(task),
            "params": sorted({str(leaf.dtype) for leaf in jax.tree.leaves(state.variables["params"])}),
            "scale": None if state.scale is None else float(state.scale.scale),
            "finite_steps": None if state.scale is None else int(state.scale.fin_steps),
            "consumed": None if state.consumed is None else int(state.consumed), "remat": compiled["remat"],
            "plan": {name: value for name, value in plan.items()
                     if name not in ("kernels", "probes", "derived")}}


def main() -> None:
    import os

    from lane_environment import configure_lane

    # The runs are written under the suite's environment, its simulated devices among it, which a
    # per-share read resumes on alone.
    configure_lane(os.environ)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout the corpus is written into")
    args = parser.parse_args()

    import numpy as np
    from fetched_fixtures import save
    from saved_run_kinds import SAVED_KINDS

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    written = Path(CORPUS) / commit[:12]
    for kind, build in SAVED_KINDS.items():
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            # The run keeps its last step alone: a later main pins and resumes from the latest. Its own
            # trainer, its clock and numerics, otherwise.
            built = build(root, drawn=False)
            steps = None if built.trainer.examples else 2
            trainer = dataclasses.replace(built.trainer, checkpoint_dir=str(root / "runs"), steps=steps,
                                          log_every=1, eval_every=None, checkpoint_every=None,
                                          multi_host=False, compilation_cache_dir=None)
            run = dataclasses.replace(built, trainer=trainer)
            # A source the repo holds is recorded by its path in the repo, which a later checkout reads
            # where it sits; a resume refuses another source (`RunConfig.check_resume`).
            if getattr(run, "pretrained", None) is not None:
                held = (ROOT / run.pretrained).resolve()
                run = dataclasses.replace(run, pretrained=held.relative_to(ROOT).as_posix())
            state = run.prepare().train(kind)
            seen = evidence(run, state, root / "runs" / kind)
            packed = anonymous(root / "runs" / kind, kind, {str(root): ".", str(ROOT): "."})
            save(written / f"{kind}.npz", {"run": np.frombuffer(packed, np.uint8)},
                 reference=f"Dew {commit[:12]}, tools/saved_runs_reference.py", license="MIT", root=args.root)
            (args.root / written / f"{kind}.evidence.json").write_text(json.dumps(seen, indent=1) + "\n")
    source = {"commit": commit, "kinds": sorted(SAVED_KINDS), "evidence": EVIDENCE}
    (args.root / written / "source.json").write_text(json.dumps(source, indent=1) + "\n")


if __name__ == "__main__":
    main()
