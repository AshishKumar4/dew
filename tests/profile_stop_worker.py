"""Workers for a profile window a pool cannot stop on every rank (`dew.training.profiling.export_wait`): each
process a rank of a run whose window rank 0 cannot write its trace for.

dew is imported inside the modes, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

from pathlib import Path

from multiprocess_worker import BATCH, indexed_loader, make_objective
from process_support import serve_mode, worker_parser


def mode_profile_stop_failure(args) -> dict:
    """A profile window whose capture rank 0 cannot stop: its capture's directory made unwritable while the
    window traces (by an observer, which runs at each step on the loop), so its trace cannot be written as
    the window stops, while rank 1's stop starts its export. Every rank fails the stop together, and the
    run's cleanup still waits for the export rank 1 started (`export_wait`): none outlives the fit."""
    import os
    import stat

    import jax
    import optax

    from dew._coordination import agree_process_phase
    from dew.data.dataset import Loader
    from dew.telemetry import profile
    from dew.training import Observe, ProfileWindow, Trainer

    profile_dir = Path(args.profile_dir)

    def locked(state) -> int:
        # On rank 0, the window's capture unwritable, each directory of it, at the window's first step,
        # before it stops.
        if jax.process_index() == 0:
            for capture in (profile_dir / "process-0").glob("capture-*"):
                for directory in (capture, *(path for path in capture.rglob("*") if path.is_dir())):
                    os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR)
        return 0

    loader = indexed_loader(args.records, BATCH // args.processes)
    trainer = Trainer(make_objective(), optax.adam(1e-3), key=jax.random.key(0),
                      profile=ProfileWindow(str(profile_dir), steps=2, warmup=0))
    error = None
    try:
        trainer.fit(Loader(lambda partition: iter(loader), None, batch=BATCH, records=None),
                    steps=args.steps, log_every=1, observe=Observe(1, locked))
    except BaseException as failure:
        error = failure
    exporting = len(profile._exporting)
    manifests = len(list((profile_dir / f"process-{args.process_id}").glob("capture-*/manifest.json")))
    for capture in (profile_dir / "process-0").glob("capture-*"):
        for directory in (capture, *(path for path in capture.rglob("*") if path.is_dir())):
            os.chmod(directory, stat.S_IRWXU)
    recovered = agree_process_phase(None, phase="after the profile stop failure")
    return {"failed": None if error is None else
            f"{type(error).__name__}: {error} | {'; '.join(getattr(error, '__notes__', ()))}",
            "exporting": exporting, "manifests": manifests, "recovered": recovered}


MODES = {"profile_stop_failure": mode_profile_stop_failure}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--records", type=int, default=BATCH * 16)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
