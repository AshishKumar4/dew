"""One process of a real pool whose preview, which its tracker on process 0 alone takes, the model refuses
(audit P02).

Driven by tests/test_multiprocess.py. The objective is `affine_run`'s regression, its preview refused; the
mode records what each process's fit ended in into --out, and the test reads it back. dew is imported inside
the mode, as `multiprocess_worker`'s are, so the pool joins before any JAX backend opens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from process_support import refusal, serve_mode, worker_parser


def preview_refusal(args) -> dict:
    """Fits of an objective that cannot sample, validated on a split of batches, with a tracker on process
    0 alone and a checkpointer both share, each ending in its refusal (`refusal`) or the step it trained to:
    a preview every process asks, one process 0 alone asks, and none (the control, which trains); then one
    more agreement every process passes, which a rank left in a checkpoint's agreement would block."""
    import jax
    import optax
    from affine_run import FEATURES, Regression
    from recording import RecordingTracker

    from dew import _coordination
    from dew.data import Memory
    from dew.training import Checkpoints, MeshSpec, Trainer

    class Unsampled(Regression):
        """The regression, its preview refused: a model that cannot generate (`preview_refusal`)."""

        def preview_refusal(self) -> str | None:
            return "this regression generates nothing"

    def records():
        x = np.repeat(np.arange(64, dtype=np.float32)[:, None], FEATURES, axis=1)
        return Memory({"x": x, "y": 2 * x[:, :2]}).batch(8).workers(0)

    rank = jax.process_index()

    def fitted(name: str, *, preview: bool) -> dict:
        trainer = Trainer(Unsampled(), optax.sgd(1e-2), key=jax.random.key(0), mesh=MeshSpec(),
                          checkpoints=Checkpoints(str(Path(args.out).parent / name)),
                          tracker=RecordingTracker() if rank == 0 else None)
        try:
            state = trainer.fit(records(), steps=2, eval_every=1, preview=preview,
                                validation={"val": records()})
        except (ValueError, RuntimeError) as error:
            return {"refused": refusal(error), "step": None}
        return {"refused": None, "step": int(state.step)}

    ended = {"every": fitted("every", preview=True), "zero": fitted("zero", preview=rank == 0),
             "none": fitted("none", preview=False)}
    _coordination.agreed("the pool after its previews", lambda: True)
    return ended


MODES = {"preview_refusal": preview_refusal}


def main(argv=None) -> None:
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args(argv))


if __name__ == "__main__":
    main()
