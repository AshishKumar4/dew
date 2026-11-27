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
    """A fit asked to preview, at a validation of batches, an objective that cannot sample, with a tracker on
    process 0 alone and a checkpointer both share: how each process's fit ended (`refusal`, None where it
    trained), then one more agreement every process passes, which a rank left in the checkpoint's own
    agreement would block."""
    import jax
    import optax
    from affine_run import FEATURES, Regression
    from recording import RecordingTracker

    from dew import _coordination
    from dew.data import Loading, Memory
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.training import Checkpoints, MeshSpec, Trainer

    class Unsampled(Regression):
        """The regression, its preview refused: a model that cannot generate (`preview_refusal`)."""

        def preview_refusal(self) -> str | None:
            return "this regression generates nothing"

    def stream():
        x = np.repeat(np.arange(64, dtype=np.float32)[:, None], FEATURES, axis=1)
        return compile(Pipeline(Memory({"x": x, "y": 2 * x[:, :2]})).shuffle(0).repeat().batch(8), seed=0,
                       loading=Loading(workers=0))

    rank = jax.process_index()
    trainer = Trainer(Unsampled(), optax.sgd(1e-2), key=jax.random.key(0), mesh=MeshSpec(),
                      checkpoints=Checkpoints(str(Path(args.out).parent / "previewed")),
                      tracker=RecordingTracker() if rank == 0 else None)
    try:
        trainer.fit(stream(), steps=2, eval_every=1, preview=True, validation={"val": stream()})
        ended = None
    except (ValueError, RuntimeError) as error:
        ended = refusal(error)
    _coordination.agreed("the pool after its preview", lambda: True)
    return {"refused": ended}


MODES = {"preview_refusal": preview_refusal}


def main(argv=None) -> None:
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args(argv))


if __name__ == "__main__":
    main()
