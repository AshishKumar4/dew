"""One process of a real pool whose tracker fails off the step loop (row 2313).

Driven by tests/test_multiprocess.py. The objective is `multiprocess_worker`'s scoring objective;
the mode records what it saw into --out, and the test reads it back. dew is imported inside the
mode, as `multiprocess_worker`'s are, so the pool joins before any JAX backend opens.
"""

from __future__ import annotations

import numpy as np
from multiprocess_worker import _evaluation_parts, _EvaluationCase
from process_support import serve_mode, worker_parser


def training_log_failure(args) -> dict:
    """Two processes fit one tiny objective, logging every step; rank 0's tracker fails at its second
    log, written off the step loop (row 2313). Each process reports the kind of error it ended with
    and the steps it ran, observed on every process alike (`Observe`)."""
    import jax
    import optax

    from dew.data.dataset import Loader
    from dew.training import Trainer
    from dew.training.observe import Observe

    rank = jax.process_index()
    case = _EvaluationCase(rank)
    Numerical, _, _, _, _ = _evaluation_parts(case)

    class Failing:
        def log(self, scalars, step):
            if step == 2:
                raise ValueError("tracker logging failed")

        def artifact(self, artifact, step):
            pass

    def rows():
        while True:
            yield {"x": np.arange(rank * 4, (rank + 1) * 4, dtype=np.float32)[:, None]}

    observed = Observe(1, lambda state: int(state.step))
    trainer = Trainer(Numerical(), optax.sgd(.01), key=jax.random.key(3),
                      tracker=Failing() if rank == 0 else None)
    from dew._coordination import PeerFailure

    try:
        trainer.fit(Loader(train=lambda partition: rows(), val=None, records=None, batch=8), steps=8,
                    log_every=1, eval_every=None, checkpoint_every=None, observe=observed)
        error = None
    except (ValueError, PeerFailure) as failure:
        error = type(failure).__name__
    return {"error": error, "steps": [step for step, _ in observed.values]}


MODES = {"training_log_failure": training_log_failure}


def main(argv=None) -> None:
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args(argv))


if __name__ == "__main__":
    main()
