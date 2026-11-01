"""Workers for the rounds of agreement a pool's fit runs beside its reads (`dew.training._fit.FitRun.agrees`):
each process a rank of the same fits.

dew is imported inside the mode, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import numpy as np
from multiprocess_worker import BATCH
from process_support import serve_mode, worker_parser


def read_rounds(args) -> dict:
    """The rounds of agreement a pool's fit of the affine regression runs, for a run of `--steps` steps
    and one of twice as many, over each stream a fit reads in a pool: a finite global order (a batch of
    records for each of those steps, repeated twice, read to its end), which every share runs out of at
    the same read;
    arrays handed whole, the same batch every step; a Grain map dataset handed as it is, repeated without
    end; and an endless order a cooldown branches from halfway, the branch half the run long. Each kind's
    rounds over its two runs, the difference the steps of the longer run add, which a read agreed at
    each step makes one a step."""
    import grain.python as pygrain
    import jax
    import optax
    from affine_run import FEATURES, Regression

    from dew import _coordination
    from dew.data import Loading, Memory
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.training import Cooldowns, MeshSpec, Trainer

    def rows(count: int) -> dict[str, np.ndarray]:
        x = np.repeat(np.arange(count, dtype=np.float32)[:, None], FEATURES, axis=1) / count
        return {"x": x, "y": 2 * x[:, :2]}

    def rounds(kind: str, steps: int) -> int:
        trainer = Trainer(Regression(), optax.inject_hyperparams(optax.sgd)(learning_rate=1e-2),
                          key=jax.random.key(0), mesh=MeshSpec())
        flags: dict = {"steps": 2 * steps}
        if kind == "global":
            data = compile(Pipeline(Memory(rows(BATCH * steps))).repeat(2).batch(BATCH), seed=0,
                           loading=Loading(workers=0))
            # More steps than the order holds: the run ends where every share runs out of it.
            flags["steps"] = 4 * steps
        elif kind == "whole":
            data = rows(BATCH)
        elif kind == "map":
            data = pygrain.MapDataset.source([{name: column[index] for name, column in rows(BATCH).items()}
                                              for index in range(BATCH)]).batch(BATCH)
        else:
            data = compile(Pipeline(Memory(rows(BATCH))).shuffle(0).repeat().batch(BATCH), seed=0,
                           loading=Loading(workers=0))
            flags.update(validation=Pipeline(Memory(rows(BATCH))).batch(BATCH),
                         cooldowns=Cooldowns(at=(steps,), length=steps))
        before = next(_coordination._agreements)
        state = trainer.fit(data, log_every=10 * steps, **flags)
        assert int(state.step) == 2 * steps, (kind, int(state.step))
        return next(_coordination._agreements) - before - 1

    return {kind: [rounds(kind, steps) for steps in (args.steps, 2 * args.steps)]
            for kind in ("global", "whole", "map", "cooldown")}


MODES = {"read_rounds": read_rounds}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--steps", type=int, default=4)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
