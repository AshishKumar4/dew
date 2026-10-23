"""The null runs of check 1 on the order sites' operation classes (tests/reference_error.py
`null_runs`): per class, the reference against
itself on disjoint draws of the site's own draw law, run after run on fresh draws, every nonpassing verdict
counted, and the one-sided Clopper-Pearson bound of the red rate.

Each class is the site's computation as its test builds it, its draw an orbit of the test's inputs (the
exact symmetry the test orders), at the site's real size. Run chunks are independent: chunk c takes runs
OFFSET + c·runs on, past every seed a site's sizing (`seeds_for`, from 0) reads, so the validation is held
out from it. Prints one JSON line of the chunk's verdicts.

Usage, from the checkout: python tools/gate_null_runs.py <class> --seeds K --runs N --chunk C
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "src"), str(REPO)]

OFFSET = 1 << 16
"""The first run a chunk's grid takes, past the seeds of any sizing."""


def cross_entropy_columns():
    """tests/test_chunked_cross_entropy.py's head over 512 columns, ordered by its columns: each
    token's loss and log Z, and the gradients of a loss of both for the states, the head, the scale
    and the bias, `jax.nn.logsumexp`'s whole logits the reference."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from test_chunked_cross_entropy import columns, eighths, inputs

    from dew.nn.protocols import OutputTable

    states, matrix, targets = inputs(vocab=512, features=16, tokens=(4, 5))
    states, matrix = eighths(states, matrix)
    scale, bias = columns(512)

    def whole(states, matrix, scale, bias, targets):
        logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        return log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0], log_z

    def loss(*arguments):
        losses, log_z = whole(*arguments)
        return jnp.sum(losses + 1e-2 * jnp.square(log_z)), (losses, log_z)

    scored = jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True)

    def reference(states, matrix, scale, bias, targets, relabel):
        with jax.default_matmul_precision("highest"):
            (d_states, d_matrix, d_scale, d_bias), (losses, log_z) = scored(states, matrix, scale, bias,
                                                                             targets)
        return {"losses": losses, "log Z": log_z, "states": d_states, "head": d_matrix[:, relabel],
                "scale": d_scale[relabel], "bias": d_bias[relabel]}

    def draw(seed):
        order = np.random.default_rng(seed).permutation(512)
        relabel = np.argsort(order)
        return (states, matrix[:, order], scale[order], bias[order], relabel[targets], relabel)

    return jax.jit(reference), draw, None


def decoder_residual_orders():
    """tests/test_decoder_export.py's qwen3_next round trip, ordered by its residual stream
    (tests/residual_orders.py): the trained model's logits on the trip's tokens, the model itself on
    both sides, as equally exact as itself at any order, against the trip's float64 truth (the
    reference run in float64, the same logits at every order)."""
    import tempfile

    import jax
    import numpy as np
    from residual_orders import permuted, residual_width

    from tools import decoder_export_reference as tool

    case = next(case for case in tool.CASES if case.name == "qwen3_next")
    trip = tool.round_trip(case, Path(tempfile.mkdtemp(prefix="null-runs-")))
    width, apply = residual_width(trip.trained), jax.jit(trip.source.model.apply)

    def forward(variables, ids):
        return {"logits": apply(variables, ids)}

    def draw(seed):
        return permuted(trip.trained, np.random.default_rng(seed).permutation(width)), trip.ids

    def truth(variables, ids):
        return {"logits": np.asarray(trip.truth, np.float64)}

    return forward, draw, truth


CLASSES = {"cross-entropy-columns": cross_entropy_columns, "decoder-residual-orders": decoder_residual_orders}


def _faulted(change):
    """A fault of a class's reference: `change(arguments)` applied to its inputs before it runs."""
    def fault(reference):
        return lambda *arguments: reference(*change(arguments))
    return fault


def _scaled(factor):
    def change(arguments):
        states, *rest = arguments
        return (states * factor, *rest)
    return change


def _changed(before, after):
    """`after`, refused where it equals `before`: a fault that changes nothing is no fault."""
    import numpy as np

    assert not np.array_equal(np.asarray(before), np.asarray(after)), "the fault changes nothing"
    return after


def _bfloat16(arguments):
    import jax.numpy as jnp

    states, matrix, scale, bias, *rest = arguments
    rounded = _changed(scale, jnp.asarray(scale, jnp.bfloat16).astype(jnp.asarray(scale).dtype))
    return (states, matrix, rounded, bias, *rest)


def _parameters(change):
    """A decoder's parameters changed by `change`, leaf by leaf, its tokens as they are."""
    def changed(arguments):
        import jax

        variables, ids = arguments
        return (jax.tree.map(change, variables), ids)
    return changed


def _parameters_scaled(factor):
    return _parameters(lambda leaf: leaf * factor)


def _parameters_bfloat16(leaf):
    import jax.numpy as jnp

    return jnp.asarray(leaf, jnp.bfloat16).astype(jnp.asarray(leaf).dtype)


def _dropped(arguments):
    import numpy as np

    states, matrix, scale, bias, *rest = arguments
    bias = np.asarray(bias).copy()
    bias[0] = -np.inf
    return (states, matrix, scale, bias, *rest)


FAULTS = {
    "cross-entropy-columns": {
        "scale-2^-10": _faulted(_scaled(1 + 2.0 ** -10)),
        "scale-2^-20": _faulted(_scaled(1 + 2.0 ** -20)),
        "scale-2^-22": _faulted(_scaled(1 + 2.0 ** -22)),
        "bf16-scale": _faulted(_bfloat16),
        "dropped-column": _faulted(_dropped),
    },
    "decoder-residual-orders": {
        "scale-2^-20": _faulted(_parameters_scaled(1 + 2.0 ** -20)),
        "bf16-parameters": _faulted(_parameters(_parameters_bfloat16)),
    },
}
"""Real faults of each class, its own computation changed before it runs, for the procedures' power on
it (`null_runs`'s `dew_fn`): the cross-entropy head's states scaled by a wrong factor at three sizes,
its column scale rounded to bfloat16 where float32 is claimed, a column of its vocabulary dropped; the
decoder's parameters scaled, or rounded to bfloat16."""


def _law(name: str, rng, size: int):
    """`size` per-draw mean squared errors of a synthetic law: "exact-f" a squared standard normal, so
    a pooled ratio's square over its own is F(K, K); "flip" 1 but 70 at one draw in 1,000, a rare bf16
    flip; "gamma-1" a gamma of shape 1/2, a chi-square of one degree."""
    import numpy as np

    if name == "exact-f":
        return np.square(rng.standard_normal(size))
    if name == "flip":
        return np.where(rng.random(size) < 1e-3, 70.0, 1.0)
    if name == "gamma-1":
        return rng.gamma(0.5, 2.0, size)
    raise ValueError(name)


def controls(law: str, ratio: float, seeds: int, runs: int, chunk: int, two_stage: bool) -> list:
    """Check 1's verdicts (`_staged`) on known-ratio controls, `runs` runs from `chunk`'s own stream:
    Dew's per-draw errors ratio² times a draw of the law and the reference's an independent draw of it,
    so the population RMS ratio is `ratio` and every run's is random. A law named for a class takes its
    whole-field draws: each draw of Dew and of the reference one of the class's own draws, every leaf
    together (512 of them, drawn past every run's seeds)."""
    import numpy as np
    from reference_error import REDRAW, _drawn, _Draws, _staged

    # The same draws under either procedure, so their verdicts compare run by run.
    rng = np.random.default_rng([chunk, int(ratio * 1000), seeds])
    span = (1 + REDRAW) * seeds
    if law in CLASSES:
        reference, draw, truth = CLASSES[law]()
        pool = _drawn(reference, reference, draw, range(3 * OFFSET * 1024, 3 * OFFSET * 1024 + 512), truth)
        names, values = pool.names, np.asarray(pool.mine)
    verdicts = []
    for _ in range(runs):
        if law in CLASSES:
            mine = ratio ** 2 * values[rng.integers(0, len(values), span)]
            theirs = values[rng.integers(0, len(values), span)]
        else:
            names = ["leaf"]
            mine, theirs = ratio ** 2 * _law(law, rng, span)[:, None], _law(law, rng, span)[:, None]
        leaves = mine.shape[1]

        def drawn(start, stop, mine=mine, theirs=theirs, leaves=leaves, names=names):
            count = stop - start
            return _Draws(names, mine[start:stop], theirs[start:stop], np.zeros((count, leaves, 2)),
                          np.zeros((count, leaves), int), np.ones((count, leaves), bool))

        over, unresolved, stage = _staged(drawn, seeds, law, two_stage=two_stage)
        verdicts.append(("over" if over else "inconclusive" if unresolved else "pass", stage))
    return verdicts


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help=f"a class ({', '.join(sorted(CLASSES))}), or with --ratio a control's "
                        "law: a class's or exact-f, flip, gamma-1")
    parser.add_argument("--ratio", type=float, help="run known-ratio controls of this population ratio")
    parser.add_argument("--today", action="store_true", help="the procedure before the two stages")
    parser.add_argument("--fault", help="run the class's reference with this fault of its own (FAULTS) "
                        "as Dew, for power")
    parser.add_argument("--seeds", type=int, help="the class's declared count; without it, size it "
                        "(`seeds_for`, on the seeds below OFFSET) and print it")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--draws", type=int, help="instead, print the reference's own mean squared error "
                        "from float64 by leaf over this many draws past every run's, for a sizing model")
    parser.add_argument("--chunk", type=int, default=0)
    options = parser.parse_args(argv)
    from reference_error import null_runs, seeds_for

    if options.ratio is not None:
        verdicts = controls(options.name, options.ratio, options.seeds, options.runs, options.chunk,
                            not options.today)
        print(json.dumps({"control": options.name, "ratio": options.ratio, "seeds": options.seeds,
                          "today": options.today, "chunk": options.chunk, "verdicts": verdicts}), flush=True)
        return 0
    reference, draw, truth = CLASSES[options.name]()
    if options.draws is not None:
        import numpy as np
        from reference_error import _drawn

        seeds = range(2 * OFFSET * 1024 + options.chunk * options.draws,
                      2 * OFFSET * 1024 + (options.chunk + 1) * options.draws)
        draws = _drawn(reference, reference, draw, seeds, truth)
        print(json.dumps({"class": options.name, "chunk": options.chunk, "names": draws.names,
                          "mse": np.asarray(draws.mine).tolist(),
                          "worst": np.asarray(draws.worst[..., 0]).tolist()}), flush=True)
        return 0
    if options.seeds is None:
        sized = seeds_for(reference, draw, options.name, truth_fn=truth)
        print(json.dumps({"class": options.name, "seeds": sized}), flush=True)
        return 0
    verdicts = null_runs(reference, draw, options.name, seeds=options.seeds, runs=options.runs,
                         first=OFFSET + options.chunk * options.runs, truth_fn=truth,
                         two_stage=not options.today,
                         dew_fn=FAULTS[options.name][options.fault](reference) if options.fault else None)
    print(json.dumps({"class": options.name, "chunk": options.chunk, "seeds": options.seeds,
                      "today": options.today, "fault": options.fault, "verdicts": verdicts}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
