"""The bible-section-10 null runs of check 1 on the order sites' operation classes (tests/reference_error.py
`null_runs`; docs of the design: the gate's orders-through-check1 review): per class, the reference against
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
sys.path[:0] = [str(REPO / "tests"), str(REPO / "src")]

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

    return jax.jit(reference), draw


def decoder_residual_orders():
    """tests/test_decoder_export.py's qwen3_next round trip, ordered by its residual stream
    (tests/residual_orders.py): the trained model's logits on the trip's tokens, the model itself on
    both sides, as equally exact as itself at any order."""
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

    return forward, draw


CLASSES = {"cross-entropy-columns": cross_entropy_columns, "decoder-residual-orders": decoder_residual_orders}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=sorted(CLASSES))
    parser.add_argument("--seeds", type=int, help="the class's declared count; without it, size it "
                        "(`seeds_for`, on the seeds below OFFSET) and print it")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=0)
    options = parser.parse_args(argv)
    from reference_error import null_runs, seeds_for

    reference, draw = CLASSES[options.name]()
    if options.seeds is None:
        sized = seeds_for(reference, draw, options.name)
        print(json.dumps({"class": options.name, "seeds": sized}), flush=True)
        return 0
    verdicts = null_runs(reference, draw, options.name, seeds=options.seeds, runs=options.runs,
                         first=OFFSET + options.chunk * options.runs)
    print(json.dumps({"class": options.name, "chunk": options.chunk, "seeds": options.seeds,
                      "verdicts": verdicts}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
