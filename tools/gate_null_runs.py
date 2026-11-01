"""The null runs of check 1 on operation classes (tests/reference_error.py `null_runs`): per class,
check 1 run after run on fresh draws, every nonpassing verdict counted with its findings, and the
one-sided Clopper-Pearson bound of the red rate.

- By default the reference against itself on disjoint draws of the class's draw law. The order classes
  (cross-entropy-columns, decoder-residual-orders) draw over an orbit of the test's inputs, the exact
  symmetry the test orders, at the site's real size.
- `--paired` runs both arms on the same draws, as a site does: Dew the class's reference at another
  order of its orbit (REORDERED), an implementation as exact as its own, or with `--fault` a fault of it.
  This is the control of the gate as sites run it. The orbit classes (contraction-terms, mean-loss-rows,
  attention-keys) take it, and `--orders K` judges their rows over K orders of each draw
  (`assert_as_exact`'s `orders`).
- `--fault` runs a fault of the class's own computation (FAULTS) as Dew, for power.

Run chunks are independent: chunk c takes runs OFFSET + c·runs on, past every seed a site's sizing
(`seeds_for`, from 0) reads, so the validation is held out from it. Prints one JSON line of the chunk's
verdicts.

Usage, from the checkout: python tools/gate_null_runs.py <class> --seeds K --runs N --chunk C
    [--paired] [--orders K] [--fault NAME]
or, for the worst-row rule's controls (`row_controls`): python tools/gate_null_runs.py <rows a draw>
    --rows <values a row> --seeds K --runs N --chunk C, printing its red runs and the site statistic's
    quantiles
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
    """tests/test_chunked_cross_entropy.py's head over 512 columns, ordered by its columns, as its check-1
    site draws and scores it (`column_order_case`): each token's loss and log Z, and the gradients for the
    states, the head, the scale and the bias under the site's cotangent, `jax.nn.logsumexp`'s whole logits
    the reference, its column-wise gradients relabelled back so every order's are the same values."""
    import jax
    from test_chunked_cross_entropy import column_order_case

    draw, _, whole = column_order_case()

    def reference(*drawn):
        *arguments, relabel = drawn
        (losses, log_z), d_states, d_matrix, d_scale, d_bias = whole(*arguments)
        return {"losses": losses, "log Z": log_z, "states": d_states, "head": d_matrix[:, relabel],
                "scale": d_scale[relabel], "bias": d_bias[relabel]}

    return jax.jit(reference), draw, None, None


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

    return forward, draw, truth, None


ORBIT_ORDERS = 4
"""The orders of an orbit class's draws (`orbited`): the draw's own and ORBIT_ORDERS - 1 others. Sites
declare their own; the paired controls run at 2, 4 and 8 (`--orders`)."""

REORDERED = 1 << 10
"""The order an orbit class's equally exact pair runs at (`--paired`), outside every orbit's own."""


def orbited(permuting, count: int = ORBIT_ORDERS):
    """`assert_as_exact`'s `orders` of `count` orders, the draw's own first, `permuting(inputs, order)`
    the inputs at a permutation of the order's own stream."""
    def orders(inputs):
        return [inputs, *(permuting(inputs, order) for order in range(1, count))]
    return orders


def permutation(tag: int, order: int, size: int):
    """The permutation of `size` an orbit class's order `order` takes, from the class's stream `tag`."""
    import numpy as np

    return np.random.default_rng([tag, order]).permutation(size)


def contraction_terms():
    """A float32 matrix product, [64, 1024] by [1024, 64], over the orbit of its contraction's orders:
    the terms of every sum taken in other orders, the same sums."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    @jax.jit
    def reference(x, w):
        return {"y": jnp.dot(x, w, precision="highest")}

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((64, 1024)).astype(np.float32),
                rng.standard_normal((1024, 64)).astype(np.float32))

    def permuting(inputs, order):
        x, w = inputs
        terms = permutation(0x0C, order, x.shape[1])
        return x[:, terms], w[terms]

    return reference, draw, None, permuting


def mean_loss_rows():
    """A two-layer tanh network's gradients of a mean squared loss over 256 rows, over the orbit of the
    rows' orders: the mean's terms, every gradient's sum over the rows, in other orders. Its parameters
    are inputs, the same at every draw, so a fault can narrow them with the rows."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from reference_error import with_gradients

    rng = np.random.default_rng(0x0E)
    params = {"inner": (rng.standard_normal((64, 256)) / 8).astype(np.float32),
              "outer": (rng.standard_normal((256, 1)) / 16).astype(np.float32)}

    @jax.jit
    def reference(params, x, targets, one):
        def loss(params):
            hidden = jnp.tanh(jnp.dot(x, params["inner"], precision="highest"))
            return jnp.mean(jnp.square(jnp.dot(hidden, params["outer"], precision="highest")[:, 0] - targets))

        # The draw's one, at the loss's dtype (float16 under the float16 fault).
        _, gradient = with_gradients(loss)(params, jnp.asarray(one, x.dtype))
        return gradient

    def draw(seed):
        rng = np.random.default_rng([0x0F, seed])
        return (params, rng.standard_normal((256, 64)).astype(np.float32),
                rng.standard_normal(256).astype(np.float32), np.float32(1))

    def permuting(inputs, order):
        params, x, targets, one = inputs
        rows = permutation(0x0E, order, len(x))
        return params, x[rows], targets[rows], one

    return reference, draw, None, permuting


def attention_keys():
    """Softmax attention, 4 heads of 128 queries over 256 keys 64 wide, over the orbit of its keys'
    orders: each softmax's and each weighted sum's terms in other orders, the same outputs."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    @jax.jit
    def reference(q, k, v):
        logits = jnp.einsum("hqd,hkd->hqk", q, k, precision="highest") / 8
        return {"out": jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(logits, -1), v, precision="highest")}

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(rng.standard_normal(shape).astype(np.float32)
                     for shape in ((4, 128, 64), (4, 256, 64), (4, 256, 64)))

    def permuting(inputs, order):
        q, k, v = inputs
        keys = permutation(0x0A, order, k.shape[1])
        return q, k[:, keys], v[:, keys]

    return reference, draw, None, permuting


def attention_key_bias(heads: int = 16, queries: int = 64, keys: int = 128, width: int = 32):
    """Softmax attention's three bias gradients, packed as one leaf of query, key and value rows,
    `heads` of each, over the orbit of its keys' orders. The key bias's gradient is zero in exact
    arithmetic (each row of softmax's dS sums to zero), so float64 holds its rows at its own residue,
    rows of float64 zero beside the query's and value's sized rows: the class of a site whose leaf
    packs a structural zero (`pooled_zero_rows`)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    @jax.jit
    def reference(q, k, v, bq, bk, bv, cotangent):
        def attend(bq, bk, bv):
            logits = jnp.einsum("hqd,hkd->hqk", q + bq[:, None], k + bk[:, None], precision="highest")
            weights = jax.nn.softmax(logits / np.sqrt(width), -1)
            return jnp.einsum("hqk,hkd->hqd", weights, v + bv[:, None], precision="highest")
        _, pull = jax.vjp(attend, bq, bk, bv)
        return {"bias": jnp.concatenate(pull(cotangent), axis=0)}

    def draw(seed):
        rng = np.random.default_rng([0xB1A5, seed])
        shapes = ((heads, queries, width), (heads, keys, width), (heads, keys, width), (heads, width),
                  (heads, width), (heads, width), (heads, queries, width))
        return tuple(rng.standard_normal(shape).astype(np.float32) for shape in shapes)

    def permuting(inputs, order):
        q, k, v, *rest = inputs
        at = permutation(0x0B, order, k.shape[1])
        return (q, k[:, at], v[:, at], *rest)

    return reference, draw, None, permuting


def heavy_sums(rows: int = 64, terms: int = 256):
    """Sums of `terms` float32 terms over `rows` rows, each draw's terms scaled by one heavy-tailed factor
    (a Pareto of index one, plus one), over the orbit of the terms' orders: each arm's error grows with
    the draw's scale, so across draws it is heavy-tailed (its mean square of index one half), and two
    orders of one sum share that scale draw by draw. The class of a site whose arms share their
    upstream rounding: the reference compared with itself on disjoint draws does not resolve, and the
    pair on the same draws does (`seeds_for`'s `dew_fn`)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    @jax.jit
    def reference(x):
        return {"sum": jnp.sum(x, axis=1)}

    def draw(seed):
        rng = np.random.default_rng([0x4E, seed])
        return ((rng.pareto(1.0) + 1) * rng.standard_normal((rows, terms))).astype(np.float32),

    def permuting(inputs, order):
        (x,) = inputs
        return (x[:, permutation(0x4E, order, x.shape[1])],)

    return reference, draw, None, permuting


CLASSES = {"cross-entropy-columns": cross_entropy_columns, "decoder-residual-orders": decoder_residual_orders,
           "contraction-terms": contraction_terms, "mean-loss-rows": mean_loss_rows,
           "attention-keys": attention_keys, "attention-key-bias": attention_key_bias,
           "heavy-sums": heavy_sums}
"""Each class's reference, draw, truth (None: the reference widened) and, for an orbit class, its
`permuting(inputs, order)` (`orbited`), else None."""


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

    if np.array_equal(np.asarray(before), np.asarray(after)):
        raise ValueError("the fault changes nothing")
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


def _input_fault(position, change):
    """A fault of the input at `position`, `change(array)` a copy of it changed before the reference
    runs, refused where it changes nothing."""
    def changed(arguments):
        import numpy as np

        arguments = list(arguments)
        before = np.asarray(arguments[position])
        arguments[position] = _changed(before, change(before.copy()))
        return tuple(arguments)
    return _faulted(changed)


def _zeroed(index):
    def change(array):
        array[index] = 0
        return array
    return change


def _negated(index):
    def change(array):
        array[index] = -array[index]
        return array
    return change


def _in_float16(reference):
    """A fault of the reference run on its float inputs in float16, its products and sums at float16's
    precision where float32 is claimed, returned as float32."""
    def run(*arguments):
        import jax
        import jax.numpy as jnp
        import numpy as np

        narrowed = jax.tree.map(lambda value: jnp.asarray(value, jnp.float16)
                                if np.ndim(value) and np.issubdtype(np.asarray(value).dtype, np.floating)
                                else value, list(arguments))
        return jax.tree.map(lambda leaf: np.asarray(leaf, np.float32), reference(*narrowed))
    return run


def _scaled_output(factor):
    """A fault scaling every output by `factor`, a wrong scale at the gate's threshold."""
    def fault(reference):
        def run(*arguments):
            import jax
            import numpy as np

            return jax.tree.map(lambda leaf: np.asarray(leaf) * np.asarray(factor, np.asarray(leaf).dtype),
                                reference(*arguments))
        return run
    return fault


def _row_scaled(factor: float, row: int = 0):
    """A fault scaling one row of each output leaf by `factor`: its `row`-th along the leading axes,
    a row of each leaf's last axis. Moved by 1 + 2^-14 of itself, a row is tens of times a float32 row's
    own rounding and too little of the leaf for its RMS ratio to show, so the worst-row rule alone can
    fail it; on a row of float64 zero, it is that residue `factor` times Dew's own."""
    def fault(reference):
        def run(*arguments):
            import jax
            import numpy as np

            def moved(leaf):
                leaf = np.array(leaf)
                at = np.unravel_index(row, leaf.shape[:-1])
                leaf[at] = leaf[at] * factor
                return leaf
            return jax.tree.map(moved, reference(*arguments))
        return run
    return fault


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
    "heavy-sums": {
        "dropped-term": _input_fault(0, _zeroed((slice(None), 0))),
        "flipped-sign": _input_fault(0, _negated((slice(None), 0))),
        "float16": _in_float16,
        "skipped-chunk": _input_fault(0, _zeroed((slice(None), slice(0, 32)))),
        "scale-2^-22": _scaled_output(1 + 2.0 ** -22),
    },
    "contraction-terms": {
        "dropped-term": _input_fault(0, _zeroed((slice(None), 0))),
        "flipped-sign": _input_fault(1, _negated(0)),
        "float16": _in_float16,
        "skipped-chunk": _input_fault(0, _zeroed((slice(None), slice(0, 128)))),
        "scale-2^-22": _scaled_output(1 + 2.0 ** -22),
        "one-row": _row_scaled(1 + 2.0 ** -14),
    },
    "mean-loss-rows": {
        "dropped-term": _input_fault(1, _zeroed(0)),
        "flipped-sign": _input_fault(2, _negated(0)),
        "float16": _in_float16,
        "skipped-chunk": _input_fault(1, _zeroed(slice(0, 32))),
        "scale-2^-22": _scaled_output(1 + 2.0 ** -22),
        "one-row": _row_scaled(1 + 2.0 ** -14),
    },
    "attention-key-bias": {
        "zero-row-10x": _row_scaled(10.0, row=16),
        "zero-row-2^10": _row_scaled(2.0 ** 10, row=16),
        "dropped-term": _input_fault(2, _zeroed((slice(None), 0))),
    },
    "attention-keys": {
        "dropped-term": _input_fault(2, _zeroed((slice(None), 0))),
        "flipped-sign": _input_fault(2, _negated((slice(None), 0))),
        "float16": _in_float16,
        "skipped-chunk": _input_fault(2, _zeroed((slice(None), slice(0, 32)))),
        "scale-2^-22": _scaled_output(1 + 2.0 ** -22),
        "one-row": _row_scaled(1 + 2.0 ** -14),
        "one-row-2^-20": _row_scaled(1 + 2.0 ** -20),
    },
}
"""Real faults of each class, its own computation changed before it runs, for the procedures' power on
it (`null_runs`'s `dew_fn`): the cross-entropy head's states scaled by a wrong factor at three sizes,
its column scale rounded to bfloat16 where float32 is claimed, a column of its vocabulary dropped; the
decoder's parameters scaled, or rounded to bfloat16; for each orbit class a term dropped, a term's sign
flipped, its arithmetic in float16, a chunk of its terms skipped, its outputs scaled by 1 + 2^-22, and one
row of each output moved by 2^-14 of itself; and attention's one row by 2^-20, sixteen float32 units in
its last place, which the worst-row rule alone resolves among its 512 rows; for the packed attention
biases its first key row, a row of float64 zero, ten and 1,024 times its own residue, and a value term
dropped."""


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
    from reference_error import NO_ZERO_ROWS, REDRAW, _drawn, _Draws, _staged

    # The same draws under either procedure, so their verdicts compare run by run.
    rng = np.random.default_rng([chunk, int(ratio * 1000), seeds])
    span = (1 + REDRAW) * seeds
    if law in CLASSES:
        reference, draw, truth, *_ = CLASSES[law]()
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
                          [[NO_ZERO_ROWS] * leaves] * count, np.ones((count, leaves), bool), (0,) * leaves)

        over, unresolved, stage = _staged(drawn, seeds, law, two_stage=two_stage)
        verdicts.append(("over" if over else "inconclusive" if unresolved else "pass", stage))
    return verdicts


def row_controls(width: int, rows: int, seeds: int, runs: int, chunk: int) -> tuple[list, list[float]]:
    """Check 1's verdicts (`_staged`) on its worst-row rule alone for equally exact pairs as sites run it,
    `runs` runs from `chunk`'s own stream: both arms on one truth, so each row's size is shared; `rows`
    rows a draw of `width` values, truths and errors standard normal, the arms' errors independent; the
    pooled ratio exactly one, so only the rows decide. Each run's verdict, and its site statistic, the
    least factor that would have failed it at both stages (`_staged`'s second stage is drawn only where
    the first fails), whose high quantiles calibrate NARROW_FACTORS."""
    import numpy as np
    from reference_error import NO_ZERO_ROWS, REDRAW, _Draws, _staged, _verdict

    rng = np.random.default_rng([0x5A, width, rows, seeds, chunk])
    span = (1 + REDRAW) * seeds
    verdicts, statistics = [], []
    for _ in range(runs):
        truth = np.sqrt(np.mean(np.square(rng.standard_normal((span, rows, width))), axis=2))
        worst = np.stack([(np.sqrt(np.mean(np.square(rng.standard_normal((span, rows, width))), axis=2))
                           / truth).max(axis=1) for _ in range(2)], axis=-1)[:, None, :]

        def drawn(start, stop, worst=worst):
            count = stop - start
            return _Draws(["rows"], np.ones((count, 1)), np.ones((count, 1)), np.log(worst[start:stop]),
                          [[NO_ZERO_ROWS]] * count, np.ones((count, 1), bool), (width,))

        over, unresolved, stage = _staged(drawn, seeds, f"rows of {width}")
        verdicts.append((_verdict(over, unresolved), stage))
        first = worst[:seeds, 0, 0].max() / worst[:seeds, 0, 1].max()
        statistics.append(float(min(first, worst[:, 0, 0].max() / worst[:, 0, 1].max())))
    return verdicts, statistics


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help=f"a class ({', '.join(sorted(CLASSES))}), or with --ratio a control's "
                        "law: a class's or exact-f, flip, gamma-1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ratio", type=float, help="run known-ratio controls of this population ratio")
    mode.add_argument("--draws", type=int, help="print the reference's own mean squared error from float64 "
                      "by leaf over this many draws past every run's, for a sizing model")
    mode.add_argument("--rows", type=int, help="run the worst-row rule's controls (`row_controls`) for rows "
                      "of this many values, the name the rows a draw; print the verdicts and statistics")
    parser.add_argument("--today", action="store_true", help="the procedure before the two stages")
    parser.add_argument("--fault", help="run the class's reference with this fault of its own (FAULTS) "
                        "as Dew, for power")
    parser.add_argument("--seeds", type=int, help="the class's declared count; without it, size it "
                        "(`seeds_for`, on the seeds below OFFSET) and print it")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--orders", type=int, help="judge an orbit class's rows over this many orders")
    parser.add_argument("--paired", action="store_true", help="run both arms on the same draws, as a site "
                        "does: Dew the reference at another order (REORDERED), or with --fault its fault")
    options = parser.parse_args(argv)
    from reference_error import _drawn, null_runs, seeds_for

    if options.rows is not None:
        if options.today or options.paired or options.orders or options.fault or options.seeds is None:
            parser.error("--rows takes the rows a draw as its name and --seeds, and no other mode's flags")
        verdicts, statistics = row_controls(options.rows, int(options.name), options.seeds, options.runs,
                                            options.chunk)
        import numpy as np

        print(json.dumps({"rows": int(options.name), "width": options.rows, "seeds": options.seeds,
                          "chunk": options.chunk, "runs": len(verdicts),
                          "reds": sum(verdict != "pass" for verdict, _ in verdicts),
                          "quantiles": {str(tail): float(np.quantile(statistics, 1 - tail))
                                        for tail in (5e-4, 1e-4)}}), flush=True)
        return 0
    if options.ratio is not None:
        if options.orders or options.paired or options.fault:
            parser.error("--ratio runs synthetic controls; --orders, --paired and --fault do not apply")
        verdicts = controls(options.name, options.ratio, options.seeds, options.runs, options.chunk,
                            not options.today)
        print(json.dumps({"control": options.name, "ratio": options.ratio, "seeds": options.seeds,
                          "today": options.today, "chunk": options.chunk, "verdicts": verdicts}), flush=True)
        return 0
    reference, draw, truth, permuting = CLASSES[options.name]()
    if (options.orders or options.paired) and permuting is None:
        parser.error(f"{options.name} has no orbit; --orders and --paired take an orbit class")
    if options.draws is not None:
        import numpy as np

        seeds = range(2 * OFFSET * 1024 + options.chunk * options.draws,
                      2 * OFFSET * 1024 + (options.chunk + 1) * options.draws)
        draws = _drawn(reference, reference, draw, seeds, truth)
        print(json.dumps({"class": options.name, "chunk": options.chunk, "names": draws.names,
                          "mse": np.asarray(draws.mine).tolist(),
                          "log_worst": np.asarray(draws.worst[..., 0]).tolist()}), flush=True)
        return 0
    def twin(*inputs):
        return reference(*permuting(inputs, REORDERED))

    if options.seeds is None:
        # A paired class sizes as a paired site does: its twin against it on the same draws.
        sized = seeds_for(reference, draw, options.name, truth_fn=truth,
                          dew_fn=twin if options.paired else None)
        print(json.dumps({"class": options.name, "paired": options.paired, "seeds": sized}), flush=True)
        return 0
    other = None
    if options.fault:
        other = FAULTS[options.name][options.fault](reference)
    elif options.paired:
        other = twin
    verdicts = null_runs(reference, draw, options.name, seeds=options.seeds, runs=options.runs,
                         first=OFFSET + options.chunk * options.runs, truth_fn=truth,
                         two_stage=not options.today, dew_fn=other, paired=options.paired,
                         orders=orbited(permuting, options.orders) if options.orders else None)
    print(json.dumps({"class": options.name, "chunk": options.chunk, "seeds": options.seeds,
                      "today": options.today, "fault": options.fault, "orders": options.orders,
                      "paired": options.paired, "verdicts": verdicts}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
