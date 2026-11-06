"""Check 2's native step witness for the Mosaic attention template's forward (row 2084 (f)): each
online softmax state its certify launch stores after a key tile (`attention_mosaic.certified`), held
inside one plain step's enclosure on the stored state before it and the tile's operands, and the output
inside the finish's enclosure on the last state.

One plain fold-form step of a query row over key tile j (attention_algebra.py on base/kernels, its order
confirmed by the attention lane, 11-06):

1. the products `s = q kᵀ`, bf16 operands on the tensor cores in fp32 (the native-dot contract,
   `recurrence_witness.dot`), a masked entry -inf;
2. the maximum `m_j = max(m_{j-1}, max(s) f)`, f = fp32(scale log2(e)) the walk's units, rounded once;
3. the factor `alpha = ex2(m_{j-1} - m_j)`, its subtraction fp32, zero from a first tile's -inf;
4. the weights `p = ex2(s f - m_j)`, the multiply and the subtraction one FMA or two roundings;
5. the denominator `l_j = l_{j-1} alpha + sum(p)`, the row's sum in any order (each add rounded to nearest,
   gamma_n of its terms) and its last add contracted or not;
6. the sums `acc_j = acc_{j-1} alpha + bf16(p) v`, the scaling an fp32 multiply, the product accumulated
   into it on the tensor cores (the native-dot contract with the scaled sums as one more term);
7. the output `acc / l` (div.rn, or rcp.approx and a multiply: within two steps of the fp32 grid), cast to
   the output dtype, zero where `l` is.

`ex2` lies within the lane's reach of its correctly rounded value (`recurrence_witness.exp2`), measured
over the step's own operands (`attention_lane`). Each state is held from the stored state before it, so no
enclosure grows across the walk; a bias below one step's enclosure is not visible (design-20's stated limit).
A query row's tiles past its last kept key are walked with nothing kept, which leaves its state bit for bit;
the witness holds a row's states up to the last tile it keeps a key of, and its output from there."""

import functools

import attention_oracle
import numpy as np
from native_certificates import PTX_EX2
from op_certificates import BFLOAT16, FLOAT32, nearest, stepped
from recurrence_witness import Iv, Lane, add, cast, dot, exp2, exp2_reach, fp32, hull, mul, point, within

from dew.nn.kernels.templates import attention as template, attention_mosaic
from dew.nn.kernels.templates.attention import LOG2E


def walk_factor(width: int) -> float:
    """The walk's units: the logit scale of a query `width` wide times log2(e), rounded to fp32 once."""
    return float(np.float32(attention_mosaic.logit_scale(width, None) * LOG2E))


@functools.cache
def attention_lane(interpret: bool) -> Lane:
    """The arithmetic a certify launch runs in: `ex2`'s reach over the step's exponents, (-126, 0] (an
    interpreted launch's XLA exp2 measured there, `exp2_reach`; a GPU's `ex2.approx`, `PTX_EX2`), and
    whether it flushes (`certified.lane`). Its products' column blocks are the witness's to read
    (`tiles`), not the lane's."""
    import jax
    import jax.numpy as jnp
    from certified import lane

    reach = (exp2_reach(lambda x: jax.jit(jnp.exp2)(jnp.asarray(x)), low=-126.0, high=0.0) if interpret
             else PTX_EX2)
    return Lane(reach, lane().flushes, 0)


def summed(p: Iv, lane: Lane) -> Iv:
    """A row's fp32 sum of the weights within `p` in any order: their exact sums, widened by gamma_n of
    their magnitudes (n the terms, each add rounded to nearest at fp32's unit), then on the fp32 grid."""
    terms = p.lo.shape[-1]
    unit = 2.0 ** -FLOAT32.precision
    gamma = terms * unit / (1 - terms * unit)
    magnitude = np.maximum(np.abs(p.lo), np.abs(p.hi)).sum(-1)
    slack = gamma * magnitude + (terms + 1) * 2.0 ** -52 * magnitude
    return fp32(Iv(p.lo.sum(-1) - slack, p.hi.sum(-1) + slack), lane)


def step(m_before, l_before, acc_before, q, keys, values, kept, factor: float, m_after, lane: Lane) -> dict:
    """One plain step's enclosures of a query block's rows over a key tile: `m`, `l` and `acc` after it, from
    the stored state before it (`m_before` -inf on a first tile) and the stored maximum after it, which the
    weights and the factor read (each state is held from the stored ones)."""
    products = dot(point(q), point(keys.T), lane)
    products = Iv(np.where(kept, products.lo, -np.inf), np.where(kept, products.hi, -np.inf))
    peak = fp32(mul(Iv(products.lo.max(-1), products.hi.max(-1)), point(factor)), lane)
    maximum = Iv(np.maximum(m_before, peak.lo), np.maximum(m_before, peak.hi))
    first = np.isneginf(m_before)
    alpha = exp2(fp32(point(np.where(first, 0.0, m_before - m_after)), lane), lane)
    alpha = Iv(np.where(first, 0.0, alpha.lo), np.where(first, 0.0, alpha.hi))
    logits = mul(products, point(factor))
    # The multiply and the subtraction: one FMA, or the product rounded first.
    exponents = hull(fp32(add(logits, point(-m_after[:, None])), lane),
                     fp32(add(fp32(logits, lane), point(-m_after[:, None])), lane))
    weights = exp2(Iv(np.where(kept, exponents.lo, 0.0), np.where(kept, exponents.hi, 0.0)), lane)
    weights = Iv(np.where(kept, weights.lo, 0.0), np.where(kept, weights.hi, 0.0))
    carried = mul(point(l_before), alpha)
    sums = summed(weights, lane)
    total = hull(fp32(add(carried, sums), lane), fp32(add(fp32(carried, lane), sums), lane))
    scaled = fp32(mul(point(acc_before), Iv(alpha.lo[:, None], alpha.hi[:, None])), lane)
    return {"m": maximum, "l": total,
            "acc": dot(cast(weights, BFLOAT16), point(values), lane, into=scaled)}


def finished(acc, total, fmt) -> Iv:
    """The output of the last state: `acc / l` within two steps of the fp32 grid of the exact quotient,
    cast to the output's format `fmt`, zero where `l` is."""
    quotient = np.where(total[:, None] > 0, acc / np.where(total > 0, total, 1.0)[:, None], 0.0)
    lo = hi = nearest(quotient, FLOAT32)
    for _ in range(2):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return cast(Iv(np.where(quotient == 0, 0.0, lo), np.where(quotient == 0, 0.0, hi)), fmt)


def tiles(keys: int, count: int) -> int:
    """The key tile width a launch walking `keys` keys in `count` tiles takes: the one power of two, at least
    16, of which `count` cover the keys and `count - 1` do not."""
    widths = [width for width in (16, 32, 64, 128, 256) if (count - 1) * width < keys <= count * width]
    assert len(widths) == 1, (keys, count, widths)
    return widths[0]


def witnessed(query, key, value, kept, *, interpret: bool, records=None) -> dict[str, tuple[np.ndarray, ...]]:
    """Each stored state and the output of the certify launch over `query`, `key` and `value` (`[B, S, H, D]`,
    bf16) under the mask `kept` (`[B or 1, 1, Q, K]`, `attention_oracle.kept`'s), held to its enclosure:
    for each record, the entries held and those inside. `records` replaces the launch's own (a fault)."""
    lane = attention_lane(interpret)
    q, k, v = (np.asarray(np.asarray(x, np.float32), np.float64) for x in (query, key, value))
    if records is None:
        out, rows, sums = certified(query, key, value, spans_of(kept, query.shape[0]), interpret=interpret)
        records = {"out": np.asarray(out, np.float64), "m": np.asarray(rows[0], np.float64),
                   "l": np.asarray(rows[1], np.float64), "acc": np.asarray(sums, np.float64)}
    count = records["m"].shape[2]
    width = tiles(k.shape[1], count)
    factor = walk_factor(q.shape[3])
    lane = Lane(lane.reach, lane.flushes, width)
    held = {name: [] for name in ("m", "l", "acc", "out")}
    for b in range(q.shape[0]):
        keeps = np.broadcast_to(kept, (q.shape[0], 1, *kept.shape[2:]))[b, 0]
        for h in range(q.shape[2]):
            m_before = np.full(q.shape[1], -np.inf)
            l_before, acc_before = np.zeros(q.shape[1]), np.zeros((q.shape[1], v.shape[3]))
            last = np.full(q.shape[1], -1)
            for j in range(count):
                at = slice(j * width, (j + 1) * width)
                tile_kept = keeps[:, at]
                walked = tile_kept.any(-1)
                if not walked.any():
                    continue
                kv = h * k.shape[2] // q.shape[2]
                m_after = records["m"][b, h, j]
                found = step(m_before, l_before, acc_before, q[b, :, h], k[b, at, kv], v[b, at, kv],
                             tile_kept, factor, m_after, lane)
                for name in ("m", "l", "acc"):
                    stored = records[name][b, h, j]
                    rows = walked if name != "acc" else walked[:, None] & np.ones(stored.shape, bool)
                    held[name].append((rows, within(stored, found[name]) | ~rows))
                m_before = np.where(walked, m_after, m_before)
                l_before = np.where(walked, records["l"][b, h, j], l_before)
                acc_before = np.where(walked[:, None], records["acc"][b, h, j], acc_before)
                last = np.where(walked, j, last)
            out = finished(acc_before, l_before, BFLOAT16)
            held["out"].append((np.ones(out.lo.shape, bool), within(records["out"][b, :, h], out)))
    return {name: (np.concatenate([rows.ravel() for rows, _ in found]),
                   np.concatenate([inside.ravel() for _, inside in found])) for name, found in held.items()}


def certified(query, key, value, spans, *, interpret: bool):
    """The certify launch's records (`attention_mosaic.certified`): interpreted, at a Hopper's tiles, the
    target an interpreted launch derives its config for (`generation.assumed`)."""
    from dew.nn.kernels import generation

    if not interpret:
        return attention_mosaic.certified(query, key, value, spans)
    with generation.assumed("sm90"):
        return attention_mosaic.certified(query, key, value, spans, interpret=True)


def spans_of(kept, batch: int):
    """The template's spans of a causal or full call whose mask `kept` is (`template.spans`)."""
    queries, keys = kept.shape[2:]
    causal = bool(np.array_equal(kept[0, 0], np.tril(np.ones((queries, keys), bool))))
    return template.spans(queries, keys, batch, causal=causal)


SHAPE = (1, 512, 2, 64)
"""The witnessed call's `[B, S, H, D]`: a Hopper's and an Ampere's launch walk more than one key tile at
this length, so a fault in the rescale has a second step to break."""


def operands(seed: int):
    """bf16 queries, keys and values of SHAPE, drawn; the keys scaled so the rows' maxima move across
    tiles."""
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(jax.random.key(seed), 3)
    q, k, v = (jax.random.normal(key, SHAPE, jnp.float32) for key in keys)
    return q.astype(jnp.bfloat16), (2 * k).astype(jnp.bfloat16), v.astype(jnp.bfloat16)


def records(query, key, value, mask, *, interpret: bool) -> dict[str, np.ndarray]:
    """The certify launch's own records (`attention_mosaic.certified`) under `mask`, as float64."""
    out, rows, sums = certified(query, key, value, spans_of(mask, query.shape[0]), interpret=interpret)
    return {"out": np.asarray(out, np.float64), "m": np.asarray(rows[0], np.float64),
            "l": np.asarray(rows[1], np.float64), "acc": np.asarray(sums, np.float64)}


def refused(found: dict, name: str) -> int:
    """The entries of record `name` the witness holds and finds outside their enclosures."""
    rows, inside = found[name]
    return int(np.sum(rows & ~inside))


def assert_witnessed(seed: int, causal: bool, *, interpret: bool) -> None:
    """Every stored state and the output of the certify launch over draw `seed`, causal or full, inside its
    step's enclosure."""
    q, k, v = operands(seed)
    found = witnessed(q, k, v, attention_oracle.kept(SHAPE[1], SHAPE[1], causal=causal), interpret=interpret)
    for name, (rows, inside) in found.items():
        assert rows.any(), name
        assert inside[rows].all(), f"{name}: {refused(found, name)} of {int(rows.sum())} held entries outside"


def assert_faults_refused(*, interpret: bool) -> None:
    """Each fault the witness exists for, refused at its elements: a dropped rescale, a maximum from the
    wrong tile, sums rounded to bf16 where the walk holds fp32, and a causal launch's records held under a
    mask one key short of its own."""
    q, k, v = operands(2)
    mask = attention_oracle.kept(SHAPE[1], SHAPE[1], causal=False)
    truth = records(q, k, v, mask, interpret=interpret)
    m, acc = truth["m"], truth["acc"]
    alpha = np.exp2(np.where(np.isfinite(m[:, :, :-1]), m[:, :, :-1] - m[:, :, 1:], 0.0))
    dropped = dict(truth, acc=np.concatenate(
        [acc[:, :, :1], acc[:, :, 1:] + acc[:, :, :-1] * (1 - alpha)[..., None]], 2))
    assert refused(witnessed(q, k, v, mask, interpret=interpret, records=dropped), "acc")
    wrong = dict(truth, m=np.concatenate([m[:, :, :1], m[:, :, :-1]], 2))
    assert refused(witnessed(q, k, v, mask, interpret=interpret, records=wrong), "m")
    narrowed = dict(truth, acc=np.asarray(nearest(acc, BFLOAT16)))
    assert refused(witnessed(q, k, v, mask, interpret=interpret, records=narrowed), "acc")
    causal = attention_oracle.kept(SHAPE[1], SHAPE[1], causal=True)
    shifted = causal & ~np.eye(SHAPE[1], dtype=bool)[None, None]
    claimed = records(q, k, v, causal, interpret=interpret)
    held = witnessed(q, k, v, shifted, interpret=interpret, records=claimed)
    assert sum(refused(held, name) for name in ("m", "l", "acc"))
