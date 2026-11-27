"""Check 2's native step witness for the Mosaic attention template's forward (row 2084 (f)): each
online softmax state its certify launch stores after a key tile (`attention_mosaic.certified`), held
inside one plain step's enclosure on the stored state before it and the tile's operands, and the output
inside the finish's enclosure on the last state.

One plain fold-form step of a query row over key tile j (attention_algebra.py on base/kernels, its order
confirmed by the attention lane, 11-06):

1. the products `s = q kᵀ`, bf16 operands on the tensor cores in fp32 (the native-dot contract,
   `interval_certificates.dot`), a masked entry -inf;
2. the maximum `m_j = max(m_{j-1}, max(s) f)`, f = fp32(scale log2(e)) the walk's units, rounded once;
3. the factor `alpha = ex2(m_{j-1} - m_j)`, its subtraction fp32, zero from a first tile's -inf;
4. the weights `p = ex2(s f - m_j)`, the multiply and the subtraction one FMA or two roundings;
5. the denominator `l_j = l_{j-1} alpha + sum(p)`, the row's sum in any order (each add rounded to nearest,
   gamma_n of its terms) and its last add contracted or not;
6. the sums `acc_j = acc_{j-1} alpha + bf16(p) v`, the scaling an fp32 multiply, the product accumulated
   into it on the tensor cores (the native-dot contract with the scaled sums as one more term);
7. the output `acc / l` (div.rn, or rcp.approx and a multiply: within two steps of the fp32 grid), cast to
   the output dtype, zero where `l` is.

`ex2` lies within the lane's reach of its correctly rounded value (`interval_certificates.exp2`), measured
over the step's own operands (`attention_lane`). Each state is held from the stored state before it, so no
enclosure grows across the walk; a bias below one step's enclosure is not visible (design-20's stated limit).
A query row's tiles past its last kept key are walked with nothing kept, which leaves its state bit for bit;
the witness holds a row's states up to the last tile it keeps a key of, and its output from there.

The fast backward has no certify form: its gradients, dQ, dK and dV, are held whole (`backward_witnessed`)
from the forward's stored output, maxima and denominators, each a tensor-core sum over the walked tiles of
its weights' or score cotangents' enclosures (row 6130)."""

import dataclasses
import functools

import attention_oracle
import numpy as np
from interval_certificates import (
    Iv,
    Lane,
    add,
    cast,
    dot,
    exp2,
    exp2_reach,
    fp32,
    hull,
    mul,
    point,
    tanh,
    tanh_reach,
    within,
)
from op_certificates import BFLOAT16, FLOAT32, nearest, stepped

from dew.nn.kernels.templates import attention as template, attention_mosaic
from dew.nn.kernels.templates.attention import LOG2E


def walk_factor(width: int) -> float:
    """The walk's units: the logit scale of a query `width` wide times log2(e), rounded to fp32 once."""
    return float(np.float32(attention_mosaic.logit_scale(width, None) * LOG2E))


@functools.cache
def attention_lane() -> Lane:
    """The arithmetic an interpreted certify launch runs in: `ex2`'s reach over the step's exponents,
    (-126, 0] (XLA's exp2 measured there, `exp2_reach`), and whether it flushes (`certified.lane`). Its
    products' column blocks are the witness's to read (`tiles`), not the lane's."""
    import jax
    import jax.numpy as jnp
    from certified import lane

    reach = exp2_reach(lambda x: jax.jit(jnp.exp2)(jnp.asarray(x)), low=-126.0, high=0.0)
    return Lane(reach, lane().flushes, 0, tanh_reach=tanh_reach(lambda x: jax.jit(jnp.tanh)(jnp.asarray(x))))


def summed(p: Iv, lane: Lane) -> Iv:
    """A row's fp32 sum of the weights within `p` in any order: their exact sums, widened by gamma_n of
    their magnitudes (n the terms, each add rounded to nearest at fp32's unit), then on the fp32 grid."""
    terms = p.lo.shape[-1]
    unit = 2.0 ** -FLOAT32.precision
    gamma = terms * unit / (1 - terms * unit)
    magnitude = np.maximum(np.abs(p.lo), np.abs(p.hi)).sum(-1)
    slack = gamma * magnitude + (terms + 1) * 2.0 ** -52 * magnitude
    return fp32(Iv(p.lo.sum(-1) - slack, p.hi.sum(-1) + slack), lane)


def step(m_before, l_before, acc_before, q, keys, values, kept, factor: float, m_after, lane: Lane,
         capped: tuple[float, float] | None = None) -> dict:
    """One plain step's enclosures of a query block's rows over a key tile: `m`, `l` and `acc` after it, from
    the stored state before it (`m_before` -inf on a first tile) and the stored maximum after it, which the
    weights and the factor read (each state is held from the stored ones). Under a cap (`capped`, its two
    folded constants, `capped_constants`), the logits are the products times the first rounded, their tanh,
    times the second (`attention_algebra.logits_of`'s base-2 cap), and the tile's maximum is theirs."""
    products = dot(point(q), point(keys.T), lane)
    if capped is None:
        products = Iv(np.where(kept, products.lo, -np.inf), np.where(kept, products.hi, -np.inf))
        peak = fp32(mul(Iv(products.lo.max(-1), products.hi.max(-1)), point(factor)), lane)
        logits = mul(products, point(factor))
    else:
        inner, outer = capped
        logits = mul(tanh(fp32(mul(products, point(inner)), lane), lane), point(outer))
        logits = Iv(np.where(kept, logits.lo, -np.inf), np.where(kept, logits.hi, -np.inf))
        peak = fp32(Iv(logits.lo.max(-1), logits.hi.max(-1)), lane)
    maximum = Iv(np.maximum(m_before, peak.lo), np.maximum(m_before, peak.hi))
    first = np.isneginf(m_before)
    alpha = exp2(fp32(point(np.where(first, 0.0, m_before - m_after)), lane), lane)
    alpha = Iv(np.where(first, 0.0, alpha.lo), np.where(first, 0.0, alpha.hi))
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


def witnessed(query, key, value, kept, records: dict, *, lane: Lane | None = None,
              capped: tuple[float, float] | None = None) -> dict[str, tuple[np.ndarray, ...]]:
    """Each stored state and the output of a certify launch's `records` (`launched`) over `query`, `key` and
    `value` (`[B, S, H, D]`, bf16), held to their enclosures under the mask `kept` (`[B or 1, 1, Q, K]`,
    `attention_oracle.kept`'s), under a cap's constants where `capped` (`capped_constants`): for each record,
    the entries held and those inside."""
    lane = attention_lane() if lane is None else lane
    q, k, v = (np.asarray(np.asarray(x, np.float32), np.float64) for x in (query, key, value))
    count, width = records["m"].shape[2], records["tile"]
    factor = walk_factor(q.shape[3])
    lane = dataclasses.replace(lane, block=width)
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
                             tile_kept, factor, m_after, lane, capped)
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


def certified(query, key, value, spans, hooks=None):
    """The interpreted certify launch's tiles and records (`attention_mosaic.certified`) at a Hopper's tiles,
    the target an interpreted launch derives its config for (`generation.assumed`)."""
    from dew.nn.kernels import generation

    with generation.assumed("sm90"):
        return attention_mosaic.certified(query, key, value, spans, hooks, interpret=True)


SHAPE = (1, 512, 2, 64)
"""The witnessed call's `[B, S, H, D]`: a Hopper's and an Ampere's launch walk more than one key tile at
this length, so a fault in the rescale has a second step to break."""

FORMS = {"full": {}, "causal": {"causal": True},
         "window": {"causal": True, "sliding_window": 192},
         "documents": {"causal": True,
                       "segment_ids": np.repeat(np.arange(1, 4, dtype=np.int32), (160, 224, 128))[None]},
         "capped": {"causal": True}}
"""Each witnessed form, its mask as `template.spans` and `attention_oracle.kept` both take it: full, causal, a
causal window of 192 keys (each query's tiles partly walked), three causal documents packed into the call,
their bounds off the key tiles, and a causal call under a cap (`CAPS`)."""

CAPS = {"capped": 3.0}
"""A form's logit cap (Gemma 2's, `template.Capped`), low enough that the witnessed draws' scaled logits
(their keys doubled) reach its bend, its reciprocal inexact in fp32 so each constant's rounding is read."""


def hooks_of(form: str):
    """The witnessed call's hooks under `form`: its cap as a score hook where it has one (`CAPS`)."""
    from dew.nn.kernels.templates.hooks import Hook

    return None if form not in CAPS else template.Hooks(score=Hook(template.Capped(CAPS[form])))


def capped_constants(form: str, width: int) -> tuple[float, float] | None:
    """The two fp32 constants a base-2 walk folds a cap into (`attention_algebra.logits_of`), as the witness's
    unscaled products take them: the walk's scale over the cap, `fp32(scale * Capped.reciprocal)` of the
    scale left after the queries' power of two (`template.prescale`, which scales the products exactly), times
    that power, before the tanh; and `fp32(cap) * fp32(log2(e))` after it. None for an uncapped form."""
    import jax.numpy as jnp

    if form not in CAPS:
        return None
    capped = template.Capped(CAPS[form])
    power, left = template.prescale(jnp.bfloat16, width, None)
    inner = power * float(np.float32(left * capped.reciprocal))
    return inner, float(np.float32(capped.softcap) * np.float32(LOG2E))


def mask_of(form: str):
    """The keys each query of the witnessed call keeps under `form` (`attention_oracle.kept`)."""
    return attention_oracle.kept(SHAPE[1], SHAPE[1], **FORMS[form])


def spans_of(form: str, batch: int):
    """The template's spans of the witnessed call under `form` (`template.spans`)."""
    import jax.numpy as jnp

    stated = {name: jnp.asarray(value) if name == "segment_ids" else value
              for name, value in FORMS[form].items()}
    return template.spans(SHAPE[1], SHAPE[1], batch, **stated)


def operands(seed: int):
    """bf16 queries, keys and values of SHAPE, drawn on the host in NumPy so a GPU tier's prepare computes
    nothing on its compile-only device (row 9546); the keys scaled so the rows' maxima move across tiles."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    bf16 = np.dtype(jnp.bfloat16)
    q = rng.standard_normal(SHAPE, dtype=np.float32).astype(bf16)
    k = (2.0 * rng.standard_normal(SHAPE, dtype=np.float32)).astype(bf16)
    v = rng.standard_normal(SHAPE, dtype=np.float32).astype(bf16)
    return q, k, v


def _certified_native(query, key, value, *, form: str, site: str):
    """Run the native certify launch through `ahead_arm` (row 9546) so GPU-tier prepare compiles it without
    eagerly touching a compile-only device, and return `(block_k, (out, rows, sums))`."""
    import jax
    import jax.numpy as jnp
    import pytest
    from reference_error import ahead_arm

    from dew.nn.kernels import generation

    abstract_spans = jax.eval_shape(lambda: spans_of(form, query.shape[0]))
    declined = attention_mosaic.forward_declined(
        *(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (query, key, value)), abstract_spans,
        hooks_of(form))
    if declined is not None:
        pytest.skip(f"the Mosaic mainloop declines the witnessed call here: {declined}")

    def program(q, k, v):
        tiles, result = attention_mosaic.certified(q, k, v, spans_of(form, q.shape[0]), hooks_of(form))
        return jnp.asarray(tiles.block_k, jnp.int32), result

    shapes = ", ".join(f"{np.dtype(x.dtype).name}{list(np.shape(x))}" for x in (query, key, value))
    label = f"attention witness, {site}, {form}, {generation.assumption()}, {shapes}"
    block_k, found = ahead_arm(jax.jit(program), label)(query, key, value)
    return int(np.asarray(block_k)), found


def launched(query, key, value, form: str, *, interpret: bool, site: str = "witness") -> dict:
    """The certify launch's own records (`attention_mosaic.certified`) under `form` (`FORMS`), as float64,
    and the key tile it walked (`tile`), each record's tiles that wide."""
    if not interpret:
        block_k, (out, rows, sums) = _certified_native(query, key, value, form=form, site=site)
    else:
        spans = spans_of(form, query.shape[0])
        tiles, (out, rows, sums) = certified(query, key, value, spans, hooks_of(form))
        block_k = tiles.block_k
    return {"out": np.asarray(out, np.float64), "m": np.asarray(rows[0], np.float64),
            "l": np.asarray(rows[1], np.float64), "acc": np.asarray(sums, np.float64), "tile": block_k}


def refused(found: dict, name: str) -> int:
    """The entries of record `name` the witness holds and finds outside their enclosures."""
    rows, inside = found[name]
    return int(np.sum(rows & ~inside))


def assert_witnessed(seed: int, form: str, *, interpret: bool, lane: Lane | None = None) -> None:
    """Every stored state and the output of the certify launch over draw `seed` under `form` (`FORMS`), inside
    its step's enclosure."""
    import tier_wire

    q, k, v = operands(seed)
    records = launched(q, k, v, form, interpret=interpret)
    if not interpret and tier_wire.preparing():
        return
    found = witnessed(q, k, v, mask_of(form), records, lane=lane, capped=capped_constants(form, SHAPE[3]))
    for name, (rows, inside) in found.items():
        assert rows.any(), name
        assert inside[rows].all(), f"{name}: {refused(found, name)} of {int(rows.sum())} held entries outside"


def assert_faults_refused(*, interpret: bool, lane: Lane | None = None) -> None:
    """Each fault the witness exists for, refused at its elements: a dropped rescale, a maximum from the
    wrong tile, sums rounded to bf16 where the walk holds fp32, a causal launch's records held under a mask
    one key short of its own, a window's and the documents' records held as one undivided causal call's, and a
    capped call's records held as an uncapped one's, and the reverse.
    Every native program launches before any check so GPU-tier prepare compiles them all (row 9546)."""
    import tier_wire

    q, k, v = operands(2)
    mask, causal = mask_of("full"), mask_of("causal")
    truth = launched(q, k, v, "full", interpret=interpret, site="faults")
    claimed = launched(q, k, v, "causal", interpret=interpret, site="faults")
    bounded = {form: launched(q, k, v, form, interpret=interpret, site="faults")
               for form in ("window", "documents")}
    capped = launched(q, k, v, "capped", interpret=interpret, site="faults")
    if not interpret and tier_wire.preparing():
        return
    m, acc = truth["m"], truth["acc"]
    alpha = np.exp2(np.where(np.isfinite(m[:, :, :-1]), m[:, :, :-1] - m[:, :, 1:], 0.0))
    dropped = dict(truth, acc=np.concatenate(
        [acc[:, :, :1], acc[:, :, 1:] + acc[:, :, :-1] * (1 - alpha)[..., None]], 2))
    assert refused(witnessed(q, k, v, mask, dropped, lane=lane), "acc")
    wrong = dict(truth, m=np.concatenate([m[:, :, :1], m[:, :, :-1]], 2))
    assert refused(witnessed(q, k, v, mask, wrong, lane=lane), "m")
    narrowed = dict(truth, acc=np.asarray(nearest(acc, BFLOAT16)))
    assert refused(witnessed(q, k, v, mask, narrowed, lane=lane), "acc")
    shifted = causal & ~np.eye(SHAPE[1], dtype=bool)[None, None]
    held = witnessed(q, k, v, shifted, claimed, lane=lane)
    assert sum(refused(held, name) for name in ("m", "l", "acc"))
    for form, records in bounded.items():
        # A window's or the documents' launch held as one undivided causal call: its bounds read.
        held = witnessed(q, k, v, causal, records, lane=lane)
        assert sum(refused(held, name) for name in ("m", "l", "acc")), form
    # A capped launch held as an uncapped call's, and an uncapped one as a capped call's: its cap read.
    assert sum(refused(witnessed(q, k, v, causal, capped, lane=lane), name) for name in ("m", "l", "acc"))
    constants = capped_constants("capped", SHAPE[3])
    assert sum(refused(witnessed(q, k, v, causal, claimed, lane=lane, capped=constants), name)
               for name in ("m", "l", "acc"))

def backward_launched(query, key, value, dout, form: str, *, interpret: bool, site: str = "backward") -> dict:
    """The fast backward's gradients (`attention_mosaic.backward`, not accurate: its one fused launch where
    derived) of the call under `form` (`FORMS`), from its training forward's output residuals
    (`template.Kept.OUTPUT`: the output, the rows' maxima in the walk's log2 units and their denominators),
    each as float64: what `backward_witnessed` holds them by."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation

    def program(q, k, v, do):
        spans = spans_of(form, q.shape[0])
        out, *residuals = attention_mosaic.forward(q, k, v, spans, hooks_of(form),
                                                   residuals=template.Kept.OUTPUT, interpret=interpret)
        grads = attention_mosaic.backward(q, k, v, spans, hooks_of(form), do, residuals=tuple(residuals),
                                          accurate=False, interpret=interpret)
        return out, residuals[0], residuals[1], grads.query, grads.key, grads.value

    if interpret:
        with generation.assumed("sm90"):
            found = jax.jit(program)(query, key, value, dout)
    else:
        from reference_error import ahead_arm

        shapes = ", ".join(f"{np.dtype(x.dtype).name}{list(np.shape(x))}" for x in (query, key, value, dout))
        label = f"attention backward witness, {site}, {form}, {generation.assumption()}, {shapes}"
        found = ahead_arm(jax.jit(program), label)(query, key, value, dout)
    names = ("out", "m", "l", "dq", "dk", "dv")
    return {name: np.asarray(jnp.asarray(x, jnp.float32), np.float64)
            for name, x in zip(names, found, strict=True)}


def backward_witnessed(query, key, value, dout, kept, records: dict, *, lane: Lane | None = None,
                       softcap: float | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each gradient of the fast backward's `records` (`backward_launched`) held elementwise to its
    enclosure under the mask `kept`, from the forward's stored output, maxima and denominators, natural
    units (as the forward's witness holds each state from the stored ones; the attention lane's order,
    2026-11-27): each query's and key's products `s` (fp32 on the tensor cores), the weights
    `P = exp2(s f - m log2(e)) (1/l)` (the walk's factor `f`; the maximum's units an fp32 product; the
    multiply and the subtraction one FMA or two roundings; `1/l` a divide within two steps of its correctly
    rounded value, `div.full`'s reach), `dP = dO Vᵀ` (fp32), the score cotangents `P ((dP - c) - (D - c))`
    about an anchor `c`, one of the row's `dP` as XLA sums it (`gradient.split_delta`), held for any such
    anchor: `dP - D` within three fp32 roundings of the row's magnitudes, `D`'s and `D - c`'s sums `gamma`
    of their terms; times the call's whole scale (`grad_scale`) in fp32, cast to bf16; and the gradients
    `dV = bf16(P)ᵀ dO`, `dK = dSᵀ Q`, `dQ = dS K`, each a tensor-core sum over every walked query or key in
    any order, cast to bf16. Under a cap (`softcap`), the logits are recomputed natural as the hook states
    them, `cap tanh(fp32(s scale) / cap)` (each divide within two steps, `div.full`'s reach; tanh within
    its lane's reach), their weights `exp2` of their natural units, and each score cotangent times the hook's
    slope as JAX's vjp of it takes it on ones: `cap`, times `1 - t²` (`one_minus_square`, fused or not),
    over `cap`. For each gradient, the entries held and those inside."""
    lane = attention_lane() if lane is None else lane
    lane = dataclasses.replace(lane, block=0)
    q, k, v, do = (np.asarray(np.asarray(x, np.float32), np.float64) for x in (query, key, value, dout))
    power, left = template.prescale(np.dtype(query.dtype), q.shape[3], None)
    factor, grad_scale = float(np.float32(left * LOG2E)) * power, power * left
    unit = 2.0 ** -FLOAT32.precision
    held = {name: [] for name in ("dq", "dk", "dv")}
    for b in range(q.shape[0]):
        keeps = np.broadcast_to(kept, (q.shape[0], 1, *kept.shape[2:]))[b, 0]
        for h in range(q.shape[2]):
            kv = h * k.shape[2] // q.shape[2]
            peak = fp32(mul(point(records["m"][b, h]), point(LOG2E)), lane)
            total = records["l"][b, h]
            products = dot(point(q[b, :, h]), point(k[b, :, kv].T), lane)
            below = Iv(-peak.hi[:, None], -peak.lo[:, None])
            if softcap is None:
                logits = mul(products, point(factor))
                exponents = hull(fp32(add(logits, below), lane), fp32(add(fp32(logits, lane), below), lane))
                slope = None
            else:
                folded = _divided(fp32(mul(products, point(grad_scale)), lane), softcap, lane)
                bent = tanh(folded, lane)
                natural = fp32(mul(bent, point(softcap)), lane)
                logits = mul(natural, point(LOG2E))
                exponents = hull(fp32(add(logits, below), lane), fp32(add(fp32(logits, lane), below), lane))
                exponents = hull(exponents, fp32(mul(fp32(add(natural, Iv(-records["m"][b, h][:, None],
                                                                          -records["m"][b, h][:, None])),
                                                          lane), point(LOG2E)), lane))
                square = hull(fp32(add(point(1.0), Iv(*(-x for x in reversed(fp32(mul(bent, bent), lane))))),
                                   lane),
                              fp32(mul(fp32(add(point(1.0), Iv(-bent.hi, -bent.lo)), lane),
                                       fp32(add(point(1.0), bent), lane)), lane))
                slope = _divided(fp32(mul(point(softcap), square), lane), softcap, lane)
            exponents = Iv(np.where(keeps, np.minimum(exponents.lo, 0.0), 0.0),
                           np.where(keeps, np.minimum(exponents.hi, 0.0), 0.0))
            inverse = np.where(total > 0, 1.0 / np.where(total > 0, total, 1.0), 0.0)
            low, high = nearest(inverse, FLOAT32), nearest(inverse, FLOAT32)
            for _ in range(2):
                low, high = stepped(low, FLOAT32, -np.inf), stepped(high, FLOAT32, np.inf)
            weights = fp32(mul(exp2(exponents, lane), Iv(low[:, None], high[:, None])), lane)
            weights = Iv(np.where(keeps, weights.lo, 0.0), np.where(keeps, weights.hi, 0.0))
            pulled = dot(point(do[b, :, h]), point(v[b, :, kv].T), lane)
            # D and an anchor's terms: dO times the output, and times any key's values (`D - c`'s, summed
            # whole), each bf16 product exact in fp32, their sums gamma_n of their magnitudes.
            out, width = records["out"][b, :, h], do.shape[3]
            gamma = width * unit / (1 - width * unit) + (width + 2) * 2.0 ** -52
            magnitudes = (np.abs(do[b, :, h]) * (np.abs(out) + np.abs(v[b, :, kv]).max(0))).sum(-1)
            delta = (do[b, :, h] * out).sum(-1)
            reach = np.maximum(np.abs(pulled.lo), np.abs(pulled.hi))
            anchor = reach.max(-1)
            slack = ((gamma * magnitudes)[:, None]
                     + 3 * unit * (reach + np.abs(delta)[:, None] + 3 * anchor[:, None]))
            differences = Iv(pulled.lo - delta[:, None] - slack, pulled.hi - delta[:, None] + slack)
            cotangents = fp32(mul(weights, differences), lane)
            if slope is not None:
                cotangents = fp32(mul(cotangents, slope), lane)
            cotangents = fp32(mul(cotangents, point(grad_scale)), lane)
            cotangents = cast(Iv(np.where(keeps, cotangents.lo, 0.0), np.where(keeps, cotangents.hi, 0.0)),
                              BFLOAT16)
            dv = dot(cast(Iv(weights.lo.T, weights.hi.T), BFLOAT16), point(do[b, :, h]), lane)
            dk = dot(Iv(cotangents.lo.T, cotangents.hi.T), point(q[b, :, h]), lane)
            dq = dot(cotangents, point(k[b, :, kv]), lane)
            for name, found in (("dq", dq), ("dk", dk), ("dv", dv)):
                stored = records[name][b, :, h if name == "dq" else kv]
                held[name].append((np.ones(stored.shape, bool), within(stored, cast(found, BFLOAT16))))
    return {name: (np.concatenate([rows.ravel() for rows, _ in found]),
                   np.concatenate([inside.ravel() for _, inside in found])) for name, found in held.items()}


def _divided(numerator: Iv, divisor: float, lane: Lane) -> Iv:
    """An fp32 quotient of fp32 values within `numerator` by the fp32 `divisor`, within two steps of its
    correctly rounded value (`div.full`'s reach)."""
    found = fp32(Iv(numerator.lo / divisor, numerator.hi / divisor), lane)
    lo, hi = found.lo, found.hi
    for _ in range(2):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(lo, hi)


BACKWARD_FORMS = ("full", "causal", "window", "documents", "capped")
"""The forms the backward's witness holds: every one (a capped call's its hook's slope too)."""


def backward_operands(seed: int):
    """`operands(seed)` and a bf16 cotangent of the output drawn beside them on the host (row 9546)."""
    import jax.numpy as jnp

    q, k, v = operands(seed)
    rng = np.random.default_rng(seed + 7)
    return q, k, v, rng.standard_normal(SHAPE, dtype=np.float32).astype(np.dtype(jnp.bfloat16))


def assert_backward_witnessed(seed: int, form: str, *, interpret: bool, lane: Lane | None = None) -> None:
    """Every gradient of the fast backward over draw `seed` under `form` inside its enclosure
    (`backward_witnessed`)."""
    import tier_wire

    q, k, v, do = backward_operands(seed)
    records = backward_launched(q, k, v, do, form, interpret=interpret)
    if not interpret and tier_wire.preparing():
        return
    held = backward_witnessed(q, k, v, do, mask_of(form), records, lane=lane, softcap=CAPS.get(form))
    for name, (rows, inside) in held.items():
        assert rows.any(), name
        assert inside[rows].all(), f"{name}: {int((rows & ~inside).sum())} of {int(rows.sum())} outside"


def assert_backward_faults_refused(*, interpret: bool, lane: Lane | None = None) -> None:
    """Each fault the backward's witness exists for, refused at its elements: dK at twice its scale, dQ and
    dV a sixty-fourth off, a causal call's gradients held as a full call's, and a capped call's as an uncapped
    one's. Every native program launches before any check (row 9546)."""
    import tier_wire

    q, k, v, do = backward_operands(3)
    causal = backward_launched(q, k, v, do, "causal", interpret=interpret, site="faults")
    capped = backward_launched(q, k, v, do, "capped", interpret=interpret, site="faults")
    if not interpret and tier_wire.preparing():
        return
    mask = mask_of("causal")
    for name, bent in (("dk", 2.0), ("dq", 1.0 + 2.0 ** -6), ("dv", 1.0 - 2.0 ** -6)):
        held = backward_witnessed(q, k, v, do, mask, dict(causal, **{name: causal[name] * bent}), lane=lane)
        assert refused(held, name), name
    held = backward_witnessed(q, k, v, do, mask_of("full"), causal, lane=lane)
    assert sum(refused(held, name) for name in ("dq", "dk", "dv"))
    # A capped call's gradients held as an uncapped call's: the hook's slope read.
    held = backward_witnessed(q, k, v, do, mask, capped, lane=lane)
    assert sum(refused(held, name) for name in ("dq", "dk", "dv"))
