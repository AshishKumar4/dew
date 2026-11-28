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
from typing import NamedTuple

import attention_oracle
import numpy as np
from interval_certificates import (
    Iv,
    Lane,
    add,
    cast,
    dot,
    exp,
    exp2,
    exp2_reach,
    exp_flushes,
    exp_reach,
    fp32,
    hull,
    mul,
    natural_exp,
    neg,
    point,
    tanh,
    tanh_reach,
    within,
)
from op_certificates import BFLOAT16, FLOAT32, nearest, stepped

from dew.nn.kernels.templates import attention as template, attention_algebra as algebra, attention_mosaic
from dew.nn.kernels.templates.attention import LOG2E


def walk_factor(width: int) -> float:
    """The walk's units: the logit scale of a query `width` wide times log2(e), rounded to fp32 once."""
    return float(np.float32(attention_mosaic.logit_scale(width, None) * LOG2E))


@functools.cache
def attention_lane() -> Lane:
    """The arithmetic an interpreted certify launch runs in: `ex2`'s reach over the step's exponents,
    (-126, 0] (XLA's exp2 measured there, `exp2_reach`), `exp`'s over a row's natural differences
    (`exp_reach`), and whether it flushes (`certified.lane`). Its
    products' column blocks are the witness's to read (`tiles`), not the lane's."""
    import jax
    import jax.numpy as jnp
    from certified import lane

    def natural(x):
        return jax.jit(jnp.exp)(jnp.asarray(x))

    reach = exp2_reach(lambda x: jax.jit(jnp.exp2)(jnp.asarray(x)), low=-126.0, high=0.0)
    return Lane(reach, lane().flushes, 0, exp_reach=exp_reach(natural), exp_flushes=exp_flushes(natural),
                tanh_reach=tanh_reach(lambda x: jax.jit(jnp.tanh)(jnp.asarray(x))))


def summed(p: Iv, lane: Lane) -> Iv:
    """A row's fp32 sum of the weights within `p` in any order: their exact sums, widened by gamma_n of
    their magnitudes (n the row's terms that may be nonzero, each add rounded to nearest at fp32's unit,
    an add of an exact zero exact), then on the fp32 grid."""
    terms = ((p.lo != 0) | (p.hi != 0)).sum(-1)
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
"""Each witnessed form, the forward's and the backward's, its mask as `template.spans` and
`attention_oracle.kept` both take it: full, causal, a causal window of 192 keys (each query's tiles partly
walked), three causal documents packed into the call, their bounds off the key tiles, and a causal call
under a cap (`CAPS`)."""

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


def operands(seed: int, width: int = SHAPE[3]):
    """bf16 queries, keys and values of SHAPE (`width` wide), drawn on the host in NumPy so a GPU tier's
    prepare computes nothing on its compile-only device (row 9546); the keys scaled so the rows' maxima move
    across tiles."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    bf16 = np.dtype(jnp.bfloat16)
    shape = (*SHAPE[:3], width)
    q = rng.standard_normal(shape, dtype=np.float32).astype(bf16)
    k = (2.0 * rng.standard_normal(shape, dtype=np.float32)).astype(bf16)
    v = rng.standard_normal(shape, dtype=np.float32).astype(bf16)
    return q, k, v


def _certified_native(query, key, value, *, form: str, site: str):
    """Run the native certify launch through `ahead_arm` (row 9546) so GPU-tier prepare compiles it without
    eagerly touching a compile-only device, and return `(block_k, (out, rows, sums))`. The flow runs it on
    the generations its cells name (`gpu_flow.Cell`), where the mainloop takes every form, so a form it
    declines there fails with the reason `certified` raises, never a skip."""
    import jax
    import jax.numpy as jnp
    from reference_error import ahead_arm

    from dew.nn.kernels import generation

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


def assert_faults_refused(*, interpret: bool, lane: Lane | None = None,
                          faults: tuple[str, ...] | None = None) -> None:
    """Each fault the witness exists for among `faults` (`FAULTS`' names, every one by default), refused at
    its elements: a dropped rescale ('rescale'), a maximum from the wrong tile ('maximum'), sums rounded to
    bf16 where the walk holds fp32 ('narrowed'), a causal launch's records held under a mask one key short of
    its own ('shifted'), a window's and the documents' records held as one undivided causal call's
    ('bounded'), and a capped call's records held as an uncapped one's, and the reverse ('capped').
    Every native program its faults read launches before any check so GPU-tier prepare compiles them all (row
    9546)."""
    import tier_wire

    faults = tuple(FAULTS) if faults is None else faults
    q, k, v = operands(2)
    forms = dict.fromkeys(form for fault in faults for form in FAULTS[fault][0])
    runs = {form: launched(q, k, v, form, interpret=interpret, site="faults") for form in forms}
    if not interpret and tier_wire.preparing():
        return
    for fault in faults:
        forms, check = FAULTS[fault]
        check(q, k, v, *(runs[form] for form in forms), lane=lane)


def _rescale_refused(q, k, v, truth: dict, *, lane) -> None:
    """A full launch's records (`truth`) with a dropped rescale, refused at the sums."""
    m, acc = truth["m"], truth["acc"]
    alpha = np.exp2(np.where(np.isfinite(m[:, :, :-1]), m[:, :, :-1] - m[:, :, 1:], 0.0))
    dropped = dict(truth, acc=np.concatenate(
        [acc[:, :, :1], acc[:, :, 1:] + acc[:, :, :-1] * (1 - alpha)[..., None]], 2))
    assert refused(witnessed(q, k, v, mask_of("full"), dropped, lane=lane), "acc")


def _maximum_refused(q, k, v, truth: dict, *, lane) -> None:
    """A full launch's records (`truth`) with a maximum from the wrong tile, refused at the maxima."""
    wrong = dict(truth, m=np.concatenate([truth["m"][:, :, :1], truth["m"][:, :, :-1]], 2))
    assert refused(witnessed(q, k, v, mask_of("full"), wrong, lane=lane), "m")


def _narrowed_refused(q, k, v, truth: dict, *, lane) -> None:
    """A full launch's records (`truth`) with sums rounded to bf16 where the walk holds fp32, refused at the
    sums."""
    narrowed = dict(truth, acc=np.asarray(nearest(truth["acc"], BFLOAT16)))
    assert refused(witnessed(q, k, v, mask_of("full"), narrowed, lane=lane), "acc")


def _shifted_refused(q, k, v, causal: dict, *, lane) -> None:
    """A causal launch's records held under a mask one key short of its own."""
    shifted = mask_of("causal") & ~np.eye(SHAPE[1], dtype=bool)[None, None]
    held = witnessed(q, k, v, shifted, causal, lane=lane)
    assert sum(refused(held, name) for name in ("m", "l", "acc"))


def _bounded_refused(q, k, v, *bounded: dict, lane) -> None:
    """Each bounded launch's records (a window's, the documents') held as one undivided causal call: its
    bounds read."""
    for records in bounded:
        held = witnessed(q, k, v, mask_of("causal"), records, lane=lane)
        assert sum(refused(held, name) for name in ("m", "l", "acc"))


def _capped_refused(q, k, v, causal: dict, capped: dict, *, lane) -> None:
    """A capped launch held as an uncapped call's, and an uncapped (causal) one as a capped call's: its cap
    read."""
    mask = mask_of("causal")
    assert sum(refused(witnessed(q, k, v, mask, capped, lane=lane), name) for name in ("m", "l", "acc"))
    constants = capped_constants("capped", SHAPE[3])
    assert sum(refused(witnessed(q, k, v, mask, causal, lane=lane, capped=constants), name)
               for name in ("m", "l", "acc"))


FAULTS = {"rescale": (("full",), _rescale_refused), "maximum": (("full",), _maximum_refused),
          "narrowed": (("full",), _narrowed_refused), "shifted": (("causal",), _shifted_refused),
          "bounded": (("window", "documents"), _bounded_refused),
          "capped": (("causal", "capped"), _capped_refused)}
"""The forward's faults (`assert_faults_refused`), each the forms whose launches it reads and its check of
them, a part a test may hold alone."""


def backward_launched(query, key, value, dout, form: str, *, interpret: bool, site: str = "backward",
                      accurate: bool = False) -> dict:
    """The backward's gradients (`attention_mosaic.backward`) of the call under `form` (`FORMS`), each as
    float64, beside what `backward_witnessed` holds them by: the fast form's from its training forward's
    output residuals (`template.Kept.OUTPUT`: the output, the rows' natural maxima and denominators); the
    accurate form's (two launches, `accurate`) from its anchor's split sums (`template.Kept.SPLIT`) through
    its certify form (`attention_mosaic.backward_certified`), with the rows of D its launches took
    (`template.BackwardRows`)."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation

    def program(q, k, v, do):
        spans = spans_of(form, q.shape[0])
        kept = template.Kept.SPLIT if accurate else template.Kept.OUTPUT
        # The accurate form's logits natural, as the template's accurate form is (`template.ACCURATE`).
        out, *residuals = attention_mosaic.forward(q, k, v, spans, hooks_of(form), residuals=kept,
                                                   natural=accurate, interpret=interpret)
        grads, rows, tiles = attention_mosaic.backward_certified(
            q, k, v, spans, hooks_of(form), do, residuals=tuple(residuals), natural=accurate,
            accurate=accurate, interpret=interpret)
        # Whether one launch ran or two (`BackwardTiles.fused`), an output of the program, so a prepared
        # native run's stored program gives it too (no Python of it runs there).
        fused = jnp.asarray(tiles.fused, jnp.float32)
        return out, residuals[0], residuals[1], grads.query, grads.key, grads.value, *rows, fused

    if interpret:
        with generation.assumed("sm90"):
            found = jax.jit(program)(query, key, value, dout)
    else:
        from reference_error import ahead_arm

        shapes = ", ".join(f"{np.dtype(x.dtype).name}{list(np.shape(x))}"
                           for x in (query, key, value, dout))
        form_of = f"{form}, accurate" if accurate else form
        label = f"attention backward witness, {site}, {form_of}, {generation.assumption()}, {shapes}"
        found = ahead_arm(jax.jit(program), label)(query, key, value, dout)
    names = ("out", "m", "l", "dq", "dk", "dv", *template.BackwardRows._fields, "fused")
    # On the host (a prepare's zeros are NumPy's; no device array is made under a compile-only default).
    records = {name: np.asarray(np.asarray(x).astype(np.float32), np.float64)
               for name, x in zip(names, found, strict=True)}
    return {**records, "accurate": np.asarray(accurate), "fused": records["fused"] != 0}


def backward_witnessed(query, key, value, dout, kept, records: dict, *, lane: Lane | None = None,
                       softcap: float | None = None, sloped: bool = True
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each gradient of the backward's `records` (`backward_launched`: its certify form's, with the rows of D
    its launches took, `template.BackwardRows`) held elementwise to its enclosure under the mask `kept`, from
    the forward's stored maxima and denominators, natural units (as the forward's witness holds each state
    from the stored ones; the attention lane's order, 2026-11-27), for the schedule that ran, which its rows
    state: each query's and key's products `s` (fp32 on the tensor cores), the weights
    `P = exp2(s f - m log2(e)) (1/l)` (the walk's factor `f`; the maximum's units an fp32 product; the
    multiply and the subtraction one FMA or two roundings; `1/l` a divide within two steps of its correctly
    rounded value, `div.full`'s reach), the dK/dV launch's over its own denominator (`T'` where two launches
    ran, the forward's `l` where one did), `dP = dO Vᵀ` (fp32), the score cotangents
    `P ((dP - c) - (D - c))` about the anchor and D the launch took (`_centred`: two launches' D refined from
    the dQ launch's sums, `gradient.dq_finish`); times the call's whole scale (`grad_scale`) in fp32, cast to
    bf16; and the gradients `dV = bf16(P)ᵀ dO`, `dK = dSᵀ Q`, each a tensor-core sum over every walked query
    in any order, and `dQ` as its launch forms it (`_fast_dq`), cast to bf16. Under a cap (`softcap`), the
    logits are recomputed natural as the hook states them, `cap tanh(fp32(s scale) / cap)` (each divide
    within two steps, `div.full`'s reach; tanh within its lane's reach), their weights `exp2` of their
    natural units, and each score cotangent times the hook's slope as JAX's vjp of it takes it on ones:
    `cap`, times `1 - t²` (`one_minus_square`, fused or not), over `cap` (none where not `sloped`: a fault's
    witness, all else capped). The accurate form's (`records["accurate"]`), its logits natural (raw products
    against their raw maximum, `Forward.raw`, but under a cap): its cotangents fp32 at their products as
    three bf16 parts summing to each exactly (`attention_algebra.formed_parts`), where the fast form casts
    them to bf16 once, and its dQ as its pass forms it (`_accurate_dq`). Every denominator a launch read is
    held to the weights' own sum (`_unbounded_denominators`), a row outside refusing the gradients it
    weighs. For each gradient, the
    entries held and those inside."""
    lane = attention_lane() if lane is None else lane
    lane = dataclasses.replace(lane, block=0)
    q, k, v, do = (np.asarray(np.asarray(x, np.float32), np.float64) for x in (query, key, value, dout))
    power, left = template.prescale(np.dtype(query.dtype), q.shape[3], None)
    grad_scale = power * left
    accurate = bool(records["accurate"])
    held = {name: [] for name in ("dq", "dk", "dv")}
    for b in range(q.shape[0]):
        keeps = np.broadcast_to(kept, (q.shape[0], 1, *kept.shape[2:]))[b, 0]
        for h in range(q.shape[2]):
            kv = h * k.shape[2] // q.shape[2]
            products = dot(point(q[b, :, h]), point(k[b, :, kv].T), lane)
            exponents, slope, natural = exponents_of(
                products, records["m"][b, h], lane, natural=accurate, power=power, left=left,
                softcap=softcap, sloped=sloped)
            if natural is not None:
                # The accurate form's weights natural exponentials: the forward's of its differences' exact
                # products (`natural_exp`), its total's, and the backward's `exp` of their rounded ones
                # (`attention_gradient.below`, no approximation), every other's.
                forward = natural_exp(natural.forward, lane)
                forward = Iv(np.where(keeps, forward.lo, 0.0), np.where(keeps, forward.hi, 0.0))
                unnormalized = exp(natural.recomputed, lane)
                unnormalized = Iv(np.where(keeps, unnormalized.lo, 0.0),
                                  np.where(keeps, unnormalized.hi, 0.0))
            else:
                # A weight past one where its logit passes the stored maximum by a rounding: the forward's
                # fold keeps its product's residual (`exp2(fma(s, c, -m))`, `attention_algebra`), the
                # backward recomputes `logits - peak` unclamped (`attention_backward_algebra`).
                exponents = Iv(np.where(keeps, exponents.lo, 0.0), np.where(keeps, exponents.hi, 0.0))
                unnormalized = forward = exp2(exponents, lane)

            def over(denominators: np.ndarray, unnormalized: Iv = unnormalized, keeps=keeps) -> Iv:
                # The weights over `denominators`, their reciprocal within two steps (`div.full`'s reach).
                inverse = nearest(np.where(denominators > 0, 1.0 / np.where(denominators > 0, denominators,
                                                                            1.0), 0.0), FLOAT32)
                low, high = inverse, inverse
                for _ in range(2):
                    low, high = stepped(low, FLOAT32, -np.inf), stepped(high, FLOAT32, np.inf)
                found = fp32(mul(unnormalized, Iv(low[:, None], high[:, None])), lane)
                return Iv(np.where(keeps, found.lo, 0.0), np.where(keeps, found.hi, 0.0))

            # The dQ launch's weights over the forward's denominator; the dK/dV launch's over its launch's own
            # (`BackwardRows.denominator`: two launches' the dQ pass's `T'`, `attention_gradient.dq_finish`;
            # a fused launch's the forward's), about the D it took (the refined one, or the fused launch's).
            # Each held to the weights' own sum (`_unbounded_denominators`): the forward's an online walk's
            # where the form is fast, two launches' `T'` the sum of the weights its dQ launch recomputes.
            total = records["l"][b, h]
            own = None if bool(records["fused"]) else records["denominator"][b, h]
            outside = _unbounded_denominators(forward, unnormalized, exponents, keeps, total, own, lane,
                                              online=not accurate)
            unbounded = outside["total"] | outside["denominator"]
            weights = over(total)
            weighed = over(records["denominator"][b, h])
            pulled = dot(point(do[b, :, h]), point(v[b, :, kv].T), lane)
            cotangents = _scored(fp32(mul(weighed, _centred(pulled, records, b, h, lane)), lane), slope,
                                 grad_scale, keeps, lane)
            dv = dot(cast(Iv(weighed.lo.T, weighed.hi.T), BFLOAT16), point(do[b, :, h]), lane)
            if accurate:
                # The accurate form's cotangents fp32, as three bf16 parts summing to each exactly
                # (`attention_algebra.formed_parts`, `product.bf16_parts`), each part's product exact.
                dk = dot(Iv(cotangents.lo.T, cotangents.hi.T), point(q[b, :, h]), lane, parts=(3, 1))
                dq = _accurate_dq(pulled, weights, slope, grad_scale, keeps, k[b, :, kv], records, b, h, lane)
            else:
                cotangents = cast(cotangents, BFLOAT16)
                dk = dot(Iv(cotangents.lo.T, cotangents.hi.T), point(q[b, :, h]), lane)
                dq = _fast_dq(pulled, weights, slope, grad_scale, keeps, k[b, :, kv], records, b, h, lane)
                for name, inside in _fast_rows(pulled, weights, ~outside["denominator"], keeps, do[b, :, h],
                                               records["out"][b, :, h], v[b, :, kv], records, b, h,
                                               lane).items():
                    held.setdefault(name, []).append((np.ones(inside.shape, bool), inside))
            weighs = (keeps & unbounded[:, None]).any(0)
            for name, found in (("dq", dq), ("dk", dk), ("dv", dv)):
                stored = records[name][b, :, h if name == "dq" else kv]
                read = unbounded if name == "dq" else weighs
                inside = within(stored, cast(found, BFLOAT16)) & ~read[:, None]
                held[name].append((np.ones(stored.shape, bool), inside))
    return {name: (np.concatenate([rows.ravel() for rows, _ in found]),
                   np.concatenate([inside.ravel() for _, inside in found])) for name, found in held.items()}


class Natural(NamedTuple):
    """An accurate launch's natural arguments to its weights' exponentials (`exponents_of`): the forward's,
    exact (`natural_exp` scales its fp32 difference by the whole scale, its product's residual kept), and
    those its backward recomputes, fp32."""

    forward: Iv
    recomputed: Iv


def exponents_of(products: Iv, peak_m: np.ndarray, lane: Lane, *, natural: bool, power: float, left: float,
               softcap: float | None, sloped: bool) -> tuple[Iv | None, Iv | None, Natural | None]:
    """A row's weights' exponents against its stored maximum `peak_m` (`[Q]`) from its query's `products`
    with each key (`[Q, K]`, unscaled), and under a cap the cotangents' slope (`backward_witnessed`): the
    fast fold's, base 2, `products` times the scale and log2(e) (`template.prescale`'s power and left) less
    the maximum in base 2, none natural; the natural accurate form's, none base 2, its natural differences
    (`Natural`), as its backward recomputes them (`attention_algebra.exponent`), the prescaled query's raw
    products (the power, exact) less their stored raw maximum, the difference scaled after it is taken by
    the scale left (`Forward.raw`; the forward's that difference times the scale left exact), or past the
    raw form's scales the scaled logits less the stored natural maximum; under a cap the hook's natural
    logits, `cap tanh(fp32(s scale) / cap)`, its slope where `sloped`, the accurate form's natural
    differences those logits less the stored maximum."""
    grad_scale = power * left
    peak = fp32(mul(point(peak_m), point(LOG2E)), lane)
    below = Iv(-peak.hi[:, None], -peak.lo[:, None])
    stored = point(peak_m[:, None])
    if softcap is None and natural:
        prescaled = mul(products, point(power))
        low, high = algebra.RAW_SCALES
        if low <= left <= high:
            difference = fp32(add(prescaled, neg(stored)), lane)
            difference = Iv(np.minimum(difference.lo, 0.0), np.minimum(difference.hi, 0.0))
            logits = fp32(mul(difference, point(float(np.float32(left)))), lane)
            exact = mul(difference, point(left))
        else:
            logits = fp32(add(fp32(mul(prescaled, point(float(np.float32(left)))), lane), neg(stored)), lane)
            logits = exact = Iv(np.minimum(logits.lo, 0.0), np.minimum(logits.hi, 0.0))
        return None, None, Natural(exact, logits)
    if softcap is None:
        logits = mul(products, point(float(np.float32(left * LOG2E)) * power))
        return hull(fp32(add(logits, below), lane), fp32(add(fp32(logits, lane), below), lane)), None, None
    folded = _divided(fp32(mul(products, point(grad_scale)), lane), softcap, lane)
    bent = tanh(folded, lane)
    capped = fp32(mul(bent, point(softcap)), lane)
    square = hull(fp32(add(point(1.0), Iv(*(-x for x in reversed(fp32(mul(bent, bent), lane))))), lane),
                  fp32(mul(fp32(add(point(1.0), Iv(-bent.hi, -bent.lo)), lane),
                           fp32(add(point(1.0), bent), lane)), lane))
    slope = _divided(fp32(mul(point(softcap), square), lane), softcap, lane) if sloped else None
    if natural:
        differences = fp32(add(capped, neg(stored)), lane)
        differences = Iv(np.minimum(differences.lo, 0.0), np.minimum(differences.hi, 0.0))
        return None, slope, Natural(differences, differences)
    logits = mul(capped, point(LOG2E))
    exponents = hull(fp32(add(logits, below), lane), fp32(add(fp32(logits, lane), below), lane))
    exponents = hull(exponents, fp32(mul(fp32(add(capped, neg(stored)), lane), point(LOG2E)), lane))
    return exponents, slope, None


def _unbounded_denominators(forward: Iv, recomputed: Iv, exponents: Iv | None, keeps: np.ndarray,
                            total: np.ndarray, denominator: np.ndarray | None, lane: Lane, *,
                            online: bool) -> dict[str, np.ndarray]:
    """The rows whose recorded denominators lie outside their unnormalized weights' fp32 sum
    (`backward_witnessed`, by name: `"total"` and `"denominator"`, none where no `denominator`), a NaN outside
    every bound (`within`): the forward's `total` the sum of its own
    visible weights (`forward`), a dQ launch's own `denominator`, where two launches ran, the sum of the
    weights it recomputes (`recomputed`, at the stored maximum). Each an fp32 row sum in any order
    (`summed`: gamma_n of its terms, each add rounded to nearest); the forward's, where it walked `online`
    (the fast form's), its terms each within its online error of itself through its row's live combines
    alone (`certified.online_spread`, an exp2 of the lane's reach: a masked key's term an exact zero, its
    combine no rounding, so a causal row of one key is charged one combine, not the row's width), what
    underflow loses beside (four a live key)."""
    from certified import online_spread, underflow

    def visible(weights: Iv) -> Iv:
        return Iv(np.where(keeps, weights.lo, 0.0), np.where(keeps, weights.hi, 0.0))

    row = summed(visible(forward), lane)
    if online:
        live, unit = keeps.sum(-1), FLOAT32.unit_roundoff
        shifted = np.minimum(exponents.lo, 0.0) * np.log(2.0)
        spread = online_spread(shifted, live[..., None], 2 * unit * lane.reach, FLOAT32)
        lost = (visible(forward).hi * spread).sum(-1) + 4 * live * underflow(FLOAT32, lane.flushes)
        row = Iv(row.lo - lost, row.hi + lost)
    outside = {"total": ~within(total, row), "denominator": np.zeros(total.shape, bool)}
    if denominator is not None:
        outside["denominator"] = ~within(denominator, summed(visible(recomputed), lane))
    return outside


def _centred(pulled: Iv, records: dict, b: int, h: int, lane: Lane, *, prior: bool = False) -> Iv:
    """The backward's `(dP - c) - delta` of each query's row (`backward_witnessed`), about the centre its
    launch takes (`attention_gradient.split_delta`'s `chosen`: an anchor key's, or 0 where it is -1), from
    that launch's own anchor product `c` and refined `delta` (the records'): the rounded `delta` alone is
    what either form's dK/dV launch receives (`attention_mosaic_backward.py`'s `refined`), which the fast
    form's witness takes; the accurate form's keeps its existing allowance of anything to `delta + lost`
    (its low part consumed elsewhere, as the initial state's pullback): `c` as split_delta sums it or as the
    walk's own `dP` at the anchor key, each difference rounded to fp32, or `c` and `delta` summed first."""
    chosen = records["chosen"][b, h].astype(np.int64)
    centre = records["anchor"][b, h]
    # About the prior D (`split_delta`'s, dQ's pass's sums), or the refined one, the accurate form's within
    # its existing allowance of what its rounding lost.
    delta = records["prior"][b, h] if prior else records["delta"][b, h]
    widened = delta if prior or not bool(records["accurate"]) else delta + records["lost"][b, h]
    refined = Iv(np.minimum(delta, widened), np.maximum(delta, widened))
    rows = np.arange(chosen.shape[0])
    own = Iv(pulled.lo[rows, np.maximum(chosen, 0)], pulled.hi[rows, np.maximum(chosen, 0)])
    anchors = hull(point(centre), own)
    anchors = Iv(np.where(chosen < 0, 0.0, anchors.lo), np.where(chosen < 0, 0.0, anchors.hi))
    shift = Iv(-refined.hi[:, None], -refined.lo[:, None])
    stepwise = fp32(add(fp32(add(pulled, Iv(-anchors.hi[:, None], -anchors.lo[:, None])), lane), shift), lane)
    centred = fp32(add(Iv(anchors.lo[:, None], anchors.hi[:, None]),
                       Iv(refined.lo[:, None], refined.hi[:, None])), lane)
    summed = fp32(add(pulled, Iv(-centred.hi, -centred.lo)), lane)
    return hull(stepwise, summed)


def _scored(cotangents: Iv, slope: Iv | None, grad_scale: float, keeps: np.ndarray, lane: Lane) -> Iv:
    """Softmax cotangents as a launch finishes its score cotangents (`backward_witnessed`): times the cap's
    slope where there is one, then the call's whole scale, each in fp32, zero where a query sees no key."""
    if slope is not None:
        cotangents = fp32(mul(cotangents, slope), lane)
    cotangents = fp32(mul(cotangents, point(grad_scale)), lane)
    return Iv(np.where(keeps, cotangents.lo, 0.0), np.where(keeps, cotangents.hi, 0.0))


def _fast_dq(pulled: Iv, weights: Iv, slope: Iv | None, grad_scale: float, keeps: np.ndarray,
             keys: np.ndarray, records: dict, b: int, h: int, lane: Lane) -> Iv:
    """The fast backward's dQ of one head (`backward_witnessed`) as its dQ launch forms it: its score
    cotangents over the forward's denominator about the D `split_delta` took (`records["prior"]`), cast to
    bf16, their tensor-core sum with the keys, times the forward's denominator over the launch's own
    (`attention_gradient.dq_finish`'s ratio, one division within `div.full`'s reach), rounded, where two
    launches ran (row 13697: the GPU tier's under deterministic ops); a fused launch's sum alone."""
    first = _scored(fp32(mul(weights, _centred(pulled, records, b, h, lane, prior=True)), lane), slope,
                    grad_scale, keeps, lane)
    sums = dot(cast(first, BFLOAT16), point(keys), lane)
    if records["fused"]:
        return sums
    ratio = _quotient(point(records["l"][b, h]), point(records["denominator"][b, h]), lane)
    return fp32(mul(sums, Iv(ratio.lo[:, None], ratio.hi[:, None])), lane)


def _fast_rows(pulled: Iv, weights: Iv, summed_own: np.ndarray, keeps: np.ndarray, do: np.ndarray,
               out: np.ndarray, values: np.ndarray, records: dict, b: int, h: int, lane: Lane
               ) -> dict[str, np.ndarray]:
    """The fast backward's rows of D (`template.BackwardRows`) held to enclosures of their own, apart from
    the gradients they centre (a launch that took and stated a wrong D refused): the prior `D - c` within
    `gamma` of its terms' magnitudes of `dO · O` less the chosen key's `dO · v_c` (exact, float64; 0 where
    it took none), and the anchor `c` within the same of its own, exactly 0 where it took none (the fast
    form's output residuals, `attention.output_split`); where two launches ran (the denominator
    the dQ launch's own), that denominator within its exponentials' fp32 sum in any order (`summed_own`,
    `_unbounded_denominators`' verdict, the one the gradients it weighs take), and the refined
    `D - c` within `fl((D - c) + r)` of `dq_finish`'s refinement `r = fl(fl(R ratio) / fl(T' fl(1/T')))`,
    `R` the first pass's cotangents' fp32 sum; where one did (`records["fused"]`, the certify launch's
    tiles', never read off its rows: two launches may take the forward's denominator to the bit), the
    forward's denominator and its D, nothing lost. Each `[Q]` the entries inside."""
    width = do.shape[-1]
    gamma = width * FLOAT32.unit_roundoff / (1 - width * FLOAT32.unit_roundoff) + (width + 2) * 2.0 ** -52
    chosen = records["chosen"][b, h].astype(np.int64)
    centre = np.where(chosen < 0, 0.0, (do * values[np.maximum(chosen, 0)]).sum(-1))
    magnitudes = (np.abs(do) * (np.abs(out) + np.abs(values).max(0))).sum(-1)
    delta = (do * out).sum(-1) - centre
    prior, anchor = records["prior"][b, h], records["anchor"][b, h]
    # Each sum within `gamma` of its terms' magnitudes, `D - c` one more rounding.
    reach = gamma * magnitudes + 2 * FLOAT32.unit_roundoff * np.abs(delta)
    own_reach = gamma * magnitudes
    # A row of no anchor key (`chosen` -1: the fast form's output residuals, `attention.output_split`) its
    # anchor exactly 0; else its key's product within its sum's reach.
    found = {"prior": within(prior, Iv(delta - reach, delta + reach)),
             "anchor": np.where(chosen < 0, anchor == 0.0,
                                within(anchor, Iv(centre - own_reach, centre + own_reach)))}
    total, own = records["l"][b, h], records["denominator"][b, h]
    if records["fused"]:
        # One launch (`BackwardTiles.fused`, as the certify form derived it): the forward's own rows.
        found["denominator"] = own == total
        found["delta"] = (records["delta"][b, h] == prior) & (records["lost"][b, h] == 0)
        return found
    found["denominator"] = summed_own
    first = fp32(mul(weights, _centred(pulled, records, b, h, lane, prior=True)), lane)
    first = Iv(np.where(keeps, first.lo, 0.0), np.where(keeps, first.hi, 0.0))
    # `1 / T'` and `l / T'` each one division within `div.full`'s reach, as `dq_finish` forms them.
    inverse = _quotient(point(np.ones_like(own)), point(own), lane)
    mass = fp32(mul(point(own), inverse), lane)
    scaled = fp32(mul(summed(first, lane), _quotient(point(total), point(own), lane)), lane)
    refinement = _quotient(scaled, mass, lane)
    found["delta"] = within(records["delta"][b, h], fp32(add(point(prior), refinement), lane))
    return found


def _quotient(numerator: Iv, divisor: Iv, lane: Lane) -> Iv:
    """An fp32 quotient of values within `numerator` by positive ones within `divisor`, within two steps of
    its correctly rounded value (`div.full`'s reach); 0 where the divisor may be (`dq_finish`'s guard)."""
    positive = divisor.lo > 0
    safe = Iv(np.where(positive, divisor.lo, 1.0), np.where(positive, divisor.hi, 1.0))
    ends = np.stack(np.broadcast_arrays(numerator.lo / safe.lo, numerator.lo / safe.hi,
                                        numerator.hi / safe.lo, numerator.hi / safe.hi))
    found = fp32(Iv(ends.min(0), ends.max(0)), lane)
    lo, hi = found.lo, found.hi
    for _ in range(2):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(np.where(positive, lo, np.minimum(lo, 0.0)), np.where(positive, hi, np.maximum(hi, 0.0)))


def _accurate_dq(pulled: Iv, weights: Iv, slope: Iv | None, grad_scale: float, keeps: np.ndarray,
                 keys: np.ndarray, records: dict, b: int, h: int, lane: Lane) -> Iv:
    """The accurate backward's dQ of one head (`backward_witnessed`) as its pass computes it
    (`attention_gradient.dq_finish`): the tensor-core sum of its score cotangents at the prior D (the
    centre `split_delta` gave, `records["prior"]`), their fp32 operands three bf16 parts, less the
    refinement (the refined D and what its rounding lost, less the prior) times the weights' product with
    the keys (`keyed`, each weight times the call's scale and the hook's slope), that product two fp32
    roundings, the difference rounded, times the prior denominator over the pass's own (one division). The
    keyed sum's operand is one bf16 part (`attention_algebra.weights_parts`), a score hook's coefficients
    each over its row's running mass, the sums rescaled by it each key tile and the mass taken back
    (`attention_gradient.dq_issued`): each term within bf16's unit of its own (a flushed one within its
    least normal of the mass), the rescales, the mass's sum and its inverse within three fp32 roundings a
    key of the row's magnitudes. Where dQ cancels across its keys the sum's rounding stays at the size of
    its terms at the prior D, which an enclosure about the refined D alone does not hold."""
    centred = _centred(pulled, records, b, h, lane, prior=True)
    cotangents = fp32(mul(weights, centred), lane)
    coefficients = weights if slope is None else fp32(mul(weights, slope), lane)
    if slope is not None:
        cotangents = fp32(mul(cotangents, slope), lane)
    cotangents = fp32(mul(cotangents, point(grad_scale)), lane)
    cotangents = Iv(np.where(keeps, cotangents.lo, 0.0), np.where(keeps, cotangents.hi, 0.0))
    sums = dot(cotangents, point(keys), lane, parts=(3, 1))
    coefficients = fp32(mul(coefficients, point(grad_scale)), lane)
    coefficients = Iv(np.where(keeps, coefficients.lo, 0.0), np.where(keeps, coefficients.hi, 0.0))
    reach = np.maximum(np.abs(coefficients.lo), np.abs(coefficients.hi))
    rounded = reach * BFLOAT16.unit_roundoff
    keyed = dot(Iv(coefficients.lo - rounded, coefficients.hi + rounded), point(keys), lane)
    slack = ((3 * keys.shape[0] + 8) * FLOAT32.unit_roundoff * (reach @ np.abs(keys))
             + 2.0 ** FLOAT32.emin * reach.sum(-1)[:, None] * np.abs(keys).sum(0)[None, :])
    keyed = Iv(keyed.lo - slack, keyed.hi + slack)
    refined, lost, prior = records["delta"][b, h], records["lost"][b, h], records["prior"][b, h]
    steps = refined - prior, refined + lost - prior
    increment = Iv(np.minimum(*steps)[:, None], np.maximum(*steps)[:, None])
    product = mul(increment, keyed)
    # The increment's product with the mass, then with the sums (`exponent_product`), each rounded.
    twice = 3 * FLOAT32.unit_roundoff * np.maximum(np.abs(product.lo), np.abs(product.hi))
    correction = fp32(Iv(product.lo - twice, product.hi + twice), lane)
    difference = fp32(add(sums, Iv(-correction.hi, -correction.lo)), lane)
    ratio = _quotient(point(records["l"][b, h]), point(records["denominator"][b, h]), lane)
    return fp32(mul(difference, Iv(ratio.lo[:, None], ratio.hi[:, None])), lane)


def _divided(numerator: Iv, divisor: float, lane: Lane) -> Iv:
    """An fp32 quotient of fp32 values within `numerator` by the fp32 `divisor`, within two steps of its
    correctly rounded value (`div.full`'s reach)."""
    found = fp32(Iv(numerator.lo / divisor, numerator.hi / divisor), lane)
    lo, hi = found.lo, found.hi
    for _ in range(2):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(lo, hi)


def backward_operands(seed: int, width: int = SHAPE[3]):
    """`operands(seed, width)` and a bf16 cotangent of the output drawn beside them on the host (row 9546)."""
    import jax.numpy as jnp

    q, k, v = operands(seed, width)
    rng = np.random.default_rng(seed + 7)
    return q, k, v, rng.standard_normal(q.shape, dtype=np.float32).astype(np.dtype(jnp.bfloat16))


def assert_backward_witnessed(seed: int, form: str, *, interpret: bool, lane: Lane | None = None,
                              accurate: bool = False, width: int = SHAPE[3]) -> None:
    """Every gradient of the fast backward over draw `seed` under `form`, or the accurate one's (two
    launches, `accurate`), `width` wide, inside its enclosure (`backward_witnessed`)."""
    import tier_wire

    q, k, v, do = backward_operands(seed, width)
    records = backward_launched(q, k, v, do, form, interpret=interpret, accurate=accurate)
    if not interpret and tier_wire.preparing():
        return
    held = backward_witnessed(q, k, v, do, mask_of(form), records, lane=lane, softcap=CAPS.get(form))
    for name, (rows, inside) in held.items():
        assert rows.any(), name
        assert inside[rows].all(), f"{name}: {int((rows & ~inside).sum())} of {int(rows.sum())} outside"


def assert_two_launches_held_to_their_own_d(*, interpret: bool, lane: Lane | None = None, held: bool = True,
                                            faults: tuple[str, ...] | None = None) -> None:
    """The backward's two launches (deterministic ops, under which `MosaicBackward.legal` refuses the fused
    one: the GPU tier's, row 13697) held to their own schedule: with `held`, their gradients about the D they
    refined and over the dQ launch's own denominator held; and each of `faults` (`TWO_LAUNCH_FAULTS`' names,
    every one by default) refused at its record: the same gradients held as the fused launch's (the D
    `split_delta` took, nothing lost, the forward's denominator, 'fused'), a D off ('delta'), a `lost` its
    launch never reads widening dK ('lost'), and an anchor where it took none ('anchor')."""
    import tier_wire

    faults = tuple(TWO_LAUNCH_FAULTS) if faults is None else faults
    q, k, v, do = backward_operands(3)
    records = backward_launched(q, k, v, do, "causal", interpret=interpret, site="two launches")
    if not interpret and tier_wire.preparing():
        return
    assert (records["denominator"] != records["l"]).any(), "two launches ran: the dQ launch's own denominator"
    mask = mask_of("causal")
    if held:
        found = backward_witnessed(q, k, v, do, mask, records, lane=lane)
        assert not sum(refused(found, name) for name in ("dq", "dk", "dv"))
    for fault in faults:
        faulty, at = TWO_LAUNCH_FAULTS[fault]
        assert refused(backward_witnessed(q, k, v, do, mask, faulty(records), lane=lane), at), fault


def _anchored(records: dict) -> dict:
    """The fast form's output residuals take no anchor key (`chosen` -1, `attention.output_split`): an anchor
    of 1 there, refused at its row, where `chosen < 0` passed any."""
    assert (records["chosen"] < 0).all(), "the fast form's output residuals take no anchor key"
    return dict(records, anchor=records["anchor"] + 1.0)


TWO_LAUNCH_FAULTS = {
    "fused": (lambda records: dict(records, delta=records["prior"], lost=np.zeros_like(records["lost"]),
                                   denominator=records["l"]), "dk"),
    # A refined D taken and stated a sixty-fourth off, refused at the row whatever the gradients.
    "delta": (lambda records: dict(records, delta=records["delta"] * (1 + 2.0 ** -6) + 2.0 ** -6), "delta"),
    # A `lost` its dK/dV launch never reads (it receives the rounded D alone) widens nothing: dK a sixteenth
    # off beside a `lost` of a thousand times each row's D refused at dK, where D between `delta` and
    # `delta + lost` admitted it (train 1394's P1).
    "lost": (lambda records: dict(records, lost=1e3 * (np.abs(records["delta"]) + 1.0),
                                  dk=records["dk"] * (1 + 2.0 ** -4) + 2.0 ** -4), "dk"),
    "anchor": (_anchored, "anchor"),
}
"""Each two-launch fault (`assert_two_launches_held_to_their_own_d`): its records from the launches' own and
the record it is refused at, a part a test may hold alone."""


def assert_equal_totals_held(*, interpret: bool, lane: Lane | None = None) -> None:
    """Two launches (deterministic ops) whose dQ launch's own denominator is the forward's to the bit: a full
    call of zero queries and keys over every key (each weight `1/K`), its values 1 in their first column but
    one 2, a unit cotangent there. Its output `1 + 1/K` rounds to bf16's 1, so the forward's D is 1, and
    the dQ launch's residual `1/K` refines it to `1 + 1/K` all the same: its gradients and rows held by the
    schedule its certify launch derived (`records["fused"]`), not by its rows, whose equal denominators read
    as one launch's refused the refined D."""
    import jax.numpy as jnp
    import tier_wire

    bf16 = np.dtype(jnp.bfloat16)
    query = np.zeros(SHAPE, bf16)
    value = np.zeros(SHAPE, bf16)
    value[:, :, :, 0] = 1
    value[:, 0, :, 0] = 2
    dout = np.zeros(SHAPE, bf16)
    dout[:, :, :, 0] = 1
    records = backward_launched(query, query, value, dout, "full", interpret=interpret, site="equal totals")
    if not interpret and tier_wire.preparing():
        return
    assert not records["fused"], "two launches ran"
    held = backward_witnessed(query, query, value, dout, mask_of("full"), records, lane=lane)
    for name, (rows, inside) in held.items():
        assert inside[rows].all(), f"{name}: {int((rows & ~inside).sum())} of {int(rows.sum())} outside"


def assert_logit_past_its_maximum_held(*, interpret: bool, lane: Lane | None = None) -> None:
    """Two launches (deterministic ops) over a causal call whose first row's one visible key's logit passes
    the stored maximum by a rounding: its one product 1 x 180 at width 64, which the folded forward stores
    as a maximum of 22.499998 and the backward recomputes as 22.5 (`logits - peak` unclamped). Its weight,
    and so the dQ launch's own denominator, past one: its gradients and rows held, none refused."""
    import jax.numpy as jnp
    import tier_wire

    bf16 = np.dtype(jnp.bfloat16)
    _, _, value, dout = backward_operands(5)
    query, key = np.zeros(SHAPE, bf16), np.zeros(SHAPE, bf16)
    query[:, 0, :, 0], key[:, 0, :, 0] = 1, 180
    records = backward_launched(query, key, value, dout, "causal", interpret=interpret,
                                site="past its maximum")
    if not interpret and tier_wire.preparing():
        return
    assert not records["fused"], "two launches ran"
    held = backward_witnessed(query, key, value, dout, mask_of("causal"), records, lane=lane)
    for name, (rows, inside) in held.items():
        assert inside[rows].all(), f"{name}: {int((rows & ~inside).sum())} of {int(rows.sum())} outside"


def assert_contracted_row_held(*, interpret: bool, lane: Lane | None = None) -> None:
    """A full call whose every row's 512 products are one, 1 x 184320 at width 64: the forward's fold
    stores a maximum of 33239.69 against a score of 23040, its FMA keeping the product's residual, so each
    weight is 2^0.0019 and the denominator about 512.67, past the keys' count: the forward's states and the
    backward's gradients and rows held, none refused. Native alone: an interpreted fold rounds its product
    before the subtraction, its weights one."""
    import jax.numpy as jnp
    import tier_wire

    bf16 = np.dtype(jnp.bfloat16)
    _, _, value, dout = backward_operands(6)
    query, key = np.zeros(SHAPE, bf16), np.zeros(SHAPE, bf16)
    query[..., 0], key[..., 0] = 1, 184320
    records = backward_launched(query, key, value, dout, "full", interpret=interpret, site="contracted row")
    if not interpret and tier_wire.preparing():
        return
    # The case it is for reached: every denominator past the keys' count, the fold's FMA its residual's.
    assert (records["l"] > SHAPE[1]).all(), f"no fold residual kept: {np.unique(records['l'])}"
    held = backward_witnessed(query, key, value, dout, mask_of("full"), records, lane=lane)
    for name, (rows, inside) in held.items():
        assert inside[rows].all(), f"{name}: {int((rows & ~inside).sum())} of {int(rows.sum())} outside"


def _exact_scores(query, key, value, dout, kept) -> np.ndarray:
    """The exact score cotangents `dS` `[B, H, Q, K]` of the call (`attention_oracle.float64_weights`,
    float64 on the CPU): `P (dP - D)` over sqrt(D)."""
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        q, k, v, do = (jnp.asarray(np.asarray(x, np.float32), jnp.float64) for x in (query, key, value, dout))
        weights, _ = attention_oracle.float64_weights(q, k, jnp.asarray(kept), None, None)
        group = q.shape[2] // k.shape[2]
        pulled = jnp.einsum("bqhd,bkhd->bhqk", do, jnp.repeat(v, group, axis=2))
        centred = pulled - (weights * pulled).sum(-1, keepdims=True)
        return np.asarray(weights * centred / np.sqrt(q.shape[3]))


def _exact_dk(query, key, scores: np.ndarray) -> np.ndarray:
    """dK `dSᵀ Q` of score cotangents `scores` (`_exact_scores`), a key head's query heads summed, cast to
    bf16 once from float64."""
    q, k = (np.asarray(np.asarray(x, np.float32), np.float64) for x in (query, key))
    group = q.shape[2] // k.shape[2]
    dk = np.einsum("bhqk,bqhd->bkhd", scores, q)
    return np.asarray(nearest(dk.reshape(*dk.shape[:2], k.shape[2], group, -1).sum(3), BFLOAT16), np.float64)


def assert_backward_faults_refused(*, interpret: bool, lane: Lane | None = None,
                                   faults: tuple[str, ...] | None = None) -> None:
    """Each fault the backward's witness exists for among `faults` (`BACKWARD_FAULTS`' names, every one by
    default), refused at its
    elements: dK at twice its scale, dQ and dV a sixty-fourth off ('scaled'), a causal call's gradients held
    as a full call's ('masked'), the fast form's D taken and stated a sixty-fourth off, refused at its row
    ('delta'), a capped call's dQ and dK held without the cap's slope (`sloped`; its dV held, 'capped'); the
    accurate form's at twice its scale and a sixty-fourth off alike ('accurate'), and a dK of the exact score
    cotangents cast to one bf16 part where the accurate form takes three (the same cotangents whole held),
    and the fast form's dQ and dK held as the accurate form's (one part, about the first pass's D,
    'parts'); a dQ pass's denominator doubled with the dV it weighs halved, and nudged or NaN ('doubled');
    and a fixed maximum's denominator past its fp32 row sum's bound, its masked keys no terms of it ('row
    sums'); and the fast form's one-key causal row's denominator 2^-13 past its one live combine's bound
    ('combine'). Every native program a part reads launches before any check (row 9546)."""
    import tier_wire

    faults = tuple(BACKWARD_FAULTS) if faults is None else faults
    q, k, v, do = backward_operands(3)
    names = dict.fromkeys(name for fault in faults for name in BACKWARD_FAULTS[fault][0])
    # The accurate form's run is the causal call's in that form.
    runs = {name: backward_launched(q, k, v, do, "causal" if name == "accurate" else name,
                                    interpret=interpret, site="faults", accurate=name == "accurate")
            for name in names}
    if not interpret and tier_wire.preparing():
        return
    mask = mask_of("causal")
    for fault in faults:
        names, check = BACKWARD_FAULTS[fault]
        check(q, k, v, do, mask, *(runs[name] for name in names), lane=lane)


def _bent_refused(q, k, v, do, mask, records: dict, *, lane) -> None:
    """A run's dK at twice its scale, dQ and dV a sixty-fourth off, each refused at its gradient."""
    for name, bent in (("dk", 2.0), ("dq", 1.0 + 2.0 ** -6), ("dv", 1.0 - 2.0 ** -6)):
        held = backward_witnessed(q, k, v, do, mask, dict(records, **{name: records[name] * bent}), lane=lane)
        assert refused(held, name), name


def _masked_refused(q, k, v, do, mask, causal: dict, *, lane) -> None:
    """A causal call's gradients held as a full call's."""
    held = backward_witnessed(q, k, v, do, mask_of("full"), causal, lane=lane)
    assert sum(refused(held, name) for name in ("dq", "dk", "dv"))


def _delta_refused(q, k, v, do, mask, causal: dict, *, lane) -> None:
    """The fast form's D taken and stated a sixty-fourth off (its prior and the D its launch read alike),
    refused at the row whatever the gradients."""
    off = dict(causal, prior=causal["prior"] * (1 + 2.0 ** -6) + 2.0 ** -6,
               delta=causal["delta"] * (1 + 2.0 ** -6) + 2.0 ** -6)
    assert refused(backward_witnessed(q, k, v, do, mask, off, lane=lane), "prior")


def _cap_refused(q, k, v, do, mask, capped: dict, *, lane) -> None:
    """The cap's slope read: a capped call's gradients held with it, and without it, its logits and weights
    capped (`sloped`), its dQ and dK refused and its dV (no slope in it) held."""
    for sloped in (True, False):
        held = backward_witnessed(q, k, v, do, mask, capped, lane=lane, softcap=CAPS["capped"],
                                  sloped=sloped)
        assert [bool(refused(held, name)) for name in ("dq", "dk", "dv")] == [not sloped, not sloped, False]


def _parts_refused(q, k, v, do, mask, causal: dict, accurate: dict, *, lane) -> None:
    """The accurate dK of the exact cotangents whole held, cast to one bf16 part first refused; the fast
    form's dQ and dK held as the accurate form's refused."""
    scores = _exact_scores(q, k, v, do, mask)
    one_part = nearest(nearest(scores, FLOAT32), BFLOAT16)
    for cotangents, wrong in ((scores, False), (one_part, True)):
        exact = dict(accurate, dk=_exact_dk(q, k, cotangents))
        assert bool(refused(backward_witnessed(q, k, v, do, mask, exact, lane=lane), "dk")) == wrong
    held = backward_witnessed(q, k, v, do, mask, dict(accurate, dq=causal["dq"], dk=causal["dk"]), lane=lane)
    assert refused(held, "dq") and refused(held, "dk")


def _combine_refused(q, k, v, do, mask, causal: dict, *, lane) -> None:
    """The fast form's one-key causal row's forward denominator past its one live combine's bound."""
    # The fast form's first causal row, its one key at its maximum: its forward's denominator exactly 1 and
    # nudged by 2^-13, refused, its online walk charged its one live combine, 8e-6 beyond its sum's own
    # enclosure (its exponent's recomputed rounding, about 8e-5); charged the row's 512 keys' combines, 4e-3
    # beyond it, it passed (Train 1444's review; FS-62-DENOM3).
    assert causal["l"][0, 0, 0] == 1.0
    nudged = causal["l"].copy()
    nudged[0, 0, 0] = 1 + 2.0 ** -13
    assert refused(backward_witnessed(q, k, v, do, mask, dict(causal, l=nudged), lane=lane), "dq")


def _doubled_refused(q, k, v, do, mask, accurate: dict, *, lane) -> None:
    """The accurate form's dQ pass denominators bent: doubled with the dV it weighs halved, and a row's
    nudged or NaN."""
    # A dQ pass's denominator twice its own with the dV it weighs half its own, each consistent with the other
    # (`dq_finish` doubling its total): refused by the denominators' bound on the weights' sum.
    doubled = dict(accurate, denominator=accurate["denominator"] * 2, dv=accurate["dv"] / 2)
    assert refused(backward_witnessed(q, k, v, do, mask, doubled, lane=lane), "dv")
    # The first causal row's one key at its maximum: its denominator exactly 1, nudged by 2^-13 (within the
    # fast walk's allowance, past a fixed maximum's sum), and NaN, each refusing its dQ.
    for wrong in (1 + 2.0 ** -13, np.nan):
        nudged = accurate["denominator"].copy()
        nudged[0, 0, 0] = wrong
        assert refused(backward_witnessed(q, k, v, do, mask, dict(accurate, denominator=nudged), lane=lane),
                       "dq")


def _row_sums_bounded(q, k, v, do, mask, *, lane) -> None:
    """A fixed maximum's row sums against their bound: its denominators' rows held and refused."""
    del q, k, v, do, mask
    # A fixed maximum's row of 512 exact unit weights: its fp32 sum 512 held, and 512 + 3 2^-7, past an fp32
    # row sum's gamma_n (512 + 2^-6) though within a tensor-core sum's twice that, refused; one such weight
    # beside 511 masked keys, its sum 1 held and 1 + 2^-16 refused (an add of a masked zero no rounding);
    # each as the forward's total and as the dQ pass's own denominator.
    units, zeros = point(np.ones((1, 512))), point(np.zeros((1, 512)))
    summing = attention_lane() if lane is None else lane
    for live, recorded, wrong in ((512, 512.0, False), (512, 512.0 + 3 * 2.0 ** -7, True),
                                  (1, 1.0, False), (1, 1.0 + 2.0 ** -16, True)):
        exact, read = np.float32([live]), np.float32([recorded])
        for total, denominator in ((read, exact), (exact, read)):
            unbounded = _unbounded_denominators(units, units, zeros, np.arange(512)[None] < live, total,
                                                denominator, summing, online=False)
            assert list(unbounded["total"] | unbounded["denominator"]) == [wrong]


BACKWARD_FAULTS = {
    "scaled": (("causal",), _bent_refused),
    "masked": (("causal",), _masked_refused),
    "delta": (("causal",), _delta_refused),
    "capped": (("capped",), _cap_refused),
    "accurate": (("accurate",), _bent_refused),
    "parts": (("causal", "accurate"), _parts_refused),
    "doubled": (("accurate",), _doubled_refused),
    "row sums": ((), _row_sums_bounded),
    "combine": (("causal",), _combine_refused),
}
"""The backward's faults (`assert_backward_faults_refused`), each the runs it reads and its check of them
from the operands, the causal mask and those runs in their order, a part a test may hold alone."""
