"""Single-draw instrumentation; never changes the production source."""

import argparse
import functools
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

import jax
import jax.numpy as jnp
import numpy as np

from reference_error import widened, with_gradients, without_ynn
from test_recurrence_stability import CHUNK, SITES, cotangents
from test_recurrence_chunk import token_by_token

from dew.nn.kernels import fla_gdn as port
from dew.nn.kernels import recurrence, recurrence_algebra as algebra
from dew.nn.kernels.recurrence_chunk import chunk


def stats(got, truth):
    got, truth = np.asarray(got, np.float64), np.asarray(truth, np.float64)
    return {
        "rms": float(np.sqrt(np.mean(truth**2))),
        "rms_error": float(np.sqrt(np.mean((got - truth)**2))),
        "relative_rms_error": float(np.linalg.norm(got - truth) / np.linalg.norm(truth)),
        "max_error": float(np.max(np.abs(got - truth))),
    }


def separated(hi, lo):
    whole = jnp.floor(hi)
    return jnp.ldexp(jnp.exp2((hi - whole) + lo), whole.astype(jnp.int32))


def split_between(rows, rows_lo, columns, columns_lo):
    return separated(rows[:, None] - columns[None, :], rows_lo[:, None] - columns_lo[None, :])


def rewrite(module, name, replacements):
    code = inspect.getsource(getattr(module, name))
    for before, after in replacements:
        if before not in code:
            raise ValueError((name, before))
        code = code.replace(before, after)
    module.__dict__["probe_between"] = split_between
    module.__dict__["probe_separated"] = separated
    exec(compile(code, f"<probe:{module.__name__}.{name}>", "exec"), module.__dict__)


def install(scope="all"):
    if scope == "dots":
        def widened_dot(a, b, **kwargs):
            with jax.enable_x64(new_val=True):
                return jnp.dot(a.astype(jnp.float64), b.astype(jnp.float64),
                               preferred_element_type=jnp.float64).astype(jnp.float32)

        port._dot = widened_dot
        return
    if scope in ("all", "absolute"):
        algebra.raised = separated
    if scope == "absolute":
        return
    for name in ("solved", "chunk_output"):
        before = ("jnp.exp2(apart(gates[i], lows[i], gates[j], lows[j]))" if name == "solved"
                  else "jnp.exp2(apart(gate, low, gate, low))")
        rewrite(algebra, name, [(before, before.replace("jnp.exp2(apart(", "probe_between(")[:-1])])
    rewrite(algebra, "scalar_write", [
        ("jnp.exp2((last - gate) + (last_lo - gate_lo))",
         "probe_separated(last - gate, last_lo - gate_lo)")])
    rewrite(port, "_dv_local_kernel", [
        ("jnp.exp2(-algebra.apart(gate, low, gate, low))",
         "probe_between(-gate, -low, -gate, -low)")])
    rewrite(port, "_dqkwg_kernel", [
        ("jnp.exp2((last - gate) + (last_low - low))",
         "probe_separated(last - gate, last_low - low)"),
        ("jnp.exp2(algebra.apart(gate, low, gate, low))",
         "probe_between(gate, low, gate, low)")])
    rewrite(port, "_wy_bwd_kernel", [
        ("jnp.exp2(algebra.apart(gate, low, gate, low))",
         "probe_between(gate, low, gate, low)")])
    rewrite(port, "_dhu_kernel", [
        ("jnp.exp2(apart)", "probe_separated(last - gate, _row(gl_ref, final) - _gates(gl_ref, begin, taken, chunk))")])


def exponent_probe(drawn):
    g = drawn[0].log_decay
    hi, lo = recurrence.cumulated(g, chunk=CHUNK, scale=port.LOG2_E, compensated=True, interpret=True)
    # Adjacent pairs excluding chunk boundaries: exact exponent is one fp32 input token.
    eligible = np.arange(g.shape[1]) % CHUNK != 0
    dh, dl = hi[:, 1:] - hi[:, :-1], lo[:, 1:] - lo[:, :-1]
    summed = np.asarray(dh + dl)[:, eligible[1:]]
    first = np.asarray(jnp.exp2(dh + dl))[:, eligible[1:]]
    split = np.asarray(separated(dh, dl))[:, eligible[1:]]
    separately_exp2 = np.asarray(jnp.exp2(dh) * jnp.exp2(dl))[:, eligible[1:]]
    exact_g = np.asarray(g[:, 1:])[:, eligible[1:]].astype(np.float64)
    exact = np.exp(exact_g)
    native = np.asarray(jnp.exp(g[:, 1:]))[:, eligible[1:]]
    combined64 = (np.asarray(dh).astype(np.float64) + np.asarray(dl).astype(np.float64))[:, eligible[1:]]
    print(json.dumps({
        "exponent_rounding_max": float(np.max(np.abs(summed - combined64))),
        "prefix_pair_max_error": float(np.max(np.abs(combined64 - exact_g * port.LOG2_E))),
        "exp2_summed": stats(first, exact),
        "exp2_separated": stats(split, exact),
        "exp2_two_factors": stats(separately_exp2, exact),
        "native_exp": stats(native, exact),
    }), flush=True)


def contributions(drawn):
    operands, (do, dht) = drawn
    q, k, v, g, beta, h0 = (getattr(operands, name) for name in
                             ("query", "key", "value", "log_decay", "beta", "state"))
    q, k = (jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 64 - x.shape[-1]))) for x in (q, k))
    v, do = (jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 64 - x.shape[-1]))) for x in (v, do))
    h0, dht = (jnp.pad(x, ((0, 0), (0, 0), (0, 64 - x.shape[-2]), (0, 64 - x.shape[-1])))
               for x in (h0, dht))
    _, residual = port._rule_fwd(q, k, v, g, beta, h0, None, 0.25, 32, True)
    q, k, v, prefix, beta, a, h0, memory, spans = residual
    w, h, corrected = memory
    kernels = dict(chunk=32, spans=spans, interpret=True)
    dv = port.chunk_bwd_dv_local(q, k, prefix, do, scale=0.25, **kernels)
    dh, _, dv = port.chunk_bwd_dhu(q, k, w, prefix, dht, do, dv, scale=0.25, **kernels)
    _, _, dw, terms = port.chunk_bwd_dqkwg(q, k, corrected, w, prefix, h, dv, do, dh,
                                          scale=0.25, **kernels)
    hi, lo = prefix
    parts = {name: [] for name in ("dqkwg_pairs", "dqkwg_after", "wy_pairs", "wy_gate", "total")}
    for slot in range(len(spans.chunks)):
        start, taken, entered, left = map(int, np.asarray(spans.chunks[slot]))
        if taken <= 0:
            continue
        # The first value head suffices to expose magnitudes; retain the original fp32 boundaries.
        def at(array):
            return np.asarray(array).reshape(-1, *array.shape[2:])[start:start + taken, 0].astype(np.float64)

        keys, values, b, inv = at(k), at(v), at(beta), at(a)
        gradw, gradu = at(dw), at(dv)
        gates = at(hi) + at(lo)
        grown = np.exp2(gates)
        scaled = keys * (b * grown)[:, None]
        dx = gradw @ scaled.T + gradu @ (values * b[:, None]).T
        dx = np.tril(dx, -1)
        diff = np.where(np.tri(taken, k=-1, dtype=bool), gates[:, None] - gates[None, :], 0)
        dl = -(inv.T @ dx @ inv.T) * np.exp2(diff)
        dl = np.tril(dl, -1)
        pairwy = dl * (keys @ keys.T) * b[:, None]
        gatewy = np.sum((inv.T @ gradw) * scaled, axis=1)
        pairdq, afterdq = at(terms[0]), at(terms[1])
        index = np.arange(taken)
        def summed(pairs, after):
            through = np.cumsum(pairs, axis=1) - pairs
            return np.sum(np.where(index[:, None] >= index[None, :], through + after[:, None], 0), axis=0)
        pq, aq = summed(pairdq, np.zeros(taken)), summed(np.zeros_like(pairdq), afterdq)
        pw, aw = summed(pairwy, np.zeros(taken)), summed(np.zeros_like(pairwy), gatewy)
        for name, data in zip(parts, (pq, aq, pw, aw, pq + aq + pw + aw), strict=True):
            parts[name].append(data)
    measured = {}
    for name, data in parts.items():
        flat = np.concatenate(data)
        measured[name] = {"rms": float(np.sqrt(np.mean(flat**2))),
                          "max_abs": float(np.max(np.abs(flat)))}
    print(json.dumps({"terms": measured}), flush=True)


def pytest_addoption(parser):
    parser.addoption("--near-zero-regroup", action="store_true")


def pytest_configure(config):
    if config.getoption("--near-zero-regroup"):
        install()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--exponents-only", action="store_true")
    parser.add_argument("--contributions", action="store_true")
    parser.add_argument("--scope", choices=("all", "pairs", "absolute", "dots"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    hooks = SITES["scalar delta"]
    drawn = cotangents(hooks, "decays near zero", args.seed)
    exponent_probe(drawn)
    if args.exponents_only:
        return
    if args.contributions:
        contributions(drawn)
        return
    if args.split:
        install(args.scope)
    dew = without_ynn(with_gradients(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=True)))
    reference = without_ynn(with_gradients(functools.partial(token_by_token, hooks)))
    ours = dew(*drawn)
    theirs = reference(*drawn)
    with jax.enable_x64(new_val=True):
        truth = reference(*widened(drawn))
    for name in ("log_decay", "state"):
        a, b, c = (getattr(x[1], name) for x in (ours, theirs, truth))
        print(json.dumps({"leaf": name, "split": args.split, "port": stats(a, c),
                          "reference": stats(b, c),
                          "error_ratio": float(np.linalg.norm(np.asarray(a) - np.asarray(c)) /
                                               np.linalg.norm(np.asarray(b) - np.asarray(c)))}), flush=True)
        if name == "log_decay" and not args.split:
            hi, lo = recurrence.cumulated(drawn[0].log_decay, chunk=CHUNK, scale=port.LOG2_E,
                                          compensated=True, interpret=True)
            adjacent = np.asarray(jnp.exp2((hi[:, 1:] - hi[:, :-1]) + (lo[:, 1:] - lo[:, :-1])))
            actual = np.exp(np.asarray(drawn[0].log_decay[:, 1:]).astype(np.float64))
            pred = np.asarray(c)[:, 1:] * (adjacent / actual - 1)
            measured = np.asarray(a)[:, 1:] - np.asarray(c)[:, 1:]
            mask = np.arange(1, drawn[0].value.shape[1]) % CHUNK != 0
            print(json.dumps({"error_prediction": stats(pred[:, mask], measured[:, mask])}), flush=True)
    print(f"elapsed {time.monotonic() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
