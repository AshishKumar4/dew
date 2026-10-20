"""The fused SwiGLU's per-operation certificates on a GPU (a prototype; tests/op_certificates.py).

    python tools/op_certificates_swiglu.py --out=DIR [--sweep] [--interpret --elements=N --sweep-limit=N]

Each value the kernel rounds is a node computed by a probe: a Pallas kernel of that one operation,
launched as the fused kernel is (`norm_swiglu._launch`: the same block, warps and stages). For every
config of the kernel's tuning space, the probes composed from the operands must equal the fused
kernel's outputs bit for bit; where they do not, the compiler crossed a cut (a multiply and an add
contracted to one FMA), which is reported, not assumed away. The kernel's PTX names each node's
instruction; this script reads none of it, so its matching is a diagnostic until the PTX agrees.

On the probes' native values every node is then checked against its own contract: the IEEE
operations bit for bit against their exact rounding, the library ones (exp, and a reciprocal the PTX
may show approximate) by their distance in ulps from the correctly rounded value. `--sweep` runs
those unary nodes over every fp32 input, which certifies their constants on this GPU. The report is
DIR/swiglu.json. `--interpret` runs every launch interpreted, on a CPU, to test the script itself
before a GPU session; its numbers are the CPU's, not the kernel's.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import op_certificates as oc

from dew.nn.kernels import norm_swiglu

INTERPRET = False
"""Whether launches are interpreted (`--interpret`)."""
CONFIGS = sorted(norm_swiglu._Elementwise()._get_autotuning_configs(None), key=lambda c: (c.block, c.warps))


def _kernel(fn, arity: int, outputs: int):
    """A Pallas body applying `fn` to `arity` flat fp32 operands, element by element, as the fused
    kernels load and store them."""
    def body(*refs, block: int, interpret: bool):
        index = pl.program_id(0) * block + jnp.arange(block)
        mask = index < refs[0].shape[0]
        operands = [plgpu.load(ref.at[index], mask=mask, other=0).astype(jnp.float32) for ref in refs[:arity]]
        results = fn(*operands)
        for ref, result in zip(refs[arity:], results if outputs > 1 else (results,), strict=True):
            plgpu.store(ref.at[index], result.astype(ref.dtype), mask=mask)
    return body


def _launched(body, operands, dtypes, config):
    shape = operands[0].shape
    outputs = tuple(jax.ShapeDtypeStruct(shape, dtype) for dtype in dtypes)
    result, _ = norm_swiglu._launch(body, "probe", operands, outputs if len(outputs) > 1 else outputs[0],
                                     (pl.cdiv(shape[0], config.block),), config, interpret=INTERPRET)
    return result


@functools.cache
def _compiled(fn, arity: int, dtypes: tuple, config):
    body = _kernel(fn, arity, len(dtypes))
    return jax.jit(functools.partial(_launched, body, dtypes=dtypes, config=config))


def _identity(x):
    return x


def probe(fn, operands, config, dtypes=(jnp.float32,)):
    """`fn` of `operands` (flat device arrays) as its own launch at `config`, one compiled program
    for each operation, config and output dtypes."""
    return _compiled(fn, len(operands), tuple(dtypes), config)(list(operands))


# Each node: its name, its parents, the operation as the kernel writes it, and its contract: "exact"
# (RN of the exact result, the oracle named) or "ulps" (the correctly rounded value, the oracle named).
FORWARD = [
    ("neg", ("g",), lambda g: -g, ("exact", "neg")),
    ("e", ("neg",), jnp.exp, ("ulps", "exp")),
    ("d", ("e",), lambda e: 1 + e, ("exact", "add1")),
    ("s", ("d",), lambda d: 1 / d, ("exact", "recip")),
    ("a", ("g", "s"), lambda g, s: g * s, ("exact", "mul")),
]
"""The forward's sigmoid and its product with the gate, as Pallas lowers `jax.nn.sigmoid`
(1 / (1 + exp(-a))) and `_product` writes it; the product with `up` and the casts follow per dtype."""
BACKWARD = [
    ("q", ("s",), lambda s: 1.0 - s, ("exact", "rsub1")),
    ("k", ("s", "q"), lambda s, q: s * q, ("exact", "mul")),
    ("t", ("g", "k"), lambda g, k: g * k, ("exact", "mul")),
    ("slope", ("s", "t"), lambda s, t: s + t, ("exact", "add")),
    ("r", ("dy", "slope"), lambda dy, slope: dy * slope, ("exact", "mul")),
    ("dg", ("r", "up"), lambda r, up: r * up, ("exact", "mul")),
    ("du", ("dy", "a"), lambda dy, a: dy * a, ("exact", "mul")),
]
"""`_gradients` through `_activated('silu')`: slope = s + g·(s·(1 - s)), a candidate for one FMA."""


def oracle(kind: str, parents, fmt=oc.FLOAT32):
    x = [np.asarray(parent, np.float64) for parent in parents]
    return {
        "neg": lambda: -x[0],
        "add1": lambda: oc.exact_add(np.ones_like(x[0]), x[0], fmt),
        "rsub1": lambda: oc.exact_add(np.ones_like(x[0]), -x[0], fmt),
        "recip": lambda: oc.exact_div(np.ones_like(x[0]), x[0], fmt),
        "mul": lambda: oc.exact_mul(x[0], x[1], fmt),
        "add": lambda: oc.exact_add(x[0], x[1], fmt),
        "fma": lambda: oc.exact_fma(x[0], x[1], x[2], fmt),
        "exp": lambda: oc.exp_rounded(x[0], fmt),
    }[kind]()


def ulps(native, reference) -> np.ndarray:
    """Each fp32 value's distance from the reference in fp32 steps, signed zeros apart by none."""
    def ordered(values):
        bits = np.asarray(values, np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)
    return np.abs(ordered(native) - ordered(reference))


def gates(count: int, rng) -> np.ndarray:
    """Gate values over the regions SwiGLU's arithmetic changes in: the bulk, exp's overflow and
    underflow, the slope's cancellation near -1.28, tiny and subnormal values, and zeros."""
    eighth, sixteenth = count // 8, count // 16
    signs = rng.choice([-1, 1], sixteenth)
    parts = [rng.normal(0, 3, count // 2), rng.uniform(-110, 110, eighth), rng.uniform(-1.4, -1.15, eighth),
             rng.uniform(85, 105, sixteenth) * signs,
             np.ldexp(rng.random(sixteenth) + 1, rng.integers(-149, -100, sixteenth)) * signs]
    values = np.concatenate([*parts, np.zeros(count - sum(len(part) for part in parts))]).astype(np.float32)
    return rng.permutation(values)


def nodes_of(table, values: dict, config) -> dict:
    """Each node of `table` probed from `values`' native parents, in table order."""
    for name, parents, fn, _ in table:
        values[name] = probe(fn, [values[parent] for parent in parents], config)
    return values


def check(table, values: dict) -> dict:
    """Each node's contract on its native parents: flags for exact nodes, the ulp distribution for
    the rest."""
    report = {}
    for name, parents, _, (contract, kind) in table:
        native = np.asarray(values[name], np.float32)
        want = oracle(kind, [np.asarray(values[parent], np.float32) for parent in parents])
        if contract == "exact":
            wrong = oc.mismatches(native, want, oc.FLOAT32)
            first = np.asarray(values[parents[0]])[wrong][:4]
            report[name] = {"contract": f"exact {kind}", "flags": int(wrong.sum()),
                            "elements": int(native.size), "first": [float(value) for value in first]}
        else:
            distance = ulps(native, want)
            finite = np.isfinite(want)
            steps, counts = np.unique(np.minimum(distance[finite], 8), return_counts=True)
            histogram = {str(step): int(count) for step, count in zip(steps, counts, strict=True)}
            edges = oc.mismatches(native[~finite], want[~finite], oc.FLOAT32)
            report[name] = {"contract": f"ulps {kind}", "max": int(distance[finite].max()),
                            "histogram": histogram, "nonfinite_mismatch": int(edges.sum())}
    return report


def fused(gate, up, dy, config):
    y = norm_swiglu._launch(norm_swiglu._swiglu_forward, "swiglu_forward", (gate, up),
                            jax.ShapeDtypeStruct(gate.shape, gate.dtype), (pl.cdiv(gate.size, config.block),),
                            config, interpret=INTERPRET)[0]
    outputs = (jax.ShapeDtypeStruct(gate.shape, gate.dtype), jax.ShapeDtypeStruct(up.shape, up.dtype))
    dg, du = norm_swiglu._launch(norm_swiglu._swiglu_backward, "swiglu_backward", (gate, up, dy), outputs,
                                 (pl.cdiv(gate.size, config.block),), config, interpret=INTERPRET)[0]
    return y, dg, du


def composed(values: dict, dtype, config) -> tuple:
    """The fused kernel's outputs rebuilt from the probes: for bf16, the product rounded to bf16 and
    back before `up` multiplies it, and every output stored by one cast."""
    narrow = jnp.dtype(dtype) != jnp.dtype(jnp.float32)
    a = probe(_narrowed[jnp.dtype(dtype).name], [values["a"]], config) if narrow else values["a"]
    return (probe(_product, [a, values["up"]], config, dtypes=(dtype,)),
            probe(_product, [values["r"], values["up"]], config, dtypes=(dtype,)),
            probe(_product, [values["dy"], values["a"]], config, dtypes=(dtype,)))


def _product(left, right):
    return left * right


_narrowed = {"bfloat16": lambda a: a.astype(jnp.bfloat16).astype(jnp.float32)}


def bitwise(left, right) -> int:
    left, right = np.asarray(left), np.asarray(right)
    bits = np.uint32 if left.dtype == np.float32 else np.uint16
    return int((left.view(bits) != right.view(bits)).sum())


def sweep(name: str, fn, kind: str, config, chunk: int = 1 << 26, limit: int = 1 << 32) -> dict:
    """`fn` over every fp32 bit pattern (the first `limit`), each against its correctly rounded
    value: the node's largest distance in ulps on this GPU, and where it sits."""
    worst, where, mismatched_nonfinite, started = 0, None, 0, time.time()
    chunk = min(chunk, limit)
    for start in range(0, limit, chunk):
        bits = np.arange(start, start + chunk, dtype=np.uint64).astype(np.uint32)
        x = bits.view(np.float32)
        native = np.asarray(probe(fn, [jnp.asarray(x)], config), np.float32)
        want = oracle(kind, [x])
        finite = np.isfinite(want) & np.isfinite(x)
        distance = np.where(finite, ulps(native, want), 0)
        if distance.max() > worst:
            worst, where = int(distance.max()), float(x[int(distance.argmax())])
        edge = ~finite & ~np.isnan(x)
        mismatched_nonfinite += int(oc.mismatches(native[edge], want[edge], oc.FLOAT32).sum())
    return {"node": name, "max_ulps": worst, "at": where, "nonfinite_mismatch": mismatched_nonfinite,
            "seconds": round(time.time() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=1 << 24)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-limit", type=int, default=1 << 32)
    parser.add_argument("--interpret", action="store_true")
    args = parser.parse_args()
    global INTERPRET
    INTERPRET = args.interpret
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {"device": str(jax.devices()[0]), "jax": jax.__version__, "configs": {}, "nodes": {}}
    base = {"g": gates(args.elements, rng), "up": rng.normal(0, 1, args.elements).astype(np.float32),
            "dy": rng.normal(0, 1, args.elements).astype(np.float32)}
    for dtype in (jnp.float32, jnp.bfloat16):
        name = jnp.dtype(dtype).name
        operands = {key: jnp.asarray(value.astype(ml_dtypes.bfloat16) if name == "bfloat16" else value)
                    for key, value in base.items()}
        for config in CONFIGS:
            values = {key: probe(_identity, [value], config) for key, value in operands.items()}
            values = nodes_of(FORWARD + BACKWARD, values, config)
            native = fused(operands["g"], operands["up"], operands["dy"], config)
            rebuilt = composed(values, dtype, config)
            pairs = zip(("y", "dg", "du"), native, rebuilt, strict=True)
            report["configs"][f"{name} block={config.block} warps={config.warps}"] = {
                output: bitwise(left, right) for output, left, right in pairs}
            if config == norm_swiglu.BlockConfig(norm_swiglu.ELEMENT_BLOCK, 4):
                held = {key: np.asarray(value) for key, value in values.items()}
                report["nodes"][name] = check(FORWARD + BACKWARD, held)
        print(name, json.dumps(report["nodes"].get(name, {}), default=str)[:2000], flush=True)
    if args.sweep:
        config = norm_swiglu.BlockConfig(norm_swiglu.ELEMENT_BLOCK, 4)
        report["sweeps"] = [sweep("e", FORWARD[1][2], "exp", config, limit=args.sweep_limit),
                            sweep("s", FORWARD[3][2], "recip", config, limit=args.sweep_limit)]
    (args.out / "swiglu.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["configs"], indent=0))
    print(json.dumps(report.get("sweeps", []), indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
