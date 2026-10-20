"""CPU-only, one-variable arithmetic probes of the interpreted SSD kernel."""

import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import lane_environment  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dew.nn.kernels import ssd  # noqa: E402
from reference_error import widened  # noqa: E402
from test_ssd_kernel import case_draw, parity_functions  # noqa: E402


SHAPES = {
    "one-short": (2, 63, 4, 8, 8, 2),
    "whole-chunk": (2, 64, 4, 8, 8, 2),
    "one-over": (2, 65, 4, 16, 8, 2),
}

REPLACEMENTS = {
    "baseline": (),
    "prefix-cumsum": (
        ("entered = jnp.exp(jnp.sum(spread, axis=-1))",
         "entered = jnp.exp(jnp.cumsum(a))"),
    ),
    "total-cumsum": (
        ("jnp.exp(jnp.sum(a))", "jnp.exp(jnp.cumsum(a)[-1])"),
    ),
    "prefix-total-cumsum": (
        ("entered = jnp.exp(jnp.sum(spread, axis=-1))",
         "entered = jnp.exp(jnp.cumsum(a))"),
        ("jnp.exp(jnp.sum(a))", "jnp.exp(jnp.cumsum(a)[-1])"),
    ),
    "dstate-scale-c": (
        ("(dy * entered[:, None]).T @ c", "dy.T @ (c * entered[:, None])"),
    ),
    "dstate-dot-transposed": (
        ("(dy * entered[:, None]).T @ c", "(c.T @ (dy * entered[:, None])).T"),
    ),
    "dstate-reduce": (
        ("(dy * entered[:, None]).T @ c",
         "jnp.sum((dy * entered[:, None])[:, :, None] * c[:, None, :], axis=0)"),
    ),
    "dstate-dot-split": (
        ("(dy * entered[:, None]).T @ c",
         "((dy[:32] * entered[:32, None]).T @ c[:32]"
         " + (dy[32:] * entered[32:, None]).T @ c[32:])"),
    ),
    "write-scale-x": (
        ("x.T @ (b * leaving[:, None])", "(x * leaving[:, None]).T @ b"),
    ),
    "write-suffix": (
        ("jnp.exp(_last_row(segments))",
         "jnp.exp(jnp.pad(jax.lax.cumsum(a, axis=0, reverse=True)[1:], (0, 1)))"),
    ),
}
ORIGINAL = {function.__name__: inspect.getsource(function)
            for function in (ssd._chunk_forward, ssd._chunk_backward)}


def install_probe(name):
    """Change only the declared expression, leaving product files untouched."""
    for original_source in ORIGINAL.values():
        source = original_source
        for before, after in REPLACEMENTS[name]:
            source = source.replace(before, after)
        exec(compile(source, f"<ssd arithmetic probe {name}>", "exec"), ssd.__dict__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=REPLACEMENTS, default="baseline")
    parser.add_argument("--case", choices=SHAPES, default="whole-chunk")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    install_probe(args.variant)
    shape, chunk = SHAPES[args.case], 64
    draw = case_draw({}, "shape", (shape, chunk))
    reference, truth, evaluated = parity_functions(chunk, sum((shape[1], shape[3], shape[4])))
    kernel = jax.jit(lambda operands, seeded: evaluated(
        lambda *o: ssd.ssd_chunk_scan(*o, chunk, "tpu"), operands, seeded))
    errors = {name: {"mine": [], "reference": [], "truth": []} for name in ("state", "final")}
    for seed in range(args.seeds):
        operands = draw(seed)
        mine, theirs = kernel(*operands), reference(*operands)
        with jax.enable_x64(True):
            exact = truth(*widened(operands))
        for name, results in errors.items():
            results["truth"].append(np.asarray(exact[name], np.float64))
            for which, values in (("mine", mine), ("reference", theirs)):
                results[which].append(np.asarray(values[name], np.float64)
                                      - np.asarray(exact[name], np.float64))
    results = {}
    for name, arrays in errors.items():
        arrays = {which: np.asarray(value) for which, value in arrays.items()}
        mine_rms = float(np.sqrt(np.mean(np.square(arrays["mine"]))))
        their_rms = float(np.sqrt(np.mean(np.square(arrays["reference"]))))
        results[name] = {
            "mine_rms": mine_rms,
            "reference_rms": their_rms,
            "ratio": mine_rms / their_rms,
            "fraction_farther": float(np.mean(np.abs(arrays["mine"]) > np.abs(arrays["reference"]))),
        }
        if args.save:
            np.savez(args.save.with_name(f"{args.save.stem}-{name}.npz"), **arrays)
    print(json.dumps({"variant": args.variant, "case": args.case, "seeds": args.seeds,
                      "backend": jax.default_backend(), "leaves": results}, sort_keys=True), flush=True)
    if args.gate:
        assert all(value["ratio"] <= 1.5 for value in results.values()), results


@pytest.mark.parametrize("variant", REPLACEMENTS)
@pytest.mark.parametrize("case", SHAPES)
def test_probe(case, variant):
    main(["--case", case, "--variant", variant, "--seeds", "32", "--gate"])


if __name__ == "__main__":
    main()
