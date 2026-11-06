#!/usr/bin/env python3
"""The fixture tests/test_selective_template.py holds the selective scan's template to: mamba_ssm's own
`selective_scan_fn` (its CUDA kernels, Mamba's selective scan) on a CUDA GPU, forward and every gradient,
over seeded draws in each regime a scan is hardest to hold.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu \
        python tools/mamba_selective_reference.py <checkout> <torch python>

on a GPU, Dew's environment recording and a second Python with torch (`--serve`) running mamba_ssm and the
oracle, its `selective_scan_cuda` built from the checkout for this GPU where none is installed. What lands is
tests/fixtures/mamba/selective_mamba_ssm_seeded.npz, a summary recording (`record_case(..., summary=True)`):
per case, seed and leaf, mamba_ssm's distance from float64 at the case's inputs, which the test draws
again (`operands`), and the truth's sketch, which Dew's float64 path is held to.

mamba_ssm's scan reads no documents: a packed row runs each document alone from no state, as Dew's scan
resets at a document's first token, and the row is their concatenation. The oracle is the recurrence one
token at a time in torch float64 (`minimal_truth`), written apart from Dew's and mamba_ssm's chunked forms.
"""

import itertools
import os
import sys
from pathlib import Path

import numpy as np

FIXTURE = (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba"
           / "selective_mamba_ssm_seeded.npz")
SEEDED_CASES = {
    # Each case's rows, tokens, channels, states, operands' dtype and regime.
    "bf16": (2, 200, 256, 16, "bfloat16", "plain"),
    "fp32": (2, 200, 256, 16, "float32", "plain"),
    "decays-near-one": (2, 200, 256, 16, "float32", "decays near one"),
    "decays-near-zero": (2, 200, 256, 16, "float32", "decays near zero"),
    "saturated-steps": (2, 200, 256, 16, "float32", "saturated steps"),
    "documents": (2, 200, 256, 16, "float32", "documents"),
    "long": (1, 4096, 256, 16, "float32", "plain"),
}
"""The recorded cases: Mamba's 16 states over two of the template's channel blocks, a ragged last chunk,
bf16 and fp32 operands as the mixer hands them, and in fp32 each regime: steps a thousandth of the
draw's (every decay near one), a thousand times it (near zero), saturated at 10, documents starting at
random tokens, and a sequence of 4096 tokens."""


def operands(rng, rows, tokens, channels, states, cast, regime) -> dict[str, np.ndarray]:
    """A case's inputs as Mamba's mixer hands them to its scan (the step through softplus), with `cast`
    narrowing `x`, the step, `B`, `C` and the cotangent and storing their values in fp32 (mamba_ssm takes
    the step in `x`'s dtype); `seq` each token's document where the regime packs them."""
    def narrow(shape):
        return cast(rng.standard_normal(shape))

    steps = np.log1p(np.exp(rng.standard_normal((rows, tokens, channels)) - 3.0))
    steps = {"decays near one": steps * 1e-3, "decays near zero": steps * 1e3,
             "saturated steps": np.full_like(steps, 10.0)}.get(regime, steps)
    arrays = {
        "x": narrow((rows, tokens, channels)), "dt": cast(steps),
        "A": -np.exp(0.5 * rng.standard_normal((channels, states))).astype(np.float32),
        "B": narrow((rows, tokens, states)), "C": narrow((rows, tokens, states)),
        "D": rng.standard_normal(channels).astype(np.float32), "dy": narrow((rows, tokens, channels)),
    }
    if regime == "documents":
        starts = rng.random((rows, tokens)) < 4.0 / tokens
        starts[:, 0] = True
        arrays["seq"] = np.cumsum(starts, axis=1).astype(np.int32)
    return arrays


def documents_of(arrays) -> list[list[tuple[int, int]]]:
    """Each row's documents as `(start, end)` token spans: the row whole where it packs none."""
    rows, tokens = arrays["x"].shape[:2]
    seq = arrays.get("seq")
    if seq is None:
        return [[(0, tokens)]] * rows
    found = []
    for row in range(rows):
        cuts = [0, *(1 + np.flatnonzero(seq[row, 1:] != seq[row, :-1])), tokens]
        found.append(list(itertools.pairwise(cuts)))
    return found


def minimal_truth(arrays) -> dict[str, np.ndarray]:
    """The scan one token at a time in torch float64, each document from no state, and its gradients under
    the recorded cotangent: `h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t`, `y_t = h_t C_t + D x_t`."""
    import torch

    names = ("x", "dt", "A", "B", "C", "D")
    inputs = tuple(torch.tensor(arrays[name], dtype=torch.float64, device="cuda", requires_grad=True)
                   for name in names)
    x, dt, A, Bm, Cm, D = inputs
    rows = []
    for row, spans in enumerate(documents_of(arrays)):
        outputs = []
        for start, end in spans:
            h = torch.zeros(A.shape, dtype=torch.float64, device="cuda")
            for t in range(start, end):
                h = torch.exp(dt[row, t, :, None] * A) * h + (dt[row, t] * x[row, t])[:, None] * Bm[row, t]
                outputs.append(h @ Cm[row, t] + D * x[row, t])
        rows.append(torch.stack(outputs))
    output = torch.stack(rows)
    cotangent = torch.tensor(arrays["dy"], dtype=torch.float64, device="cuda")
    grads = torch.autograd.grad(output, inputs, cotangent)
    return {name: value.detach().cpu().numpy() for name, value in
            zip(("y", *names), (output, *grads), strict=True)}


def mamba_ssm_run(scan, arrays, dtype) -> dict[str, np.ndarray]:
    """mamba_ssm's `selective_scan_fn` on a case's inputs, each document alone, its operands as Mamba
    hands them (`x`, the step, `B`, `C` in `dtype`, the rest fp32; channels before tokens), and its
    gradients."""
    import torch

    cuda = {name: torch.tensor(value, device="cuda") for name, value in arrays.items()}
    inputs = {name: cuda[name].to(dtype if name in ("x", "dt", "B", "C") else torch.float32).requires_grad_()
              for name in ("x", "dt", "A", "B", "C", "D")}
    rows = []
    for row, spans in enumerate(documents_of(arrays)):
        pieces = []
        for start, end in spans:
            x, dt, b, c = (inputs[name][row:row + 1, start:end].transpose(1, 2)
                           for name in ("x", "dt", "B", "C"))
            out = scan(x, dt, inputs["A"], b, c, inputs["D"], delta_softplus=False)
            pieces.append(out.transpose(1, 2))
        rows.append(torch.cat(pieces, dim=1))
    output = torch.cat(rows, dim=0)
    names = ("x", "dt", "A", "B", "C", "D")
    grads = torch.autograd.grad(output, tuple(inputs[name] for name in names), cuda["dy"].to(output.dtype))
    return {name: value.detach().float().cpu().numpy() for name, value in
            zip(("y", *names), (output, *grads), strict=True)}


def _extension(checkout: str):
    """mamba_ssm's `selective_scan_cuda` for this GPU: the installed one, else built from the checkout's
    sources by torch's JIT build, as mamba_ssm's setup.py compiles them."""
    import importlib
    import importlib.util

    if importlib.util.find_spec("selective_scan_cuda") is not None:
        return importlib.import_module("selective_scan_cuda")
    import torch
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    source = os.path.join(checkout, "csrc", "selective_scan")
    sources = [os.path.join(source, name) for name in sorted(os.listdir(source))
               if name.endswith((".cpp", ".cu"))]
    flags = ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
             f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]
    return load(name="selective_scan_cuda", sources=sources, extra_include_paths=[source],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags)


def serve(checkout: str) -> None:
    """The torch side (tools/reference_worker.py's protocol): each request a case's inputs and its `dtype`,
    answered with mamba_ssm's outputs and the oracle's (`reference.<leaf>`, `truth.<leaf>`), after one `ready`
    line of mamba_ssm's version, the checkout's revision and the device."""
    import re

    import reference_worker
    import torch
    from reference_source import git_head, package

    sys.modules["selective_scan_cuda"] = _extension(checkout)
    # The package without its `__init__`, which imports its models and through them a transformers
    # version of its own: the scan's interface alone.
    package("mamba_ssm", Path(checkout, "mamba_ssm"))
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    init = Path(checkout, "mamba_ssm", "__init__.py").read_text()
    version = re.search(r'__version__ = "([^"]+)"', init).group(1)

    def run(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dtype = str(arrays.pop("dtype"))
        return reference_worker.paired(mamba_ssm_run(selective_scan_fn, arrays, getattr(torch, dtype)),
                                       minimal_truth(arrays))

    reference_worker.serve(run, f"{version} {git_head(checkout)} {torch.cuda.get_device_name()}")


def write_seeded(worker) -> None:
    """Each case's seeded draws with mamba_ssm's outputs and gradients and their float64 truth (`worker`,
    `serve`'s `Session`), recorded as a summary (`record_case`), Dew's portable scan the sizing proxy and
    its float64 path held to the oracle at every seed (`assert_computes_the_oracle`)."""
    import jax
    import jax.numpy as jnp
    from recorded_references import held_to_the_oracle, provenance, record_case, save
    from reference_error import chain_roundings, widened
    from reference_worker import arms
    from selective_support import selective_draw, selective_xla

    version, revision, device = worker.started.split(maxsplit=2)

    def mamba_ssm(arrays, dtype: str):
        return arms(worker.ask({**arrays, "dtype": np.asarray(dtype)}))

    stored = {}
    for case, (*shape, dtype, regime) in SEEDED_CASES.items():
        narrow_dtype = jnp.dtype(dtype)
        exact_run = jax.jit(selective_xla)

        def draw(seed, case=case):
            return selective_draw(seed, case)

        with jax.enable_x64(new_val=True):
            roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(draw(0))))

        def evaluate(arrays, dtype=dtype, exact_run=exact_run, roundings=roundings):
            reference, truth = mamba_ssm(arrays, dtype)
            held_to_the_oracle(exact_run, arrays, truth, roundings, "mamba_ssm")
            return reference, truth

        stored.update(record_case(
            case, draw, evaluate,
            lambda arrays, dtype=narrow_dtype: selective_xla(arrays, dtype),
            truth_proxy=exact_run, store_inputs=False, summary=True))
        stored[f"{case}.dtype"] = np.asarray(str(narrow_dtype))
        stored[f"{case}.shape"] = np.asarray(shape, np.int32)
        stored[f"{case}.regime"] = np.asarray(regime)
    stored["provenance"] = provenance(
        f"mamba_ssm {version}", revision, device,
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/mamba_selective_reference.py <checkout> "
        "<torch python>",
        truth="the recurrence one token at a time in torch float64, per document",
        cross_check="Dew XLA float64; assert_computes_the_oracle with its traced chain_roundings")
    written = save(FIXTURE, stored, reference=f"mamba_ssm {version}", license="Apache-2.0")
    print(f"wrote {FIXTURE.name}: {written} ({written.stat().st_size:,} bytes)", flush=True)


def main() -> None:
    from reference_worker import session

    checkout = sys.argv[1]
    if sys.argv[2] == "--serve":
        serve(checkout)
        return
    with session([sys.argv[2], __file__, checkout, "--serve"]) as worker:
        write_seeded(worker)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
