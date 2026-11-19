#!/usr/bin/env python3
"""The fixture tests/test_mamba3.py holds Mamba-3's chunked form to: mamba_ssm's own `mamba3_siso_combined`
(its Triton kernels, Mamba-3's SISO) on a CUDA GPU, forward and every gradient, at each site
tests/mamba3_sites.py names, over the tests' own seeded draws.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu \
        python tools/mamba3_reference.py <torch python> <unit> <site>...
    PYTHONPATH=src:tests:tools python tools/mamba3_reference.py --merge <unit>...

on a GPU, Dew's environment recording and a second Python with torch and mamba_ssm (`--serve`) running the
kernel, each run a unit of named sites (`--list` names them all), the units then merged. What lands is
tests/fixtures/mamba3/siso_seeded.npz, a summary recording (`record_case(..., summary=True)`): per site,
seed and leaf, mamba_ssm's distance from float64 at the site's inputs, which the test draws again
(`Site.arrays`), and the truth's sketch, which Dew's float64 recurrence is. The truth is Dew's recurrence a
token at a time in float64 (`mamba3_sites.truth`): mamba_ssm ships no reference of Mamba-3's of its own.
Its upload runs on the recording host (`fetched_fixtures.py stage`, then `dispatch`).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

COHORT = "mamba-ssm"
"""The reference releases this tool runs against (tools/reference_releases.py), and the environment
tools/check_tool_types.py checks it in."""

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba3" / "siso_seeded.npz"


def run(arrays: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """`mamba3_siso_combined` on a site's arrays (`mamba3_sites.Site.arrays`' layout, `[B, L, H]` where the
    kernel reads `[B, H, L]`; its chunk `chunk`, a 0-d array), its biases zero and no skip, gate or state
    entering: its output and each operand's gradient under `d_out`, as float32 in the site's layout. Packed
    documents as one row of sequences (`cu_seqlens`)."""
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    # Its chunk rides the request (`write_seeded`): this environment holds no Dew, the sites' module none.
    chunk = int(arrays.pop("chunk"))
    narrow = {"x": torch.bfloat16, "B": torch.bfloat16, "C": torch.bfloat16}
    operands = [name for name in arrays if name not in ("d_out", "starts")]
    leaves = {name: torch.tensor(arrays[name], device=device, dtype=narrow.get(name, torch.float32))
              .requires_grad_() for name in operands}
    cu_seqlens = None
    if "starts" in arrays:
        firsts = np.flatnonzero(arrays["starts"][0]).tolist()
        cu_seqlens = torch.tensor([*firsts, arrays["starts"].shape[1]], device=device, dtype=torch.int32)
    zero = torch.zeros((arrays["x"].shape[2], arrays["B"].shape[-1]), device=device, dtype=torch.float32)
    out = mamba3_siso_combined(
        Q=leaves["C"], K=leaves["B"], V=leaves["x"], ADT=leaves["ADT"].transpose(1, 2),
        DT=leaves["DT"].transpose(1, 2), Trap=leaves["trap"].transpose(1, 2), Q_bias=zero, K_bias=zero,
        Angles=leaves["angles"], chunk_size=chunk, cu_seqlens=cu_seqlens)
    assert isinstance(out, torch.Tensor)  # The output alone: no final state asked for.
    out.backward(torch.tensor(arrays["d_out"], device=device, dtype=out.dtype))
    grads = {name: leaves[name].grad for name in operands}
    assert all(grad is not None for grad in grads.values())  # Every operand a leaf of the backward pass.
    return {"out": out.detach().float().cpu().numpy(),
            **{name: grad.float().cpu().numpy() for name, grad in grads.items() if grad is not None}}


def serve() -> None:
    """The torch side: each request a site's arrays, its answer mamba_ssm's leaves (`run`)."""
    from importlib.metadata import version

    import reference_worker
    import torch

    torch_release = version('torch').split('+')[0]
    found = f"mamba_ssm-{version('mamba_ssm')}+torch-{torch_release}+triton-{version('triton')}"

    def answer(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        began = time.perf_counter()
        reference = run(arrays, "cuda")
        print(f"timed mamba_ssm {time.perf_counter() - began:.3f} s", file=sys.stderr, flush=True)
        return reference_worker.paired(reference, {})

    reference_worker.serve(answer, f"{found} {torch.cuda.get_device_name()}")


def write_seeded(worker, names: list[str], out: Path) -> None:
    """Each named site's seeded draws with mamba_ssm's leaves and their float64 truth (Dew's recurrence a
    token at a time, `mamba3_sites.truth`), recorded as a summary (`record_case`) into `out`: a unit of the
    recording, whose units `reference_worker.merged_units` joins into the fixture."""
    import jax
    from mamba3_sites import CHUNK, sites, truth
    from recorded_references import provenance, record_case
    from reference_worker import arms

    version, device = worker.started.split(maxsplit=1)
    stored = {}
    with jax.enable_x64(new_val=True):
        exact = jax.jit(truth)

    def evaluate(arrays):
        request = {name: np.asarray(value) for name, value in arrays.items()} | {"chunk": np.asarray(CHUNK)}
        reference, _ = arms(worker.ask(request))
        with jax.enable_x64(new_val=True):
            return reference, {leaf: np.asarray(value, np.float64) for leaf, value in exact(arrays).items()}

    for name in names:
        began = time.perf_counter()
        stored.update(record_case(name, sites()[name].arrays, evaluate, store_inputs=False, summary=True))
        print(f"{name}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        version, version, device.strip(),
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/mamba3_reference.py <torch python> <unit> "
        "<site>...",
        truth="Dew's recurrent_mamba3 a token at a time in float64 (mamba_ssm ships no Mamba-3 reference)",
        cross_check="none: the truth is Dew's own float64 recurrence")
    np.savez(out, **stored)
    print(f"wrote {out}: {out.stat().st_size:,} bytes", flush=True)


def main() -> None:
    """The recorder's command line (`reference_worker.recording_command`), its sites `mamba3_sites`'s."""
    from reference_worker import recording_command

    recording_command(__file__, "mamba3_sites", serve=serve, write_seeded=write_seeded,
                      fixture=FIXTURE, license="Apache-2.0")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
