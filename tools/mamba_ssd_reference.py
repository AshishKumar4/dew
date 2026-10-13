#!/usr/bin/env python3
"""The fixture tests/test_mamba_ssd.py holds the SSD port to: mamba_ssm 2.3.2.post1's own
`mamba_chunk_scan_combined` on a CUDA GPU, forward and every gradient, at IEEE products.

    python tools/mamba_ssd_reference.py <mamba_ssm source checkout>

in an environment with torch, Triton and einops on a GPU. What lands is
tests/fixtures/mamba2/ssd_mamba_ssm.npz, each case's arrays under its name (`CASES`): fixed-seed operands
as Mamba-2 hands them over (`x`, `B` and `C` in the case's dtype, the step through softplus and `A`,
`D` and an initial state in fp32), each token's document (`seq_idx`), the cotangents of the output and
of the final state, and mamba_ssm's output, final state and gradients of `x`, `dt`, `A`, `B`, `C`, `D`
and the initial state.

mamba_ssm's bf16 path multiplies bf16 operands, exact in its fp32 sums. Its fp32 path takes Triton's
default fp32 products, TF32, unless `TRITON_F32_DEFAULT` says otherwise; this sets it to 'ieee' before
Triton loads, so the fixture is mamba_ssm's algorithm at IEEE products, and records 'ieee'
(`reference_error.PRECISION`).
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ["TRITON_F32_DEFAULT"] = "ieee"

import torch

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba2" / "ssd_mamba_ssm.npz"
SEED = 2026
CASES = {
    # name: (batch, tokens, heads, head width, groups, state, chunk, dtype, packed documents)
    # Three heads a group, a ragged last chunk, one row starting on a document, one carrying its
    # initial state into its first: in bf16, and in fp32 at a smaller state.
    "bf16": (2, 200, 6, 16, 2, 32, 64, torch.bfloat16, True),
    "ragged-documents": (2, 100, 6, 16, 2, 16, 32, torch.float32, True),
    # Whole chunks, one group, no documents.
    "whole-chunks": (1, 128, 4, 32, 1, 32, 64, torch.float32, False),
    # A state wider than one 128-wide product, as Falcon-H1's 256.
    "wide-state-documents": (1, 48, 2, 16, 2, 256, 16, torch.float32, True),
}


def operands(rng, batch, tokens, heads, width, groups, state, dtype, documents) -> dict[str, np.ndarray]:
    """A case's inputs, those in `dtype` stored as the fp32 values they hold."""
    def narrow(shape):
        return torch.tensor(rng.standard_normal(shape), dtype=dtype).float().numpy()

    starts = rng.random((batch, tokens)) < 3.0 / tokens
    starts[0, 0] = True
    seq = {"seq": np.cumsum(starts, axis=1).astype(np.int32)} if documents else {}
    return {**seq,
        "x": narrow((batch, tokens, heads, width)), "B": narrow((batch, tokens, groups, state)),
        "C": narrow((batch, tokens, groups, state)),
        "dt": np.log1p(np.exp(rng.standard_normal((batch, tokens, heads)) - 2.0)).astype(np.float32),
        "A": -np.exp(rng.uniform(0.0, np.log(16.0), heads)).astype(np.float32),
        "D": rng.standard_normal(heads).astype(np.float32),
        "h0": (0.1 * rng.standard_normal((batch, heads, width, state))).astype(np.float32),
        "dy": narrow((batch, tokens, heads, width)),
        "dfinal": rng.standard_normal((batch, heads, width, state)).astype(np.float32),
    }


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    from mamba_ssm import __version__
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    rng = np.random.default_rng(SEED)
    stored = {}
    for name, (*shape, chunk, dtype, documents) in CASES.items():
        arrays = operands(rng, *shape, dtype, documents)
        cuda = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()}
        narrow = {key: cuda[key].to(dtype).requires_grad_() for key in ("x", "B", "C")}
        wide = {key: cuda[key].clone().requires_grad_() for key in ("dt", "A", "D", "h0")}
        y, final = mamba_chunk_scan_combined(narrow["x"], wide["dt"], wide["A"], narrow["B"], narrow["C"],
                                             chunk, D=wide["D"], initial_states=wide["h0"],
                                             seq_idx=cuda.get("seq"), return_final_states=True)
        inputs = (narrow["x"], wide["dt"], wide["A"], narrow["B"], narrow["C"], wide["D"], wide["h0"])
        grads = torch.autograd.grad((y, final), inputs, (cuda["dy"].to(dtype), cuda["dfinal"]))
        names = ("y", "final", "x", "dt", "A", "B", "C", "D", "h0")
        found = dict(zip(names, (y, final, *grads), strict=True))
        arrays.update({f"mamba.{key}": value.detach().float().cpu().numpy() for key, value in found.items()})
        arrays.update(chunk=np.int32(chunk), dtype=np.asarray(str(dtype).removeprefix("torch.")))
        stored.update({f"{name}.{key}": value for key, value in arrays.items()})
    np.savez_compressed(FIXTURE, precision=np.asarray("ieee"), version=np.asarray(__version__),
             device=np.asarray(torch.cuda.get_device_name()), **stored)
    print(f"wrote {FIXTURE} from mamba_ssm {__version__} on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
