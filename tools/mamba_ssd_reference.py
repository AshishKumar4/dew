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

import itertools
import os
import sys
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba2" / "ssd_mamba_ssm.npz"
SEED = 2026
CASES = {
    # name: (batch, tokens, heads, head width, groups, state, chunk, dtype, packed documents)
    # Three heads a group, a ragged last chunk, one row starting on a document, one carrying its
    # initial state into its first: in bf16, and in fp32 at a smaller state.
    "bf16": (2, 200, 6, 16, 2, 32, 64, "bfloat16", True),
    "ragged-documents": (2, 100, 6, 16, 2, 16, 32, "float32", True),
    # Whole chunks, one group, no documents.
    "whole-chunks": (1, 128, 4, 32, 1, 32, 64, "float32", False),
    # A state wider than one 128-wide product, as Falcon-H1's 256.
    "wide-state-documents": (1, 48, 2, 16, 2, 256, 16, "float32", True),
}

SEEDED_CASES = {
    "bf16": (2, 200, 24, 16, 8, 32, 64, "bfloat16", True),
    "ragged-documents": (2, 100, 24, 16, 8, 16, 32, "float32", True),
    "whole-chunks": (1, 128, 64, 32, 1, 32, 64, "float32", False),
    "wide-state-documents": (2, 48, 16, 16, 2, 256, 16, "float32", True),
}
"""The seeded recording's cases (`write_seeded`): `CASES`' with more heads (three a group where
`CASES` has three, one group where it has one) and the wide state over two rows, where `CASES`' own
did not resolve: `D`'s gradient is a head's one sum, and a few heads' per-head decays are each
draw's shared factors, so the reference compared with itself needed 512 seeds or more on the XLA
proxy (`seeds_for`), mamba_ssm's own more than 512 for ragged-documents; these take 64 to 128 on the
proxy (2026-10-21, on armada)."""


def operands(rng, batch, tokens, heads, width, groups, state, cast, documents) -> dict[str, np.ndarray]:
    """A case's inputs, with `cast` narrowing native operands and storing their values in fp32."""
    def narrow(shape):
        return cast(rng.standard_normal(shape))

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


def minimal_truth(minimal, arrays, chunk):
    """The upstream minimal reference in float64, composed over every native document and input.

    Expanding grouped B/C only copies shared coordinates. Documents reset to zero, and padding with
    zero-dt steps writes nothing and decays by exp(0)=1, so the final state and every native output
    are unchanged. The artificial padding is removed before applying the original cotangents.
    """
    import torch

    names = ("x", "dt", "A", "B", "C", "D", "h0")
    inputs = tuple(torch.tensor(arrays[name], dtype=torch.float64, device="cuda", requires_grad=True)
                   for name in names)
    x, dt, A, Bm, Cm, D, h0 = inputs
    batch, length, heads, _ = x.shape
    Bm, Cm = (value.repeat_interleave(heads // value.shape[2], dim=2) for value in (Bm, Cm))
    seq = arrays.get("seq")
    rows, finals = [], []
    for row in range(batch):
        cuts = ([0, length] if seq is None else
                [0, *(1 + np.flatnonzero(seq[row, 1:] != seq[row, :-1])), length])
        outputs = []
        for start, end in itertools.pairwise(cuts):
            reset = seq is not None and (start > 0 or seq[row, 0] > 0)
            initial = h0[row:row + 1, None] * (0 if reset else 1)
            values = (x[row:row + 1, start:end] * dt[row:row + 1, start:end, :, None],
                      A * dt[row:row + 1, start:end], Bm[row:row + 1, start:end], Cm[row:row + 1, start:end])
            pad = (chunk - (end - start) % chunk) % chunk
            blocked = tuple(torch.nn.functional.pad(value, (0, 0) * (value.ndim - 2) + (0, pad))
                            for value in values)
            output, final = minimal(*blocked, chunk, initial_states=initial)
            outputs.append(output[:, :end - start] + x[row:row + 1, start:end] * D[None, None, :, None])
        rows.append(torch.cat(outputs, dim=1))
        finals.append(final)
    output, final = torch.cat(rows, dim=0), torch.cat(finals, dim=0)
    cotangents = tuple(torch.tensor(arrays[name], dtype=torch.float64, device="cuda")
                       for name in ("dy", "dfinal"))
    grads = torch.autograd.grad((output, final), inputs, cotangents)
    return {name: value.detach().cpu().numpy() for name, value in
            zip(("y", "final", *names), (output, final, *grads), strict=True)}


def write_seeded(scan, minimal, version, revision):
    """Indexed fresh-input/head/P/N draws with native mamba_ssm outputs and independent float64 truth.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/mamba_ssd_reference.py <checkout> --seeded
    """
    import jax
    import jax.numpy as jnp
    import torch
    from recorded_references import provenance, record_case, save
    from reference_error import assert_computes_the_oracle, chain_roundings, widened
    from ssd_support import mamba_ssd_draw, mamba_ssd_xla, ssd_orbit

    stored = {"precision": np.asarray("ieee")}
    for case, (*shape, chunk, dtype, documents) in SEEDED_CASES.items():
        narrow_dtype, dtype = jnp.dtype(dtype), getattr(torch, dtype)
        exact_run = jax.jit(lambda arrays, chunk=chunk: mamba_ssd_xla(arrays, chunk))

        def draw(seed, shape=shape, dtype=dtype, documents=documents, narrow_dtype=narrow_dtype):
            inputs = ssd_orbit(operands(np.random.default_rng(seed), *shape,
                                        lambda value: torch.tensor(value, dtype=dtype).float().numpy(),
                                        documents), seed)
            redrawn = mamba_ssd_draw(seed, shape, narrow_dtype, documents)
            assert inputs.keys() == redrawn.keys(), f"mamba_ssm seed {seed}: regenerated inputs differ"
            for name, value in inputs.items():
                other = redrawn[name]
                assert (value.dtype == other.dtype and value.shape == other.shape
                        and value.tobytes() == other.tobytes()), f"mamba_ssm seed {seed}: {name} differs"
            return inputs

        with jax.enable_x64(new_val=True):
            roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(draw(0))))

        def evaluate(arrays, chunk=chunk, dtype=dtype, exact_run=exact_run, roundings=roundings):
            cuda = {name: torch.tensor(value, device="cuda") for name, value in arrays.items()}
            inputs = tuple(cuda[name].to(dtype if name in ("x", "B", "C") else torch.float32)
                           .requires_grad_() for name in ("x", "dt", "A", "B", "C", "D", "h0"))
            x, dt, A, Bm, Cm, D, h0 = inputs
            output, final = scan(x, dt, A, Bm, Cm, chunk, D=D, initial_states=h0,
                                 seq_idx=cuda.get("seq"), return_final_states=True)
            grads = torch.autograd.grad((output, final), inputs, (cuda["dy"].to(dtype), cuda["dfinal"]))
            names = ("y", "final", "x", "dt", "A", "B", "C", "D", "h0")
            reference = {name: value.detach().float().cpu().numpy() for name, value in
                         zip(names, (output, final, *grads), strict=True)}
            truth = minimal_truth(minimal, arrays, chunk)
            with jax.enable_x64(new_val=True):
                twin = jax.tree.map(np.asarray, exact_run(widened(arrays)))
            for name in truth:
                assert_computes_the_oracle(twin[name], truth[name], f"mamba_ssm float64 {name}",
                                          roundings=roundings)
            return reference, truth

        stored.update(record_case(
            case, draw, evaluate,
            lambda arrays, chunk=chunk, dtype=narrow_dtype: mamba_ssd_xla(arrays, chunk, dtype),
            truth_proxy=exact_run, store_inputs=False, summary=True))
        stored[f"{case}.chunk"] = np.int32(chunk)
        stored[f"{case}.dtype"] = np.asarray(str(narrow_dtype))
        stored[f"{case}.shape"] = np.asarray(shape, np.int32)
        stored[f"{case}.documents"] = np.bool_(documents)
    stored["provenance"] = provenance(
        f"mamba_ssm {version}", revision, torch.cuda.get_device_name(),
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu "
        "python tools/mamba_ssd_reference.py <checkout> --seeded",
        truth="mamba_ssm.modules.ssd_minimal.ssd_minimal_discrete in torch float64, per padded document",
        cross_check="Dew XLA float64; assert_computes_the_oracle with its traced chain_roundings")
    destination = FIXTURE.with_name("ssd_mamba_ssm_seeded.npz")
    written = save(destination, stored, reference=f"mamba_ssm {version}", license="Apache-2.0")
    print(f"wrote {destination.name}: {written} ({written.stat().st_size:,} bytes)", flush=True)


def main() -> None:
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    import torch

    sys.path.insert(0, sys.argv[1])
    from mamba_ssm import __version__
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    if sys.argv[2:] == ["--seeded"]:
        from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete
        from reference_source import git_head

        write_seeded(mamba_chunk_scan_combined, ssd_minimal_discrete, __version__, git_head(sys.argv[1]))
        return

    rng = np.random.default_rng(SEED)
    stored = {}
    for name, (*shape, chunk, dtype, documents) in CASES.items():
        dtype = getattr(torch, dtype)
        arrays = operands(rng, *shape,
                           lambda value, dtype=dtype: torch.tensor(value, dtype=dtype).float().numpy(),
                           documents)
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
