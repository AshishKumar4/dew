#!/usr/bin/env python3
"""The fixture a coupled momentum's chunked op is held to (row 1454, `recurrence.Hooks.coupled`): fla-core
0.6.0's own `chunk_momentum_delta_rule` on a CUDA GPU, forward and every gradient, at each site
tests/momentum_sites.py names, over the tests' own seeded draws.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_momentum_reference.py <fla> <unit> <site>...
    PYTHONPATH=src:tests:tools python tools/fla_momentum_reference.py --merge <unit>...

on a GPU, Dew's environment recording and a second Python with torch and fla-core 0.6.0 (`--serve`, the
"fla-0.6" cohort, tools/reference_releases.py) running fla and its reference, each run a unit of named sites,
the units then merged into tests/fixtures/fla/momentum_seeded.npz (`momentum_sites.RECORDED`), a summary
recording (`record_case(..., summary=True)`): per site, seed and leaf, fla's distance from float64 at the
site's inputs, which the test draws again (`Site.arrays`), and the truth's sketch.

fla runs at its defaults but for its keys and queries taken as given (`use_qk_l2norm_in_kernel=False`) and
its erase keys too (`use_p_times_alpha=False`), each the caller's in Dew, and its write keys' scale one
(`eta`), its sites' operands in bf16 (it refuses fp32), the log decays, the write gates and the entering pair
fp32. The oracle is fla's own reference, `recurrent_momentum_delta_rule_ref`, in torch float64
(`_dew_torch.precision.float64`) on the GPU, held at every seed to Dew's float64 token recurrence.
"""

import os
import sys
import time

import numpy as np

COHORT = "fla-0.6"
"""The reference releases this tool runs against (tools/reference_releases.py)."""

GRADIENTS = ("q", "k", "v", "p", "g", "mu", "beta")
"""The operands differentiated beside the entering pair."""


def run(arrays, *, rule, dtype, device, wide: bool) -> dict[str, np.ndarray]:
    """`rule` (fla's chunked rule, or its float64 reference where `wide`) on a site's arrays, `[B, T, H,
    ...]` as fla takes them: the queries, keys, values and erase keys in `dtype`, the rest fp32 (all float64
    where `wide`), the entering pair its memory and momentum apart; its output, final pair `[B, H, 2, Dk,
    Dv]` and each gradient under the drawn cotangents (`momentum_sites.LEAVES`)."""
    import torch

    narrow = torch.float64 if wide else torch.float32
    tensors = {name: torch.tensor(np.asarray(value, np.float64 if wide else np.float32), device=device)
               for name, value in arrays.items()}
    inputs = {name: tensors[name].to(dtype if name in ("q", "k", "v", "p") else narrow).requires_grad_()
              for name in GRADIENTS}
    entering = [tensors["h0"][:, :, part].to(narrow).requires_grad_() for part in range(2)]
    scale = inputs["q"].shape[-1] ** -0.5
    common = {"log_alpha": inputs["g"], "log_mu": inputs["mu"], "p": inputs["p"], "beta": inputs["beta"],
              "eta": torch.ones_like(inputs["beta"]), "scale": scale}
    if wide:
        out, final = rule(inputs["q"], inputs["k"], inputs["v"], initial_S=entering[0],
                          initial_M=entering[1], output_final_state=True, **common)
    else:
        out, final = rule(inputs["q"], inputs["k"], inputs["v"], initial_state=torch.stack(entering),
                          output_final_state=True, use_qk_l2norm_in_kernel=False, use_p_times_alpha=False,
                          **common)
    dfinal = tensors["dfinal"].to(final.dtype)
    grads = torch.autograd.grad((out, final[0], final[1]), (*(inputs[name] for name in GRADIENTS), *entering),
                                (tensors["dy"].to(out.dtype), dfinal[:, :, 0], dfinal[:, :, 1]))
    *operands, d_memory, d_momentum = (grad.detach().to(torch.float64).cpu().numpy() for grad in grads)
    found = {"out": out, "final": final.permute(1, 2, 0, 3, 4)}
    leaves = {name: value.detach().to(torch.float64).cpu().numpy() for name, value in found.items()}
    leaves.update({f"d{name}": grad for name, grad in zip(GRADIENTS, operands, strict=True)})
    leaves["dh0"] = np.stack([d_memory, d_momentum], axis=2)
    return leaves


def serve() -> None:
    """The torch side (tools/reference_worker.py's protocol): each request a site's inputs with its `dtype`,
    answered with fla's outputs and the reference's (`reference.<leaf>`, `truth.<leaf>`), after one `ready`
    line of fla's, torch's and Triton's releases and the device."""
    from importlib.metadata import version

    import reference_worker
    import torch
    from fla.ops.momentum_delta_rule import chunk_momentum_delta_rule, recurrent_momentum_delta_rule_ref
    from reference_releases import require

    from _dew_torch.precision import float64

    require(COHORT)
    torch_release = version('torch').split('+')[0]
    found = f"fla-core-{version('fla-core')}+torch-{torch_release}+triton-{version('triton')}"

    def answer(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dtype = getattr(torch, str(arrays.pop("dtype")))
        began = time.perf_counter()
        reference = run(arrays, rule=chunk_momentum_delta_rule, dtype=dtype, device="cuda", wide=False)
        referenced = time.perf_counter()
        with float64():
            truth = run(arrays, rule=recurrent_momentum_delta_rule_ref, dtype=torch.float64, device="cuda",
                        wide=True)
        print(f"timed fla {referenced - began:.3f} s truth {time.perf_counter() - referenced:.3f} s",
              file=sys.stderr, flush=True)
        return reference_worker.paired(reference, truth)

    reference_worker.serve(answer, f"{found} {torch.cuda.get_device_name()}")


def write_seeded(worker, names: list[str], out) -> None:
    """Each named site's seeded draws with fla's outputs and gradients and their float64 truth (`worker`,
    `serve`'s session), recorded as a summary (`record_case`) into `out`, Dew's float64 token recurrence held
    to fla's reference at every seed: a unit of the recording, which `reference_worker.merged_units` joins
    into the fixture."""
    import functools

    import jax
    from momentum_sites import ruled, sites, token_by_token
    from recorded_references import held_to_the_oracle, provenance, record_case
    from reference_error import chain_roundings, widened
    from reference_worker import arms

    version, device = worker.started.split(maxsplit=1)
    stored = {}
    for name in names:
        site, began = sites()[name], time.perf_counter()
        sample = site.arrays(0)
        dtype = str(sample["q"].dtype)

        def fla(arrays, dtype=dtype):
            request = {key: np.asarray(value, np.float32) for key, value in arrays.items()}
            return arms(worker.ask(request | {"dtype": np.asarray(dtype)}))

        exact_run = jax.jit(functools.partial(ruled, token_by_token, leaves=site.leaves))
        with jax.enable_x64(new_val=True):
            roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(sample)))

        def evaluate(arrays, fla=fla, exact_run=exact_run, roundings=roundings, name=name):
            reference, truth = fla(arrays)
            held_to_the_oracle(exact_run, arrays, truth, roundings, f"fla {name}")
            return reference, truth

        stored.update(record_case(name, site.arrays, evaluate, store_inputs=False, summary=True))
        print(f"{name}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        version, version, device.strip(),
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_momentum_reference.py "
        "<fla-0.6 python> <unit> <site>...",
        truth="fla's recurrent_momentum_delta_rule_ref in torch float64",
        cross_check="Dew's token recurrence in float64; assert_computes_the_oracle with its traced "
                    "chain_roundings")
    np.savez(out, allow_pickle=False, **stored)
    print(f"wrote {out}: {out.stat().st_size:,} bytes", flush=True)


def main() -> None:
    """The recorder's command line (`reference_worker.recording_command`), its sites `momentum_sites`'s."""
    from pathlib import Path

    from reference_worker import recording_command

    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fla" / "momentum_seeded.npz"
    recording_command(__file__, "momentum_sites", serve=serve, write_seeded=write_seeded, fixture=fixture,
                      license="MIT")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
