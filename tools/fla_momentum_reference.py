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
    """The torch side: fla's `chunk_momentum_delta_rule` and its float64 reference
    `recurrent_momentum_delta_rule_ref` on each request (`fla_recording.serve`)."""
    import fla_recording
    from fla.ops.momentum_delta_rule import chunk_momentum_delta_rule, recurrent_momentum_delta_rule_ref

    def answered(arrays, *, dtype, device, wide: bool):
        rule = recurrent_momentum_delta_rule_ref if wide else chunk_momentum_delta_rule
        return run(arrays, rule=rule, dtype=dtype, device=device, wide=wide)

    fla_recording.serve(COHORT, answered, ("dtype",))


def write_seeded(worker, names: list[str], out) -> None:
    """A unit of the recording at `names` (`fla_recording.write_seeded`), Dew's float64 token recurrence
    (`momentum_sites.token_by_token`) held to fla's reference at every seed."""
    import functools

    import fla_recording
    from momentum_sites import ruled, sites, token_by_token

    fla_recording.write_seeded(
        worker, names, out, sites=sites(), options=lambda site, sample: {"dtype": str(sample["q"].dtype)},
        exact=lambda site: functools.partial(ruled, token_by_token, leaves=site.leaves),
        command="PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_momentum_reference.py "
                "<fla-0.6 python> <unit> <site>...",
        truth="fla's recurrent_momentum_delta_rule_ref in torch float64")


if __name__ == "__main__":
    import fla_recording

    fla_recording.main(__file__, "momentum_sites", "momentum_seeded.npz", serve=serve,
                       write_seeded=write_seeded)
