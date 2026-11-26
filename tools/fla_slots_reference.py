#!/usr/bin/env python3
"""The fixture the slot compositions are held to (`dew.nn.kernels.slots`, row 6127): fla-core's own
`chunk_lightnet` and `chunk_gsa` (ABC's its `g=None` form) on a CUDA GPU, forward and every gradient, at
each site tests/slots_sites.py names, over the tests' own seeded draws.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_slots_reference.py <fla> <unit> <site>...
    PYTHONPATH=src:tests:tools python tools/fla_slots_reference.py --merge <unit>...

on a GPU, Dew's environment recording and a second Python with torch and fla-core (`--serve`, the "fla-0.6"
cohort, tools/reference_releases.py) running fla and its references, each run a unit of named sites, the
units then merged into tests/fixtures/fla/slots_seeded.npz (`slots_sites.RECORDED`), a summary recording
(`record_case(..., summary=True)`): per site, seed and leaf, fla's distance from float64 at the site's
inputs, which the test draws again (`Site.arrays`), and the truth's sketch.

fla runs at its defaults (scale `K^-1/2`, chunks of 64), its operands in the site's dtype, the slots' log
decays and the memories fp32. The oracles are fla's own naive recurrences in torch float64
(`_dew_torch.precision.float64`) on the GPU: `naive_recurrent_gsa`, ABC's with no gates, and for LightNet
`naive_recurrent_gla` of its keys' additive-gate weights and log decays (a log-cumulative sum of the keys,
fla's `fused_additive_gate`), each held at every seed to Dew's float64 token recurrences
(tests/test_slots.py).
"""

from typing import Any

import numpy as np

COHORT = "fla-0.6"
"""The reference releases this tool runs against (tools/reference_releases.py)."""

OPERANDS = {"lightnet": ("q", "k", "v"), "gsa": ("q", "k", "v", "s", "g"), "abc": ("q", "k", "v", "s")}
"""Each form's differentiated operands beside its entering memories."""


def _gated(k):
    """fla's additive gate in torch (`fused_additive_gate`'s math): the keys' weights `exp(k - z)` and log
    decays `z_{t-1} - z_t`, zero at the first token, `z` the keys' log-cumulative sum over time (axis 1)."""
    import torch

    z = k.logcumsumexp(1)
    return torch.exp(k - z), torch.cat((z[:, :1], z[:, :-1]), 1) - z


def run(arrays, *, form: str, dtype, device, wide: bool) -> dict[str, np.ndarray]:
    """fla's chunked form (or its float64 reference where `wide`) on a site's arrays, `[B, T, H, ...]` as fla
    takes them: the operands in `dtype`, the decays and memories fp32 (all float64 where `wide`); its leaves
    (`slots_sites.LEAVES`) under the drawn cotangents."""
    import torch
    from fla.ops.gla.naive import naive_recurrent_gla
    from fla.ops.gsa import chunk_gsa
    from fla.ops.gsa.naive import naive_recurrent_gsa
    from fla.ops.lightnet import chunk_lightnet

    # fla's chunked forms, decorated past what a type checker reads of their calls.
    chunked: dict[str, Any] = {"lightnet": chunk_lightnet, "gsa": chunk_gsa}
    narrow = torch.float64 if wide else torch.float32
    tensors = {name: torch.tensor(np.asarray(value, np.float64 if wide else np.float32), device=device)
               for name, value in arrays.items()}
    inputs = {name: tensors[name].to(narrow if name == "g" else dtype).requires_grad_()
              for name in OPERANDS[form]}
    memories = [tensors[name].to(narrow).requires_grad_() for name in
                (("h0",) if form == "lightnet" else ("hk0", "hv0"))]
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    if form == "lightnet":
        if wide:
            weights, log_decay = _gated(k)
            out, final = naive_recurrent_gla(q, weights, v, log_decay, initial_state=memories[0],
                                             output_final_state=True)
        else:
            out, (final, _) = chunked["lightnet"](q=q, k=k, v=v, initial_state=(memories[0], None),
                                                  output_final_state=True)
        assert final is not None
        outputs, cotangents = (out, final), (tensors["dy"].to(out.dtype), tensors["dfinal"].to(final.dtype))
        names = ("out", "final")
    else:
        s, g = inputs["s"], inputs.get("g")
        entering = (memories[0], memories[1])
        # ABC is GSA with no gates of its own (`g=None`): its slots' weights and decays fla's additive gate's,
        # differentiable (`fused_additive_gate`). fla's chunk_abc takes and returns its memories in tuples
        # its autograd function does not differentiate, so no gradient of theirs could be recorded there.
        if wide:
            out, finals = naive_recurrent_gsa(q, k, v, s, g, initial_state=entering, output_final_state=True)
        else:
            out, finals = chunked["gsa"](q=q, k=k, v=v, s=s, g=g, initial_state=entering,
                                         output_final_state=True)
        assert finals is not None
        hk, hv = finals
        outputs = (out, hk, hv)
        cotangents = (tensors["dy"].to(out.dtype), tensors["dhk"].to(hk.dtype), tensors["dhv"].to(hv.dtype))
        names = ("out", "hk", "hv")
    grads = torch.autograd.grad(outputs, (*inputs.values(), *memories), cotangents)
    leaves = {name: value.detach().to(torch.float64).cpu().numpy()
              for name, value in zip(names, outputs, strict=True)}
    gradients = [f"d{name}" for name in inputs] + (["dh0"] if form == "lightnet" else ["dhk0", "dhv0"])
    leaves.update({name: grad.detach().to(torch.float64).cpu().numpy()
                   for name, grad in zip(gradients, grads, strict=True)})
    return leaves


def serve() -> None:
    """The torch side: fla's chunked forms and their float64 references on each request
    (`fla_recording.serve`), its `form` and `dtype` the request's."""
    import fla_recording

    fla_recording.serve(COHORT, run, ("form", "dtype"))


def write_seeded(worker, names: list[str], out) -> None:
    """A unit of the recording at `names` (`fla_recording.write_seeded`), Dew's float64 token recurrences
    (`slots_sites.by_token`) held to fla's references at every seed."""
    import functools

    import fla_recording
    from slots_sites import by_token, ruled, sites

    fla_recording.write_seeded(
        worker, names, out, sites=sites(),
        options=lambda site, sample: {"form": site.form, "dtype": str(sample["q"].dtype)},
        exact=lambda site: functools.partial(ruled, by_token(site.form), site.form),
        command="PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_slots_reference.py "
                "<fla-0.6 python> <unit> <site>...",
        truth="fla's naive_recurrent_gsa (ABC's with no gates) and (LightNet's) naive_recurrent_gla in "
              "torch float64")


if __name__ == "__main__":
    import fla_recording

    fla_recording.main(__file__, "slots_sites", "slots_seeded.npz", serve=serve, write_seeded=write_seeded)
