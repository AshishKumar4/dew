"""EDM's and EDM2's training losses, for tests/fixtures/edm_loss/seeds.npz.

The references are NVlabs' own loss classes and preconditioners, read at the
pinned commits the sampler and network fixtures use and executed as
published: `EDMLoss` (training/loss.py) through `EDMPrecond`
(training/networks.py) of NVlabs/edm, and `EDM2Loss` (training/training_loop.py)
through `Precond` (training/networks_edm2.py) of NVlabs/edm2, logvar head
included. Their `torch_utils` imports are stubbed as in
tools/edm2_reference.py; their code is not vendored here (EDM is CC BY-NC-SA
4.0, as EDM2 is).

Three things are fixed so the losses can be replayed: the network the
preconditioners wrap is an affine stand-in, F(x, c_noise) = a x + b c_noise
+ c with a per channel, set as EDMPrecond's `model_type` and as Precond's
`unet`; the loss's two `torch.randn` draws (the standard normal behind each
training sigma, and the noise before it is scaled by sigma) return stored
draws; and there are no labels or augmentation. What lands, over the seeds
check 1 draws (`record_case`, tools/recorded_references.py): each loss's
per-element loss and the gradient of its mean in the
network's output and in the logvar head's weight, computed as published in
float32 and by its float64 twin (the float32 casts retargeted), whose
distance from the first is the float32 run's rounding, summarized (each
leaf's error from its truth and its truth's sketch, `record_case`'s
summary). A seed's inputs, the
stand-in and logvar-head weights among them, are a seeded NumPy draw
(`draw`) the test makes again.

    python tools/edm_loss_reference.py
"""

from __future__ import annotations

import contextlib
import functools
import types
from pathlib import Path

import numpy as np
import torch
from reference_source import fetched, ran, selected, stub_torch_utils
from torch_precision import replaced

EDM = "008a4e5316c8e3bfe61a62f874bddba254295afb"
EDM2 = "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm_loss" / "seeds.npz"
IMAGES, CHANNELS, SIZE, LOGVAR_CHANNELS = 1024, 1, 2, 16
"""A draw's images: 1024 of one channel 2 by 2, so each draw's error averages that many lognormal training
sigmas while its leaves stay small enough for a recording's sizing to hold its draws. At 6 images a draw
EDM's paired sizing took all 512 seeds and EDM2's logvar weight gradient did not resolve within them; at
64 and 256, EDM's sigmas did not; at 1024, 3 channels 4 by 4, EDM's resolved at 128 and EDM2's at 32
(armada, 10-28), and the recording's draws past 12 GiB."""
TINY_UNET = {"model_channels": 8, "channel_mult": [1], "num_blocks": 1, "attn_resolutions": []}
"""Precond builds a U-Net it is then given the stand-in in place of; a tiny one."""


def published(repo: str, commit: str, path: str, dtype: str,
              keep: set[str] | None = None) -> types.ModuleType:
    """The published module computing in `dtype`; with `keep`, only those
    top-level classes and functions of it (a training loop's loss class
    without the loop's own imports)."""
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    source = fetched(url).replace("torch.float32", f"torch.{dtype}")
    module = types.ModuleType(Path(path).stem)
    if keep is None:
        exec(source, module.__dict__)
        return module
    kept = selected(source, keep, filename=url)
    for node in kept:
        node.decorator_list = []
    module.__dict__.update(torch=torch, np=np)
    ran(kept, module.__dict__, url)
    return module


class Affine(torch.nn.Module):
    """The stand-in network: a per-channel scale of its input plus a scale of
    the noise label and a bias."""

    def __init__(self, dtype, **_):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(CHANNELS, dtype=dtype))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.c = torch.nn.Parameter(torch.zeros((), dtype=dtype))

    def forward(self, x, noise_labels, class_labels=None, **_):
        return self.a.reshape(1, -1, 1, 1) * x + self.b * noise_labels.reshape(-1, 1, 1, 1) + self.c


@contextlib.contextmanager
def replayed(normals: torch.Tensor, noise: torch.Tensor):
    """`torch.randn` returns the sigma draw and `torch.randn_like` the noise."""
    with replaced((torch, "randn", lambda *_, **__: normals.reshape(-1, 1, 1, 1)),
                  (torch, "randn_like", lambda *_, **__: noise)):
        yield


def run(loss, net, network, arrays: dict, dtype) -> dict[str, np.ndarray]:
    """One loss evaluation: its per-element values,
    and the gradient of their mean in the stand-in `network`'s output and in
    the logvar head's weight. The stand-in's own weights are a linear map of
    the first gradient, so theirs adds nothing."""
    images, normals, noise = (torch.tensor(arrays[name], dtype=dtype)
                              for name in ("images", "normals", "noise"))
    outputs = []

    def keep(_, __, output):
        output.retain_grad()
        outputs.append(output)

    hooks = (network.register_forward_hook(keep),)
    with replayed(normals, noise):
        value = loss(net=net, images=images)
    for hook in hooks:
        hook.remove()
    value.mean().backward()
    out = {"loss": value.detach().numpy(),
           "grad/network_output": outputs[0].grad.numpy()}
    head = getattr(net, "logvar_linear", None)
    if head is not None:
        out["grad/logvar_weight"] = head.weight.grad.numpy()
    return out


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's inputs: images, the standard normals behind the training sigmas, the noise, the
    stand-in's weights, and the logvar head's Fourier features and weight as MPFourier and MPConv
    initialize them (2 pi N(0, 1) frequencies, 2 pi U(0, 1) phases, an N(0, 1) weight)."""
    generator = np.random.default_rng([0xED, seed])
    return {
        "images": generator.uniform(-1, 1, (IMAGES, CHANNELS, SIZE, SIZE)).astype(np.float32),
        "normals": generator.standard_normal(IMAGES).astype(np.float32),
        "noise": generator.standard_normal((IMAGES, CHANNELS, SIZE, SIZE)).astype(np.float32),
        "a": generator.normal(0.5, 0.2, CHANNELS).astype(np.float32),
        "b": np.float32(generator.normal(0.3, 0.1)), "c": np.float32(generator.normal(-0.1, 0.1)),
        "logvar/freqs": (2 * np.pi * generator.standard_normal(LOGVAR_CHANNELS)).astype(np.float32),
        "logvar/phases": (2 * np.pi * generator.uniform(0, 1, LOGVAR_CHANNELS)).astype(np.float32),
        "logvar/weight": generator.standard_normal((1, LOGVAR_CHANNELS)).astype(np.float32),
    }


def main() -> None:
    import jax
    from recorded_references import provenance, record_case, save
    from test_edm_loss import edm2_loss, edm_loss

    stub_torch_utils()
    torch.manual_seed(5)
    built = {}
    for dtype in (torch.float32, torch.float64):
        name = str(dtype).removeprefix("torch.")
        networks = published("NVlabs/edm", EDM, "training/networks.py", name)
        networks.Affine = functools.partial(Affine, dtype)
        edm2 = published("NVlabs/edm2", EDM2, "training/networks_edm2.py", name)
        built[dtype] = (networks, published("NVlabs/edm", EDM, "training/loss.py", name, keep={"EDMLoss"}),
                        edm2, published("NVlabs/edm2", EDM2, "training/training_loop.py", name,
                                        keep={"EDM2Loss"}))

    # Each precision's two networks, built once (a Precond builds its whole UNet before the stand-in replaces
    # it) and given each draw's parameters in place.
    models = {}
    for dtype in (torch.float32, torch.float64):
        networks, _, edm2, _ = built[dtype]
        precond = edm2.Precond(SIZE, CHANNELS, 0, use_fp16=False, logvar_channels=LOGVAR_CHANNELS,
                               **TINY_UNET).eval()
        precond.unet = Affine(dtype)
        models[dtype] = (networks.EDMPrecond(SIZE, CHANNELS, model_type="Affine"), precond.to(dtype))

    def evaluated(inputs, which):
        results = []
        for dtype in (torch.float32, torch.float64):
            _, losses, _, loop = built[dtype]
            net, precond = models[dtype]
            for model in (net, precond):
                model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for model in (net.model, precond.unet):
                    for parameter in ("a", "b", "c"):
                        getattr(model, parameter).copy_(torch.tensor(inputs[parameter], dtype=dtype))
                precond.logvar_fourier.freqs.copy_(torch.tensor(inputs["logvar/freqs"], dtype=dtype))
                precond.logvar_fourier.phases.copy_(torch.tensor(inputs["logvar/phases"], dtype=dtype))
                precond.logvar_linear.weight.copy_(torch.tensor(inputs["logvar/weight"], dtype=dtype))
            if which == "edm":
                results.append(run(losses.EDMLoss(), net, net.model, inputs, dtype))
            else:
                results.append(run(loop.EDM2Loss(), precond, precond.unet, inputs, dtype))
        return tuple(results)

    arrays = {}
    for which, dew in (("edm", jax.jit(edm_loss)), ("edm2", jax.jit(edm2_loss))):
        # Sized paired, as check 1 judges: the training sigma's lognormal spread makes each arm's
        # error heavy-tailed across draws alike.
        arrays.update(record_case(which, draw, lambda inputs, which=which: evaluated(inputs, which),
                                  store_inputs=False, summary=True, dew=dew))
    written = save(FIXTURE, {**arrays, "provenance": provenance(
        "NVlabs/edm, NVlabs/edm2", f"{EDM}, {EDM2}", "cpu", "python tools/edm_loss_reference.py",
        torch=torch.__version__)}, reference="NVlabs/edm, NVlabs/edm2",
        license="CC BY-NC-SA 4.0 (results only)")
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
