"""EDM's and EDM2's training losses, for tests/fixtures/edm/loss.npz.

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
draws; and there are no labels or augmentation. What lands: the inputs, the
stand-in and logvar-head weights, and each loss's training sigmas,
per-element loss and the gradient of its mean in the network's output and
in the logvar head's weight, computed as published in float32 and by its
float64 twin (the float32 casts retargeted), whose distance from the first
is the float32 run's rounding.

    python tools/edm_loss_reference.py
"""

from __future__ import annotations

import ast
import contextlib
import functools
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch

EDM = "008a4e5316c8e3bfe61a62f874bddba254295afb"
EDM2 = "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm" / "loss.npz"
IMAGES, CHANNELS, SIZE, LOGVAR_CHANNELS = 6, 3, 4, 16
TINY_UNET = {"model_channels": 8, "channel_mult": [1], "num_blocks": 1, "attn_resolutions": []}
"""Precond builds a U-Net it is then given the stand-in in place of; a tiny one."""


def stub_torch_utils() -> None:
    persistence = types.ModuleType("torch_utils.persistence")
    persistence.persistent_class = lambda cls: cls
    misc = types.ModuleType("torch_utils.misc")
    misc.const_like = lambda ref, value: torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    package = types.ModuleType("torch_utils")
    package.persistence, package.misc = persistence, misc
    sys.modules.update({"torch_utils": package, "torch_utils.persistence": persistence,
                        "torch_utils.misc": misc})


def published(repo: str, commit: str, path: str, dtype: str,
              keep: set[str] | None = None) -> types.ModuleType:
    """The published module computing in `dtype`; with `keep`, only those
    top-level classes and functions of it (a training loop's loss class
    without the loop's own imports)."""
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    source = urllib.request.urlopen(url).read().decode()
    module = types.ModuleType(Path(path).stem)
    if keep is not None:
        tree = ast.parse(source)
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in keep]
        for node in tree.body:
            node.decorator_list = []
        source = ast.unparse(tree)
        module.__dict__.update(torch=torch, np=np)
    exec(source.replace("torch.float32", f"torch.{dtype}"), module.__dict__)
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
    randn, randn_like = torch.randn, torch.randn_like
    torch.randn = lambda *_, **__: normals.reshape(-1, 1, 1, 1)
    torch.randn_like = lambda *_, **__: noise
    try:
        yield
    finally:
        torch.randn, torch.randn_like = randn, randn_like


def run(loss, net, network, arrays: dict, dtype) -> dict[str, np.ndarray]:
    """One loss evaluation: the sigmas it hands `net`, its per-element values,
    and the gradient of their mean in the stand-in `network`'s output and in
    the logvar head's weight. The stand-in's own weights are a linear map of
    the first gradient, so theirs adds nothing."""
    images, normals, noise = (torch.tensor(arrays[name], dtype=dtype)
                              for name in ("images", "normals", "noise"))
    sigmas, outputs = [], []

    def keep(_, __, output):
        output.retain_grad()
        outputs.append(output)

    hooks = (net.register_forward_pre_hook(lambda _, args: sigmas.append(args[1].detach().clone())),
             network.register_forward_hook(keep))
    with replayed(normals, noise):
        value = loss(net=net, images=images)
    for hook in hooks:
        hook.remove()
    value.mean().backward()
    out = {"sigma": sigmas[0].reshape(-1).numpy(), "loss": value.detach().numpy(),
           "grad/network_output": outputs[0].grad.numpy()}
    head = getattr(net, "logvar_linear", None)
    if head is not None:
        out["grad/logvar_weight"] = head.weight.grad.numpy()
    return out


def main() -> None:
    stub_torch_utils()
    generator = np.random.default_rng(3)
    arrays = {
        "images": generator.uniform(-1, 1, (IMAGES, CHANNELS, SIZE, SIZE)).astype(np.float32),
        "normals": generator.standard_normal(IMAGES).astype(np.float32),
        "noise": generator.standard_normal((IMAGES, CHANNELS, SIZE, SIZE)).astype(np.float32),
        "a": generator.normal(0.5, 0.2, CHANNELS).astype(np.float32),
        "b": np.asarray(0.3, np.float32), "c": np.asarray(-0.1, np.float32),
    }
    torch.manual_seed(5)
    edm2_32 = published("NVlabs/edm2", EDM2, "training/networks_edm2.py", "float32")
    head = edm2_32.Precond(SIZE, CHANNELS, 0, use_fp16=False, logvar_channels=LOGVAR_CHANNELS, **TINY_UNET)
    arrays.update({"logvar/freqs": head.logvar_fourier.freqs.numpy(),
                   "logvar/phases": head.logvar_fourier.phases.numpy(),
                   "logvar/weight": head.logvar_linear.weight.detach().numpy()})
    for dtype, tail in ((torch.float32, ""), (torch.float64, "_f64")):
        name = str(dtype).removeprefix("torch.")
        networks = published("NVlabs/edm", EDM, "training/networks.py", name)
        networks.Affine = functools.partial(Affine, dtype)
        net = networks.EDMPrecond(SIZE, CHANNELS, model_type="Affine")
        losses = published("NVlabs/edm", EDM, "training/loss.py", name, keep={"EDMLoss"})
        edm2 = published("NVlabs/edm2", EDM2, "training/networks_edm2.py", name)
        precond = edm2.Precond(SIZE, CHANNELS, 0, use_fp16=False, logvar_channels=LOGVAR_CHANNELS,
                               **TINY_UNET).eval()
        precond.unet = Affine(dtype)
        loop = published("NVlabs/edm2", EDM2, "training/training_loop.py", name, keep={"EDM2Loss"})
        with torch.no_grad():
            for model in (net.model, precond.unet):
                for parameter in ("a", "b", "c"):
                    getattr(model, parameter).copy_(torch.tensor(arrays[parameter], dtype=dtype))
            precond.logvar_fourier.freqs.copy_(torch.tensor(arrays["logvar/freqs"], dtype=dtype))
            precond.logvar_fourier.phases.copy_(torch.tensor(arrays["logvar/phases"], dtype=dtype))
            precond.logvar_linear.weight.copy_(torch.tensor(arrays["logvar/weight"], dtype=dtype))
        precond.to(dtype)
        for key, value in run(losses.EDMLoss(), net, net.model, arrays, dtype).items():
            arrays[f"edm/{key}{tail}"] = value
        for key, value in run(loop.EDM2Loss(), precond, precond.unet, arrays, dtype).items():
            arrays[f"edm2/{key}{tail}"] = value
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
