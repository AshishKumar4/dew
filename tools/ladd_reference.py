"""One LADD step, with ADD's R1 and distillation terms, as an oracle in torch,
for tests/fixtures/ladd/step.npz.

LADD (Sauer et al. 2024) and ADD (Sauer et al. 2023) publish no training
code, so the step is written from the papers' equations, with their
published parts run as published:

- the discriminator heads are StyleGAN-T's `DiscHead` (with `make_block`,
  `BatchNormLocal`, `SpectralConv1d`, `ResidualBlock` and
  `FullyConnectedLayer`, autonomousvision/stylegan-t at a pinned commit),
  in training mode, one per teacher feature layer, over sequences of one
  token. torch cannot pad a one-token sequence circularly by four, so the
  residual block is `make_block(channels, kernel_size=1)` where `DiscHead`
  builds 9, and Dew's heads take `kernel_size=(1, 1)`; the kernel of 9 is
  held by tests/test_adversarial.py's head test;
- the heads' condition is DiT's `TimestepEmbedder.timestep_embedding`
  (facebookresearch/DiT at a pinned commit) of the renoising level's model
  time;
- the equations: the student's clean prediction x_0 = x_t - t v at a drawn
  student time; renoising at a logit-normal level s, (1 - s) x + s eps;
  the teacher's token features of the renoised real sample, of the renoised
  student sample (gradient to the student) and of it stopped; the hinge
  losses relu(1 - D(real)) + relu(1 + D(held)) for the heads and -D(fake)
  for the student, each meaned over every head's logits; R1, gamma times
  the squared gradient of each head's summed mean logit at its real input;
  and ADD's distillation, lambda (1 - s) ||x_0 - sg(teacher's x_0 of the
  renoised sg(x_0))||^2 summed over the sample. Each row's sum is meaned
  over the batch.

The spectral norms follow Dew's documented cadence: the real pass runs one
power iteration and keeps its `u`; the held, fake and R1 passes run one
from that `u` and keep nothing. torch's own `spectral_norm` does each
iteration, its `u` restored after the passes that keep nothing.

The draws are the ones `AdversarialDistillationObjective.loss` makes from
`jax.random.key(KEY)`: of the split's five keys the third picks each row's
student time, the fourth draws the noise, and the fifth, split again, the
level's normal and the renoising noise. The student and teacher networks
are stand-ins with one token per sample. Both runs, float32 and float64,
take their own precision wherever the published code names one (DiT's
embedding computes in float32). What lands: the inputs, the draws, every
weight in Flax's layout and the `u`s, and the loss, every gradient and the
kept `u`s in float32 and float64.

    PYTHONPATH=src python tools/ladd_reference.py
"""

from __future__ import annotations

import ast
import json
import math
import types
import urllib.request
from pathlib import Path

import jax
import numpy as np
import torch
from torch.nn.utils.spectral_norm import SpectralNorm

STYLEGAN_T = ("https://raw.githubusercontent.com/autonomousvision/stylegan-t/"
              "36ab80ce76237fefe03e65e9b3161c040ae888e3/")
DIT = "https://raw.githubusercontent.com/facebookresearch/DiT/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ladd" / "step.npz"
SETTINGS = {"batch": 4, "side": 4, "channels": 3, "width": 6, "cmap_dim": 8, "time_features": 16,
            "student_times": [1.0, 0.75, 0.5, 0.25], "renoise_times": [1.0, 1.0], "distillation_weight": 2.5,
            "r1_weight": 0.5, "layers": ["layer_a", "layer_b"], "key": 5}
"""R1's weight is far above ADD's 1e-5, so its term moves the loss."""


def definitions(url: str, names: tuple[str, ...], scope: dict, *, within: str | None = None) -> dict:
    """The named top-level definitions (or methods of the class `within`) of
    the file at `url`, run as published."""
    text = urllib.request.urlopen(url).read().decode()
    body = ast.parse(text).body
    if within is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == within).body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            source = ast.get_source_segment(text, node)
            if within is not None:
                first, *rest = source.splitlines()
                source = "\n".join([first, *(line[4:] for line in rest)])
            exec(source, scope)
    return scope


class Precision:
    """`.float()` reads as the run's own precision, as `torch.float32` does
    in the scopes the published code runs in."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        self.kept = torch.Tensor.float
        torch.Tensor.float = lambda tensor, *args, **kwargs: tensor.to(self.dtype)
        return self

    def __exit__(self, *_):
        torch.Tensor.float = self.kept


class Network(torch.nn.Module):
    """The stand-in velocity network: the sample as one token plus the model
    time, two tanh layers whose tokens are the features, and a head back."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.weights = torch.nn.ParameterDict({name: torch.nn.Parameter(value.clone())
                                               for name, value in weights.items()})

    def forward(self, x, time):
        b, side, _, channels = x.shape
        tokens = (x.reshape(b, 1, -1) @ self.weights["embed"]
                  + (time / 1000).reshape(-1, 1, 1) * self.weights["time"])
        layer_a = torch.tanh(tokens @ self.weights["layer_a"])
        layer_b = torch.tanh(layer_a @ self.weights["layer_b"])
        return (layer_b @ self.weights["head"]).reshape(b, side, side, channels), [layer_a, layer_b]


def spectral(head) -> list[tuple[torch.nn.Module, str]]:
    return [(module, name) for name, module in head.named_modules() if isinstance(module, torch.nn.Conv1d)]


def kept(heads) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(module.weight_u.clone(), module.weight_v.clone())
            for head in heads for module, _ in spectral(head)]


def restore(heads, held) -> None:
    for (module, _), (u, v) in zip([pair for head in heads for pair in spectral(head)], held, strict=True):
        module.weight_u.copy_(u)
        module.weight_v.copy_(v)


def flax_head(head) -> tuple[dict, dict]:
    """A `DiscHead`'s weights and `u`s in the layout of Dew's `Head`."""
    def conv(module):
        return {"kernel": module.weight_orig.detach().double().numpy().transpose(2, 1, 0)[None],
                "bias": module.bias.detach().double().numpy()}

    def norm(module):
        return {"weight": module.weight.detach().double().numpy(),
                "bias": module.bias.detach().double().numpy()}

    params = {"block_0": {"conv": conv(head.main[0][0]), "norm": norm(head.main[0][1])},
              "block_1": {"conv": conv(head.main[1].fn[0]), "norm": norm(head.main[1].fn[1])},
              "cls": conv(head.cls), "cmapper_weight": head.cmapper.weight.detach().double().numpy().T,
              "cmapper_bias": head.cmapper.bias.detach().double().numpy()}
    us = {"block_0": {"conv": {"u": head.main[0][0].weight_u.double().numpy()}},
          "block_1": {"conv": {"u": head.main[1].fn[0].weight_u.double().numpy()}},
          "cls": {"u": head.cls.weight_u.double().numpy()}}
    return params, us


def flax_gradient(head) -> dict:
    def conv(module):
        return {"kernel": module.weight_orig.grad.double().numpy().transpose(2, 1, 0)[None],
                "bias": module.bias.grad.double().numpy()}

    def norm(module):
        return {"weight": module.weight.grad.double().numpy(), "bias": module.bias.grad.double().numpy()}

    return {"block_0": {"conv": conv(head.main[0][0]), "norm": norm(head.main[0][1])},
            "block_1": {"conv": conv(head.main[1].fn[0]), "norm": norm(head.main[1].fn[1])},
            "cls": conv(head.cls), "cmapper_weight": head.cmapper.weight.grad.double().numpy().T,
            "cmapper_bias": head.cmapper.bias.grad.double().numpy()}


def flatten(tree: dict, prefix: str) -> dict[str, np.ndarray]:
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}/{key}"))
        else:
            out[f"{prefix}/{key}"] = np.asarray(value)
    return out


def published() -> dict:
    scope = {"torch": torch, "nn": torch.nn, "np": np, "SpectralNorm": SpectralNorm, "F": torch.nn.functional,
             "misc": None, "Callable": object, "Any": object}
    definitions(STYLEGAN_T + "networks/shared.py", ("ResidualBlock", "FullyConnectedLayer"), scope)
    definitions(STYLEGAN_T + "networks/discriminator.py",
                ("SpectralConv1d", "BatchNormLocal", "make_block", "DiscHead"), scope)
    return scope


def made(scope: dict, dtype):
    """A `DiscHead` over one-token sequences in `dtype`, its residual block's
    kernel 1."""
    settings = SETTINGS
    head = scope["DiscHead"](settings["width"], settings["time_features"], settings["cmap_dim"])
    head.main[1] = scope["ResidualBlock"](scope["make_block"](settings["width"], kernel_size=1))
    return head.to(dtype).train()


def step(dtype, pixels, draws, student, teacher, heads_state) -> dict[str, np.ndarray]:
    settings = SETTINGS
    scope = published()
    narrowed = types.SimpleNamespace(**{**vars(torch), "float32": dtype})
    embedding = definitions(DIT, ("timestep_embedding",), {"torch": narrowed, "math": math},
                            within="TimestepEmbedder")["timestep_embedding"]
    heads = []
    for state in heads_state:
        head = made(scope, dtype)
        head.load_state_dict(state)
        heads.append(head)
    net = Network({name: torch.as_tensor(value, dtype=dtype) for name, value in student.items()})
    teacher_net = Network({name: torch.as_tensor(value, dtype=dtype) for name, value in teacher.items()})
    teacher_net.requires_grad_(requires_grad=False)
    x = torch.as_tensor(pixels, dtype=dtype) / 127.5 - 1
    chosen = np.asarray(settings["student_times"])[draws["indices"]]
    t = torch.as_tensor(chosen, dtype=dtype).reshape(-1, 1, 1, 1)
    noise, renoise = (torch.as_tensor(draws[name], dtype=dtype) for name in ("noise", "renoise"))
    mean, std = settings["renoise_times"]
    level = torch.sigmoid(mean + std * torch.as_tensor(draws["level"], dtype=dtype))
    s = level.reshape(-1, 1, 1, 1)

    with Precision(dtype):
        noisy = (1 - t) * x + t * noise
        velocity, _ = net(noisy, t.flatten() * 1000)
        clean = noisy - t * velocity

        def renoised(sample):
            return (1 - s) * sample + s * renoise

        def features(sample):
            return [f.permute(0, 2, 1) for f in teacher_net(renoised(sample), level * 1000)[1]]

        condition = embedding(level * 1000, settings["time_features"])
        before = kept(heads)
        real = [f.detach().requires_grad_() for f in features(x)]
        fake, held = features(clean), [f.detach() for f in features(clean.detach())]

        def scores(features):
            return [head(f, condition).reshape(f.shape[0], -1)
                    for head, f in zip(heads, features, strict=True)]

        def logits(features):
            return torch.cat(scores(features), -1)

        # Each head's hinge has live terms, real and held, and dead ones: with
        # every term live, a head's last bias enters real and held alike and
        # its gradient cancels to zero.
        for real_head, held_head in zip(scores(real), scores(held), strict=True):
            live = torch.cat([real_head < 1, held_head > -1], -1)
            assert bool((real_head < 1).any() and (held_head > -1).any() and not live.all()), \
                "a head's hinge terms must be partly live"
        restore(heads, before)
        real_scores = logits(real)
        after = kept(heads)
        held_scores = logits(held)
        restore(heads, after)
        frozen = [parameter for head in heads for parameter in head.parameters()]
        for parameter in frozen:
            parameter.requires_grad_(requires_grad=False)
        fake_scores = logits(fake)
        for parameter in frozen:
            parameter.requires_grad_(requires_grad=True)
        restore(heads, after)
        per_head = [head(f, condition).reshape(f.shape[0], -1).mean(-1).sum()
                    for head, f in zip(heads, real, strict=True)]
        restore(heads, after)
        gradients = torch.autograd.grad(sum(per_head), real, create_graph=True)
        r1 = sum(torch.square(g).reshape(g.shape[0], -1).sum(-1) for g in gradients)
        discriminator = (torch.relu(1 - real_scores).mean(-1) + torch.relu(1 + held_scores).mean(-1))
        generator = -fake_scores.mean(-1)
        target = renoised(clean.detach())
        teacher_velocity, _ = teacher_net(target, level * 1000)
        teacher_clean = target - s * teacher_velocity
        distillation = settings["distillation_weight"] * (1 - level) * torch.square(
            clean - teacher_clean.detach()).reshape(clean.shape[0], -1).sum(-1)
        total = (discriminator + generator + settings["r1_weight"] * r1 + distillation).mean()
        total.backward()

    out = {"loss": total.detach().double().numpy()}
    for name, parameter in net.weights.items():
        out[f"grad/student/{name}"] = parameter.grad.double().numpy()
    for index, head in enumerate(heads):
        out.update(flatten(flax_gradient(head), f"grad/heads/head_{index}"))
        out.update(flatten(flax_head(head)[1], f"spectral/head_{index}"))
    return out


def main() -> None:
    settings = SETTINGS
    b, side, channels, width = settings["batch"], settings["side"], settings["channels"], settings["width"]
    generator = np.random.default_rng(21)
    pixels = generator.integers(0, 256, (b, side, side, channels), dtype=np.uint8)
    flat = side * side * channels
    teacher = {"embed": generator.standard_normal((flat, width)) * 0.2,
               "time": generator.standard_normal(width),
               "layer_a": generator.standard_normal((width, width)) * 0.6,
               "layer_b": generator.standard_normal((width, width)) * 0.6,
               "head": generator.standard_normal((width, flat)) * 0.3}
    student = {name: value + generator.standard_normal(value.shape) * 0.05 for name, value in teacher.items()}
    # Every input a float32 value, so both runs start from the same numbers.
    teacher, student = ({name: value.astype(np.float32) for name, value in tree.items()}
                        for tree in (teacher, student))
    scope = published()
    torch.manual_seed(4)
    heads = [made(scope, torch.float64) for _ in settings["layers"]]
    with torch.no_grad():
        for head in heads:
            # Biases and the local batch norms' affine maps off their identity.
            for name, parameter in head.named_parameters():
                affine = "main" in name and name.endswith("weight") and parameter.ndim == 1
                if name.endswith("bias") or affine:
                    parameter.add_(torch.randn(parameter.shape, dtype=parameter.dtype) * 0.2)
            # Logits of order one or more, so a hinge term can be dead.
            head.cmapper.weight.mul_(5)
            for value in head.state_dict().values():
                value.copy_(value.float().double())
    keys = jax.random.split(jax.random.key(settings["key"]), 5)
    level_key, renoise_key = jax.random.split(keys[4])
    draws = {"indices": np.array(jax.random.randint(keys[2], (b,), 0, len(settings["student_times"]))),
             "noise": np.array(jax.random.normal(keys[3], pixels.shape)),
             "level": np.array(jax.random.normal(level_key, (b,))),
             "renoise": np.array(jax.random.normal(renoise_key, pixels.shape))}
    arrays = {"pixels": pixels, "settings": np.asarray(json.dumps(settings)),
              **{f"draws/{name}": value for name, value in draws.items()},
              **{f"teacher/{name}": value for name, value in teacher.items()},
              **{f"student/{name}": value for name, value in student.items()}}
    for index, head in enumerate(heads):
        params, us = flax_head(head)
        arrays.update(flatten(params, f"heads/head_{index}"))
        arrays.update(flatten(us, f"initial/head_{index}"))
    states = [{name: value.clone() for name, value in head.state_dict().items()} for head in heads]
    for dtype, suffix in ((torch.float64, "_f64"), (torch.float32, "")):
        typed = [{name: value.to(dtype) for name, value in state.items()} for state in states]
        out = step(dtype, pixels, draws, student, teacher, typed)
        arrays.update({f"{name}{suffix}": value for name, value in out.items()})
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: one LADD step over {b} rows, loss {float(arrays['loss_f64']):.6f}")


if __name__ == "__main__":
    main()
