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

A seed's inputs are a seeded NumPy draw (`draw`) the test makes again: the
pixels; the step's draws, each row's student time, its noise, its level's
normal and its renoising noise, which the test hands to
`AdversarialDistillationObjective.loss` by name (`Step.tape`), so both runs
step on the same draws; the student and teacher stand-ins' weights, the
networks one token per sample; and each head's weights and `u`s in the
layout of Dew's `Head`. Both runs, float32 and float64, take their own
precision wherever the published code names one (DiT's embedding computes
in float32). What lands, over the seeds check 1 draws (`record_case`,
tools/recorded_references.py): the loss, the student's gradient, each head's
and the kept `u`s, each tree's leaves in one vector in their paths' order
(`judged`), in float32 and float64, kept whole: Dew's objective computes its
heads, their condition and their features in float32 whatever its inputs'
precision, so its own float64 path is no truth for a summary to check.

    PYTHONPATH=src:tests python tools/ladd_reference.py
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import jax
import numpy as np
import torch
from reference_source import definitions
from torch.nn.utils.spectral_norm import SpectralNorm
from torch_precision import float_as, replaced

STYLEGAN_T = ("https://raw.githubusercontent.com/autonomousvision/stylegan-t/"
              "36ab80ce76237fefe03e65e9b3161c040ae888e3/")
DIT = "https://raw.githubusercontent.com/facebookresearch/DiT/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ladd" / "seeds.npz"
SETTINGS = {"batch": 64, "side": 4, "channels": 3, "width": 6, "cmap_dim": 8, "time_features": 16,
            "student_times": [1.0, 0.75, 0.5, 0.25], "renoise_times": [1.0, 1.0], "distillation_weight": 2.5,
            "r1_weight": 0.5, "layers": ["layer_a", "layer_b"]}
"""R1's weight is far above ADD's 1e-5, so its term moves the loss. 64 rows a draw, eight of the local batch
norms' groups of eight, as the loss, one number a draw, at 4 rows did not resolve paired within 512 seeds."""


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


def us(head) -> dict:
    """A `DiscHead`'s spectral `u`s in the layout of Dew's `Head`."""
    return {"block_0": {"conv": {"u": head.main[0][0].weight_u.double().numpy()}},
            "block_1": {"conv": {"u": head.main[1].fn[0].weight_u.double().numpy()}},
            "cls": {"u": head.cls.weight_u.double().numpy()}}


def loaded(head, params: dict, initial: dict, dtype) -> None:
    """`head` given weights and `u`s in the layout of Dew's `Head`, each `v` the one its `u` gives, which
    the training-mode power iteration computes again before it reads it."""
    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=dtype)

    def conv(module, weights, u):
        module.weight_orig.copy_(tensor(weights["kernel"][0].transpose(2, 1, 0)))
        module.bias.copy_(tensor(weights["bias"]))
        module.weight_u.copy_(tensor(u["u"]))
        module.weight_v.copy_(torch.nn.functional.normalize(module.weight_orig.flatten(1).T @ module.weight_u,
                                                            dim=0))

    def norm(module, weights):
        module.weight.copy_(tensor(weights["weight"]))
        module.bias.copy_(tensor(weights["bias"]))

    with torch.no_grad():
        conv(head.main[0][0], params["block_0"]["conv"], initial["block_0"]["conv"])
        norm(head.main[0][1], params["block_0"]["norm"])
        conv(head.main[1].fn[0], params["block_1"]["conv"], initial["block_1"]["conv"])
        norm(head.main[1].fn[1], params["block_1"]["norm"])
        conv(head.cls, params["cls"], initial["cls"])
        head.cmapper.weight.copy_(tensor(params["cmapper_weight"]).T)
        head.cmapper.bias.copy_(tensor(params["cmapper_bias"]))


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


def nested(arrays: dict[str, np.ndarray], prefix: str) -> dict:
    """`flatten`'s inverse: the arrays under `prefix/` as a tree."""
    tree: dict = {}
    for key, value in arrays.items():
        if key.startswith(prefix + "/"):
            *path, leaf = key[len(prefix) + 1:].split("/")
            node = tree
            for name in path:
                node = node.setdefault(name, {})
            node[leaf] = value
    return tree


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


def step(dtype, inputs: dict[str, np.ndarray], scope: dict) -> dict[str, np.ndarray]:
    """One LADD step in `dtype` on a seed's inputs (`draw`): the loss, every gradient and the kept `u`s."""
    settings = SETTINGS
    narrowed = types.SimpleNamespace(**{**vars(torch), "float32": dtype})
    embedding = definitions(DIT, ("timestep_embedding",), {"torch": narrowed, "math": math},
                            within="TimestepEmbedder")["timestep_embedding"]
    heads = []
    for index in range(len(settings["layers"])):
        head = made(scope, dtype)
        loaded(head, nested(inputs, f"heads/head_{index}"), nested(inputs, f"initial/head_{index}"), dtype)
        heads.append(head)
    net, teacher_net = (Network({name: torch.as_tensor(value, dtype=dtype)
                                 for name, value in nested(inputs, side).items()})
                        for side in ("student", "teacher"))
    teacher_net.requires_grad_(requires_grad=False)
    x = torch.as_tensor(inputs["pixels"], dtype=dtype) / 127.5 - 1
    chosen = np.asarray(settings["student_times"])[inputs["draws/indices"]]
    t = torch.as_tensor(chosen, dtype=dtype).reshape(-1, 1, 1, 1)
    noise, renoise = (torch.as_tensor(inputs[f"draws/{name}"], dtype=dtype) for name in ("noise", "renoise"))
    mean, std = settings["renoise_times"]
    level = torch.sigmoid(mean + std * torch.as_tensor(inputs["draws/level"], dtype=dtype))
    s = level.reshape(-1, 1, 1, 1)

    with replaced(float_as(dtype)):
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

        def logits(features):
            return torch.cat([head(f, condition).reshape(f.shape[0], -1)
                              for head, f in zip(heads, features, strict=True)], -1)

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
        out.update(flatten(us(head), f"spectral/head_{index}"))
    return out


def judged(out: dict, concatenate=np.concatenate) -> dict:
    """What check 1 judges of a step's arrays by their paths (`step`'s, or Dew's under the same paths): the
    loss, and the student's gradient, each head's and the kept `u`s, each tree's leaves in one vector in
    their paths' order. A few entries' root mean square is no estimate of rounding, and the two biases a
    batch norm follows have a gradient of exactly zero, rounding alone in both runs."""
    def vector(prefix):
        return concatenate([out[key].reshape(-1) for key in sorted(out) if key.startswith(prefix + "/")])

    return {"loss": out["loss"].reshape(1), "student": vector("grad/student"),
            **{f"head_{index}": vector(f"grad/heads/head_{index}")
               for index in range(len(SETTINGS["layers"]))},
            "spectral": vector("spectral")}


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's inputs: the pixels; the step's draws, each row's student time, its noise, its level's
    normal and its renoising noise; the teacher stand-in and the student a little off it; and each head's
    weights and `u`s as torch makes a `DiscHead`'s, its biases and its local batch norms' affine maps
    off their identity and its condition mapper's weight five times its own, so its logits are of order
    one or more and a hinge term can be dead, in the layout of Dew's `Head` (`heads/`, `initial/`).
    Every input a float32 value, so both runs start from the same numbers."""
    settings = SETTINGS
    b, side, channels, width = settings["batch"], settings["side"], settings["channels"], settings["width"]
    rng = np.random.default_rng([2104, seed])
    pixels = rng.integers(0, 256, (b, side, side, channels), dtype=np.uint8)
    flat = side * side * channels
    teacher = {"embed": rng.standard_normal((flat, width)) * 0.2, "time": rng.standard_normal(width),
               "layer_a": rng.standard_normal((width, width)) * 0.6,
               "layer_b": rng.standard_normal((width, width)) * 0.6,
               "head": rng.standard_normal((width, flat)) * 0.3}
    student = {name: value + rng.standard_normal(value.shape) * 0.05 for name, value in teacher.items()}

    def conv(channels_in, channels_out):
        # torch's Conv1d: weight and bias uniform within one over the root of the fan-in.
        bound = 1 / math.sqrt(channels_in)
        return {"kernel": rng.uniform(-bound, bound, (1, 1, channels_in, channels_out)),
                "bias": rng.uniform(-bound, bound, channels_out) + rng.standard_normal(channels_out) * 0.2}

    def norm():
        return {"weight": 1 + rng.standard_normal(width) * 0.2, "bias": rng.standard_normal(width) * 0.2}

    def unit(size):
        u = rng.standard_normal(size)
        return {"u": u / np.linalg.norm(u)}

    arrays = {"pixels": pixels, "draws/indices": rng.integers(0, len(settings["student_times"]), b),
              "draws/noise": rng.standard_normal(pixels.shape), "draws/level": rng.standard_normal(b),
              "draws/renoise": rng.standard_normal(pixels.shape), **flatten(teacher, "teacher"),
              **flatten(student, "student")}
    for index in range(len(settings["layers"])):
        heads = {"block_0": {"conv": conv(width, width), "norm": norm()},
                 "block_1": {"conv": conv(width, width), "norm": norm()},
                 "cls": conv(width, settings["cmap_dim"]),
                 "cmapper_weight": rng.standard_normal((settings["time_features"], settings["cmap_dim"])) * 5,
                 "cmapper_bias": rng.standard_normal(settings["cmap_dim"]) * 0.2}
        initial = {"block_0": {"conv": unit(width)}, "block_1": {"conv": unit(width)},
                   "cls": unit(settings["cmap_dim"])}
        arrays.update(flatten(heads, f"heads/head_{index}") | flatten(initial, f"initial/head_{index}"))
    kinds = {"pixels": np.uint8, "draws/indices": np.int32}
    return {name: value.astype(kinds.get(name, np.float32)) for name, value in arrays.items()}


def main() -> None:
    from recorded_references import provenance, record_case, save
    from test_adversarial import one_step

    scope = published()
    dew = jax.jit(lambda inputs: one_step(np.float32, inputs))
    arrays = record_case("ladd", draw, lambda inputs: tuple(judged(step(dtype, inputs, scope))
                                                            for dtype in (torch.float32, torch.float64)),
                         store_inputs=False, dew=dew)
    written = save(FIXTURE, {**arrays, "provenance": provenance(
        "autonomousvision/stylegan-t, facebookresearch/DiT", f"{STYLEGAN_T}, {DIT}", "cpu",
        "python tools/ladd_reference.py", torch=torch.__version__)},
        reference="autonomousvision/stylegan-t, facebookresearch/DiT", license="StyleGAN-T: NVIDIA's own "
        "terms; DiT: CC BY-NC 4.0; both fetched at record time, neither vendored")
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
