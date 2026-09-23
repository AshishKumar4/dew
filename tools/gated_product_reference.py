#!/usr/bin/env python3
"""Write transformers' bf16 gated product, forward and backward, for the
gated-product rounding test.

A transformers gated MLP computes `act_fn(gate_proj(x)) * up_proj(x)`. With
bf16 tensors (a bf16 model, or autocast's bf16 projections) torch computes
the activation in fp32 and rounds it to bf16, then rounds the product; its
backward rounds the gradient of each bf16 tensor. For each of Dew's three
named activations (silu for 'swiglu', the tanh gelu for 'geglu', the erf
gelu for 'geglu_exact') the fixture holds bf16 gate, up and cotangent
values, torch's product and torch's gradients of gate and up.

Every kept element has its two transcendental values, the activation and
the gate's gradient, further in float64 from a bf16 rounding midpoint than
64 fp32 roundings of the formula's own scale can move them: |gate| for the
activation, |d activation| (1 + |gate|) for the gradient. Any fp32
evaluation then rounds as float64 does, however it orders its terms and
whichever exp, tanh or erf it calls; torch's `0.5 x (1 + erf)` and XLA's
`x erfc / 2` disagree in the tail where one of them cancels, and such
elements are redrawn. The products (bf16 times bf16) are exact in fp32.

Run with Torch 2.14.0 CPU:
  python tools/gated_product_reference.py --out tests/fixtures/gated_product/bf16.npz
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

COUNT = 2048
SLACK = 64 * 2.0 ** -24
ROOT_2_OVER_PI = math.sqrt(2 / math.pi)
ERFC = np.frompyfunc(math.erfc, 1, 1)


def silu(g):
    s = 1 / (1 + np.exp(-g))
    return g * s, s * (1 + g * (1 - s))


def gelu_tanh(g):
    inner = ROOT_2_OVER_PI * (g + 0.044715 * g ** 3)
    t = np.tanh(inner)
    return 0.5 * g * (1 + t), 0.5 * (1 + t) + 0.5 * g * (1 - t * t) * ROOT_2_OVER_PI * (1 + 3 * 0.044715 * g * g)


def gelu_erf(g):
    cdf = 0.5 * ERFC(-g / math.sqrt(2)).astype(np.float64)
    return g * cdf, cdf + g * np.exp(-g * g / 2) / math.sqrt(2 * math.pi)


ACTIVATIONS = {"swiglu": (silu, F.silu), "geglu": (gelu_tanh, lambda x: F.gelu(x, approximate="tanh")),
               "geglu_exact": (gelu_erf, F.gelu)}


def bf16(values):
    return torch.from_numpy(np.asarray(values, np.float32)).bfloat16().float().numpy().astype(np.float64)


def decided(values, scale):
    """Whether each float64 value sits further from a bf16 rounding midpoint
    than SLACK times `scale`, the most an fp32 evaluation can move it."""
    magnitude = np.abs(values)
    exponent = np.floor(np.log2(np.where(magnitude > 0, magnitude, 1.0)))
    ulp = 2.0 ** (exponent - 7)
    position = magnitude / ulp
    distance = np.abs(position - np.floor(position) - 0.5) * ulp
    return (magnitude == 0) | (distance > SLACK * scale)


def draw(rng, oracle):
    """COUNT (gate, up, cotangent) bf16 triples whose roundings are decided."""
    kept = []
    while sum(len(k) for k in kept) < COUNT:
        gate, up, cotangent = (bf16(values) for values in (
            3 * rng.standard_normal(COUNT), rng.standard_normal(COUNT), rng.standard_normal(COUNT)))
        activated, slope = oracle(gate)
        d_activated = bf16(cotangent * up)
        keep = (decided(activated, np.abs(gate))
                & decided(slope * d_activated, np.abs(d_activated) * (1 + np.abs(gate))))
        kept.append(np.stack([gate, up, cotangent])[:, keep])
    return np.concatenate(kept, axis=1)[:, :COUNT]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if torch.__version__.split("+")[0] != "2.14.0":
        raise RuntimeError("This fixture is pinned to Torch 2.14.0")
    rng = np.random.default_rng(2112)
    arrays = {}
    for name, (oracle, activate) in ACTIVATIONS.items():
        gate, up, cotangent = (torch.from_numpy(values.astype(np.float32)).bfloat16()
                               for values in draw(rng, oracle))
        gate.requires_grad_()
        up.requires_grad_()
        output = activate(gate) * up
        output.backward(cotangent)
        for key, value in (("gate", gate), ("up", up), ("cotangent", cotangent), ("output", output),
                           ("d_gate", gate.grad), ("d_up", up.grad)):
            assert value.dtype == torch.bfloat16, (name, key, value.dtype)
            arrays[f"{name}/{key}"] = value.detach().float().numpy()
    np.savez_compressed(args.out, torch=torch.__version__, **arrays)


if __name__ == "__main__":
    main()
