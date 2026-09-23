#!/usr/bin/env python3
"""The first step's loss under each attention kernel, torch's and Dew's.

A step-0 loss difference between two frameworks mixes rounding points with
kernel choice. This holds the rounding points fixed: torch runs at Dew's
(`torch_lm.bf16_residual`: autocast, the residual stream and every RMSNorm
output in bf16, the rotary table in fp32) under each sdpa backend, and Dew
under each of its attention implementations, on the first batch of a
windows file. Every run's final hidden states are scored by one head,
torch's: fp32 logits of bf16 operands rounded to bf16, then fp32 cross
entropy. The table gives each run's loss minus the fp32 truth's and the
distance between every two runs' final hidden states.

The two sides run in their own venvs, as `torch_lm.py` and `dew_lm.py` do;
the torch side writes each backend's hidden states as bf16 bits to
`--hidden-dir`, which the Dew side reads.

    python tools/reference_runs/step0_attention.py torch --model <hf dir> \\
        --data windows.npz --batch 4 --hidden-dir hidden/
    PYTHONPATH=src python tools/reference_runs/step0_attention.py dew --model <hf dir> \\
        --data windows.npz --batch 4 --hidden-dir hidden/ --truth torch-fp32.json --out step0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from common import window_rows, write_record

BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH")
IMPLEMENTATIONS = ("cudnn", "xla")


def first_batch(args) -> np.ndarray:
    data = np.load(args.data)
    return data["windows"][window_rows(data["order"], 0, args.batch)].astype(np.int64)


def torch_side(args) -> None:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch_lm import bf16_residual
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="sdpa").cuda()
    bf16_residual(model)
    kept = {}
    model.model.norm.register_forward_hook(lambda module, inputs, output: kept.__setitem__("hidden", output))
    ids = torch.from_numpy(first_batch(args)).cuda()
    out = Path(args.hidden_dir)
    out.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), \
                    sdpa_kernel(getattr(SDPBackend, backend)):
                model(input_ids=ids[:, :-1])
        except RuntimeError as error:
            print(f"torch {backend}: {str(error).splitlines()[0][:100]}")
            continue
        bits = kept["hidden"].to(torch.bfloat16).cpu().view(torch.int16).numpy()
        np.save(out / f"torch-{backend.lower()}.npy", bits)
        print(f"wrote {out / f'torch-{backend.lower()}.npy'}")


def dew_side(args) -> None:
    import jax
    import jax.numpy as jnp

    from dew.interop import load_pretrained

    rows = first_batch(args)
    targets = jnp.asarray(rows[:, 1:].reshape(-1))
    hidden = {path.stem: np.load(path).view(jnp.bfloat16).astype(np.float32)
              for path in sorted(Path(args.hidden_dir).glob("torch-*.npy"))}
    table = None
    for implementation in IMPLEMENTATIONS:
        pretrained = load_pretrained(args.model, dtype="bfloat16", param_dtype="float32",
                                     attention_impl=implementation)
        model = pretrained.model

        def states(variables, tokens, model=model):
            return model.apply(variables, tokens, method=type(model).hidden_states)

        values = jax.jit(states)(pretrained.variables, jnp.asarray(rows[:, :-1]))
        hidden[f"dew-{implementation}"] = np.asarray(jnp.asarray(values, jnp.float32))
        if table is None:
            table = jnp.asarray(model.apply(pretrained.variables, pretrained.variables["params"],
                                            method=type(model).head_weight), jnp.bfloat16)

    width = next(iter(hidden.values())).shape[-1]
    # A tied head is the embedding table, [vocab, width]; a Dense kernel is [width, vocab].
    kernel = table.T if table.shape[-1] == width and table.shape[0] != width else table

    @jax.jit
    def loss(states):
        states = states.reshape(-1, width).astype(jnp.bfloat16)
        logits = jnp.dot(states, kernel, preferred_element_type=jnp.float32)
        logits = jax.lax.reduce_precision(logits, exponent_bits=8, mantissa_bits=7)
        return jnp.mean(jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, targets[:, None], -1)[:, 0])

    truth = json.loads(Path(args.truth).read_text())["loss"][0]
    names = list(hidden)
    record = {"truth_loss": truth, "model": args.model, "data": args.data, "batch": args.batch,
              "loss_minus_truth": {name: float(loss(jnp.asarray(hidden[name]))) - truth for name in names},
              "hidden_distance": {a: {b: float(np.linalg.norm(hidden[a] - hidden[b]) / np.linalg.norm(hidden[b]))
                                      for b in names} for a in names}}
    print(f"{'run':<26} {'loss - fp32':>12}" + "".join(f" {name[:14]:>15}" for name in names))
    for a in names:
        print(f"{a:<26} {record['loss_minus_truth'][a]:>+12.3e}"
              + "".join(f" {record['hidden_distance'][a][b]:>15.2e}" for b in names))
    write_record(args.out, record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=("torch", "dew"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--truth", help="dew side: the torch fp32 run's record, whose first loss is the truth")
    parser.add_argument("--out", help="dew side: where the table is written as JSON")
    args = parser.parse_args()
    if args.side == "dew" and not (args.truth and args.out):
        parser.error("the dew side takes --truth and --out")
    (torch_side if args.side == "torch" else dew_side)(args)


if __name__ == "__main__":
    main()
