#!/usr/bin/env python3
"""Write tests/fixtures/token_accuracy: TRL's own token accuracy, on the
logits of a fixed tiny Dew decoder and on logits with planted exact ties.

The accuracy is `SFTTrainer.compute_loss`'s, fetched from TRL v1.12.0
(trl/trainer/sft_trainer.py, with utils.py's `entropy_from_logits`) and
run as published: the statements of its
no-grad block from the shifted logits and labels to `accuracy`, the argmax
of the shifted logits against the labels where they are not -100, the
counts gathered (one process: `gather_for_metrics` returns its input), and
the accuracy their ratio. `SFTTrainer.log` then reports the mean of the
batches' accuracies; that mean is recorded beside the counts.

- `decoder.npz`: a two-layer `CausalTransformer` at `jax.random.key(0)` over
  two batches of uneven size, its full-row logits, the ids, and chat roles
  (0 padding, 2 user, 3 assistant) whose assistant targets are the labels
  and every other target -100, as an assistant-only SFT run labels them.
  The smallest gap between a counted position's top two logits is
  recorded, so a comparison knows rounding cannot reorder them.
- `ties.npz`: float32 logits over a 64-wide vocabulary on which the top
  value is planted at two columns per row, apart by more than a quarter of
  the vocabulary, and labels with -100 on a third of them.

    PYTHONPATH=src python tools/token_accuracy_reference.py
"""

from __future__ import annotations

import ast
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dew.nn.backbones import CausalTransformer

TRL = "https://raw.githubusercontent.com/huggingface/trl/59c4a8e104413fa9f4ca1a54eaf2ff93c0f299be/trl/trainer/"
"""TRL v1.12.0's commit."""
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "token_accuracy"
MODEL = {"vocab_size": 32, "emb_features": 16, "num_layers": 2, "num_heads": 2, "mlp_features": 32,
         "max_seq_len": 16, "attention_impl": "reference"}
BATCHES = (3, 5)
LENGTH = 12
USER, ASSISTANT = 2, 3


def entropy() -> object:
    """`entropy_from_logits` from trl/trainer/utils.py, which those
    statements call."""
    text = urllib.request.urlopen(TRL + "utils.py").read().decode()
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "entropy_from_logits")
    scope = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "utils.py", "exec"), scope)
    return scope["entropy_from_logits"]


def accuracy_statements() -> list[ast.stmt]:
    """`compute_loss`'s statements from the entropy to the accuracy."""
    text = urllib.request.urlopen(TRL + "sft_trainer.py").read().decode()
    method = next(node for node in ast.walk(ast.parse(text))
                  if isinstance(node, ast.FunctionDef) and node.name == "compute_loss")
    block = next(node for node in ast.walk(method) if isinstance(node, ast.With)
                 and ast.unparse(node.items[0].context_expr) == "torch.no_grad()")
    body = block.body
    first = next(index for index, node in enumerate(body)
                 if ast.unparse(node).startswith("per_token_entropy = "))
    last = next(index for index, node in enumerate(body) if ast.unparse(node).startswith("accuracy = "))
    return body[first:last + 1]


def trl_accuracy(statements, logits: np.ndarray, labels: np.ndarray) -> dict:
    """TRL's statements on one batch's full-row `logits` and `labels`."""
    scope = {"torch": torch, "entropy_from_logits": entropy(),
             "self": types.SimpleNamespace(accelerator=types.SimpleNamespace(gather_for_metrics=lambda x: x)),
             "shift_logits": torch.as_tensor(logits)[..., :-1, :],
             "shift_labels": torch.as_tensor(labels)[..., 1:]}
    with torch.no_grad():
        exec(compile(ast.Module(body=statements, type_ignores=[]), "sft_trainer.py", "exec"), scope)
    return {"correct": int(scope["correct_tokens"].sum()), "total": int(scope["total_sum"]),
            "accuracy": float(scope["accuracy"])}


def decoder(statements) -> dict:
    model = CausalTransformer(**MODEL)
    params = model.init(jax.random.key(0), jnp.ones((1, LENGTH), jnp.int32))
    rng = np.random.default_rng(5)
    arrays, accuracies = {}, []
    for index, rows in enumerate(BATCHES):
        # Each next id is the model's own top choice for it more often than
        # not, so the batches count many right predictions and many wrong.
        tokens = rng.integers(0, MODEL["vocab_size"], (rows, LENGTH)).astype(np.int32)
        for position in range(1, LENGTH):
            chosen = np.asarray(jnp.argmax(model.apply(params, jnp.asarray(tokens))[:, position - 1], -1))
            greedy = rng.random(rows) < 0.6
            tokens[greedy, position] = chosen[greedy]
        roles = np.full((rows, LENGTH), USER, np.int32)
        for row in range(rows):
            start, end = rng.integers(2, 6), rng.integers(8, LENGTH + 1)
            roles[row, start:end] = ASSISTANT
            roles[row, end:] = 0
        logits = np.asarray(model.apply(params, jnp.asarray(tokens)), np.float32)
        labels = np.where(roles == ASSISTANT, tokens, -100).astype(np.int64)
        counted = labels[:, 1:] != -100
        top = np.sort(logits[:, :-1], axis=-1)[..., -2:]
        gap = float(np.min((top[..., 1] - top[..., 0])[counted]))
        result = trl_accuracy(statements, logits, labels)
        accuracies.append(result["accuracy"])
        arrays.update({f"{index}/tokens": tokens, f"{index}/roles": roles, f"{index}/logits": logits,
                       f"{index}/correct": np.asarray(result["correct"]),
                       f"{index}/total": np.asarray(result["total"]), f"{index}/gap": np.asarray(gap)})
    arrays["logged"] = np.asarray(sum(accuracies) / len(accuracies))
    return arrays


def ties(statements) -> dict:
    """Each position's label is one of its two tied columns, the label of
    position p + 1 scoring the logits at p as the shift reads them."""
    rng = np.random.default_rng(6)
    logits = rng.normal(size=(4, 10, 64)).astype(np.float32)
    first = rng.integers(0, 24, (4, 10))
    second = first + 24 + rng.integers(0, 16, (4, 10))
    peak = logits.max(-1) + 1
    np.put_along_axis(logits, first[..., None], peak[..., None], -1)
    np.put_along_axis(logits, second[..., None], peak[..., None], -1)
    labels = np.full((4, 10), -100, np.int64)
    labels[:, 1:] = np.where(rng.random((4, 10)) < 0.5, first, second)[:, :-1]
    labels[:, 1:][rng.random((4, 9)) < 1 / 3] = -100
    result = trl_accuracy(statements, logits, labels)
    return {"logits": logits, "labels": labels, "correct": np.asarray(result["correct"]),
            "total": np.asarray(result["total"])}


def main() -> None:
    statements = accuracy_statements()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "decoder.npz", **decoder(statements))
    np.savez(FIXTURE / "ties.npz", **ties(statements))
    print(f"{FIXTURE}: decoder.npz and ties.npz")


if __name__ == "__main__":
    main()
