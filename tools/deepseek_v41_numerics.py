"""What Dew's DeepSeek-V4.1 comparisons with the release's reference rest on.

tests/test_deepseek_v41.py holds Dew to tests/fixtures/hf/deepseek-v41-tiny,
which tools/deepseek_v41_reference.py writes from the release's own code.
Each mode measures one thing, on the backend JAX picks:

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py residuals

every compared output's largest distance from the reference in fp32;

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py noise

how far Dew's inputs to each rounding and selection sit from the ones the
reference recorded, in the units of its margins (a quantizer's input over
its block's amax, a top-k row over its largest finite magnitude), which the
reference tool's NOISE bounds at twice the largest over CPU and GPU.

The tests share `unquantized`, `loss_and_gradient`, `stepped`,
`cached_run`, `captured` and the noise measures from here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from dew.interop import load_pretrained
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective

TINY = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf" / "deepseek-v41-tiny"


def unquantized(model):
    """The model with its quantization-aware rounding off, as the reference's
    architecture outputs were taken."""
    kinds = {name: dataclasses.replace(kind, mixer=dataclasses.replace(kind.mixer, kv_qat=False))
             for name, kind in model.kinds.items()}
    return model.clone(kinds=flax.core.freeze(kinds))


def loss_and_gradient(model, variables, ids):
    """The next-token loss LMObjective reports and its gradient in the parameters."""
    objective = LMObjective(model, ids.shape[1] - 1, pretrained=variables, ema_decay=None)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**variables, "params": params}, {"text": ModelInputs(ids)}, step)
        return objective.reduce_loss(statistics)[0]

    return jax.jit(jax.value_and_grad(loss))(variables["params"])


def stepped(variables, gradient, rate):
    """The variables one SGD step of `rate` along `gradient` away."""
    return {**variables, "params": jax.tree.map(
        lambda weight, grad: weight - rate * grad, variables["params"], gradient)}


def cached_run(model, variables, ids, prompt):
    """Prefill `prompt` tokens, then one token per step; the prompt's logits,
    each step's logits, and DSpark's drafts after each step."""
    rows = ids.shape[0]
    cache = model.apply(variables, rows, method=model.init_cache, mutable=["cache"])[1]
    drafts = model.apply(variables, rows, method=model.init_draft_cache, mutable=["cache"])[1]
    (hidden, prompt_logits), cache = model.apply(
        {**variables, **cache}, ids[:, :prompt], decode=True, method=model.states_and_logits, mutable=["cache"])
    _, drafts = model.apply({**variables, **drafts}, hidden, None, method=model.draft, mutable=["cache"])
    steps, drafted = [], []
    for position in range(prompt, ids.shape[1]):
        (hidden, logits), cache = model.apply(
            {**variables, **cache}, ids[:, position:position + 1], decode=True,
            method=model.states_and_logits, mutable=["cache"])
        steps.append(np.asarray(logits[:, 0]))
        out, drafts = model.apply({**variables, **drafts}, hidden, jnp.argmax(logits[:, -1], -1),
                                  method=model.draft, mutable=["cache"])
        drafted.append([np.asarray(value) for value in out])
    return np.asarray(prompt_logits), np.stack(steps, 1), [np.stack(value, 1) for value in zip(*drafted, strict=True)]


@dataclasses.dataclass
class Captured:
    """Dew's quantizer inputs per site, and its top-k rows and sorted picks,
    in call order."""

    blocks: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)
    rows: list[np.ndarray] = dataclasses.field(default_factory=list)
    picks: list[np.ndarray] = dataclasses.field(default_factory=list)


def captured(model, variables, ids) -> Captured:
    """One compiled forward with every quantizer input and every top-k's rows
    and picks recorded in call order.

    Wrappers stand in for the quantizers `dew.nn.deepseek_v4` calls and for
    `jax.lax.top_k` while the forward traces, hand their operands to the
    host through ordered callbacks and call the originals; they are removed
    again after."""
    from dew.nn import deepseek_v4

    record = Captured()
    fp8, fp4, top_k = deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k

    def keep(site: str, block: int):
        def store(x):
            record.blocks.setdefault(site, []).append(np.asarray(x, np.float32).reshape(-1, block))
        return store

    def selected(rows, picks):
        rows, picks = np.asarray(rows, np.float32), np.asarray(picks, np.int32)
        record.rows.append(rows.reshape(-1, rows.shape[-1]))
        record.picks.append(np.sort(picks.reshape(-1, picks.shape[-1]), -1))

    def window(x, block):
        jax.debug.callback(keep("window", block), x, ordered=True)
        return fp8(x, block)

    def fourbit(x, block, e4m3_scale):
        jax.debug.callback(keep("entries" if e4m3_scale else "index", block), x, ordered=True)
        return fp4(x, block, e4m3_scale)

    def ranked(values, k, **kwargs):
        out = top_k(values, k, **kwargs)
        jax.debug.callback(selected, values, out[1], ordered=True)
        return out

    deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k = window, fourbit, ranked
    try:
        jax.block_until_ready(jax.jit(lambda held, tokens: model.apply(held, tokens))(variables, ids))
        jax.effects_barrier()
    finally:
        deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k = fp8, fp4, top_k
    return record


def padded(parts: list[np.ndarray], width: int, fill) -> np.ndarray:
    """Rows of every call, each padded to `width` with `fill`, as the
    reference records them."""
    return np.concatenate([np.pad(part, ((0, 0), (0, width - part.shape[1])), constant_values=fill)
                           for part in parts])


def input_noise(ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """Each quantizer input's distance from the reference's, over the
    reference block's amax; an all-zero block is zero apart."""
    if ours.shape != theirs.shape:
        raise ValueError(f"Dew quantizes {ours.shape} where the reference quantized {theirs.shape}")
    amax = np.max(np.abs(theirs), -1, keepdims=True)
    return np.abs(ours - theirs) / np.where(amax > 0, amax, np.inf)


def dew_rows(record: Captured, theirs: np.ndarray) -> np.ndarray:
    """Dew's top-k rows laid out as the reference's, with the lowest float
    Dew's top-k gives a forbidden key read as the reference's -inf; refused
    where the two allow different keys."""
    ours = padded(record.rows, theirs.shape[1], np.nan)
    if ours.shape != theirs.shape:
        raise ValueError(f"Dew ranks {ours.shape} where the reference ranked {theirs.shape}")
    ours = np.where(ours <= np.finfo(np.float32).min / 2, -np.inf, ours)
    if np.any(np.isfinite(ours) != np.isfinite(theirs)):
        raise ValueError("Dew's top-k and the reference's allow different keys")
    return ours


def row_noise(ours: np.ndarray, theirs: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Each finite top-k score's distance from the reference's over the
    row's recorded scale."""
    finite = np.isfinite(theirs)
    apart = np.zeros(theirs.shape)
    apart[finite] = np.abs(ours[finite] - theirs[finite])
    return apart / scale[:, None]


def noise() -> dict[str, float]:
    """The largest distance between Dew's and the reference's inputs to the
    roundings and selections of both forwards: a quantizer's input over its
    block's amax, a top-k row over its largest finite magnitude."""
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    reference = np.load(TINY / "reference.npz")
    ids = jnp.asarray(reference["input_ids"])
    measured = {}
    for prefix, model in (("qat_", loaded.model), ("", unquantized(loaded.model))):
        record = captured(model, loaded.variables, ids)
        for site, blocks in record.blocks.items():
            measured[f"{prefix}{site}"] = float(np.max(input_noise(
                np.concatenate(blocks), reference[f"{prefix}{site}_in"])))
        theirs = reference[f"{prefix}selection_rows"]
        measured[f"{prefix}selection"] = float(np.max(row_noise(
            dew_rows(record, theirs), theirs, reference[f"{prefix}selection_scale"])))
    return measured


def distances(model, variables, reference, prefix: str, fixture) -> dict[str, float]:
    """How far one run's outputs lie from the reference's `prefix` ones: the
    forward, the loss, the update (read through the forward without
    quantization) and the cached run with its drafts, each as its largest
    absolute difference, the draft ids as the count that differ. `fixture`
    supplies the ids, the update's rate and the prompt length."""
    ids = jnp.asarray(fixture["input_ids"])
    plain = unquantized(model)

    def apart(name, actual):
        return float(np.max(np.abs(np.asarray(actual) - reference[prefix + name])))

    value, gradient = loss_and_gradient(model, variables, ids)
    moved = stepped(variables, gradient, fixture["learning_rate"])
    prompt_logits, steps, (draft_ids, draft_logits, confidence) = cached_run(
        model, variables, ids, int(fixture["decode_prompt"]))
    return {prefix + name: value for name, value in {
        "logits": apart("logits", jax.jit(model.apply)(variables, ids)),
        "loss": apart("loss", value),
        "updated_logits": apart("updated_logits", jax.jit(plain.apply)(moved, ids)),
        "prompt_logits": apart("prompt_logits", prompt_logits),
        "decode_logits": apart("decode_logits", steps),
        "draft_logits": apart("draft_logits", draft_logits),
        "draft_confidence": apart("draft_confidence", confidence),
        "draft_ids": float(np.sum(draft_ids != reference[prefix + "draft_ids"]))}.items()}


def residuals() -> dict[str, float]:
    """Every output the tests compare, as its distance from the reference's."""
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    reference = np.load(TINY / "reference.npz")
    return {**distances(unquantized(loaded.model), loaded.variables, reference, "", reference),
            **distances(loaded.model, loaded.variables, reference, "qat_", reference)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", choices=("residuals", "noise"))
    measure = parser.parse_args().measure
    # The reference multiplies in fp32, where a GPU's default is TF32.
    with jax.default_matmul_precision("highest"):
        measured = residuals() if measure == "residuals" else noise()
    sys.stdout.write(json.dumps({"backend": jax.default_backend(), **measured}, indent=1) + "\n")


if __name__ == "__main__":
    main()
