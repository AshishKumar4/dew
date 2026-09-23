"""What Dew's DeepSeek-V4.1 comparisons with the release's reference rest on.

tests/test_deepseek_v41.py holds Dew to tests/fixtures/hf/deepseek-v41-tiny,
which tools/deepseek_v41_reference.py writes from the release's own code,
with its float64 truth (reference_f64.npz, `--fp64`). Each mode measures
one thing, on the backend JAX picks:

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py residuals

every compared output's and every quantizer input's and top-k row's RMS
distance from the float64 truth, Dew's beside the reference's, whose
ratio tests/reference_error.py bounds;

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py noise

how far Dew's inputs to each rounding and selection sit from the ones the
reference recorded, in the units of its margins (a quantizer's input over
its block's amax, a top-k row over its largest finite magnitude), which the
reference tool's NOISE, its seed rank's estimate, takes twice the largest of;

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py fp64

the plain outputs with Dew widened to fp64 as the truth is, every fp32 pin
included, where two implementations of the same arithmetic agree to fp64
rounding: what remains in fp32 is rounding.

The tests share `unquantized`, `loss_and_gradient`, `stepped`,
`cached_run`, `captured`, `selections` and the vision bundle from here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from dew.interop import load_pretrained
from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from tests.reference_error import distance

TINY = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf" / "deepseek-v41-tiny"


# The release's three quantizer sites: FP8 over 32 channels under
# power-of-two scales, FP4 over 16 under E4M3 scales, FP4 over 32 under
# power-of-two scales.
QUANTIZERS = {"window": lambda v: fake_quant_fp8(v, 32),
              "entries": lambda v: fake_quant_fp4(v, 16, e4m3_scale=True),
              "index": lambda v: fake_quant_fp4(v, 32, e4m3_scale=False)}


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
    each step's logits, and DSpark's drafts after each step over the context
    each call recorded. Every call is compiled once per shape."""
    rows = ids.shape[0]
    cache = model.apply(variables, rows, method=model.init_cache, mutable=["cache"])[1]
    drafts = model.apply(variables, rows, method=model.init_draft_cache, mutable=["cache"])[1]

    @jax.jit
    def step(cache, tokens):
        (_, logits), updated = model.apply(
            {**variables, **cache}, tokens, decode=True, method=model.states_and_logits,
            mutable=["cache", "prediction_inputs"])
        context = model.apply(variables, updated["prediction_inputs"], method=model.draft_context)
        return logits, {"cache": updated["cache"]}, context

    @jax.jit
    def draft(drafts, context, tokens):
        return model.apply({**variables, **drafts}, context, tokens, method=model.draft, mutable=["cache"])

    prompt_logits, cache, context = step(cache, ids[:, :prompt])
    _, drafts = draft(drafts, context, None)
    steps, drafted = [], []
    for position in range(prompt, ids.shape[1]):
        logits, cache, context = step(cache, ids[:, position:position + 1])
        steps.append(np.asarray(logits[:, 0]))
        out, drafts = draft(drafts, context, jnp.argmax(logits[:, -1], -1))
        drafted.append([np.asarray(value) for value in out])
    return np.asarray(prompt_logits), np.stack(steps, 1), [np.stack(value, 1) for value in zip(*drafted, strict=True)]


@dataclasses.dataclass
class Captured:
    """Dew's quantizer inputs per site, and its top-k rows and sorted picks,
    in call order."""

    blocks: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)
    rows: list[np.ndarray] = dataclasses.field(default_factory=list)
    picks: list[np.ndarray] = dataclasses.field(default_factory=list)


def captured(model, variables, ids, forced: Mapping[str, np.ndarray] | None = None) -> Captured:
    """One compiled forward with every quantizer input and every top-k's rows
    and picks recorded in call order.

    Wrappers stand in for the quantizers `dew.nn.deepseek_v4` calls and for
    `jax.lax.top_k` while the forward traces, hand their operands to the
    host through ordered callbacks and call the originals; they are removed
    again after. `forced` holds each site's recorded outputs in call order
    for the quantizers to return in place of their own rounding, so a value
    that rounds the other way within the noise cannot move what the
    forward's later calls read."""
    from dew.nn import deepseek_v4

    record = Captured()
    fp8, fp4, top_k = deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k
    taken = dict.fromkeys(forced or (), 0)

    def rounded(site: str, x, block: int, own):
        """`own` rounding, or the next `forced` rows of the site in its place."""
        if forced is None:
            return own()
        rows = x.size // block
        held = forced[site][taken[site]:taken[site] + rows]
        taken[site] += rows
        return jnp.asarray(held.reshape(x.shape), x.dtype)

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
        return rounded("window", x, block, lambda: fp8(x, block))

    def fourbit(x, block, e4m3_scale):
        site = "entries" if e4m3_scale else "index"
        jax.debug.callback(keep(site, block), x, ordered=True)
        return rounded(site, x, block, lambda: fp4(x, block, e4m3_scale))

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
    for site, count in taken.items():
        if count != len(forced[site]):
            raise ValueError(f"the forward rounds {count} {site} blocks where the reference rounded "
                             f"{len(forced[site])}")
    return record


def recorded_outputs(reference, prefix: str) -> dict[str, np.ndarray]:
    """The reference's quantizer outputs of one forward, per site in call
    order, for `captured` to force."""
    return {site: reference[f"{prefix}{site}_out"] for site in ("window", "entries", "index")
            if f"{prefix}{site}_out" in reference}


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


def candidates(rows: np.ndarray, picks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each top-k row's finite scores in key order, NaN after them, and
    which of those its picks name; a pick of a forbidden key names none."""
    finite = np.isfinite(rows)
    order = np.argsort(~finite, axis=1, kind="stable")
    compact = np.where(np.take_along_axis(finite, order, 1), np.take_along_axis(rows, order, 1), np.nan)
    rank = np.where(finite, np.cumsum(finite, 1) - 1, -1)
    named = np.where(picks >= 0, np.take_along_axis(rank, np.maximum(picks, 0), 1), -1)
    chosen = np.zeros(rows.shape, bool)
    row, slot = np.nonzero(named >= 0)
    chosen[row, named[row, slot]] = True
    return compact, chosen


def selections(record: Captured, reference, prefix: str):
    """Dew's and the reference's top-k calls of one forward, each as
    `candidates` reads it: `(ours, our picks, theirs, their picks)`.

    Dew scores a Reindex layer's candidate pool alone, where the reference
    scores the whole row with every key outside it at -inf, and Dew's top-k
    gives a forbidden key the lowest float; both keep the same keys in the
    same order, so the calls compare by their finite scores. Refused where
    the two keep different keys."""
    theirs, their_picks = reference[f"{prefix}selection_rows"], reference[f"{prefix}selection_picks"]
    rows = padded(record.rows, theirs.shape[1], np.nan)
    if rows.shape != theirs.shape:
        raise ValueError(f"Dew ranks {rows.shape} where the reference ranked {theirs.shape}")
    rows = np.where(rows <= np.finfo(np.float32).min / 2, -np.inf, rows)
    ours, our_picks = candidates(rows, padded(record.picks, their_picks.shape[1], -1))
    theirs, their_picks = candidates(theirs, their_picks)
    if np.any(np.isfinite(ours) != np.isfinite(theirs)):
        raise ValueError("Dew's top-k and the reference's keep different keys")
    return ours, our_picks, theirs, their_picks


def scores(record: Captured, reference, truth, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every finite top-k score of one forward's calls, Dew's, the
    reference's and the float64 truth's, in the same order (`selections`)."""
    ours, _, theirs, _ = selections(record, reference, prefix)
    wide = candidates(truth[f"{prefix}selection_rows"], reference[f"{prefix}selection_picks"])[0]
    finite = np.isfinite(theirs)
    return ours[finite], theirs[finite], wide[finite]


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
        record = captured(model, loaded.variables, ids, recorded_outputs(reference, prefix))
        for site, blocks in record.blocks.items():
            measured[f"{prefix}{site}"] = float(np.max(input_noise(
                np.concatenate(blocks), reference[f"{prefix}{site}_in"])))
        ours, _, theirs, _ = selections(record, reference, prefix)
        measured[f"{prefix}selection"] = float(np.max(row_noise(
            ours, theirs, reference[f"{prefix}selection_scale"])))
    return measured


def outputs(model, variables, fixture, prefix: str) -> dict[str, np.ndarray]:
    """One run's outputs under the reference's `prefix` names: the forward,
    the loss, the update (read through the forward without quantization)
    and the cached run with its drafts. `fixture` supplies the ids, the
    update's rate and the prompt length."""
    ids = jnp.asarray(fixture["input_ids"])
    value, gradient = loss_and_gradient(model, variables, ids)
    moved = stepped(variables, gradient, fixture["learning_rate"])
    prompt_logits, steps, (draft_ids, draft_logits, confidence) = cached_run(
        model, variables, ids, int(fixture["decode_prompt"]))
    return {prefix + name: np.asarray(value) for name, value in {
        "logits": jax.jit(model.apply)(variables, ids), "loss": value,
        "updated_logits": jax.jit(unquantized(model).apply)(moved, ids),
        "prompt_logits": prompt_logits, "decode_logits": steps, "draft_logits": draft_logits,
        "draft_confidence": confidence, "draft_ids": draft_ids}.items()}


def apart(ours, reference, truth) -> dict[str, float]:
    """Dew's and the reference's RMS distances from the float64 truth and
    their ratio (tests/reference_error.py)."""
    dew, theirs = distance(ours, truth), distance(reference, truth)
    return {"dew": dew, "reference": theirs, "ratio": dew / theirs}


def vision_bundle(directory: Path) -> Path:
    """The fixture with its vision half in `directory` as the release ships a
    bundle: vision_config inside config.json, one safetensors file."""
    from safetensors.numpy import load_file, save_file

    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(TINY / name, directory / name)
    config = json.loads((TINY / "config.json").read_text())
    config["vision_config"] = json.loads((TINY / "vision_config.json").read_text())
    (directory / "config.json").write_text(json.dumps(config))
    save_file({**load_file(TINY / "model.safetensors"), **load_file(TINY / "vision.safetensors")},
              directory / "model.safetensors")
    return directory


def vision_run(model, variables, reference) -> tuple[jax.Array, jax.Array, np.ndarray]:
    """The span the fixture's image fills, the prefill's logits with it and
    the text steps' after it, with the quantizers off as the reference ran."""
    model = model.clone(language_model=unquantized(model.language_model))
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    media = reference["token_types"] >= 0
    image_indices = jnp.asarray(np.where(media, np.cumsum(media, -1) - 1, -1))
    conditioning = {"pixel_values": jnp.asarray(reference["pixels"])[None, None]}
    span = model.apply(variables, conditioning, method=lambda module, held: module.conditioner(held))
    cache = model.apply(variables, ids.shape[0], method=model.init_cache, mutable=["cache"])[1]
    prompt_logits, cache = model.apply(
        {**variables, **cache}, ids[:, :prompt], decode=True, image_indices=image_indices[:, :prompt],
        conditioning=conditioning, mutable=["cache"])
    steps = []
    for position in range(prompt, ids.shape[1]):
        logits, cache = model.apply({**variables, **cache}, ids[:, position:position + 1],
                                    decode=True, mutable=["cache"])
        steps.append(np.asarray(logits[:, 0]))
    return span[0], prompt_logits, np.stack(steps, 1)


def residuals() -> dict[str, object]:
    """Every output the tests compare, every quantizer input and every top-k
    row, as Dew's and the reference's RMS distances from the float64 truth
    (`apart`); the draft ids as the count that differ, and each quantizer
    site as the count of recorded values its compiled quantizer does not
    reproduce bit for bit."""
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    reference, truth = np.load(TINY / "reference.npz"), np.load(TINY / "reference_f64.npz")
    ids = jnp.asarray(reference["input_ids"])
    measured: dict[str, object] = {f"qat_{site}_mismatches": int(np.sum(
        np.asarray(jax.jit(quantize)(jnp.asarray(reference[f"qat_{site}_in"]))).view(np.uint32)
        != reference[f"qat_{site}_out"].view(np.uint32))) for site, quantize in QUANTIZERS.items()}
    for prefix, model in (("", unquantized(loaded.model)), ("qat_", loaded.model)):
        for name, value in outputs(model, loaded.variables, reference, prefix).items():
            measured[name] = (int(np.sum(value != reference[name])) if name.endswith("draft_ids")
                              else apart(value, reference[name], truth[name]))
        record = captured(model, loaded.variables, ids, recorded_outputs(reference, prefix))
        for site, blocks in record.blocks.items():
            measured[f"{prefix}{site}_in"] = apart(np.concatenate(blocks), reference[f"{prefix}{site}_in"],
                                                   truth[f"{prefix}{site}_in"])
        measured[f"{prefix}selection_rows"] = apart(*scores(record, reference, truth, prefix))
    vision = np.load(TINY / "vision.npz")
    with tempfile.TemporaryDirectory() as directory:
        bundle = load_pretrained(vision_bundle(Path(directory)), dtype="float32", attention_impl="reference")
    for name, value in zip(("vision_span", "vision_prompt_logits", "vision_decode_logits"),
                           vision_run(bundle.model, bundle.variables, vision), strict=True):
        measured[name] = apart(np.asarray(value), vision[name], truth[name])
    return measured


def fp64() -> dict[str, float]:
    """The plain outputs' largest distances from the float64 truth with Dew
    widened alike: the fixture's fp32 weights in fp64, the model's dtype
    fp64 and every fp32 pin Dew names as `jnp.float32` (the rotary tables,
    the norms, the softmaxes, the mHC mixing, the pooling, the router and
    the engram gate) read as fp64 while the model traces."""
    fixture, truth = np.load(TINY / "reference.npz"), np.load(TINY / "reference_f64.npz")
    with jax.enable_x64(new_val=True):
        loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, loaded.variables)
        single = jnp.float32
        jnp.float32 = jnp.float64
        try:
            measured = outputs(unquantized(loaded.model).clone(dtype=jnp.float64), variables, fixture, "")
        finally:
            jnp.float32 = single
    return {name: float(np.sum(value != fixture[name]) if name == "draft_ids"
                        else np.max(np.abs(value - truth[name]))) for name, value in measured.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", choices=("residuals", "noise", "fp64"))
    measure = {"residuals": residuals, "noise": noise, "fp64": fp64}[parser.parse_args().measure]
    # The reference multiplies in fp32, where a GPU's default is TF32.
    with jax.default_matmul_precision("highest"):
        measured = measure()
    sys.stdout.write(json.dumps({"backend": jax.default_backend(), **measured}, indent=1) + "\n")


if __name__ == "__main__":
    main()
