"""What Dew's DeepSeek-V4.1 comparisons with the release's reference rest on.

tests/test_deepseek_v41.py holds Dew to tests/fixtures/hf/deepseek-v41-tiny,
which tools/deepseek_v41_reference.py writes from the release's own code,
with its float64 truth (reference_f64.npz, `--fp64`). Each mode measures
one thing, on the backend JAX picks:

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py residuals

(on a GPU with JAX_PLATFORMS=cuda,cpu: the float64 twins' decisions cross
through host callbacks, which JAX places on a CPU device)

every compared output's and every quantizer input's and top-k row's RMS
distance from the float64 truth, Dew's beside the reference's, whose
ratio tests/reference_error.py bounds, and each output's float64 twin's
(`decided`), which agrees with the truth to float64 rounding: what remains
in fp32 is rounding;

    PYTHONPATH=src:. python tools/deepseek_v41_numerics.py noise

how far Dew's inputs to each rounding and selection sit from the ones the
reference recorded, in the units of its margins (a quantizer's input over
its block's amax, a top-k row over its largest finite magnitude), which the
reference tool's NOISE, its seed rank's estimate, takes twice the largest of.

The tests share the runs (`forward`, `updated`, `cached_run`,
`vision_run`), `decided`, `selections` and the vision bundle from here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from dew.interop import load_pretrained
from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8, straight_through
from dew.nn.inputs import ModelInputs
from dew.nn.multimodal import MultimodalTransformer
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


def forward(model, variables, ids):
    """The model's logits over `ids`, compiled from a function made for the
    call, so what a run under `decided` stands in for traces afresh."""
    return jax.jit(lambda held, tokens: model.apply(held, tokens))(variables, ids)


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


def updated(model, variables, ids, rate):
    """The loss, and the logits after one SGD step of `rate` along its
    gradient, read through the forward without quantization, so the step
    compares the gradient alone."""
    value, gradient = loss_and_gradient(model, variables, ids)
    return value, forward(unquantized(model), stepped(variables, gradient, rate), ids)


def cached_run(model, variables, ids, prompt):
    """Prefill `prompt` tokens, then one token per step: the prompt's logits,
    each step's logits, and DSpark's draft ids, logits and confidence after
    each step over the context each call recorded. Every call is compiled
    once per shape."""
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
    return (np.asarray(prompt_logits), np.stack(steps, 1),
            *(np.stack(value, 1) for value in zip(*drafted, strict=True)))


@dataclasses.dataclass
class Captured:
    """Dew's quantizer inputs per site, and its top-k rows and sorted picks,
    in call order."""

    blocks: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)
    rows: list[np.ndarray] = dataclasses.field(default_factory=list)
    picks: list[np.ndarray] = dataclasses.field(default_factory=list)


def widened(model, variables):
    """The model and its variables in float64, a media model's language
    model included; x64 has to be on. A media model's towers also ask for
    HIGHEST precision, at which float64 multiplies anyway: it sends their
    attention down Dew's reference path (`reference_only`), where jax's own
    attention takes its softmax in float32 whatever its inputs."""
    wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                        if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, variables)
    if isinstance(model, MultimodalTransformer):
        return model.clone(dtype=jnp.float64, precision=jax.lax.Precision.HIGHEST,
                           language_model=model.language_model.clone(dtype=jnp.float64)), wide
    return model.clone(dtype=jnp.float64), wide


def decided(run, model, variables):
    """`run(model, variables)` with every rounding and top-k pick taken from
    its float64 twin: the same run with x64 on, the model and its variables
    widened (`widened`) and every fp32 pin Dew names as `jnp.float32` (the
    rotary tables, the norms, the softmaxes, the mHC mixing, the pooling, the
    router and the engram gate) read as float64 while it traces. There each
    quantizer rounds its input's float32, which rounds as the float64 value
    does but at a tie.

    A value within fp32 noise of where its rounding or pick changes then goes
    the way the exact value goes, as the reference's did: `--fp64` refuses a
    fixture whose recorded roundings and picks its float64 run makes
    otherwise. So no comparison rests on the seed keeping every value clear
    of such a boundary, and the given run is held to the reference's
    rounding error alone.

    Wrappers stand in for `dew.nn.deepseek_v4`'s quantizers and for
    `jax.lax.top_k` while the runs trace or execute: the twin's hand each
    decision to the host through an ordered callback, and the given run's
    take them back through one in the same order and record the quantizer
    inputs and top-k rows they meet. Returns the given run's result, the
    twin's, and that record."""
    from dew.nn import deepseek_v4

    fp8, fp4, top_k = deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k
    single, made, record = jnp.float32, [], Captured()

    def keep(value):
        made.append(np.asarray(value))

    def exact(quantize):
        def call(*args, **kwargs):
            jnp.float32 = single  # the quantizer's own fp32 arithmetic
            try:
                out = quantize(*args, **kwargs)
            finally:
                jnp.float32 = jnp.float64
            jax.debug.callback(keep, out, ordered=True)
            return out
        return call

    def picked(values, k, is_stable=True):
        out = top_k(values, k, is_stable=is_stable)
        jax.debug.callback(keep, out[1], ordered=True)
        return out

    def taken(shape, dtype):
        """The twin's next decision, which has to be of `shape`."""
        def fetch():
            value = next(pending, None)
            if value is None or value.shape != shape:
                raise ValueError(f"the run makes a {shape} decision where its float64 twin made "
                                 f"{None if value is None else value.shape}")
            return value.astype(dtype)
        return io_callback(fetch, jax.ShapeDtypeStruct(shape, dtype), ordered=True)

    def kept(site: str, block: int):
        def store(x):
            record.blocks.setdefault(site, []).append(np.asarray(x, np.float32).reshape(-1, block))
        return store

    def selected(rows, picks):
        rows, picks = np.asarray(rows, np.float32), np.asarray(picks, np.int32)
        record.rows.append(rows.reshape(-1, rows.shape[-1]))
        record.picks.append(np.sort(picks.reshape(-1, picks.shape[-1]), -1))

    def window(x, block):
        jax.debug.callback(kept("window", block), x, ordered=True)
        return straight_through(x, taken(x.shape, x.dtype))

    def fourbit(x, block, e4m3_scale):
        jax.debug.callback(kept("entries" if e4m3_scale else "index", block), x, ordered=True)
        return straight_through(x, taken(x.shape, x.dtype))

    def ranked(values, k, is_stable=True):
        picks = taken((*values.shape[:-1], k), jnp.int32)
        jax.debug.callback(selected, values, picks, ordered=True)
        return jnp.take_along_axis(values, picks, -1), picks

    try:
        deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k = exact(fp8), exact(fp4), picked
        with jax.enable_x64(new_val=True):
            jnp.float32 = jnp.float64
            try:
                wide = run(*widened(model, variables))
                jax.effects_barrier()
            finally:
                jnp.float32 = single
        pending = iter(made)
        deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k = window, fourbit, ranked
        given = run(model, variables)
        jax.effects_barrier()
    finally:
        deepseek_v4.fake_quant_fp8, deepseek_v4.fake_quant_fp4, jax.lax.top_k = fp8, fp4, top_k
    if next(pending, None) is not None:
        raise ValueError("the run leaves decisions its float64 twin made untaken")
    return given, wide, record


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
        _, _, record = decided(lambda model, variables: forward(model, variables, ids), model, loaded.variables)
        for site, blocks in record.blocks.items():
            measured[f"{prefix}{site}"] = float(np.max(input_noise(
                np.concatenate(blocks), reference[f"{prefix}{site}_in"])))
        ours, _, theirs, _ = selections(record, reference, prefix)
        measured[f"{prefix}selection"] = float(np.max(row_noise(
            ours, theirs, reference[f"{prefix}selection_scale"])))
    return measured


def outputs(model, variables, fixture, prefix: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], Captured]:
    """One run's outputs under the reference's `prefix` names, each as the
    run makes it and as its float64 twin does (`decided`): the forward, the
    loss, the update (read through the forward without quantization) and the
    cached run with its drafts; with the forward's quantizer inputs and
    top-k rows. `fixture` supplies the ids, the update's rate and the prompt
    length."""
    ids, prompt = jnp.asarray(fixture["input_ids"]), int(fixture["decode_prompt"])
    runs = {("logits",): lambda model, variables: (forward(model, variables, ids),),
            ("loss", "updated_logits"): lambda model, variables: updated(
                model, variables, ids, fixture["learning_rate"]),
            ("prompt_logits", "decode_logits", "draft_ids", "draft_logits", "draft_confidence"):
                lambda model, variables: cached_run(model, variables, ids, prompt)}
    measured, records = {}, []
    for names, run in runs.items():
        given, wide, record = decided(run, model, variables)
        measured.update({prefix + name: (np.asarray(value), np.asarray(twin))
                         for name, value, twin in zip(names, given, wide, strict=True)})
        records.append(record)
    return measured, records[0]


def apart(ours, reference, truth, wide=None) -> dict[str, float]:
    """Dew's and the reference's RMS distances from the float64 truth and
    their ratio (tests/reference_error.py), with the float64 twin's distance
    where there is one (`decided`)."""
    dew, theirs = distance(ours, truth), distance(reference, truth)
    return {"dew": dew, "reference": theirs, "ratio": dew / theirs,
            **({} if wide is None else {"float64": distance(wide, truth)})}


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
    the text steps' after it, with the quantizers off as the reference ran.
    Every call is compiled once per shape."""
    model = model.clone(language_model=unquantized(model.language_model))
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    media = reference["token_types"] >= 0
    image_indices = jnp.asarray(np.where(media, np.cumsum(media, -1) - 1, -1))
    conditioning = {"pixel_values": jnp.asarray(reference["pixels"])[None, None]}
    span = jax.jit(lambda held, media: model.apply(
        held, media, method=lambda module, pixels: module.conditioner(pixels)))(variables, conditioning)
    cache = model.apply(variables, ids.shape[0], method=model.init_cache, mutable=["cache"])[1]
    prompt_logits, cache = jax.jit(lambda cache, tokens, indices, media: model.apply(
        {**variables, **cache}, tokens, decode=True, image_indices=indices, conditioning=media,
        mutable=["cache"]))(cache, ids[:, :prompt], image_indices[:, :prompt], conditioning)
    step = jax.jit(lambda cache, tokens: model.apply({**variables, **cache}, tokens, decode=True, mutable=["cache"]))
    steps = []
    for position in range(prompt, ids.shape[1]):
        logits, cache = step(cache, ids[:, position:position + 1])
        steps.append(np.asarray(logits[:, 0]))
    return span[0], prompt_logits, np.stack(steps, 1)


def residuals() -> dict[str, object]:
    """Every output the tests compare, every quantizer input and every top-k
    row, as Dew's and the reference's RMS distances from the float64 truth
    with each output's float64 twin's (`apart`); the draft ids as the count
    that differ, and each quantizer site as the count of recorded values its
    compiled quantizer does not reproduce bit for bit. The twins agree with
    the truth to float64 rounding, so the fp32 runs differ from it by fp32
    rounding alone."""
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    reference, truth = np.load(TINY / "reference.npz"), np.load(TINY / "reference_f64.npz")
    measured: dict[str, object] = {f"qat_{site}_mismatches": int(np.sum(
        np.asarray(jax.jit(quantize)(jnp.asarray(reference[f"qat_{site}_in"]))).view(np.uint32)
        != reference[f"qat_{site}_out"].view(np.uint32))) for site, quantize in QUANTIZERS.items()}
    for prefix, model in (("", unquantized(loaded.model)), ("qat_", loaded.model)):
        run, record = outputs(model, loaded.variables, reference, prefix)
        for name, (value, wide) in run.items():
            measured[name] = (int(np.sum(value != reference[name])) if name.endswith("draft_ids")
                              else apart(value, reference[name], truth[name], wide))
        for site, blocks in record.blocks.items():
            measured[f"{prefix}{site}_in"] = apart(np.concatenate(blocks), reference[f"{prefix}{site}_in"],
                                                   truth[f"{prefix}{site}_in"])
        measured[f"{prefix}selection_rows"] = apart(*scores(record, reference, truth, prefix))
    vision = np.load(TINY / "vision.npz")
    with tempfile.TemporaryDirectory() as directory:
        bundle = load_pretrained(vision_bundle(Path(directory)), dtype="float32", attention_impl="reference")
        given, wide, _ = decided(lambda model, variables: vision_run(model, variables, vision),
                                 bundle.model, bundle.variables)
    for name, value, twin in zip(("vision_span", "vision_prompt_logits", "vision_decode_logits"),
                                 given, wide, strict=True):
        measured[name] = apart(np.asarray(value), vision[name], truth[name], np.asarray(twin))
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", choices=("residuals", "noise"))
    measure = {"residuals": residuals, "noise": noise}[parser.parse_args().measure]
    # The reference multiplies in fp32, where a GPU's default is TF32.
    with jax.default_matmul_precision("highest"):
        measured = measure()
    sys.stdout.write(json.dumps({"backend": jax.default_backend(), **measured}, indent=1) + "\n")


if __name__ == "__main__":
    main()
