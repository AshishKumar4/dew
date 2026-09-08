"""Network-gated released-weight checks for load, scoring, generation, SGD and export.

SmolLM2-135M and Qwen3-0.6B are revision-pinned below. Dew and Transformers
5.16.1 load the same snapshot in float32; the Torch reference stays on CPU.
The GPU run uses the suite's highest matmul precision. Lowering precision to
TF32 made four Qwen cases fail at the unchanged bounds.

`LOGITS` is per checkpoint because the fp32 forward residual is per
checkpoint, and it is not 1e-4. What sets it, measured on these two
revisions over the prompts below:

                  CPU fp32   GPU fp32   fp64, both sides widened
  SmolLM2-135M    1.469e-4   1.590e-4   2.149e-13
  Qwen3-0.6B      7.820e-5   3.586e-4   2.487e-13

The last column is the exactness proof, and it says the arithmetic is the
same arithmetic. Both implementations deliberately pin several operations to
fp32 whatever the model's compute dtype: the rotary tables
(`dew.nn.attention.rotary_freqs`, `modeling_llama.py:108-125`), RMSNorm
(`dew.nn.attention.RMSNorm`, `modeling_llama.py:62-67`) and the attention
softmax (`flax/linen/attention.py:143-144`, `modeling_llama.py:208`). Widen
exactly those on both sides, run the trunk in double, and the two agree to
2e-13, five decimal orders inside 1e-4: rope, norm epsilon, attention
scaling, grouped-query repetition and the gated MLP are not merely close,
they are the same operators. The ids are identical too, which
`test_the_load_hands_back_the_checkpoints_own_tokenizer` asserts.

What remains in fp32 is therefore rounding, not a difference, and no change
to dew drives it under 1e-4. The residual stream of these checkpoints
carries massive activations, peaking near 19279 for SmolLM2 and 6560 for
Qwen3, where one fp32 ulp is already 1.95e-3 and 4.88e-4; the measured
per-layer residuals sit at one to four of those ulps, and the final norm and
unembedding carry that into logits of magnitude 36.1 and 20.7 as a few tens
of eps. TF32 is not the cause either: the CPU lane has no TF32 at all and
lands within 1.1x (SmolLM2) and 4.6x (Qwen3) of the GPU lane, both lanes
being accumulation order over the same operators.

The bounds below are the tightest the numerics support, each about twice its
checkpoint's worst measured lane. That leaves room for a backend
reassociating a sum without leaving room for a wrong operator: an eps-scale
bug in any operation above moves these residuals by orders of magnitude, not
by a factor of two.

Each prompt is tokenized individually, then the public numeric-input seam
assembles padded rows. This does not exercise raw-text batch padding when
the source tokenizer has no pad token; masked fill uses its EOS id instead.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.interop import load_pretrained
from dew.objectives.base import Step, scalar_loss
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

# Pinned: the numbers above are these commits' weights, and a repository
# that moved would compare dew against a different checkpoint.
CHECKPOINTS = {
    "SmolLM2-135M": ("HuggingFaceTB/SmolLM2-135M",
                     "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "Qwen3-0.6B": ("Qwen/Qwen3-0.6B",
                   "c1899de289a04d12100db370d81485cdf75e47ca"),
}
PROMPTS = ("The Cascade Range runs from northern California through Oregon and Washington, "
           "and its tallest volcano is",
           "The capital of France is")
SEQ = 32
NEW_TOKENS = 12
RATE = 1e-4
LOGITS = {"SmolLM2-135M": 3e-4, "Qwen3-0.6B": 7e-4}
TOKEN_LOSS = 5e-3
UPDATE = 1e-6


def available(repo: str, revision: str) -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo, "model.safetensors", revision=revision), str)


pytestmark = pytest.mark.network


def flat(tree):
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def stop_ids(generation_config) -> tuple[int, ...] | None:
    value = generation_config.get("eos_token_id")
    if isinstance(value, list):
        return tuple(value)
    return (value,) if isinstance(value, int) else None


@pytest.fixture(scope="module", params=list(CHECKPOINTS), ids=list(CHECKPOINTS))
def checkpoint(request):
    """Which pinned checkpoint the case runs on: the key into `CHECKPOINTS`
    and `LOGITS`, since the forward residual is a property of the weights."""
    return request.param


@pytest.fixture(scope="module")
def bundle(checkpoint):
    """One released checkpoint as a native model, loaded once per checkpoint."""
    repo, revision = CHECKPOINTS[checkpoint]
    if not available(repo, revision):
        pytest.skip(f"{repo} at {revision[:8]} is neither cached nor DEW_NETWORK_TESTS=1")
    return load_pretrained(repo, dtype="float32", attention_impl="reference",
                           max_seq_len=SEQ, revision=revision)


@pytest.fixture(scope="module")
def batch(bundle):
    """The prompts as one right-padded batch, built without asking the
    tokenizer to pad.

    Each prompt goes through the processor alone, and the rows are handed
    back through `Processor.from_hf`, the seam that validates processor
    output and derives the positions. `fill` is the id the short row is
    padded with: the tokenizer's pad token when it declares one, its
    end-of-text id otherwise, since a released base checkpoint need not
    name a pad token and this changes no tokenizer.
    """
    assert bundle.processor is not None, "the released checkpoint carries no tokenizer"
    tokenizer = bundle.processor.reference
    fill = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
               else tokenizer.eos_token_id)
    rows = [np.asarray(bundle.processor(text).tokens)[0] for text in PROMPTS]
    lengths = [int(row.shape[0]) for row in rows]
    width = max(lengths)
    ids = np.stack([np.concatenate([row, np.full(width - len(row), fill, row.dtype)])
                    for row in rows])
    valid = np.stack([np.arange(width) < length for length in lengths])
    return bundle.processor.from_hf({"input_ids": ids, "attention_mask": valid}), lengths, fill


@pytest.fixture(scope="module")
def reference(bundle):
    """transformers over the same snapshot directory the loader resolved."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(bundle.source), dtype=torch.float32,
                                                 local_files_only=True)
    model.eval()
    model.set_attn_implementation("eager")
    return model


def reference_logits(model, ids, mask=None, positions=None) -> np.ndarray:
    """`model`'s logits for exactly the ids, mask and positions dew was given."""
    import torch

    def rows(value):
        return torch.from_numpy(np.asarray(value).astype(np.int64))

    fields = {"input_ids": rows(ids)}
    if mask is not None:
        fields["attention_mask"] = rows(mask)
        fields["position_ids"] = rows(positions)
    with torch.no_grad():
        return model(**fields).logits.to(torch.float32).numpy()


@pytest.fixture(scope="module")
def scoring(bundle, batch):
    """The objective the scoring and the training step share, and its rows.

    Module-scoped because a second copy of a real parameter tree buys these
    tests nothing. No EMA: the averaged weights are not what the export or
    the gradient check reads.
    """
    inputs, _, fill = batch
    ids = np.asarray(inputs.tokens)
    rows = np.concatenate(
        [ids, np.full((ids.shape[0], SEQ + 1 - ids.shape[1]), fill, ids.dtype)], axis=1)
    objective = LMObjective(bundle.model, seq_len=SEQ, ema_decay=None, pad_id=fill,
                            pretrained=bundle.variables)
    return objective, objective.init(jax.random.key(0)), rows


@pytest.fixture(scope="module")
def trained(scoring):
    """One real `Trainer` step of plain SGD over the released weights."""
    import grain

    from dew.data.dataset import Dataset
    from dew.training import Trainer

    objective, _, rows = scoring
    count = math.lcm(rows.shape[0], jax.device_count())
    tokens = np.concatenate([rows] * (count // rows.shape[0]), axis=0)
    entries = [{"text": tokens[row]} for row in range(count)]
    stream = (grain.MapDataset.source(entries).repeat().to_iter_dataset()
              .batch(count, drop_remainder=True))
    data = Dataset(train=lambda: iter(stream), val=None, records=count, batch=count)
    state = Trainer(objective, optax.sgd(RATE), key=jax.random.key(2)).fit(
        data, steps=1, log_every=1)
    return state, {"text": tokens}


def test_the_load_hands_back_the_checkpoints_own_tokenizer(bundle, batch):
    """A text prompt becomes the ids the checkpoint was trained on, and comes
    back as its own text, without the caller naming a vocabulary."""
    from transformers import AutoTokenizer

    inputs, lengths, _ = batch
    ids = np.asarray(inputs.tokens)
    tokenizer = AutoTokenizer.from_pretrained(str(bundle.source), local_files_only=True)

    for row, text in enumerate(PROMPTS):
        assert list(ids[row, :lengths[row]]) == tokenizer(text)["input_ids"], text
        assert bundle.processor.decode(ids[row:row + 1, :lengths[row]]) == [text]
    assert lengths[1] < lengths[0], "both prompts tokenize to the same length; nothing is padded"


def test_the_released_logits_match_transformers_on_the_same_ids(
        bundle, batch, reference, checkpoint):
    """The parity claim on real weights: dew's fp32 forward over the released
    checkpoint against transformers over the same directory, the same ids,
    the same mask and the same positions."""
    inputs, _, _ = batch
    valid = np.asarray(inputs.token_fields["attention_mask"])
    ours = np.asarray(bundle.model.apply(bundle.variables, inputs.tokens, **inputs.kwargs()),
                      np.float32)
    theirs = reference_logits(reference, inputs.tokens, valid,
                              inputs.token_fields["positions"])

    assert np.array_equal(np.argmax(ours[valid], -1), np.argmax(theirs[valid], -1))
    difference = float(np.max(np.abs(ours[valid] - theirs[valid])))
    assert difference < LOGITS[checkpoint], f"max |logit difference| {difference:.3e}"


def test_the_objectives_token_scores_are_the_references_cross_entropy(
        batch, reference, scoring):
    """`LMObjective.evaluate` is the shifted cross entropy of the reference's
    own logits, position by position, and the fill it is handed counts for
    nothing."""
    import torch

    inputs, _, _ = batch
    objective, variables, rows = scoring
    valid = np.asarray(inputs.token_fields["attention_mask"])
    scores = objective.evaluate(variables, {"text": jnp.asarray(rows)},
                                Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None))
    losses, weights = np.asarray(scores.losses), np.asarray(scores.weights)
    expected = torch.nn.functional.cross_entropy(
        torch.from_numpy(reference_logits(reference, rows[:, :-1])).transpose(1, 2),
        torch.from_numpy(rows[:, 1:].astype(np.int64)), reduction="none").numpy()

    counted = weights > 0
    # Every real token but each row's first is a target; the fill is not.
    assert int(counted.sum()) == int(valid.sum()) - rows.shape[0]
    difference = float(np.max(np.abs(losses[counted] - expected[counted])))
    assert difference < TOKEN_LOSS, f"max |token loss difference| {difference:.3e}"


def test_greedy_generation_draws_the_references_greedy_continuation(bundle, batch, reference):
    """Temperature zero is argmax, so the native sampler and the reference's
    greedy search read the same distribution and draw the same tokens. One
    prompt at a time: a right-padded batch is not what the reference's
    generate reads."""
    import torch

    inputs, lengths, fill = batch
    ids = np.asarray(inputs.tokens)
    task = bundle.text_generation(
        sampling=Sampling(temperature=0.0, eos_id=stop_ids(bundle.generation_config), pad_id=fill))

    for row, text in enumerate(PROMPTS):
        generation = task(text, NEW_TOKENS, key=jax.random.key(0))
        drawn = np.asarray(generation.tokens)
        width = drawn.shape[1] - np.asarray(generation.behavior_log_probs).shape[1]
        length = int(generation.lengths[0])
        expected = reference.generate(
            input_ids=torch.from_numpy(ids[row:row + 1, :lengths[row]].astype(np.int64)),
            do_sample=False, num_beams=1, max_new_tokens=NEW_TOKENS, pad_token_id=fill)

        assert length == NEW_TOKENS and not bool(generation.terminated[0]), text
        assert (list(drawn[0, width:width + length])
                == [int(token) for token in expected[0][lengths[row]:]]), text


def test_one_trainer_step_moves_the_weights_by_the_objectives_gradient(scoring, trained):
    """A real `Trainer.fit(steps=1)` over the released weights: with plain
    SGD the new parameters are the loaded ones less the rate times the
    objective's own gradient, on every leaf."""
    objective, variables, _ = scoring
    state, batch = trained
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    gradient = jax.grad(lambda values: scalar_loss(objective, values, batch, step)[0])(variables)

    held, updated, grads = flat(variables), flat(state.params), flat(gradient)
    assert int(state.updates) == 1 and held.keys() == updated.keys()
    difference = max(float(np.max(np.abs(np.asarray(updated[name])
                                         - (np.asarray(leaf) - RATE * np.asarray(grads[name])))))
                     for name, leaf in held.items())
    assert difference < UPDATE, f"max |update difference| {difference:.3e}"


def test_the_trained_export_reloads_and_transformers_reads_it(
        bundle, batch, trained, checkpoint, tmp_path):
    """`Pretrained.save` writes the trained weights back into the released
    layout: `load_pretrained` reads them back leaf for leaf, rebuilds the
    same model and still carries a tokenizer, and transformers loads the
    same directory, consuming every tensor and wanting none, and computes
    the same logits."""
    import torch
    from transformers import AutoModelForCausalLM

    inputs, _, _ = batch
    state, _ = trained
    valid = np.asarray(inputs.token_fields["attention_mask"])
    export = tmp_path / "trained"

    bundle.save(export, variables=state.params)
    ours = np.asarray(bundle.model.apply(state.params, inputs.tokens, **inputs.kwargs()),
                      np.float32)
    again = load_pretrained(export, dtype="float32", attention_impl="reference", max_seq_len=SEQ)

    assert again.model == bundle.model, "the exported config rebuilds a different model"
    assert again.processor is not None, "the export carries no tokenizer"
    held = flat(state.params)
    for name, leaf in flat(again.variables).items():
        assert np.array_equal(np.asarray(leaf), np.asarray(held[name])), name
    reloaded = np.asarray(again.model.apply(again.variables, inputs.tokens, **inputs.kwargs()),
                          np.float32)
    np.testing.assert_array_equal(reloaded[valid], ours[valid])

    # The loading report, as `tools/decoder_export_reference.py` reads it:
    # silent name drift leaves a tensor behind and a randomly initialized
    # parameter in its place, which agreeing logits below would then have to
    # catch by luck. These two families declare no module transformers lacks,
    # so nothing may be missing, mismatched or left over.
    read_back, report = AutoModelForCausalLM.from_pretrained(
        str(export), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    for category in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not report.get(category), f"{category}: {report[category]}"

    read_back.eval()
    read_back.set_attn_implementation("eager")
    theirs = reference_logits(read_back, inputs.tokens, valid, inputs.token_fields["positions"])
    assert np.array_equal(np.argmax(theirs[valid], -1), np.argmax(ours[valid], -1))
    difference = float(np.max(np.abs(theirs[valid] - ours[valid])))
    assert difference < LOGITS[checkpoint], f"max |logit difference| {difference:.3e}"
