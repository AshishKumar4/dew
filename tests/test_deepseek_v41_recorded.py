"""DeepSeek-V4.1-Flash's tiny held to the release's own recorded runs (tests/reference_error.py).

recording.npz (tools/deepseek_v41_reference.py --record, pinned in the fetched store) holds the release's
outputs over drawn rows whose every selection clears the gate's margin, summarized over seeds with their
float64 truth, for check 1: the plain forward's logits and a cached walk's, the quantization-aware
forward's, and the cached run with DSpark's drafts. Dew's quantization-aware runs are held against the
float64 run that rounds where each rounded, every quantizer's input a leaf of its own
(docs/design/testing.md, a quantized computation's truth): a value within float32 noise of where a rounding
changes goes either way in two float32 runs, and in a float32 run and its float64 twin, so no truth that
rounds for itself holds every arm. step_truths.npz and quantized_step_truths.npz (--record-step-truths)
hold the training step's float64 truth alone at a few draws, which Dew's float64 step is held to leaf by
leaf: the step's float32 errors no count of draws resolves (testing.md, the training step's finding)."""


import deepseek_v41_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import float64_held
from deepseek_recorded import (
    assert_cached_run_as_recorded,
    assert_selections_clear,
    cleared_rows,
    held_to_the_release,
)
from deepseek_v41_support import TINY, training_step
from reference_error import as_given, assert_as_exact

from dew.nn.inputs import ModelInputs

fp32_matmuls = pytest.fixture(scope="module", autouse=True)(deepseek_v41_support.fp32_matmuls)
source = pytest.fixture(scope="module")(deepseek_v41_support.source)


RECORDING = TINY / "recording.npz"
"""The release's plain logits over drawn rows whose every selection clears `selection_margins.MARGIN`,
every position's and a cached walk's past its prompt, summarized over seeds, beside the rows each seed drew
(recording_rows.npz; tools/deepseek_v41_reference.py --record)."""


@pytest.fixture(scope="module")
def recorded_rows(source):
    """The rows the recording drew (recording_rows.npz), each of whose selections Dew's plain forward clears
    `selection_margins.MARGIN` by too (`cleared_rows`)."""
    loaded, plain, _ = source
    with np.load(TINY / "recording_rows.npz") as stored:
        return cleared_rows(plain, loaded.variables, loaded.model_config, stored["rows"], "recorded row")


VISION_RECORDING = TINY / "vision_recording.npz"
"""The vision half's recording (tools/deepseek_v41_reference.py --record-vision): the release's span, prefill
and teacher-forced steps over drawn image prompts its selections clear, summarized over seeds, the prompts
in vision_rows.npz."""


@pytest.fixture(scope="module")
def vision_source(tmp_path_factory):
    """The tiny with its vision half, loaded as the release ships the bundle (`vision_bundle`), and the
    recording's image prompts, each of whose selections, the release's cleared when it was recorded,
    Dew's plain forward over the whole prompt clears `selection_margins.MARGIN` by too, its prefill's and
    every step's among them (`assert_selections_clear`)."""
    from dew.interop import Pretrained
    from tools.deepseek_v41_numerics import unquantized, vision_bundle

    loaded = Pretrained.load(vision_bundle(tmp_path_factory.mktemp("bundle")), dtype="float32",
                             attention_impl="reference")
    plain = loaded.model.clone(language_model=unquantized(loaded.model.language_model))
    with np.load(TINY / "vision.npz") as fixture:
        # The site's shape, alike for every prompt (tools/deepseek_v41_reference.py `vision_layout`).
        shape = {"token_types": fixture["token_types"], "decode_prompt": fixture["decode_prompt"]}
    with np.load(TINY / "vision_rows.npz") as stored:
        prompts = [{name: stored[name][index] for name in stored.files}
                   for index in range(len(stored["input_ids"]))]
    media = shape["token_types"] >= 0
    indices = jnp.asarray(np.where(media, np.cumsum(media, -1) - 1, -1))
    calls = (((jnp.asarray(prompt["input_ids"]),),
              {"image_indices": indices,
               "conditioning": {"pixel_values": jnp.asarray(prompt["pixels"])[None, None]}})
             for prompt in prompts)
    assert_selections_clear(plain, loaded.variables, loaded.model_config, calls, "image prompt")
    return loaded, prompts, shape


def test_an_image_prompt_is_on_par_with_the_releases(vision_source):
    """The vision half over drawn image prompts its selections clear: the span the image fills, the
    prefill's logits with it and each teacher-forced text step's after it, through Dew's cache
    (`vision_run`), on par with the release's from its float64 run's (check 1, the recording's), Dew's
    own float64 run of the bundle its truth (row 354)."""
    from model_support import float64_twin, recorded_site

    from tools.deepseek_v41_numerics import vision_run

    loaded, prompts, shape = vision_source
    wide = float64_twin(loaded.model)

    def leaves(model, variables, prompt) -> dict:
        span, prompt_logits, steps = vision_run(model, variables, {**prompt, **shape})
        return {"vision_span": span, "vision_prompt_logits": prompt_logits, "vision_decode_logits": steps}

    def twin(inputs):
        with jax.enable_x64(new_val=True):
            return leaves(wide, float64_held(loaded.variables),
                          {**inputs, "pixels": jnp.asarray(inputs["pixels"], jnp.float64)})

    site = "deepseek v41 vision"
    draw, recorded, truth, seeds = recorded_site(VISION_RECORDING, site, lambda seed: prompts[int(seed)],
                                                 twin)
    # Dew's run compiled as its users compile it, one program a draw's shape.
    assert_as_exact(lambda index, inputs: leaves(loaded.model, loaded.variables, inputs), recorded, draw,
                    site, seeds=seeds, truth_fn=truth)


@pytest.mark.shared_setup("recorded_rows")
def test_an_interior_gap_leaves_every_window_on_its_rows_real_tokens(source, recorded_rows):
    """Uncached, three padding slots inside each row leave each real token's sliding window counting
    its row's real tokens (`counted_places`), as a cache's slots count them, and its compressed windows
    and n-grams on them: over drawn rows the release selects over as it would without the gap, the real
    tokens' logits are on par with the release's of the rows without it from its float64 run's (check 1,
    the recording's), where a window counted by index let the gap crowd out the keys before it (audit
    NN-02)."""
    loaded, plain, _ = source
    at, gap = 5, 3
    draw, reference, truth, seeds = held_to_the_release(RECORDING, plain, loaded.variables, recorded_rows,
                                                         "deepseek v41 plain rows")

    @jax.jit
    def gapped(ids):
        filler = jnp.zeros((ids.shape[0], gap), ids.dtype)
        tokens = jnp.concatenate([ids[:, :at], filler, ids[:, at:]], axis=1)
        slots = jnp.arange(tokens.shape[1])
        real = (slots < at) | (slots >= at + gap)
        logits = plain.apply(loaded.variables, tokens, attention_mask=jnp.broadcast_to(real, tokens.shape))
        return jnp.concatenate([logits[:, :at], logits[:, at + gap:]], axis=1)

    assert_as_exact(as_given(lambda index, inputs: {"logits": gapped(jnp.asarray(inputs["input_ids"]))}),
                    reference, draw, "an interior gap", seeds=seeds, truth_fn=truth)


SHIFT = 8
"""How far the shifted rows' given positions sit past their indices: a multiple of every compression
rate, so a query that read entries by its given place would see SHIFT // rate windows of later tokens."""


@pytest.mark.shared_setup("recorded_rows")
def test_shifted_positions_compute_the_unshifted_rows_logits(source, recorded_rows):
    """Uncached, positions given SHIFT past the tokens' indices: every rotation is relative, each
    compressed entry is visible by its window's tokens' ranks and rotates at its window's first given
    place, where its queries rotate (`window_origins`), so over drawn rows the logits are on par with
    the release's of the rows at their indices from its float64 run's (check 1, the recording's);
    entries visible by the given places showed each query windows of later tokens, and entries rotated
    at zero-based origins sat SHIFT away from their queries (audit NN-03)."""
    loaded, plain, _ = source
    draw, reference, truth, seeds = held_to_the_release(RECORDING, plain, loaded.variables, recorded_rows,
                                                         "deepseek v41 plain rows")

    @jax.jit
    def shifted(ids):
        return plain.apply(loaded.variables, ids,
                           positions=jnp.broadcast_to(jnp.arange(ids.shape[1]) + SHIFT, ids.shape))

    assert_as_exact(as_given(lambda index, inputs: {"logits": shifted(jnp.asarray(inputs["input_ids"]))}),
                    reference, draw, "shifted positions", seeds=seeds, truth_fn=truth)


@pytest.mark.shared_setup("recorded_rows")
def test_a_shifted_prompt_decodes_the_unshifted_rows_logits(source, recorded_rows):
    """Cached, a prompt at positions SHIFT past its indices, then the rest of each row one token at a
    time: each compressed entry rotates at its window's first given place, which the cache carries
    across the calls a window spans (`append_windows`), so over drawn rows every step's logits are on
    par with the release's of the rows at their indices from its float64 run's (check 1, the
    recording's), where entries rotated at their slots sat SHIFT away from their queries (audit NN-03).
    The walk is one compiled program, its steps scanned."""
    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    prompt = int(reference["decode_prompt"])
    draw, recorded, truth, seeds = held_to_the_release(RECORDING, plain, loaded.variables, recorded_rows,
                                                       "deepseek v41 plain walk", prompt - 1)

    @jax.jit
    def walked(ids):
        ops = decode_ops(plain, loaded.variables, 0, None)
        positions = jnp.broadcast_to(jnp.arange(prompt) + SHIFT, (ids.shape[0], prompt))
        prompted = ModelInputs(ids[:, :prompt], {"positions": positions})
        state, _ = prefill_state(plain, loaded.variables, prompted, ops)

        def step(state, column):
            state = ops.advance(state, column, jnp.ones(ids.shape[0], bool))
            return state, state.logits

        _, steps = jax.lax.scan(step, state, ids[:, prompt:].T)
        return jnp.concatenate([state.logits[:, None], jnp.moveaxis(steps, 0, 1)], axis=1)

    assert_as_exact(as_given(lambda index, inputs: {"logits": walked(jnp.asarray(inputs["input_ids"]))}),
                    recorded, draw, "a shifted cached walk", seeds=seeds, truth_fn=truth)


@pytest.fixture(scope="module")
def quantized_rows(source):
    """The rows the recorded quantization-aware forward drew (recording_rows.npz), each of whose selections
    the release's cleared when it was recorded (`clearing_quantized_rows`), and Dew's clear
    `selection_margins.MARGIN` by too, in its own forward with its quantizers on: so Dew's float32 run and
    its float64 one, following its roundings, select alike."""
    loaded, _, _ = source
    with np.load(TINY / "recording_rows.npz") as stored:
        return cleared_rows(loaded.model, loaded.variables, loaded.model_config, stored["quantized_rows"],
                            "quantized row")


@pytest.mark.shared_setup("quantized_rows")
def test_a_quantization_aware_forward_is_on_par_with_the_releases(source, quantized_rows):
    """The tiny's forward, its activations rounding through the release's quantizers, over drawn rows its
    selections clear: its logits on par with the release's from the float64 forward that rounds where each
    rounded, every quantizer's input a leaf of its own and each decision Dew's quantizer's of it (check 1,
    the recording's; docs/design/testing.md, a quantized computation's truth)."""
    import quantized_arms
    from model_support import float64_twin, recorded_site
    from reference_error import Followed, Rounded

    loaded, _, _ = source
    model, wide_model = loaded.model, float64_twin(loaded.model)
    rows = quantized_rows

    def wide(decisions, inputs):
        with quantized_arms.rounding(decisions) as met:
            logits = wide_model.apply(float64_held(loaded.variables), jnp.asarray(inputs["input_ids"]))
        return {"logits": logits, **met.rounded()}

    site = "deepseek v41 quantized forward"
    draw, recorded, truth, seeds = recorded_site(RECORDING, site, lambda seed: {"input_ids": rows[int(seed)]},
                                                 wide)

    @jax.jit
    def forward(ids):
        with quantized_arms.rounding() as met:
            logits = model.apply(loaded.variables, ids)
        return {"logits": logits, **met.rounded()}, met.decisions()

    def dew(index, inputs):
        return Rounded(*forward(jnp.asarray(inputs["input_ids"])))

    assert_as_exact(as_given(dew), recorded, draw, site, seeds=seeds,
                    followed=Followed(truth, quantized_arms.rounds))


def test_a_training_steps_float64_gradients_are_the_releases(source):
    """The tiny's training step in float64, the next-token loss and its gradient in every tensor the
    release trains (the MTP layers', the DSpark stages', the routing biases' among them), by the release's
    name for it, the release's own float64 step's at each recorded draw, each leaf within float64's
    rounding of the step (`assert_truths_as_recorded`): the step's function, whatever its float32 run
    rounds (docs/design/testing.md, the training step's finding)."""
    from model_support import float64_twin
    from recorded_truths import assert_truths_as_recorded
    from reference_error import recording

    loaded, plain, _ = source
    wide_step = training_step(loaded, float64_twin(plain))

    def truth(inputs):
        with jax.enable_x64(new_val=True):
            return wide_step(float64_held(loaded.variables), jnp.asarray(inputs["input_ids"]))[0]

    assert_truths_as_recorded(recording(TINY / "step_truths.npz"), "deepseek v41 plain step", truth)


def test_a_quantization_aware_training_steps_float64_gradients_are_the_releases(source):
    """The tiny's training step with its activations rounding through the release's quantizers, in float64
    following the release's recorded decisions: its loss, its gradient in every tensor the release trains
    and every quantizer's input, the release's own followed float64 step's at each recorded draw, each
    within float64's rounding of the step (`assert_truths_as_recorded`). The release's quantizers are
    inference kernels with no backward: its recording passes a rounding's gradient straight through
    (tools/deepseek_v41_kernels.py), so Dew's (`dew.nn.fake_quant.straight_through`) is held to that rule
    here, and a rounding's input to the release's before it."""
    from model_support import float64_twin
    from recorded_truths import assert_truths_as_recorded
    from reference_error import recording

    loaded, _, _ = source
    wide_step = training_step(loaded, float64_twin(loaded.model), quantized=True)

    def truth(decisions, inputs):
        with jax.enable_x64(new_val=True):
            leaves, (rounded, _) = wide_step(float64_held(loaded.variables), jnp.asarray(inputs["input_ids"]),
                                             decisions)
            return {**leaves, **rounded}

    assert_truths_as_recorded(recording(TINY / "quantized_step_truths.npz"), "deepseek v41 quantized step",
                              truth)


# The recording's declared 32 seeds, a stage's 32 draws a map task: a draw takes about 4.5 s in the two arms
# (2026-11-13, cached-cost1) beside a task's 85 s of setup, so a task of 32 draws runs within the plan's site
# targets (v41-fit2: 226 to 233 s) where 8 a task spent half its seconds on setup (v41-fit1).
@pytest.mark.check1_shards(seeds=32, per_task=32)
def test_a_cached_run_and_its_drafts_are_on_par_with_the_releases(source):
    """The release's driver through Dew's cache (`cached_leaves`): the prefill of the prompt, each
    teacher-forced step's logits after it and DSpark's draft after each, fed the step's argmax, its logits
    and confidence, over drawn rows whose every selection and fed argmax clears (`cleared_cached_run`), on
    par with the release's from its float64 run's (check 1, the recording's)."""
    loaded, plain, reference = source
    with np.load(TINY / "recording_rows.npz") as stored:
        rows = stored["cached_rows"]
    assert_cached_run_as_recorded(RECORDING, loaded, plain, rows, int(reference["decode_prompt"]),
                                  "deepseek v41 plain cached")
