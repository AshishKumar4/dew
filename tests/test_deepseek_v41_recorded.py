"""DeepSeek-V4.1-Flash's tiny held to the release's own recorded runs by check 1 (tests/reference_error.py).

recording.npz (tools/deepseek_v41_reference.py --record, pinned in the fetched store) holds the release's
outputs over drawn rows whose every selection clears the gate's margin, summarized over seeds with their
float64 truth: the plain forward's logits and a cached walk's, the quantization-aware forward's, and the
cached run with DSpark's drafts. Dew's quantization-aware forward is held against the float64 run that
rounds where it rounded, every quantizer's input a leaf of its own (docs/design/testing.md, a quantized
computation's truth): a value within float32 noise of where a rounding changes goes either way in two
float32 runs, and in a float32 run and its float64 twin, so no truth that rounds for itself holds every
arm."""

import functools

import deepseek_v41_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import float64_held
from deepseek_v41_support import TINY
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
    """The rows the recording drew (recording_rows.npz), each of whose selections, the release's cleared
    when it was recorded, Dew's plain forward clears `selection_margins.MARGIN` by too: so Dew's float32
    runs over them, its float64 one and the release's select alike, whichever positions a variant of the
    forward computes them at (a gap, shifted positions, a cached walk), each of whose selections is one
    of the plain forward's."""
    import selection_margins

    loaded, plain, _ = source
    with np.load(TINY / "recording_rows.npz") as stored:
        rows = stored["rows"]
    selecting = jax.jit(functools.partial(selection_margins.capturing, plain))
    for index, row in enumerate(rows):
        _, captured = selecting(loaded.variables, jnp.asarray(row))
        smallest = selection_margins.margin(loaded.model_config, captured, f"recorded row {index}")
        assert smallest > selection_margins.MARGIN, (
            f"Dew's selections over recorded row {index} clear {smallest:.3g}, not MARGIN")
    return rows


def held_to_the_release(plain, variables, rows, site: str, start: int = 0):
    """A recorded site's draw of `rows`, the release's evidence and truth over its declared seeds, Dew's
    plain forward in float64 past `start` its truth, held to the release's (`recorded_site`), and the
    count."""
    from model_support import float64_twin, recorded_site

    wide = jax.jit(float64_twin(plain).apply)

    def twin(inputs):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, variables)
            return {"logits": wide(held, jnp.asarray(inputs["input_ids"]))[:, start:]}

    return recorded_site(RECORDING, site, lambda seed: {"input_ids": rows[int(seed)]}, twin)


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
    draw, reference, truth, seeds = held_to_the_release(plain, loaded.variables, recorded_rows,
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
    draw, reference, truth, seeds = held_to_the_release(plain, loaded.variables, recorded_rows,
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
    draw, recorded, truth, seeds = held_to_the_release(plain, loaded.variables, recorded_rows,
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
    import selection_margins

    loaded, _, _ = source
    with np.load(TINY / "recording_rows.npz") as stored:
        rows = stored["quantized_rows"]
    selecting = jax.jit(functools.partial(selection_margins.capturing, loaded.model))
    for index, row in enumerate(rows):
        label = f"quantized row {index}"
        captured = selecting(loaded.variables, jnp.asarray(row))[1]
        smallest = selection_margins.margin(loaded.model_config, captured, label)
        assert smallest > selection_margins.MARGIN, f"{label}: Dew's selections clear {smallest:.3g}"
    return rows


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


CACHED_LEAVES = ("prompt_logits", "decode_logits", "draft_logits", "draft_confidence")
"""The recorded cached run's leaves (tools/deepseek_v41_reference.py `cached_evaluated`): its draft ids are
its draft logits' argmax, which the rows' margins hold apart (`cleared_cached_run`)."""


def cached_leaves(model, variables, ids, prompt: int, captured: list | None = None) -> dict[str, np.ndarray]:
    """Dew's cached run of `ids` (`deepseek_v41_numerics.cached_run`, the release's driver: the prefill of
    `prompt` tokens, teacher-forced steps after it, DSpark's draft after each fed the step's argmax), its
    `CACHED_LEAVES`; with `captured`, every call's selections appended to it."""
    from tools.deepseek_v41_numerics import cached_run

    prompt_logits, steps, _, draft_logits, confidence = cached_run(model, variables, jnp.asarray(ids), prompt,
                                                                   captured)
    return dict(zip(CACHED_LEAVES, (prompt_logits, steps, draft_logits, confidence), strict=True))


def cleared_cached_run(model, variables, config, ids, prompt: int, label: str) -> dict:
    """Dew's cached run of `ids` (`cached_leaves`), each of whose selections and fed argmaxes clears
    `selection_margins.MARGIN`: every selection its calls make, the trunk's top-k and each DSpark stage's
    experts, captured as they run (`cached_run`'s `captured`), the main logits' argmax at the prompt's last
    position and every step, which DSpark drafts from, and each draft's at every position but its last,
    which the next position reads. The release's cleared twice MARGIN where it recorded the rows
    (`clearing_cached_rows`), so Dew's float32 run and its float64 one decide alike; a row Dew's run does
    not clear is refused, its verdict not one of the release's computation."""
    import selection_margins

    captured: list = []
    leaves = cached_leaves(model, variables, ids, prompt, captured)
    fed = (np.asarray(leaves["prompt_logits"])[:, -1:], np.asarray(leaves["decode_logits"]),
           np.asarray(leaves["draft_logits"])[:, :, :-1])
    smallest = min(min(selection_margins.margin(config, call, label) for call in captured),
                   *(float(selection_margins.top_margin(logits, 1).min()) for logits in fed))
    assert smallest > selection_margins.MARGIN, f"{label}: Dew's clear {smallest:.3g}"
    return leaves


# The recording's declared 32 seeds, a stage's 32 draws a map task: a draw takes about 4.5 s in the two arms
# (2026-11-13, cached-cost1) beside a task's 85 s of setup, so a task of 32 draws runs within the plan's site
# targets (v41-fit2: 226 to 233 s) where 8 a task spent half its seconds on setup (v41-fit1).
@pytest.mark.check1_shards(seeds=32, per_task=32)
def test_a_cached_run_and_its_drafts_are_on_par_with_the_releases(source):
    """The release's driver through Dew's cache (`cached_leaves`): the prefill of the prompt, each
    teacher-forced step's logits after it and DSpark's draft after each, fed the step's argmax, its logits
    and confidence, over drawn rows whose every selection and fed argmax clears (`cleared_cached_run`), on
    par with the release's from its float64 run's (check 1, the recording's)."""
    from model_support import float64_twin, recorded_site

    loaded, plain, reference = source
    prompt = int(reference["decode_prompt"])
    wide = float64_twin(plain)
    with np.load(TINY / "recording_rows.npz") as stored:
        rows = stored["cached_rows"]

    @as_given
    def twin(inputs):
        return cached_leaves(wide, float64_held(loaded.variables), inputs["input_ids"], prompt)

    site = "deepseek v41 plain cached"
    draw, recorded, truth, seeds = recorded_site(RECORDING, site, lambda seed: {"input_ids": rows[int(seed)]},
                                                 twin)
    assert_as_exact(as_given(lambda index, inputs: cleared_cached_run(
        plain, loaded.variables, loaded.model_config, inputs["input_ids"], prompt, f"cached row {index}")),
        recorded, draw, site, seeds=seeds, truth_fn=truth)
