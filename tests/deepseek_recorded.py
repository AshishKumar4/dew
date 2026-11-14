"""Dew's DeepSeek runs held to a release's own recorded runs (tools/deepseek_v41_reference.py and
tools/deepseek_v4_dspark_reference.py, `--record`; tests/test_deepseek_v41_recorded.py and
tests/test_deepseek_v4_dspark.py): over drawn rows whose every selection clears the gate's margin
(tests/selection_margins.py), a forward's logits and a cached run with DSpark's drafts, each summarized over
seeds with its float64 truth for check 1."""

import functools
from collections.abc import Iterable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import float64_held
from reference_error import as_given, assert_as_exact


def assert_selections_clear(model, variables, config, calls: Iterable[tuple[tuple, dict]],
                            label: str) -> None:
    """Every selection `model`'s forward makes in each of `calls` (its inputs past the variables, a
    recording's drawn row's or prompt's, each of whose selections the release's run cleared when it was
    recorded) clears `selection_margins.MARGIN`: so Dew's float32 runs over them, its float64 one and the
    release's select alike, whichever positions a variant of the forward computes them at (a gap, shifted
    positions, a cached walk), each of whose selections is one of the forward's."""
    import selection_margins

    selecting = jax.jit(functools.partial(selection_margins.capturing, model))
    for index, (inputs, fields) in enumerate(calls):
        _, captured = selecting(variables, *inputs, **fields)
        smallest = selection_margins.margin(config, captured, f"{label} {index}")
        assert smallest > selection_margins.MARGIN, (
            f"Dew's selections over {label} {index} clear {smallest:.3g}, not MARGIN")


def cleared_rows(model, variables, config, rows: np.ndarray, label: str) -> np.ndarray:
    """`rows`, a recording's drawn token rows, each of whose selections `model`'s forward clears
    (`assert_selections_clear`)."""
    assert_selections_clear(model, variables, config, (((jnp.asarray(row),), {}) for row in rows), label)
    return rows


def held_to_the_release(recording: Path, plain, variables, rows: np.ndarray, site: str, start: int = 0):
    """`recording`'s site's draw of `rows`, the release's evidence and truth over its declared seeds, Dew's
    plain forward in float64 past `start` its truth, held to the release's (`recorded_site`), and the
    count."""
    from model_support import float64_twin, recorded_site

    wide, held = jax.jit(float64_twin(plain).apply), float64_held(variables)

    def twin(inputs):
        with jax.enable_x64(new_val=True):
            return {"logits": wide(held, jnp.asarray(inputs["input_ids"]))[:, start:]}

    return recorded_site(recording, site, lambda seed: {"input_ids": rows[int(seed)]}, twin)


CACHED_LEAVES = ("prompt_logits", "decode_logits", "draft_logits", "draft_confidence")
"""A recorded cached run's leaves (tools/deepseek_v41_reference.py `cached_leaves`): its draft ids are its
draft logits' argmax, which the rows' margins hold apart (`cleared_cached_run`)."""


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


def assert_cached_run_as_recorded(recording: Path, loaded, plain, rows: np.ndarray, prompt: int,
                                  site: str) -> None:
    """The release's driver through Dew's cache (`cached_leaves`) over `recording`'s cached rows `rows`, each
    of whose selections and fed argmaxes clears (`cleared_cached_run`): Dew's `plain` model's run of
    `loaded`'s variables on par with the release's from its float64 run's (check 1, `site`'s), Dew's own
    float64 run its truth."""
    from model_support import float64_twin, recorded_site

    wide = float64_twin(plain)

    @as_given
    def twin(inputs):
        return cached_leaves(wide, float64_held(loaded.variables), inputs["input_ids"], prompt)

    draw, recorded, truth, seeds = recorded_site(recording, site, lambda seed: {"input_ids": rows[int(seed)]},
                                                 twin)
    assert_as_exact(as_given(lambda index, inputs: cleared_cached_run(
        plain, loaded.variables, loaded.model_config, inputs["input_ids"], prompt, f"cached row {index}")),
        recorded, draw, site, seeds=seeds, truth_fn=truth)
