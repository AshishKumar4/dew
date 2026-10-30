"""The draws Dream's own sampling (generation_utils.py `_sample`, `alg='origin'`) is recorded over
(tools/dream_walk_reference.py) and tests/test_dream_walk.py feeds Dew's walk again: NumPy alone, as the
recording tool reads it beside the release's code. Each step of a walk reveals each masked position whose
uniform falls below the step's reveal chance, the uniforms each prompt's own stream gives (`uniforms`)."""

import numpy as np

PROMPTS = ((5, 17, 33, 2), (64, 9, 9), (1, 2, 3, 4, 5, 6), (7, 8, 9))
"""The recorded prompts' token ids, below dream-tiny's mask id (120)."""

BETWEEN = 3
"""The prompt whose first step's uniforms all fall between the two releases' first reveal chances,
`1 - s/t` at their grids' first step: Dream-v0's, its grid ending at its `eps` 1e-3, is 0.124875, and
DiffuCoder's, ending at 1e-12, 0.125, so DiffuCoder's walk reveals its whole response at that step and
Dream's none of it."""

NEW_TOKENS = 6
"""The response each prompt is walked to."""

STEPS = 8
"""The reveal steps a recorded walk takes."""

RELEASES = {
    "dream": ("dream-7b", "Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
              "apache-2.0"),
    "diffucoder": ("diffucoder-7b", "apple/DiffuCoder-7B-Base", "a2c33054cbe99b9ab7af074f09b1d13943157ff1",
                   "apple-amlr"),
}
"""Each release whose sampling code a walk is recorded under: its fixture, its Hub repo, the commit its
`generation_utils.py` is read at and its license. Dream-v0's grid ends at its generation config's
`eps`, DiffuCoder's at 1e-12 whatever its config states."""


def uniforms(prompt: int, width: int) -> list[np.ndarray]:
    """The `STEPS` arrays of `width` float32 uniforms a walk of prompt `prompt` reads, one a step, one a
    canvas position: a step reveals a masked position whose uniform falls below its reveal chance."""
    rng = np.random.default_rng([2263, prompt])
    drawn = [rng.random(width, dtype=np.float32) for _ in range(STEPS)]
    if prompt == BETWEEN:
        drawn[0] = np.full(width, 0.12494, np.float32)
    return drawn
