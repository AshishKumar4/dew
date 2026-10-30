"""Dream's own sampling and Dew's walk of it (ledger 2263): a Dream release's task builds from its generation
config and its own sampling code (`generation_utils.py`), and walks as that code's `diffusion_generate`
does at `alg='origin'` (its `_sample`), which tools/dream_walk_reference.py records on dream-tiny
(tests/fixtures/hf/dream-tiny/origin.json)."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dream_walk_support import NEW_TOKENS, PROMPTS, RELEASES, STEPS, uniforms
from tool_support import load

from dew.diffusion.discrete import Unmask
from dew.interop import Pretrained

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
SAMPLING = ("alg", "alg_temp", "eps", "steps", "temperature", "top_p", "top_k")
"""The generation config's fields Dream's `_sample` reads."""


def snapshot(release: str, root: Path, generation: dict | None = None) -> Path:
    """dream-tiny as `release` ships it, under `root`: its config naming its own code (`auto_map`), its
    generation config's sampling fields (`generation`, else the release's pinned file's) at the tiny's
    ids, and the release's `generation_utils.py` (the fetched store's pinned copy)."""
    fixture = RELEASES[release][0]
    directory = Path(shutil.copytree(FIXTURES / "dream-tiny", root / release))
    config = json.loads((directory / "config.json").read_text())
    config["auto_map"] = json.loads((FIXTURES / fixture / "config.json").read_text())["auto_map"]
    (directory / "config.json").write_text(json.dumps(config))
    pinned = json.loads((FIXTURES / fixture / "generation_config.json").read_text())
    stated = {name: pinned[name] for name in SAMPLING if name in pinned} if generation is None else generation
    tiny = {"bos_token_id": 1, "pad_token_id": 0, "mask_token_id": config["mask_token_id"]}
    (directory / "generation_config.json").write_text(json.dumps({**stated, **tiny}))
    code = load("fetched_fixtures").local(FIXTURES / fixture / "release" / "generation_utils.py")
    shutil.copy(code, directory / "generation_utils.py")
    return directory


@pytest.mark.parametrize("release", sorted(RELEASES))
def test_a_dream_release_walks_its_responses_as_its_own_sampling_does(release, tmp_path, monkeypatch):
    """A Dream release's task (`text_generation`, the public path a request takes) walks as its own
    `diffusion_generate` does at its generation config's settings: `alg='origin'`, 512 steps at
    temperature 0 (`Unmask`), over its code's grid, which ends at the config's `eps` for Dream-v0 and at
    1e-12 for DiffuCoder, whose config states 0.001 its code does not read. At the recording's steps, fed
    the uniforms the release's loop read (`uniforms`, the draw each step reveals by), it generates each
    recorded prompt's response, the release's tokens: among them one whose first step's uniforms fall
    between the two grids' reveal chances, which the two releases walk apart."""
    recorded = json.loads((FIXTURES / "dream-tiny" / "origin.json").read_text())
    assert recorded["settings"] == {"steps": STEPS, "new_tokens": NEW_TOKENS}
    source = Pretrained.load(snapshot(release, tmp_path), dtype="float32", attention_impl="reference")
    task = source.text_generation()
    assert (task.solver, task.steps, task.block) == (Unmask(temperature=0.0), 512, None)
    assert task.process.endpoint == {"dream": 1e-3, "diffucoder": 1e-12}[release]
    walked = replace(task, steps=STEPS)
    original = jax.random.uniform
    for index, (prompt, response) in enumerate(zip(PROMPTS, recorded["responses"][release], strict=True)):
        width = len(prompt) + NEW_TOKENS
        stream = iter(uniforms(index, width))

        def drawn(key, shape=(), dtype=jnp.float32, *args, stream=stream, width=width, **kwargs):
            # A walk's reveal draw: one uniform a canvas position, the step's from the release's stream.
            if tuple(shape) == (width,):
                return jnp.asarray(next(stream))
            return original(key, shape, dtype, *args, **kwargs)

        monkeypatch.setattr(jax.random, "uniform", drawn)
        with jax.disable_jit():
            generation = walked(jnp.asarray([prompt], jnp.int32), NEW_TOKENS, key=0).host()
        assert next(stream, None) is None
        assert np.asarray(generation.tokens)[0, len(prompt):].tolist() == response, (release, prompt)


def test_a_dream_release_asking_what_its_walk_does_not_take_is_refused_by_name(tmp_path):
    """A Dream release whose generation config asks its sampling for what `alg='origin'` does not take
    (another `alg`, a `top_k`) builds no task: the control is refused by name. One whose sampling code
    Dew does not know walks Dew's MDLM instead, refusing the controls that code would read."""
    for index, (asked, refused) in enumerate((({"alg": "entropy"}, "alg"),
                                              ({"alg": "origin", "top_k": 5}, "top_k"))):
        source = Pretrained.load(snapshot("diffucoder", tmp_path / str(index), asked), dtype="float32",
                                 attention_impl="reference")
        assert source.masked_walk is None
        with pytest.raises(ValueError, match=refused):
            source.text_generation()
    unknown = snapshot("dream", tmp_path / "unknown", {"steps": 64})
    (unknown / "generation_utils.py").write_text("# another sampling\n")
    source = Pretrained.load(unknown, dtype="float32", attention_impl="reference")
    assert source.masked_walk is None
    with pytest.raises(ValueError, match="steps"):
        source.text_generation()
