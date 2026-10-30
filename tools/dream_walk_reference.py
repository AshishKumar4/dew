#!/usr/bin/env python3
"""Record Dream's own sampling on dream-tiny: `diffusion_generate`'s `alg='origin'` (generation_utils.py
`_sample`) of each release in tests/dream_walk_support.py's `RELEASES`, which tests/test_dream_walk.py
holds Dew's walk to (ledger 2263).

Each release's `generation_utils.py`, read at its pinned commit, walks each of `PROMPTS` to
`NEW_TOKENS` tokens over `STEPS` steps at temperature 0 (its configs' default) over Dew's own logits of
dream-tiny, which tests/test_remote_code_releases.py holds to the release's own model code: the loop,
its time grid, its reveal rule and its token rule are the release's, the denoiser the same on both
sides. Each step's reveals are decided by the uniforms `uniforms` gives (`torch.rand`, which the loop
draws one a masked position, reads them at the masked positions in order), which the test feeds Dew's
walk too. The mask token's logit is held out (the loop's own `generation_logits_hook_func`), as Dew's
walk never draws the mask token. Writes tests/fixtures/hf/dream-tiny/origin.json.

`--vendor` pins each release's `generation_utils.py` with its license (`fetched_fixtures.pin_release`),
which the test reads offline: Dream-v0's (apache-2.0) into the fetched store, DiffuCoder's (apple-amlr,
which is not re-hosted) at its own repo's commit (`AT_ORIGIN`).

  PYTHONPATH=src:tools python tools/dream_walk_reference.py [--vendor]
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from dream_walk_support import NEW_TOKENS, PROMPTS, RELEASES, STEPS, uniforms

ROOT = Path(__file__).resolve().parents[1]

FIXTURES = ROOT / "tests" / "fixtures" / "hf"
AT_ORIGIN = {"diffucoder"}
"""The releases whose sampling code is pinned at their own repo rather than re-hosted in the store
(`pin_release(origin=True)`): DiffuCoder's, under the Apple Machine Learning Research License."""


def sampling_code(release: str) -> Path:
    """`release`'s own `generation_utils.py` at its pinned commit."""
    _, repo, revision, _ = RELEASES[release]
    return Path(hf_hub_download(repo, "generation_utils.py", revision=revision))


class _Released:
    """The release's `self` in `_sample`: its device, and its call, which gives Dew's logits of
    `source` (dream-tiny) for the canvas `x`."""

    device = torch.device("cpu")

    def __init__(self, source):
        self.source = source

    def __call__(self, x, attention_mask, tok_idx):
        import jax.numpy as jnp

        scores = self.source.model.apply(self.source.variables, jnp.asarray(x.numpy(), jnp.int32),
                                         method="logits")
        return types.SimpleNamespace(logits=torch.from_numpy(np.asarray(scores, np.float32)))


def walked(release: str) -> list[list[int]]:
    """Each prompt's response, as `release`'s own `_sample` walks it over Dew's logits of dream-tiny."""
    from dew.interop import Pretrained

    spec = importlib.util.spec_from_file_location(f"{release}_generation_utils", sampling_code(release))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Pretrained.load(FIXTURES / "dream-tiny", dtype="float32", attention_impl="reference")
    mask = source.model.mask_token_id

    def held_out(step, x, scores):
        scores[..., mask] = -torch.inf
        return scores

    responses = []
    for index, prompt in enumerate(PROMPTS):
        width = len(prompt) + NEW_TOKENS
        stream, canvas = iter(uniforms(index, width)), {}

        def rand(*shape, device=None, stream=stream, canvas=canvas):
            # The step's uniforms at the positions masked now, in the loop's order.
            return torch.from_numpy(next(stream)[(canvas["x"][0] == mask).numpy()]).reshape(shape)

        def seen(step, x, scores, canvas=canvas):
            canvas["x"] = x
            return x

        config = module.DreamGenerationConfig(steps=STEPS, max_length=width, mask_token_id=mask,
                                              temperature=0.0, alg="origin", return_dict_in_generate=False)
        with mock.patch.object(torch, "rand", rand):
            out = module.DreamGenerationMixin._sample(_Released(source), torch.tensor([prompt]), None, config,
                                                      seen, held_out)
        responses.append(out[0, len(prompt):].tolist())
    return responses


def vendor() -> None:
    """Each release's `generation_utils.py`, pinned into the fetched store with its license."""
    from fetched_fixtures import pin_release

    for release, (fixture, repo, revision, license) in RELEASES.items():
        print(pin_release(FIXTURES / fixture / "release" / "generation_utils.py",
                          sampling_code(release).read_bytes(),
                          reference=f"{repo}@{revision}/generation_utils.py", license=license,
                          origin=release in AT_ORIGIN), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--vendor"]:
        vendor()
    else:
        recording = {"settings": {"steps": STEPS, "new_tokens": NEW_TOKENS},
                     "responses": {release: walked(release) for release in RELEASES}}
        (FIXTURES / "dream-tiny" / "origin.json").write_text(json.dumps(recording, indent=1) + "\n")
        print(json.dumps(recording))
