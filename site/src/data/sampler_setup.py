from functools import cache

import jax.numpy as jnp

from dew.interop import PretrainedDecoder
from dew.sampling import CFG, DPMSolverMultistep, EulerAncestral, Heun, Sampling, TextToImage


@cache
def from_pretrained(repo_id, *, revision=None):
    return TextToImage.from_pretrained(repo_id, revision=revision)


@cache
def text_model(name):
    model = PretrainedDecoder.load(f"/opt/models/{name}", dtype=jnp.float32, max_seq_len=256)
    return model.text_generation(sampling=Sampling(temperature=0))


pipe = from_pretrained("dewml/hybrid-dit-176m", revision="434d7e8a940a906ef5d422b31d12b965944a5562")
