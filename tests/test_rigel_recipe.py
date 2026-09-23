"""recipes/lm/rigel.py: the published parameter counts."""

import importlib.util
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from dew.registry import models

PATH = Path(__file__).resolve().parents[1] / "recipes" / "lm" / "rigel.py"


def load_rigel():
    spec = importlib.util.spec_from_file_location("recipe_rigel", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rigel_has_its_published_parameter_counts():
    """2,345,567,552 in all and 260,998,464 active outside the embeddings, the
    lm-eval record's count and the 6N of its 5.4e21 FLOPs."""
    rigel = load_rigel()
    model = models.build("causal_transformer", **rigel.model_config(1024),
                         vocab_size=rigel.VOCAB, max_seq_len=4096)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))
    total = active = embedding = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(shapes["params"]):
        names, count = [entry.key for entry in path], math.prod(leaf.shape)
        total += count
        embedding += count if "embed_tokens" in names else 0
        active += count * 2 // 128 if "experts" in names else count
    assert (total, active - embedding) == (2_345_567_552, 260_998_464)
