"""Dreamer 4's tokenizer trained as tutorial 12 trains it, on a GPU: the scale at which its reconstruction is
judged takes minutes there and hours on a CPU."""

import ast
import itertools
import json
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from test_world_model import assert_lights_what_it_shows

from dew.data.dataset import Loader
from dew.objectives.world_model import CausalTokenizer, TokenizerObjective
from dew.training import Trainer

pytestmark = pytest.mark.gpu(reason="tutorial 12's tokenizer at its own scale, 8,000 updates of 32 clips")

TUTORIAL = Path(__file__).parent.parent / "tutorials" / "12-world-models.ipynb"


def tutorial_function(name: str):
    """The function `name` tutorial 12 defines, as its notebook's cell defines it: its clips are the
    tutorial's own, drawn by its code."""
    for cell in json.loads(TUTORIAL.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                scope = {"np": np}
                exec(compile(ast.Module([node], []), str(TUTORIAL), "exec"), scope)
                return scope[name]
    raise LookupError(f"tutorial 12 defines no {name}")


def test_tutorial_12s_tokenizer_reconstructs_the_balls_it_was_trained_on():
    """Tutorial 12's tokenizer, trained as the tutorial trains it on its own clips (2,048 clips of 12 frames
    of two bouncing balls, 32 clips an update, Adam 3e-4) for 8,000 updates, reconstructs the balls of 8
    held-out clips (`assert_lights_what_it_shows`). Its tokens know their places by Dreamer 4's rotary
    positions (`positions`, row 4669): by learned vectors alone it reconstructed the mean frame (an H100 at
    fdd1da52c9: squared error 0.1406 against the mean frame's 0.1417), where rotary positions reach 0.0067
    and a correlation with the frames of 0.976."""
    bouncing = tutorial_function("bouncing")
    videos, held = bouncing(2048, 12, 32, 0), bouncing(8, 12, 32, 1)
    clips = Loader(train=lambda partition: ({"video": videos[start:start + 32]}
                                            for start in itertools.cycle(range(0, 2048, 32))),
                   val=None, records=2048, batch=32)
    tokenizer = CausalTokenizer((32, 32, 1), width=128, depth=3, latents=16, bottleneck=8)
    state = Trainer(TokenizerObjective(tokenizer), optax.adam(3e-4), key=jax.random.key(9)).fit(
        clips, steps=8000, log_every=4000)
    assert_lights_what_it_shows(tokenizer, state.variables, held.reshape(-1, 32, 32, 1))
