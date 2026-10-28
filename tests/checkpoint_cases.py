"""The tiny checkpoints Dew's registry-wide model tests run: each one under tests/fixtures/hf whose config
names a registered decoder family, itself or as a wrapper's decoder (`CHECKPOINTS`), the smallest of
each code path (`PATHS`), each loaded once (`loaded`) and read once (`reads`). tests/test_model_protocols.py
holds each one's reads to its forward, and tests/test_remat_rungs.py each one's remat rungs to it."""

import functools
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.interop import Pretrained, hf_decoders
from dew.nn.protocols import OutputTable
from dew.registry import models

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
ROWS, LENGTH = 2, 9


def config_of(name: str) -> dict:
    return json.loads((FIXTURES / name / "config.json").read_text())


def is_checkpoint(directory: Path) -> bool:
    """Whether `directory` is a tiny source of a registered decoder family: a
    decoder's config, or a wrapper's whose text_config names its decoder. A
    tower fixture's text_config states the width its projector reads and no
    decoder (mimo-vision-tiny, under MiMo-V2's own model_type)."""
    if not (directory / "config.json").is_file() or not (directory / "model.safetensors").is_file():
        return False
    config, families = config_of(directory.name), hf_decoders.families()
    text = config.get("text_config") or {}
    return ("architectures" in config or "model_type" in text) and (
        config.get("model_type") in families or text.get("model_type") in families)


CHECKPOINTS = sorted(directory.name for directory in FIXTURES.iterdir() if is_checkpoint(directory))
OWN = [name for name in CHECKPOINTS if config_of(name)["model_type"] in models]
"""The checkpoints of a Dew model of their own, DiffusionGemma's."""
DECODERS = [name for name in CHECKPOINTS if name not in OWN]


def _layers(value, found: set[str], under: bool = False) -> set[str]:
    """The mixer classes a translated record names, by the `class` of each
    record under `mixer` or `kinds` (KDA beside MLA is not a delta net beside
    attention), and its layer types."""
    if isinstance(value, dict):
        for key, item in value.items():
            if under and key == "class" and isinstance(item, str):
                found.add(item)
            _layers(item, found, under or key in ("mixer", "kinds"))
            if key == "layer_types" and isinstance(item, (list, tuple)):
                found.update(map(str, item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _layers(item, found, under)
    return found


@functools.cache
def code_path(name: str) -> tuple:
    """The code paths a checkpoint takes: the model it loads into, its token
    mixers and layer types, and its routing, head and attention direction.
    A family whose fields only its own translator reads is its own path."""
    config = config_of(name)
    wrapper = config["model_type"] if name in OWN or config.get("text_config") else "decoder"
    try:
        record = dict(hf_decoders.translate_config(config.get("text_config") or config))
    except ValueError:
        return wrapper, name
    return (wrapper, frozenset(_layers(record, set())),
            (record.get("mixture") is not None, bool(record.get("tie_embeddings", True)),
             record.get("head_transform") is not None, bool(record.get("head_bias")),
             bool(record.get("causal", True))))


def smallest_of_each(names) -> list[str]:
    """The smallest checkpoint of each code path (`code_path`), in name order."""
    chosen: dict[object, str] = {}
    for name in sorted(names, key=lambda name: (FIXTURES / name / "model.safetensors").stat().st_size):
        chosen.setdefault(code_path(name), name)
    return sorted(chosen.values())


PATHS = smallest_of_each(CHECKPOINTS)
"""One checkpoint per wrapper, mixer and layer set, routing, head and direction."""


@functools.cache
def loaded(name: str, dtype: str = "float32") -> Pretrained:
    return Pretrained.load(FIXTURES / name, dtype=dtype, attention_impl="reference")


class Reads(NamedTuple):
    tokens: jax.Array
    logits: jax.Array
    hidden: jax.Array
    table: OutputTable | None


@functools.cache
def reads(name: str, dtype: str) -> Reads:
    source = loaded(name, dtype)
    model, variables = source.model, source.variables
    tokens = jnp.asarray(np.random.RandomState(0).randint(0, model.vocab_size, (ROWS, LENGTH)), jnp.int32)
    return Reads(tokens, model.apply(variables, tokens, method="logits"),
                 model.apply(variables, tokens, method="hidden_states"),
                 model.apply(variables, method="output_table"))
