"""Packed rows and cached decoding against transformers' own Llama and Mistral.

tools/hf_packed_cached_reference.py runs transformers' `LlamaForCausalLM`
and `MistralForCausalLM` on the committed llama-tiny and mistral-tiny
checkpoints: packed rows that restart their positions per document, from
which transformers builds its own packed mask (each document's logits its
logits alone), and a prompt prefilled into its cache and then fed a token a
step, past Mistral's sliding window (each step the full sequence's logits).
In float32 and in float64 (tests/fixtures/hf/packed_cached.npz). Dew's
decoder loaded from the same checkpoints, on both attention kernels, is held
to tests/reference_error.py's rule: no further from float64 than twice
transformers' float32.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from test_causal_transformer import decode_logits

from dew.interop.pretrained import Pretrained

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURES / "packed_cached.npz") as loaded:
        arrays = dict(loaded)
    return arrays, json.loads(arrays.pop("meta").tobytes())


def segments(packing) -> tuple[np.ndarray, np.ndarray]:
    """Each packed row's segment ids and positions, documents numbered from one."""
    ids = [[index + 1 for index, length in enumerate(row) for _ in range(length)] for row in packing]
    positions = [[step for length in row for step in range(length)] for row in packing]
    return np.asarray(ids, np.int32), np.asarray(positions, np.int32)


@pytest.mark.parametrize("attention_impl", ["reference", "xla"])
@pytest.mark.parametrize("family", ["llama-tiny", "mistral-tiny"])
def test_packed_rows_are_as_exact_as_transformers(reference, family, attention_impl):
    arrays, meta = reference
    loaded = Pretrained.load(FIXTURES / family, dtype="float32", attention_impl=attention_impl)
    segment_ids, positions = segments(meta["packing"])
    logits = loaded.model.apply(loaded.variables, jnp.asarray(arrays[f"{family}/packed_ids"]),
                                positions=jnp.asarray(positions), segment_ids=jnp.asarray(segment_ids))
    assert_as_exact_as_the_reference(np.asarray(logits), arrays[f"{family}/fp32.packed"],
                                     arrays[f"{family}/fp64.packed"], f"{family} packed")


@pytest.mark.parametrize("attention_impl", ["reference", "xla"])
@pytest.mark.parametrize("family", ["llama-tiny", "mistral-tiny"])
def test_cached_decoding_is_as_exact_as_transformers(reference, family, attention_impl):
    arrays, meta = reference
    loaded = Pretrained.load(FIXTURES / family, dtype="float32", attention_impl=attention_impl)
    ids = jnp.asarray(arrays[f"{family}/decode_ids"])
    prompt = meta["prompt"]
    logits = decode_logits(loaded.model, loaded.variables, ids[:, :prompt], ids[:, prompt:])
    assert_as_exact_as_the_reference(np.asarray(logits), arrays[f"{family}/fp32.cached"],
                                     arrays[f"{family}/fp64.cached"], f"{family} cached")
