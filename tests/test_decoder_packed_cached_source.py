"""Packed rows and cached decoding against transformers' own Llama and Mistral.

transformers' `LlamaForCausalLM` and `MistralForCausalLM` run in the test on the committed llama-tiny
and mistral-tiny checkpoints (tools/hf_packed_cached_reference.py): packed rows that restart their
positions per document, from which transformers builds its own packed mask (each document's logits its
logits alone), and a prompt prefilled into its cache and then fed a token a step, past Mistral's sliding
window (each step the full sequence's logits). Dew's decoder loaded from the same checkpoints, on both
attention kernels, is on par with transformers' float32 run from its float64 twin's over drawn ids
(check 1).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from interop_support import transformers_arms, transformers_model
from reference_error import assert_as_exact
from test_causal_transformer import decode_logits

from dew.interop.pretrained import Pretrained

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

PACKED_SEEDS = 0
"""Each family's declared draws of packed rows against transformers' packed forward (check 1), sized by
`seeds_for` paired with Dew's packed logits on the CPU."""

CACHED_SEEDS = 0
"""Each family's declared draws of decoded rows against transformers' cached steps (check 1), sized by
`seeds_for` paired with Dew's cached steps on the CPU."""


def segments(packing) -> tuple[np.ndarray, np.ndarray]:
    """Each packed row's segment ids and positions, documents numbered from one."""
    ids = [[index + 1 for index, length in enumerate(row) for _ in range(length)] for row in packing]
    positions = [[step for length in row for step in range(length)] for row in packing]
    return np.asarray(ids, np.int32), np.asarray(positions, np.int32)


def drawn(family: str, shape):
    """Check 1's draw of `shape` ids over the family's vocabulary, the same on every run."""
    vocabulary = json.loads((FIXTURES / family / "config.json").read_text())["vocab_size"]
    return lambda seed: (np.random.default_rng(seed).integers(0, vocabulary, shape).astype(np.int32),)


@pytest.mark.parametrize("attention_impl", ["reference", "xla"])
@pytest.mark.parametrize("family", ["llama-tiny", "mistral-tiny"])
def test_packed_rows_are_as_exact_as_transformers(family, attention_impl):
    from tools.hf_packed_cached_reference import PACKING, packed

    loaded = Pretrained.load(FIXTURES / family, dtype="float32", attention_impl=attention_impl)
    segment_ids, positions = segments(PACKING)
    reference, truth = transformers_arms(transformers_model(FIXTURES / family), packed)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids, positions=positions,
                                                   segment_ids=segment_ids),
                    reference, drawn(family, segment_ids.shape), f"{family} packed", seeds=PACKED_SEEDS,
                    truth_fn=truth)


@pytest.mark.parametrize("attention_impl", ["reference", "xla"])
@pytest.mark.parametrize("family", ["llama-tiny", "mistral-tiny"])
def test_cached_decoding_is_as_exact_as_transformers(family, attention_impl):
    from tools.hf_packed_cached_reference import PROMPT, STEPS, cached

    loaded = Pretrained.load(FIXTURES / family, dtype="float32", attention_impl=attention_impl)
    reference, truth = transformers_arms(transformers_model(FIXTURES / family), cached)
    def ours(ids):
        return decode_logits(loaded.model, loaded.variables, ids[:, :PROMPT], ids[:, PROMPT:])

    assert_as_exact(ours, reference, drawn(family, (2, PROMPT + STEPS)), f"{family} cached",
                    seeds=CACHED_SEEDS, truth_fn=truth)
