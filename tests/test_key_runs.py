"""Key runs (`AttentionMetadata.key_runs`, the template's spans' `runs`): each query's key intervals, whose
union it reads, in place of a pairwise mask. The decision layout's runs (`dew.decision.model.key_runs`) read
exactly what its pairwise mask reads, causal or not, over packed rows and padding; a layer reading them
computes what it computes over that mask, a windowed layer windowing each run over the key positions, whose
places restart at each option (a mixer that reads no runs refuses them: `test_mixers`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER

from dew.decision.layout import _parallel
from dew.decision.model import key_runs
from dew.nn.backbones.causal_transformer import CausalTransformer

VOCAB = 37
OPTIONS = 4


def layout(rows: list[list[tuple[int, int, int, int]]], length: int) -> tuple[np.ndarray, ...]:
    """Rows of packed decision documents, each `(prefix, options, width, suffix)` tokens, then padding:
    their `slots`, `segments` and `positions` `[B, length]`, as `layout._parallel` lays a document out (its
    places restart at each option, the suffix after the widest)."""
    slots, segments, positions = (np.zeros((len(rows), length), np.int32) for _ in range(3))
    for row, documents in enumerate(rows):
        at = 0
        for number, (prefix, options, width, suffix) in enumerate(documents, start=1):
            size = prefix + options * width + suffix
            spans = [(prefix + k * width, prefix + (k + 1) * width) for k in range(options)]
            places, held = _parallel(spans, size)
            slots[row, at:at + size], positions[row, at:at + size] = held, places
            segments[row, at:at + size] = number
            at += size
        # Padding continues the last place, as a packed window pads.
        positions[row, at:] = np.arange(length - at) + (positions[row, at - 1] + 1 if at else 0)
    return slots, segments, positions


def visible(slots, segments, *, causal: bool) -> np.ndarray:
    """What `DecisionModel._read` gives the backbone, `[B, L, L]`: a token reads its own document's
    tokens, the padding its own, and an option no other option's; under `causal` none after it."""
    other = (slots[:, :, None] > 0) & (slots[:, None, :] > 0) & (slots[:, :, None] != slots[:, None, :])
    seen = (segments[:, :, None] == segments[:, None, :]) & ~other
    if causal:
        order = np.arange(slots.shape[1])
        seen &= order[None, :, None] >= order[None, None, :]
    return seen


def dense(runs: np.ndarray, keys: int) -> np.ndarray:
    """The keys `runs` `[B, Q, R, 2]` let each query read, `[B, Q, keys]`."""
    index = np.arange(keys)[None, None, :, None]
    return np.any((index >= runs[:, :, None, :, 0]) & (index < runs[:, :, None, :, 1]), axis=-1)


ROWS = [[(2, 4, 3, 4), (3, 2, 2, 2)], [(1, 3, 4, 3)]]
"""Two rows: Sol's document (a prefix of two, four options of three, a suffix) packed with a smaller one,
and one of three options beside trailing padding."""
LENGTH = 32


@pytest.mark.parametrize("causal", [False, True], ids=["bidirectional", "causal"])
def test_the_decision_runs_read_what_its_pairwise_mask_reads(causal):
    """`key_runs` over packed documents and padding equals `_read`'s mask, bitwise, and keeps its static
    count of runs, the option capacity and two."""
    slots, segments, _ = layout(ROWS, LENGTH)
    # The layout's options lie back to back in each document, no shared token between two of them.
    for row in range(len(slots)):
        for document in set(segments[row].tolist()) - {0}:
            options = np.nonzero((segments[row] == document) & (slots[row] > 0))[0]
            assert np.all(slots[row, options.min():options.max() + 1] > 0)
    runs = np.asarray(key_runs(jnp.asarray(slots), jnp.asarray(segments), OPTIONS, causal=causal))
    assert runs.shape == (2, LENGTH, OPTIONS + 2, 2)
    np.testing.assert_array_equal(dense(runs, LENGTH), visible(slots, segments, causal=causal))


def model(**kinds):
    """A two-layer float32 decoder of the kinds `kinds` names, in turn, on xla's kernel."""
    types = tuple(kinds) * (TINY_DECODER["num_layers"] // len(kinds))
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32", "layer_types": types,
                                "kinds": kinds, "attention_impl": "xla"})


@pytest.mark.parametrize("causal", [False, True], ids=["bidirectional", "causal"])
@pytest.mark.parametrize("kinds", [{"full_attention": {}}, {"sliding_attention": {"window": 2}}],
                         ids=["full", "window"])
def test_a_layer_reading_key_runs_computes_what_it_computes_over_the_same_mask(kinds, causal):
    """A decoder given the layout's runs and key positions computes what it computes given its pairwise
    mask and key positions, its logits and QK-Clip maxima alike, bitwise on the one kernel: under a window
    of two the shared prefix's second token, at place 1, reads places 0 to 2 of the prefix and of each of
    the four options, which restart at 2, five runs a single interval could not hold."""
    slots, segments, positions = layout(ROWS, LENGTH)
    decoder = model(**kinds)
    ids = jax.random.randint(jax.random.key(0), (2, LENGTH), 0, VOCAB)
    params = decoder.init(jax.random.key(1), ids)
    runs = key_runs(jnp.asarray(slots), jnp.asarray(segments), OPTIONS, causal=causal)
    places = jnp.asarray(positions)

    def read(**visibility):
        return decoder.apply(params, ids, positions=places, attention_key_positions=places, mutable=["qk"],
                             method=CausalTransformer.hidden_states, **visibility)

    ran, sown = read(attention_key_runs=runs)
    masked, maxima = read(attention_pairwise_mask=jnp.asarray(visible(slots, segments, causal=causal)))
    np.testing.assert_array_equal(np.asarray(ran), np.asarray(masked))
    jax.tree.map(lambda mine, theirs: np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs)),
                 sown, maxima)


