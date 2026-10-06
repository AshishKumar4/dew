"""A source's decoding controls are held to what its model declares
(`dew.sampling.text.Bounded` and `Predicting`), so a module no registry
names is read as a registered decoder is (tests/test_inference_sampling.py)."""

import flax.linen as nn
import pytest

from dew.interop.generation_config import source_decoding
from dew.sampling.strategies import Speculative


class Drafter(nn.Module):
    """A decoder of the user's own, declaring its cache capacity and its prediction depths."""

    vocab_size: int = 8
    max_seq_len: int = 16
    num_nextn_predict_layers: int = 1

    @nn.compact
    def __call__(self, tokens):
        return nn.Embed(self.vocab_size, self.vocab_size)(tokens)


def test_a_user_decoders_declared_depths_and_capacity_hold_its_sources_decoding():
    _, _, strategy = source_decoding({}, {"use_mtp": True, "num_assistant_tokens": 2}, Drafter(), 1, None)
    assert strategy == Speculative(block=3)
    with pytest.raises(ValueError, match="no prediction-depth weights"):
        source_decoding({}, {"use_mtp": True}, Drafter(num_nextn_predict_layers=0), 1, None)
    policy, _, _ = source_decoding({}, {"max_cache_len": 16}, Drafter(), 1, None)
    assert policy == source_decoding({}, {}, Drafter(), 1, None)[0]
    with pytest.raises(ValueError, match="max_cache_len 17 exceeds the model's max_seq_len 16"):
        source_decoding({}, {"max_cache_len": 17}, Drafter(), 1, None)
