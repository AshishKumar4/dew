"""A source's decoding controls are held to what its model declares.

`source_decoding` checks a generation config's cache length and its
multi-token prediction request against the capacity and the prediction
depths the model declares (`dew.sampling.text.Bounded` and `Predicting`),
which are what the sampler reads, so a registered decoder and a module no
registry names are read alike.
"""

import flax.linen as nn
import pytest

from dew.interop.generation_config import source_decoding
from dew.registry import models
from dew.sampling.strategies import Speculative


class Drafter(nn.Module):
    """A decoder of the user's own, declaring its vocabulary, its cache
    capacity and its prediction depths."""

    vocab_size: int = 8
    max_seq_len: int = 16
    num_nextn_predict_layers: int = 1

    @nn.compact
    def __call__(self, tokens):
        return nn.Embed(self.vocab_size, self.vocab_size)(tokens)


DECODERS = {
    "registered": lambda depths: models.build(
        "causal_transformer", vocab_size=8, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
        max_seq_len=16, num_nextn_predict_layers=depths),
    "user": lambda depths: Drafter(num_nextn_predict_layers=depths),
}


@pytest.mark.parametrize("decoder", list(DECODERS.values()), ids=list(DECODERS))
def test_a_source_asking_for_prediction_depths_drafts_with_the_ones_its_model_declares(decoder):
    _, _, strategy = source_decoding({}, {"use_mtp": True, "num_assistant_tokens": 2}, decoder(1), 1, None)
    assert strategy == Speculative(block=3)
    with pytest.raises(ValueError, match="no prediction-depth weights"):
        source_decoding({}, {"use_mtp": True}, decoder(0), 1, None)


@pytest.mark.parametrize("decoder", list(DECODERS.values()), ids=list(DECODERS))
def test_a_source_cache_length_is_held_to_the_capacity_its_model_declares(decoder):
    policy, _, _ = source_decoding({}, {"max_cache_len": 16}, decoder(0), 1, None)
    assert policy == source_decoding({}, {}, decoder(0), 1, None)[0]
    with pytest.raises(ValueError, match="max_cache_len 17 exceeds the model's max_seq_len 16"):
        source_decoding({}, {"max_cache_len": 17}, decoder(0), 1, None)
