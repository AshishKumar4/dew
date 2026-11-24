"""A user's own decoder module served: packing reaches the projections it names at their paths in its
variables, a tree split for training packs as its whole tree does, a projection quantized apart from its
group stays unpacked, and a server sized below the module's context refuses what passes its own capacity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import Digits, decoder, serving_task as task
from serving_support import assert_same_draws

from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.nn.protocols import held_groups
from dew.sampling import Sampling

PROMPTS = ["12", "5", "1234567", "98", "321"]
"""The prompts each test serves, `tests/test_serving.py`'s."""


class UserDecoder(nn.Module):
    """A user's own decoder module, no Dew class: a plain module around a
    decoder that forwards the calls a task and a server make, and answers
    nothing a server asks of a model."""

    decoder: nn.Module

    @property
    def vocab_size(self):
        return self.decoder.vocab_size

    @property
    def max_seq_len(self):
        return self.decoder.max_seq_len

    def init_cache(self, batch_size):
        self.decoder.init_cache(batch_size)

    def __call__(self, tokens, **fields):
        return self.decoder(tokens, **fields)

    def states_and_logits(self, tokens, **fields):
        return self.decoder.states_and_logits(tokens, **fields)

    def states_and_logits_at(self, tokens, slots, **fields):
        return self.decoder.states_and_logits_at(tokens, slots, **fields)


class AnsweringDecoder(UserDecoder):
    """`UserDecoder`, answering what a server asks of a model from the decoder it holds."""

    @nn.nowrap
    def with_cache_capacity(self, capacity):
        return self.clone(decoder=self.decoder.with_cache_capacity(capacity))

    @nn.nowrap
    def mixed_admission_refusal(self):
        return self.decoder.mixed_admission_refusal()

    @nn.nowrap
    def split_prefill_refusal(self):
        return self.decoder.split_prefill_refusal()

    @property
    def cache_rebuild_position(self):
        return self.decoder.cache_rebuild_position

    @nn.nowrap
    def inference_projection_groups(self, variables):
        return held_groups(self.decoder, "decoder", variables)


def wrapped(kind, model, variables, sampling, processor=None):
    """A task over `kind` around `model`, its `variables` under `decoder`."""
    held = {collection: {"decoder": tree} for collection, tree in variables.items()}
    return TextGeneration(kind(model), held, processor, sampling=sampling)


def test_a_user_decoder_module_is_packed_where_it_names_its_groups():
    """Packing reaches the projections a model names (`Serving`)
    at their paths in its variables: a user's module around a decoder that
    names the decoder's is packed, draws what the unpacked weights draw, and
    is served and reloaded from them; one that names none keeps its layout."""
    from dew.nn.projections import inference_projections

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    answering = wrapped(AnsweringDecoder, bound.model, bound.variables, bound.sampling, bound.processor)
    plain = wrapped(UserDecoder, bound.model, bound.variables, bound.sampling, bound.processor)
    members = {"q_proj", "k_proj", "v_proj"}
    packed = inference_projections(answering.model, answering.variables)
    attention = set(packed["params"]["decoder"]["layers_0"]["self_attn"])
    assert "qkv_proj" in attention and not members & attention
    assert members <= set(
        inference_projections(plain.model, plain.variables)["params"]["decoder"]["layers_0"]["self_attn"])
    served = TextGeneration(answering.model, packed, bound.processor, sampling=bound.sampling)
    for index, prompt in enumerate(PROMPTS[:3]):
        assert_same_draws(served(prompt, 5, key=index), bound(prompt, 5, key=index))
    server = Server.from_task(served, slots=2, capacity=128)
    trained = jax.tree.map(lambda leaf: leaf * 1.5, answering.variables)
    server.reload(trained)
    retrained = wrapped(UserDecoder, bound.model, jax.tree.map(lambda leaf: leaf * 1.5, bound.variables),
                        bound.sampling, bound.processor)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_draws(actual, retrained(prompt, 5, key=3))


def test_a_tree_split_for_training_packs_as_its_whole_tree_and_splits_back_as_it_was():
    """A tree split for training (`dew.batch.freeze`), its first layer's keys and values frozen and its
    queries trained and its second layer trained whole, packs as its whole tree does (ledger 2439): thawed,
    it is the unsplit weights packed, the flags of its frozen members (`PACKED_FROZEN`) gone with the
    split, so a model reads the same tree whatever its layers' membership; split, it is the tree it was,
    each member in its own collection; and a server of it reloads its unsplit source weights."""
    from dew.batch import PACKED_FROZEN, freeze, thaw
    from dew.nn.projections import inference_projections, packed_owned, unpacked

    model = decoder().clone(num_layers=2, max_seq_len=128)
    whole = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    split = freeze(whole, lambda path: not ("layers_0" in path and {"k_proj", "v_proj"} & set(path)))
    packed = packed_owned(model, jax.tree.map(jnp.array, split))
    assert PACKED_FROZEN in packed and PACKED_FROZEN not in thaw(packed)

    def assert_trees_equal(tree, expected):
        assert jax.tree.structure(tree) == jax.tree.structure(expected)
        for leaf, alike in zip(jax.tree.leaves(tree), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(alike), strict=True)

    assert_trees_equal(thaw(packed), inference_projections(model, whole))
    assert_trees_equal(unpacked(model, packed), split)
    sampling = Sampling(temperature=0, eos_token_ids=None)
    server = Server.from_task(TextGeneration(model, packed, RunProcessor(Digits()), sampling=sampling),
                              slots=2, capacity=128)
    server.reload(whole)
    alone = TextGeneration(model, whole, RunProcessor(Digits()), sampling=sampling)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_draws(actual, alone(prompt, 5, key=3))


def test_a_projection_quantized_apart_from_its_group_is_served_unpacked():
    """A model quantized by its modules' paths (`Quantization.patterns`) selecting its query
    projection alone keeps its query, key and value apart, each quantized as the pattern selects it,
    and packs its gate and up projections, which no pattern selects: a server draws what the task
    draws on its own. Packed, the query was served as `qkv_proj`, which the pattern does not
    select, so unquantized."""
    from dew.nn.projections import projection_groups
    from dew.quantization import Quantization

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    quantized = Quantization(patterns=(".*/q_proj",)).apply(bound.model)
    groups = {group.packed for group in projection_groups(quantized, bound.variables)}
    assert "qkv_proj" not in groups and "gate_up_proj" in groups
    alone = TextGeneration(quantized, bound.variables, bound.processor, sampling=bound.sampling)
    server = Server.from_task(alone, slots=2, capacity=128)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_draws(actual, alone(prompt, 5, key=3))


def test_a_request_past_the_servers_own_capacity_is_refused_at_submit():
    """A user's decoder declares a context of 128 and is served at a
    capacity of 64: a request that fits 64 draws what the task draws alone,
    and a budget or a prompt past it is refused at submission, leaving the
    server serving."""
    bound = task(Sampling(temperature=0, eos_token_ids=None))
    plain = wrapped(UserDecoder, bound.model, bound.variables, bound.sampling, bound.processor)
    server = Server.from_task(plain, slots=2, capacity=64)
    assert plain.model.max_seq_len == 128
    with pytest.raises(ValueError):
        server.submit("12", 63, key=0)
    with pytest.raises(ValueError):
        server.submit("1" * 64, 1, key=0)
    ticket = server.submit("12", 62, key=0)
    server.run()
    assert_same_draws(ticket.result(), plain("12", 62, key=0))
