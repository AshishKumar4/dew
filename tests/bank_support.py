"""Nested decoder-bank fixtures and sources that record selected layer reads."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.inference.banks import LayerBanks, at_namespace, entry_tree, in_namespace, narrowed
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import DecoderBank
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import MultimodalTransformer, VisionConditioner
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.objectives.base import merge
from dew.training import Layout


def fixture(kind):
    text = CausalTransformer(vocab_size=32, emb_features=16, num_layers=4,
                             num_heads=2, num_kv_heads=2, head_dim=8, mlp_features=32,
                             max_seq_len=16, scan_layers=True, bank_layers=2,
                             dtype=jnp.float32, attention_impl="reference")
    tokens = jnp.asarray([[2, 1, 3, 4]], jnp.int32)
    indices = jnp.asarray([[-1, 0, -1, -1]], jnp.int32)
    media = {"pixel_values": jnp.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8)}
    vision = SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1,
                         num_heads=2, image_size=8, patch_size=4)
    projection = GemmaProjector(text_width=16,
                               patches_per_side=2, tokens_per_side=1)
    if kind == "root":
        return text, text.init(jax.random.key(0), tokens), tokens, indices, media
    if kind == "multimodal":
        model = MultimodalTransformer(text, vision, projection, family="gemma3",
                                      image_token_id=1, dtype=jnp.float32)
        variables = model.init(jax.random.key(0), tokens, image_indices=indices, conditioning=media)
        return model, variables, tokens, indices, media
    model = DiffusionGemma(text, canvas_length=2, conditioner=VisionConditioner(
        vision, projection, dtype=jnp.float32))

    def initialize(bound):
        bound(tokens)
        assert bound.conditioner is not None
        bound.conditioner(media)

    variables = model.init(jax.random.key(0), method=initialize)
    return model, {name: tree for name, tree in variables.items() if name != "cache"}, tokens, indices, media


def identical(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.dtype == reference.dtype
        np.testing.assert_array_equal(value, reference)


def unpack(store, sites, shapes):
    logical = narrowed(store, entry_tree(shapes, sites))
    for site in sites:
        logical = merge(logical, at_namespace(
            site.view.unstack(in_namespace(store, site.namespace)), site.namespace))
    return logical


def host_layout(site):
    return Layout(min_shard=1, tolerance=1.0,
                  host_parameters=("/".join(("params", *site.namespace, "layers_*")),))


class SelectedReads(LayerBanks):
    """A storage boundary that forbids transferring decoder layers as entries."""

    def __init__(self, source: LayerBanks, sites: tuple[DecoderBank, ...]):
        self.source = source
        self.sites = sites
        self.reads: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def shapes(self):
        return self.source.shapes()

    def entry(self, placement):
        for path, _ in jax.tree_util.tree_flatten_with_path(placement)[0]:
            keys = tuple(entry.key for entry in path)
            for site in self.sites:
                start = 1 + len(site.namespace)
                if keys[1:start] != site.namespace or len(keys) <= start:
                    continue
                layers = {f"layers_{index}" for first, count in site.view.groups
                          for index in range(first, first + count)}
                if keys[start] in layers:
                    raise AssertionError("a decoder layer was requested as a whole-device entry")
        return self.source.entry(placement)

    def bank(self, layers, placement, *, namespace=()):
        self.reads.append((namespace, tuple(layers)))
        return self.source.bank(layers, placement, namespace=namespace)


def scores(model, variables, tokens, indices, media):
    if isinstance(model, DiffusionGemma):
        cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]
        encoded, cache = model.apply(merge(variables, cache), tokens, method=model.encode,
                                     image_indices=indices, conditioning=media, mutable=["cache"])
        decoded, _ = model.apply(merge(variables, cache), tokens[:, :2], mutable=["cache"])
        return encoded, decoded
    if isinstance(model, MultimodalTransformer):
        return model.apply(variables, tokens, image_indices=indices, conditioning=media)
    return model.apply(variables, tokens)

