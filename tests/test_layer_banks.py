"""Canonical nested decoder banks, shared ownership, and selected source reads."""
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference.banks import (
    CheckpointBanks, HeldBanks, LayerBanks, at_namespace, entry_tree, host_banked,
    in_namespace, narrowed,
)
from dew.nn.backbones.causal_transformer import CausalTransformer, DecoderBank, StackView
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import MultimodalTransformer, VisionConditioner
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.objectives.base import merge
from dew.training import Checkpoints, Layout
from dew.training.state import TrainState

DEVICE = Layout(min_shard=1, tolerance=1.0)


def fixture(kind):
    text = CausalTransformer(vocab_size=32, emb_features=16, num_layers=4,
                             num_heads=2, num_kv_heads=2, head_dim=8, mlp_features=32,
                             max_seq_len=16, scan_layers=True, bank_layers=2,
                             dtype=jnp.float32, attention_impl=None)
    tokens = jnp.asarray([[2, 1, 3, 4]], jnp.int32)
    indices = jnp.asarray([[-1, 0, -1, -1]], jnp.int32)
    media = {"pixel_values": jnp.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8)}
    vision = SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1,
                         num_heads=2, image_size=8, patch_size=4)
    projection = GemmaProjector(vision_width=16, text_width=16,
                               patches_per_side=2, tokens_per_side=1)
    if kind == "root":
        return text, text.init(jax.random.key(0), tokens), tokens, indices, media
    if kind == "multimodal":
        model = MultimodalTransformer(text, vision, projection, family="gemma3",
                                      image_token_id=1, dtype=jnp.float32)
        variables = model.init(jax.random.key(0), tokens, image_indices=indices, conditioning=media)
        return model, variables, tokens, indices, media
    model = DiffusionGemma(text, canvas_length=2, conditioner=VisionConditioner(
        "gemma3", vision, projection, dtype=jnp.float32))

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


class SelectedReads:
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


@pytest.mark.parametrize("kind", ["root", "multimodal", "shared-diffusion"])
def test_declared_banks_preserve_canonical_values_layout_and_model_reads(kind):
    model, variables, tokens, indices, media = fixture(kind)
    sites = model.bank_sites
    source = SelectedReads(HeldBanks(variables), sites)
    resident = host_banked(model, HeldBanks(variables), layout=DEVICE)
    hosted = host_banked(model, source, layout=host_layout(sites[0]))
    identical(unpack(hosted, sites, source.shapes()), variables)
    identical(narrowed(hosted, entry_tree(source.shapes(), sites)),
              narrowed(resident, entry_tree(source.shapes(), sites)))
    for leaf in jax.tree.leaves(narrowed(hosted, entry_tree(source.shapes(), sites))):
        assert leaf.sharding.memory_kind == "device"
    for site in sites:
        host_local = in_namespace(hosted, site.namespace)
        device_local = in_namespace(resident, site.namespace)
        for name in site.view.bank_names():
            for value, device in zip(jax.tree.leaves(host_local["params"][name]),
                                     jax.tree.leaves(device_local["params"][name]), strict=True):
                assert value.sharding.memory_kind == "pinned_host"
                assert value.sharding.spec == device.sharding.spec
                assert value.shape[0] == 2
    identical(scores(model, hosted, tokens, indices, media),
              scores(model, resident, tokens, indices, media))
    assert source.reads == [(site.namespace, tuple(range(first, first + count)))
                            for site in sites for first, count in site.view.groups]


@dataclass(frozen=True)
class DeclaredOwners:
    bank_sites: tuple[DecoderBank, ...]


def test_shared_decoder_readers_allocate_one_canonical_owner_and_refuse_conflicts():
    model, variables, tokens, indices, media = fixture("shared-diffusion")
    (site,) = model.bank_sites
    repeated = DeclaredOwners((site, site))
    source = SelectedReads(HeldBanks(variables), model.bank_sites)
    store = host_banked(repeated, source, layout=host_layout(site))
    assert len(source.reads) == len(site.view.groups)
    identical(unpack(store, model.bank_sites, source.shapes()), variables)
    reference = host_banked(model, HeldBanks(variables), layout=DEVICE)
    identical(scores(model, store, tokens, indices, media),
              scores(model, reference, tokens, indices, media))
    conflicting = DecoderBank(site.namespace, StackView(((0, 1), (1, 3))))
    with pytest.raises(ValueError, match="conflicting decoder views"):
        host_banked(DeclaredOwners((site, conflicting)), HeldBanks(variables), layout=DEVICE)


def first_leaf(tree):
    if isinstance(tree, Mapping):
        name = next(iter(tree))
        return {name: first_leaf(tree[name])}
    return tree + jnp.asarray(0.125, tree.dtype)


@pytest.mark.parametrize("kind", ["multimodal", "shared-diffusion"])
def test_nested_checkpoint_banks_restore_only_selected_leaves_and_partial_ema(tmp_path, kind):
    model, variables, tokens, indices, media = fixture(kind)
    (site,) = model.bank_sites
    local = in_namespace(variables, site.namespace)["params"]
    averaged = at_namespace({"params": {
        "layers_0": first_leaf(local["layers_0"]),
        "embed_tokens": first_leaf(local["embed_tokens"]),
    }}, site.namespace)
    zero = jnp.asarray(0, jnp.int32)
    state = TrainState(step=zero, microstep=zero, updates=zero, params=variables,
                       opt_state=(), ema=averaged, key=jax.random.PRNGKey(0),
                       scale=None, window_size=jnp.asarray(1, jnp.int32))
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    source = SelectedReads(CheckpointBanks(str(tmp_path), ema=True), model.bank_sites)
    restored = host_banked(model, source, layout=host_layout(site))
    expected = merge(variables, averaged)
    identical(unpack(restored, model.bank_sites, source.shapes()), expected)
    resident = host_banked(model, HeldBanks(expected), layout=DEVICE)
    identical(scores(model, restored, tokens, indices, media),
              scores(model, resident, tokens, indices, media))


def test_a_missing_nested_owner_or_layer_is_rejected_before_entry_transfer():
    model, variables, _, _, _ = fixture("multimodal")
    (site,) = model.bank_sites

    class MetadataOnly(HeldBanks):
        def entry(self, placement):
            raise AssertionError("invalid ownership reached an entry read")

        def bank(self, layers, placement, *, namespace=()):
            raise AssertionError("invalid ownership reached a bank read")

    wrong = DeclaredOwners((DecoderBank(("missing",), site.view),))
    with pytest.raises(ValueError, match="holds layers"):
        host_banked(wrong, MetadataOnly(variables), layout=DEVICE)
    scope = dict(in_namespace(variables, site.namespace)["params"])
    scope.pop("layers_1")
    broken = {**variables, "params": {**variables["params"], "language_model": scope}}
    with pytest.raises(ValueError, match="holds layers"):
        host_banked(model, MetadataOnly(broken), layout=DEVICE)


def test_nested_entry_offload_cannot_treat_a_decoder_owner_as_one_fetch():
    model, variables, _, _, _ = fixture("multimodal")
    with pytest.raises(ValueError, match="stack does not fetch"):
        host_banked(model, HeldBanks(variables), layout=Layout(
            min_shard=1, tolerance=1.0, host_parameters=("params/language_model/*",)))


def test_an_unscanned_decoder_banks_one_layer_at_a_time():
    """A plain loop declares singleton runs, which stream like longer ones."""
    model, variables, tokens, indices, media = fixture("multimodal")
    unscanned = model.clone(language_model=model.language_model.clone(scan_layers=False))
    (site,) = unscanned.bank_sites
    assert site.view.groups == tuple((index, 1) for index in range(4))
    source = SelectedReads(HeldBanks(variables), unscanned.bank_sites)
    resident = host_banked(unscanned, HeldBanks(variables), layout=DEVICE)
    hosted = host_banked(unscanned, source, layout=host_layout(site))
    identical(unpack(hosted, unscanned.bank_sites, source.shapes()), variables)
    layers = in_namespace(hosted, site.namespace)["params"]
    device_layers = in_namespace(resident, site.namespace)["params"]
    for index in range(4):
        for leaf, row in zip(jax.tree.leaves(layers[f"layers_{index}"]),
                             jax.tree.leaves(device_layers[f"layers_{index}"]), strict=True):
            assert leaf.sharding.memory_kind == "pinned_host"
            assert leaf.shape == row.shape
    identical(scores(unscanned, hosted, tokens, indices, media),
              scores(unscanned, resident, tokens, indices, media))
    assert source.reads == [(site.namespace, (index,)) for index in range(4)]
