"""Canonical nested decoder banks, shared ownership, and selected source reads."""
import shutil
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from bank_support import SelectedReads, fixture, host_layout, identical, scores, unpack

from dew.inference.banks import CheckpointBanks, HeldBanks, at_namespace, entry_tree, in_namespace, narrowed
from dew.nn.backbones.decoder_stack import DecoderBank, StackView
from dew.objectives.base import merge
from dew.training import Checkpoints, Layout
from dew.training.state import TrainState

DEVICE = Layout(min_shard=1, tolerance=1.0)





@pytest.mark.parametrize("kind", ["root", "multimodal", "shared-diffusion"])
def test_declared_banks_preserve_canonical_values_layout_and_model_reads(kind):
    model, variables, tokens, indices, media = fixture(kind)
    sites = model.bank_sites
    source = SelectedReads(HeldBanks(variables), sites)
    resident = HeldBanks(variables).place(model, layout=DEVICE)
    hosted = source.place(model, layout=host_layout(sites[0]))
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
    store = source.place(repeated, layout=host_layout(site))
    assert len(source.reads) == len(site.view.groups)
    identical(unpack(store, model.bank_sites, source.shapes()), variables)
    reference = HeldBanks(variables).place(model, layout=DEVICE)
    identical(scores(model, store, tokens, indices, media),
              scores(model, reference, tokens, indices, media))
    conflicting = DecoderBank(site.namespace, StackView(((0, 1), (1, 3))))
    with pytest.raises(ValueError, match="conflicting decoder views"):
        HeldBanks(variables).place(DeclaredOwners((site, conflicting)), layout=DEVICE)


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
    state = TrainState(step=zero, microstep=zero, updates=zero, variables=variables,
                       opt_state=(), ema=averaged, key=jax.random.PRNGKey(0),
                       scale=None, window_size=jnp.asarray(1, jnp.int32))
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    source = SelectedReads(CheckpointBanks(str(tmp_path), ema=True), model.bank_sites)
    restored = source.place(model, layout=host_layout(site))
    expected = merge(variables, averaged)
    identical(unpack(restored, model.bank_sites, source.shapes()), expected)
    resident = HeldBanks(expected).place(model, layout=DEVICE)
    identical(scores(model, restored, tokens, indices, media),
              scores(model, resident, tokens, indices, media))


def test_a_checkpoint_written_again_at_a_path_is_read_as_it_now_is(tmp_path):
    """A bank source reads the checkpoint that is at its path when it is
    built. One saved where an earlier run's was deleted, at the same step,
    has to be read as itself, not with the earlier one's shapes, which is
    what pytest reusing a deleted test's directory name exposed."""
    run = tmp_path / "run"
    zero = jnp.asarray(0, jnp.int32)
    for kind in ("multimodal", "shared-diffusion"):
        _, variables, _, _, _ = fixture(kind)
        state = TrainState(step=zero, microstep=zero, updates=zero, variables=variables,
                           opt_state=(), ema=None, key=jax.random.PRNGKey(0),
                           scale=None, window_size=jnp.asarray(1, jnp.int32))
        if run.exists():
            shutil.rmtree(run)
        checkpoints = Checkpoints(str(run))
        checkpoints.save(0, state, None)
        checkpoints.wait()
        shapes = CheckpointBanks(str(run)).shapes()
        assert jax.tree.structure(shapes) == jax.tree.structure(variables), kind


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
        MetadataOnly(variables).place(wrong, layout=DEVICE)
    scope = dict(in_namespace(variables, site.namespace)["params"])
    scope.pop("layers_1")
    broken = {**variables, "params": {**variables["params"], "language_model": scope}}
    with pytest.raises(ValueError, match="holds layers"):
        MetadataOnly(broken).place(model, layout=DEVICE)


def test_nested_entry_offload_cannot_treat_a_decoder_owner_as_one_fetch():
    model, variables, _, _, _ = fixture("multimodal")
    with pytest.raises(ValueError, match="stack does not fetch"):
        HeldBanks(variables).place(model, layout=Layout(
            min_shard=1, tolerance=1.0, host_parameters=("params/language_model/*",)))


def test_an_unscanned_decoder_banks_one_layer_at_a_time():
    """A plain loop declares singleton runs, which stream like longer ones."""
    model, variables, tokens, indices, media = fixture("multimodal")
    unscanned = model.clone(language_model=model.language_model.clone(scan_layers=False))
    (site,) = unscanned.bank_sites
    assert site.view.groups == tuple((index, 1) for index in range(4))
    source = SelectedReads(HeldBanks(variables), unscanned.bank_sites)
    resident = HeldBanks(variables).place(unscanned, layout=DEVICE)
    hosted = source.place(unscanned, layout=host_layout(site))
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
