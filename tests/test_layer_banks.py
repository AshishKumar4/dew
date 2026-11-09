"""Canonical nested decoder banks, shared ownership, and selected source reads."""
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from bank_support import SelectedReads, fixture, host_layout, identical, scores, unpack

from dew.batch import merge
from dew.inference.banks import CheckpointBanks, HeldBanks, at_namespace, entry_tree, in_namespace, narrowed
from dew.nn.backbones.decoder_stack import DecoderBank, StackView
from dew.state import TrainState
from dew.training import Checkpoints, Layout

DEVICE = Layout(min_shard=1, tolerance=1.0)





@pytest.mark.parametrize("kind", ["root", "multimodal", "shared-diffusion"])
def test_declared_banks_preserve_canonical_values_layout_and_model_reads(kind):
    model, variables, tokens, indices, media = fixture(kind)
    sites = model.bank_sites(variables)
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


def test_a_distillations_source_banks_its_student_and_reads_its_teacher_whole():
    """A source places a distillation's store as its student's runs banked and its frozen teacher's
    tree as it is stored, deeper than the student's: the nested model's stacks are training's host
    execution's to bank (`DecoderBank.root`), and nothing of the student's is read under it."""
    from flax import linen as nn

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.base import TEACHER
    from dew.objectives.distillation import DistillationObjective
    from dew.objectives.lm import LMObjective

    def decoder(layers):
        return CausalTransformer(vocab_size=16, emb_features=8, num_layers=layers, num_heads=2,
                                 mlp_features=16, max_seq_len=8, scan_layers=True)

    objective = DistillationObjective(LMObjective(decoder(2), 8, ema_decay=None),
                                      LMObjective(decoder(3), 8, ema_decay=None))
    variables = nn.unbox(objective.init(jax.random.key(0)))
    placed = HeldBanks(variables).place(objective, layout=DEVICE)
    student = objective.student.bank_sites(objective.student_variables(variables))
    identical(placed[TEACHER], variables[TEACHER])
    identical(unpack(objective.student_variables(placed), student, objective.student_variables(variables)),
              objective.student_variables(variables))


@dataclass(frozen=True)
class DeclaredOwners:
    sites: tuple[DecoderBank, ...]

    def bank_sites(self, store):
        return self.sites


def test_shared_decoder_readers_allocate_one_canonical_owner_and_refuse_conflicts():
    model, variables, tokens, indices, media = fixture("shared-diffusion")
    (site,) = model.bank_sites(variables)
    repeated = DeclaredOwners((site, site))
    source = SelectedReads(HeldBanks(variables), model.bank_sites(variables))
    store = source.place(repeated, layout=host_layout(site))
    assert len(source.reads) == len(site.view.groups)
    identical(unpack(store, model.bank_sites(variables), source.shapes()), variables)
    reference = HeldBanks(variables).place(model, layout=DEVICE)
    identical(scores(model, store, tokens, indices, media),
              scores(model, reference, tokens, indices, media))
    conflicting = DecoderBank(site.namespace, StackView(((0, 1), (1, 3))))
    with pytest.raises(ValueError):
        HeldBanks(variables).place(DeclaredOwners((site, conflicting)), layout=DEVICE)


def first_leaf(tree):
    if isinstance(tree, Mapping):
        name = next(iter(tree))
        return {name: first_leaf(tree[name])}
    return tree + jnp.asarray(0.125, tree.dtype)


@pytest.mark.parametrize("kind", ["multimodal", "shared-diffusion"])
def test_nested_checkpoint_banks_restore_only_selected_leaves_and_partial_ema(tmp_path, kind):
    model, variables, tokens, indices, media = fixture(kind)
    (site,) = model.bank_sites(variables)
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
    source = SelectedReads(CheckpointBanks(str(tmp_path), ema=True), model.bank_sites(variables))
    restored = source.place(model, layout=host_layout(site))
    expected = merge(variables, averaged)
    identical(unpack(restored, model.bank_sites(variables), source.shapes()), expected)
    resident = HeldBanks(expected).place(model, layout=DEVICE)
    identical(scores(model, restored, tokens, indices, media),
              scores(model, resident, tokens, indices, media))


def test_an_unbound_decoder_plans_its_banks_from_a_store_whose_rows_differ():
    """A store whose third layer holds a leaf its neighbours lack (a checkpoint that adapts or quantizes
    one layer in part) banks that layer in a run of its own, as the decoder bound to the store scans it
    (`CausalTransformer.bank_sites`), and places and scores as the resident store does; planned from
    the decoder's specs alone, its run of like specs stacked the unlike rows and was refused."""
    model, variables, tokens, indices, media = fixture("root")
    params = variables["params"]
    adapted = {**params, "layers_2": {**params["layers_2"], "part": jnp.ones((3,), jnp.float32)}}
    store = {**variables, "params": adapted}
    (site,) = model.bank_sites(store)
    assert site.view.groups == model.bind(store).groups
    assert (2, 1) in site.view.groups
    hosted = SelectedReads(HeldBanks(store), (site,)).place(model, layout=host_layout(site))
    resident = HeldBanks(store).place(model, layout=DEVICE)
    identical(unpack(hosted, (site,), HeldBanks(store).shapes()), store)
    identical(scores(model, hosted, tokens, indices, media), scores(model, resident, tokens, indices, media))


def test_an_unscanned_decoder_banks_one_layer_at_a_time():
    """A plain loop declares singleton runs, which stream like longer ones."""
    model, variables, tokens, indices, media = fixture("multimodal")
    unscanned = model.clone(language_model=model.language_model.clone(scan_layers=False))
    (site,) = unscanned.bank_sites(variables)
    assert site.view.groups == tuple((index, 1) for index in range(4))
    source = SelectedReads(HeldBanks(variables), unscanned.bank_sites(variables))
    resident = HeldBanks(variables).place(unscanned, layout=DEVICE)
    hosted = source.place(unscanned, layout=host_layout(site))
    identical(unpack(hosted, unscanned.bank_sites(variables), source.shapes()), variables)
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
