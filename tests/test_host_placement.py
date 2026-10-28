"""What a layout keeps in pinned host memory, and what reads it back.

`Layout.host` changes where the optimizer state and the EMA copy live, not
what they hold: the step fetches them to the device, runs the update it
always ran, and writes them back. JAX 0.11.1 types memory spaces, so a field
left on the host would fail the first device op that touched it, and a field
fetched but not written back would land on the device; the memory kinds on
the state leaves are asserted for that reason.

`Layout.host_parameters` keeps the layer stack's parameter banks there
instead, which only generation reads: the stack fetches one layer at a time
as it reaches it, so the values are the resident run's exactly and the device
memory is one layer's, not the model's. Those tests assert bitwise equality
against the same banks placed on the device, because where an array sits is
not a numerical choice, and they read the compiled module's own memory-space
assignment rather than trusting the placement they asked for.

The CPU backend exposes `pinned_host` beside `device`, so the placement, the
checkpoint round trip and the resume run here as written. What the CPU
backend does not do is account for the two spaces separately, since it has
one; the byte accounting of an offload is a GPU measurement, taken with
tools/benchmark_host_offload.py.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import BATCH, Counting, Features, Spread, val_batches
from process_support import run_pool

from dew.data.dataset import Loader
from dew.inference.banks import CheckpointBanks, HeldBanks
from dew.nn.backbones.decoder_stack import StackView
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling.text import Sampling, generate
from dew.training import Checkpoints, Layout, Trainer

DEVICE = Layout(min_shard=1, tolerance=1.0)
HOST = Layout(min_shard=1, tolerance=1.0, host=("opt_state", "ema"))
BANKS = Layout(min_shard=1, tolerance=1.0, host_parameters=("params/layers_*",))


def fit(layout, directory, steps):
    checkpoints = Checkpoints(str(directory), keep=3)
    # A validation pass needs a consumer, so the objective that scores
    # features stands in for the plain regression and a metric reads them.
    trainer = Trainer(Features(), optax.adam(0.1), key=jax.random.key(0), layout=layout,
                      checkpoints=checkpoints)
    data = Loader(
        train=lambda partition: Counting(), val=lambda partition: val_batches()(), records=None, batch=BATCH
    )
    state = trainer.fit(data, steps=steps, log_every=1, eval_every=2, checkpoint_every=2,
                        metrics=(Spread([]),))
    checkpoints.wait()
    return state


def identical(left, right):
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def memory_kinds(tree):
    return {leaf.sharding.memory_kind for leaf in jax.tree.leaves(tree)}


def test_host_resident_state_trains_the_same_model(tmp_path):
    on_device = fit(DEVICE, tmp_path / "device", 4)
    on_host = fit(HOST, tmp_path / "host", 4)
    assert memory_kinds(on_host.opt_state) == memory_kinds(on_host.ema) == {"pinned_host"}
    assert memory_kinds(on_host.variables) == memory_kinds(on_device.opt_state) == {"device"}
    assert identical(on_host.variables, on_device.variables)
    assert identical(on_host.opt_state, on_device.opt_state)
    assert identical(on_host.ema, on_device.ema)


def test_host_resident_state_resumes_from_its_checkpoint(tmp_path):
    saved = fit(HOST, tmp_path / "host", 4)
    fit(DEVICE, tmp_path / "device", 4)
    resumed = fit(HOST, tmp_path / "host", 6)
    reference = fit(DEVICE, tmp_path / "device", 6)
    assert int(resumed.step) == 6
    assert memory_kinds(resumed.opt_state) == memory_kinds(resumed.ema) == {"pinned_host"}
    assert identical(resumed.variables, reference.variables)
    assert identical(resumed.opt_state, reference.opt_state)
    assert identical(resumed.ema, reference.ema)
    assert not identical(resumed.variables, saved.variables)


# --------------------------------------------------------------------------
# The layer stack's parameter banks in pinned host memory
# --------------------------------------------------------------------------

VOCAB = 32
PROMPT = 5
SHAPE = {"vocab_size": VOCAB, "emb_features": 16, "num_heads": 4, "num_kv_heads": 2,
             "mlp_features": 32, "max_seq_len": 16}

# Every layer kind whose parameters a bank has to hold and a fetched run has
# to read: the dense block, the routed one, the two mixers with a state of
# their own, and the sparse indexer beside the latent attention. The last
# three shapes are about the grouping instead: runs of unequal length, a
# stack of nothing but runs of one, and one run cut into banks by
# bank_layers.
SHAPES = {
    "dense": {"num_layers": 4},
    "narrow": {"num_layers": 8, "emb_features": 4, "num_heads": 1, "num_kv_heads": 1, "mlp_features": 8},
    "moe": {"num_layers": 4, "mixture": {"experts": 4, "top_k": 2, "bias": True}},
    "gated_delta_net": {"num_layers": 4, "layer_types": ("linear_attention",) * 4,
                            "kinds": {"linear_attention": {
                                "mixer": {"class": "gated_delta_net", "fields": {}}}}},
    "latent_attention": {"num_layers": 4, "mixer": {
        "class": "mla", "fields": {"kv_lora_rank": 16, "q_lora_rank": 16, "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4, "v_head_dim": 8}}},
    "indexed_latent_attention": {"num_layers": 4, "mixer": {
        "class": "mla", "fields": {"kv_lora_rank": 16, "q_lora_rank": 16, "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4, "v_head_dim": 8, "index_n_heads": 2, "index_head_dim": 8,
        "index_topk": 4}}},
    "untied_head": {"num_layers": 4, "tie_embeddings": False},
    "unequal_runs": {"num_layers": 5, "layer_types": (
        "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"), "kinds": {"sliding_attention": {"window": 8}}},
    "runs_of_one": {"num_layers": 3, "layer_types": (
        "full_attention", "sliding_attention", "full_attention"),
        "kinds": {"sliding_attention": {"window": 8}}},
    "capped_banks": {"num_layers": 6, "bank_layers": 2},
}

# The scanned stack against the plain loop. Both are fp32 evaluations of the
# same logits, so they can differ by the sum of their errors against the
# exact values. Measured against a float64 evaluation of the same weights on
# CPU, the narrow shape's scanned logits are off by up to 7.3e-06 and its
# plain ones by up to 4.7e-06, depending on the ISA XLA compiles for
# (--xla_cpu_max_isa SSE4_2, AVX, AVX2); an AVX-512 CI runner measured a
# difference of 1.03e-05 between them. Twice the larger error, 1.5e-05,
# rounded up, is the bound.
# A fetched run reassociates nothing the scan does not; what it must not do
# is differ from the same banks on the device, which every case asserts
# bitwise.
SCAN_BOUND = 2e-5


def pair(**overrides):
    """The plain decoder, its banked twin, and the weights of both."""
    fields = {**SHAPE, **overrides}
    plain = models.build("causal_transformer", **fields)
    scanned = models.build("causal_transformer", **fields, scan_layers=True)
    tokens = jnp.asarray(
        np.random.default_rng(0).integers(1, VOCAB, size=(2, PROMPT)), jnp.int32)
    return plain, scanned, plain.init(jax.random.key(0), tokens), tokens


def stores(scanned, variables, **kwargs):
    """The same weights banked twice: resident, and with the layers on the host."""
    return (HeldBanks(variables).place(scanned, layout=DEVICE, **kwargs),
            HeldBanks(variables).place(scanned, layout=BANKS, **kwargs))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_host_resident_bank_scores_what_a_resident_bank_scores(shape):
    """The logits of a stack whose banks are in host memory, against the same
    banks on the device and against the plain loop.

    Bitwise against the resident banks: the fetch is a copy, and a copy that
    changed a value would show here whatever the layer kind, the grouping or
    the head. Against the plain loop only to the scan's own bound, which is
    where the loop's reassociation lives and is unchanged by the fetching.
    """
    plain, scanned, variables, tokens = pair(**SHAPES[shape])
    before = np.asarray(plain.apply(variables, tokens))
    resident, on_host = stores(scanned, variables)
    banks = [name for name in on_host["params"] if name.startswith("layers_")]
    assert memory_kinds({name: on_host["params"][name] for name in banks}) == {"pinned_host"}
    assert memory_kinds(resident["params"]) == {"device"}
    fetched = scanned.apply(on_host, tokens)
    assert np.array_equal(np.asarray(fetched), np.asarray(scanned.apply(resident, tokens)))
    assert all(not leaf.is_deleted() for leaf in jax.tree.leaves(variables))
    after = np.asarray(plain.apply(variables, tokens))
    np.testing.assert_array_equal(after, before)
    difference = float(jnp.max(jnp.abs(fetched - after)))
    assert difference < SCAN_BOUND, f"max |logit difference| {difference:.3e}"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_host_resident_bank_decodes_through_the_cache_it_allocates(shape):
    """A prefill and greedy steps through a fetched stack: the same
    continuation as the resident banks, out of a cache the fetched run
    allocated and wrote itself.

    The cache comes back out per layer with the batch on its first axis,
    which is what a caller between two steps holds and what the banking must
    not change.
    """
    _, scanned, variables, tokens = pair(**SHAPES[shape])
    resident, on_host = stores(scanned, variables)
    greedy = Sampling(temperature=0.0)
    fetched = generate(scanned, on_host, tokens, 4, key=0, sampling=greedy)
    assert np.array_equal(
        np.asarray(fetched.tokens),
        np.asarray(generate(scanned, resident, tokens, 4, key=0, sampling=greedy).tokens))

    cache = scanned.apply(on_host, 2, method="init_cache", mutable=["cache"])[1]["cache"]
    depth = SHAPES[shape]["num_layers"]
    assert sorted(cache) == sorted(f"layers_{index}" for index in range(depth))
    assert memory_kinds(cache) == {"device"}
    assert {leaf.shape[0] for leaf in jax.tree.leaves(cache)} == {2}


def test_a_layout_offloads_only_the_layers_it_names():
    """Banks the patterns do not name stay on the device, and a stack split
    between the two places computes what the resident one computes."""
    _plain, scanned, variables, tokens = pair(num_layers=4, bank_layers=2)
    selected = Layout(min_shard=1, tolerance=1.0,
                      host_parameters=("params/layers_0", "params/layers_1"))
    resident = HeldBanks(variables).place(scanned, layout=DEVICE)
    split = HeldBanks(variables).place(scanned, layout=selected)
    kinds = {name: memory_kinds(tree) for name, tree in split["params"].items()}
    assert kinds == {"embed_tokens": {"device"}, "norm": {"device"},
                     "layers_0_1": {"pinned_host"}, "layers_2_3": {"device"}}
    assert np.array_equal(np.asarray(scanned.apply(split, tokens)),
                          np.asarray(scanned.apply(resident, tokens)))


def test_banked_training_preserves_each_layers_dropout_rng_lineage():
    """Lifted staging and remat must not split or fold a different RNG stream.
    Compare every layer's two dropout masks to the original Linen scan, not
    merely repeated calls with the same root key."""
    from flax import linen as nn
    _, scanned, variables, tokens = pair(num_layers=4, dropout_rate=.4)
    _, on_host = stores(scanned, variables)
    captures = []
    for store in (variables, on_host):
        _, changed = scanned.apply(
            store, tokens, train=True, rngs={"dropout": jax.random.key(9)},
            capture_intermediates=lambda module, method: isinstance(module, nn.Dropout),
            mutable=["intermediates"])
        captures.append(changed["intermediates"])
    for index in range(4):
        assert len(captures[0][f"layers_{index}"]["dropout"]["__call__"]) == 2
    assert jax.tree.structure(captures[0]) == jax.tree.structure(captures[1])
    for a, b in zip(jax.tree.leaves(captures[0]), jax.tree.leaves(captures[1]), strict=True):
        np.testing.assert_array_equal(np.asarray(a) == 0, np.asarray(b) == 0)


def test_a_checkpoint_restores_bank_by_bank_into_host_memory(tmp_path):
    """A run's saved weights read one bank at a time, against the same
    checkpoint read into resident banks."""
    plain, scanned, _, tokens = pair(num_layers=8, bank_layers=4, emb_features=2,
                                      num_heads=1, num_kv_heads=1, mlp_features=4)
    directory = str(tmp_path / "run")
    checkpoints = Checkpoints(directory, keep=1)
    trainer = Trainer(LMObjective(plain, SHAPE["max_seq_len"] - 1, head_chunks=1),
                      optax.sgd(0.1), key=jax.random.PRNGKey(0), layout=DEVICE,
                      checkpoints=checkpoints)
    state, _, _ = trainer.place()
    batch = {"text": jnp.tile(
        jnp.arange(1, SHAPE["max_seq_len"] + 1, dtype=jnp.int32)[None],
        (jax.device_count(), 1))}
    state = trainer.compile(state, batch)(state, batch)[0]
    checkpoints.save(int(state.step), state, None, metrics={"loss": 1.0})
    checkpoints.wait()

    resident = CheckpointBanks(directory).place(scanned, layout=DEVICE)
    on_host = CheckpointBanks(directory).place(scanned, layout=BANKS)
    assert memory_kinds(on_host["params"]["layers_0_3"]) == {"pinned_host"}
    assert np.array_equal(np.asarray(scanned.apply(on_host, tokens)),
                          np.asarray(scanned.apply(resident, tokens)))
    trained = StackView(scanned.bind({}).groups).unstack(resident)
    assert identical(trained, state.variables)


# --------------------------------------------------------------------------
# A real pool: sharded banks, in host memory, on more than one process
# --------------------------------------------------------------------------

WORKER = Path(__file__).with_name("host_banks_worker.py")



@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_reads_its_own_shards_of_a_host_resident_bank(tmp_path):
    """Two real processes, a mesh split over fsdp and tensor, and the layer
    banks in pinned host memory.

    Each process holds the shards of the banks its own devices address and no
    others, fetches those to its own devices and issues the collectives the
    resident run issues: the logits and the greedy continuation are the
    resident placement's, bitwise, shard for shard. Then the pool writes a
    checkpoint and reads it back bank by bank onto the same placement.
    """
    reports = run_pool("banked", tmp_path / "pool", processes=2, script=WORKER, devices=2, fsdp_size=2,
                       tensor_size=2, bank_layers=2, run_dir=str(tmp_path / "run"))
    for report in reports:
        assert report["processes"] == 2 and report["devices"] == 4
        assert report["local_devices"] == 2
        assert report["banks"] == ["embed_tokens", "layers_0_1", "layers_2_3", "norm"]
        assert report["logits_equal"] and report["logits_difference"] == 0.0
        assert report["tokens_equal"]
        assert report["restored_equal"] and report["restored_difference"] == 0.0
        assert report["restored_is_trained"]
        banks = {path: value for path, value in report["host_placement"].items()
                 if "layers_" in path}
        assert {value["kind"] for value in banks.values()} == {"pinned_host"}
        assert {value["addressable"] for value in banks.values()} == {2}
        # The layer axis is whole in front of the spec the rules gave the
        # leaf, and the shard is smaller than the bank on at least one of the
        # parameter axes.
        assert all(value["spec"].startswith("P(None") for value in banks.values())
        assert any(value["shard"] != value["shape"] for value in banks.values())


@pytest.mark.parametrize("kind", ["root", "multimodal"])
def test_synthetic_banks_keep_namespace_values_across_bank_sizes(kind):
    from bank_support import SelectedReads, fixture, host_layout, identical, scores, unpack

    from dew.nn.multimodal import MultimodalTransformer
    from tools.benchmark_host_offload import SyntheticBanks

    model, variables, tokens, indices, media = fixture(kind)
    sites = model.bank_sites(variables)
    source = SyntheticBanks(HeldBanks(variables).shapes(), seed=17)
    resident = source.place(model, layout=DEVICE)
    selected = SelectedReads(source, sites)
    hosted = selected.place(model, layout=host_layout(sites[0]))
    identical(scores(model, hosted, tokens, indices, media),
              scores(model, resident, tokens, indices, media))
    one = (model.clone(language_model=model.language_model.clone(bank_layers=1))
           if isinstance(model, MultimodalTransformer) else model.clone(bank_layers=1))
    separate = source.place(one, layout=DEVICE)
    identical(unpack(separate, one.bank_sites(variables), source.shapes()),
              unpack(resident, sites, source.shapes()))
