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
from flax import linen as nn
from model_support import float64_twin
from placement_parity import layouts_agree, x64
from process_support import run_pool
from reference_error import assert_as_exact
from selection_margins import MARGIN, top_margin

from dew.data.dataset import Loader
from dew.inference.banks import CheckpointBanks, HeldBanks
from dew.nn.backbones.decoder_stack import StackView
from dew.nn.kv_cache import row_axis
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling.text import Sampling, generate
from dew.training import Checkpoints, Layout, Trainer

DEVICE = Layout(min_shard=1, tolerance=1.0)
HOST = Layout(min_shard=1, tolerance=1.0, host=("opt_state", "ema"))
BANKS = Layout(min_shard=1, tolerance=1.0, host_parameters=("params/layers_*",))


def fit(layout, directory, steps, optimizer=None, **options):
    checkpoints = Checkpoints(str(directory), keep=3)
    # A validation pass needs a consumer, so the objective that scores
    # features stands in for the plain regression and a metric reads them.
    trainer = Trainer(Features(), optimizer or optax.adam(0.1), key=jax.random.key(0), layout=layout,
                      checkpoints=checkpoints, **options)
    data = Loader(
        train=lambda partition: Counting(), val=lambda partition: val_batches()(), records=None, batch=BATCH
    )
    state = trainer.fit(data, steps=steps, log_every=1, eval_every=2, checkpoint_every=2,
                        metrics=(Spread([]),))
    checkpoints.wait()
    return state


def trained(layout, seed, steps, optimizer=None, **options):
    """`steps` updates of the affine model from the start `seed` draws, without evaluation or checkpoints:
    a check-1 arm's run (`layouts_agree`), in float64 where the process runs in it."""
    wide = x64()

    def stream(partition):
        def held(value):
            floating = np.issubdtype(np.asarray(value).dtype, np.floating)
            return np.asarray(value, np.float64) if wide and floating else value
        return (jax.tree.map(held, batch) for batch in Counting())

    trainer = Trainer(Features(), optimizer or optax.adam(0.1), key=jax.random.key(int(seed)),
                      layout=layout, param_dtype="float64" if wide else None, **options)
    return trainer.fit(Loader(train=stream, val=None, records=None, batch=BATCH), steps=steps,
                       log_every=steps)


def identical(left, right):
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def memory_kinds(tree):
    return {leaf.sharding.memory_kind for leaf in jax.tree.leaves(tree)}


def test_host_resident_state_trains_the_same_model(tmp_path):
    """A run whose optimizer state and average sit in host memory trains the model, moments and average
    the device-resident run trains, by check 1 over drawn starts, the device's run in float64 the truth."""
    layouts_agree(lambda layout, seed: trained(layout, seed, 4), lambda seed: (np.int32(seed),),
                  "four updates with the optimizer state and average in host memory", HOST, DEVICE)
    on_device = fit(DEVICE, tmp_path / "device", 4)
    on_host = fit(HOST, tmp_path / "host", 4)
    assert memory_kinds(on_host.opt_state) == memory_kinds(on_host.ema) == {"pinned_host"}
    assert memory_kinds(on_host.variables) == memory_kinds(on_device.opt_state) == {"device"}


def test_host_resident_state_trains_the_same_model_over_a_scaled_window(tmp_path):
    """A window of two microbatches commits under a conditional, and a dynamically scaled loss admits it
    on the finiteness of the whole committed state: each reads every leaf of the host-resident state at
    once, which its commit then stages as one chunk (`dew.training.staging`), and a clipped Adam's run
    trains the model, moments and average the device-resident run trains, by check 1 over drawn starts."""
    options = {"optimizer": optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.1)), "accumulation": 2,
               "dynamic_scale": True}
    layouts_agree(lambda layout, seed: trained(layout, seed, 4, **options), lambda seed: (np.int32(seed),),
                  "a scaled window of two microbatches with the state in host memory", HOST, DEVICE)
    on_host = fit(HOST, tmp_path / "host", 4, **options)
    assert memory_kinds(on_host.opt_state) == memory_kinds(on_host.ema) == {"pinned_host"}


def test_host_resident_state_resumes_from_its_checkpoint(tmp_path):
    """A host-resident run resumed from its checkpoint continues as the run that never stopped, bit for bit:
    one program over restored values."""
    saved = fit(HOST, tmp_path / "host", 4)
    resumed = fit(HOST, tmp_path / "host", 6)
    reference = fit(HOST, tmp_path / "whole", 6)
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

HOST_SCAN_SEEDS = 32
"""The drawn prompts a fetched stack's logits are judged over against the plain loop's from float64
(check 1, `assert_as_exact`)."""


def pair(**overrides):
    """The plain decoder, its banked twin, and the weights of both."""
    fields = {**SHAPE, **overrides}
    plain = models.build("causal_transformer", **fields)
    scanned = models.build("causal_transformer", **fields, scan_layers=True)
    tokens = jnp.asarray(
        np.random.default_rng(0).integers(1, VOCAB, size=(2, PROMPT)), jnp.int32)
    return plain, scanned, nn.unbox(plain.init(jax.random.key(0), tokens)), tokens


def stores(scanned, variables, **kwargs):
    """The same weights banked twice: resident, and with the layers on the host."""
    return (HeldBanks(variables).place(scanned, layout=DEVICE, **kwargs),
            HeldBanks(variables).place(scanned, layout=BANKS, **kwargs))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_host_resident_bank_scores_what_a_resident_bank_scores(shape):
    """The logits of a stack whose banks are in host memory, against the same
    banks on the device and against the plain loop.

    On par from float64 over drawn prompts (check 1) with the resident banks,
    whose program differs by the fetch's copies alone (another compiled HLO,
    so no common bits are promised), and with the plain loop: the scan
    reassociates where the loop does not, and the fetching adds nothing to it
    (row 3950: an absolute bound measured on one machine stood in for the
    check).
    """
    plain, scanned, variables, tokens = pair(**SHAPES[shape])
    before = np.asarray(plain.apply(variables, tokens))
    resident, on_host = stores(scanned, variables)
    banks = [name for name in on_host["params"] if name.startswith("layers_")]
    assert memory_kinds({name: on_host["params"][name] for name in banks}) == {"pinned_host"}
    assert memory_kinds(resident["params"]) == {"device"}
    assert all(not leaf.is_deleted() for leaf in jax.tree.leaves(variables))
    after = np.asarray(plain.apply(variables, tokens))
    np.testing.assert_array_equal(after, before)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth = jax.jit(float64_twin(plain).apply)

    def draw(seed):
        return (np.random.default_rng(seed).integers(1, VOCAB, size=tokens.shape).astype(np.int32),)

    def wide_logits(drawn):
        with jax.enable_x64(new_val=True):
            return truth(wide, jnp.asarray(drawn))

    # The fetched stack called as a caller calls it, its host-resident banks fetched per call.
    assert_as_exact(lambda drawn: scanned.apply(on_host, jnp.asarray(drawn)),
                    jax.jit(lambda drawn: plain.apply(variables, drawn)), draw,
                    f"{shape}: a fetched stack's logits", seeds=HOST_SCAN_SEEDS, truth_fn=wide_logits,
                    also={"resident banks": lambda drawn: scanned.apply(resident, jnp.asarray(drawn))})


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_host_resident_bank_decodes_through_the_cache_it_allocates(shape):
    """A prefill and greedy steps through a fetched stack: the same
    continuation as the resident banks on each row whose every step's choice
    is decided, at least one, its top logit clearing the next by the selection margin
    (`selection_margins.MARGIN`; the two compile two programs, whose
    rounding may pick another token at a near tie), out of a cache the
    fetched run allocated and wrote itself.

    The cache comes back out as the store holds the stack, each run's
    layers one bank under the run's name and a run of one per layer, the
    batch after a bank's layers and first elsewhere (`row_axis`): what a
    caller between two steps holds, which the next step reads and writes
    where it lies.
    """
    _, scanned, variables, tokens = pair(**SHAPES[shape])
    resident, on_host = stores(scanned, variables)
    greedy = Sampling(temperature=0.0)
    fetched = generate(scanned, on_host, tokens, 4, key=0, sampling=greedy).host()
    held = generate(scanned, resident, tokens, 4, key=0, sampling=greedy).host()
    # Each step's logits: the resident run's prompt and continuation read back whole, its last token aside.
    decisions = np.asarray(scanned.apply(resident, jnp.asarray(held.tokens[:, :-1])),
                           np.float64)[:, held.prompt_width - 1:]
    decided = np.all(top_margin(decisions, 1) > MARGIN, axis=-1)
    assert decided.any(), f"{shape}: no row's continuation is decided at every step"
    np.testing.assert_array_equal(fetched.tokens[decided], held.tokens[decided])

    cache = scanned.apply(on_host, 2, method="init_cache", mutable=["cache"])[1]["cache"]
    assert sorted(cache) == sorted(name for name in on_host["params"] if name.startswith("layers_"))
    assert memory_kinds(cache) == {"device"}
    assert {leaf.shape[row_axis(path)] for path, leaf in jax.tree_util.tree_leaves_with_path(cache)} == {2}


def placements_agree(model, placed, resident, tokens, label: str):
    """`model`'s logits from the `placed` store on par from float64 over drawn prompts shaped as `tokens`
    (check 1) with its logits from the `resident` one, the resident store in float64 the truth: the two
    compile two programs, so no common bits are promised."""
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), jax.device_get(resident))
        truth = jax.jit(float64_twin(model).apply)

    def draw(seed):
        return (np.random.default_rng([4405, seed]).integers(1, VOCAB, size=tokens.shape).astype(np.int32),)

    def wide_logits(drawn):
        with jax.enable_x64(new_val=True):
            return truth(wide, jnp.asarray(drawn))

    assert_as_exact(lambda drawn: model.apply(placed, jnp.asarray(drawn)),
                    lambda drawn: model.apply(resident, jnp.asarray(drawn)), draw, label,
                    seeds=HOST_SCAN_SEEDS, truth_fn=wide_logits)


def test_a_layout_offloads_only_the_layers_it_names():
    """Banks the patterns do not name stay on the device, and a stack split
    between the two places computes what the resident one computes (check 1)."""
    _plain, scanned, variables, tokens = pair(num_layers=4, bank_layers=2)
    selected = Layout(min_shard=1, tolerance=1.0,
                      host_parameters=("params/layers_0", "params/layers_1"))
    resident = HeldBanks(variables).place(scanned, layout=DEVICE)
    split = HeldBanks(variables).place(scanned, layout=selected)
    kinds = {name: memory_kinds(tree) for name, tree in split["params"].items()}
    assert kinds == {"embed_tokens": {"device"}, "norm": {"device"},
                     "layers_0_1": {"pinned_host"}, "layers_2_3": {"device"}}
    placements_agree(scanned, split, resident, tokens, "a stack split between host and device")


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
    """A run's saved weights read one bank at a time compute what the same
    checkpoint read into resident banks computes (check 1), and unstack to the
    run's weights bit for bit: values moved, not computed."""
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
    # Saved as a run saves it, each layer stored (`Trainer.stored`), as its step holds it stacked.
    checkpoints.save(int(state.step), trainer.stored(state), None, metrics={"loss": 1.0})
    checkpoints.wait()

    resident = CheckpointBanks(directory).place(scanned, layout=DEVICE)
    on_host = CheckpointBanks(directory).place(scanned, layout=BANKS)
    assert memory_kinds(on_host["params"]["layers_0_3"]) == {"pinned_host"}
    placements_agree(scanned, on_host, resident, tokens, "a checkpoint read bank by bank into host memory")
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


def test_a_banked_store_packed_for_serving_or_adapted_exports_as_its_layers_do(tmp_path):
    """A scanned decoder's banked store (`HeldBanks.place`) exports as the plain decoder's per-layer tree
    does, tensor for tensor: packed for serving (`inference_projections`, its q, k and v one kernel a bank's
    row splits back as it is written, `StoredRow.split`), and under a LoRA adapter the export merges into
    its kernels as it writes them (`Adapter.merge` over a bank's rows, `Deferred`, ledger 4103), and the
    adapter's own save writes its factors as the plain decoder's adapter does (ledger 4566)."""
    from safetensors.numpy import load_file

    from dew.batch import thaw
    from dew.inference.projections import inference_projections
    from dew.interop.pretrained import PretrainedDecoder
    from dew.lora import LoRA

    plain, scanned, variables, _ = pair(num_layers=3)
    banked = HeldBanks(variables).place(scanned, layout=DEVICE)

    def written(model, tree, name):
        PretrainedDecoder.from_model(model, tree).save(tmp_path / name)
        return load_file(tmp_path / name / "model.safetensors")

    expected = written(plain, variables, "plain")
    packed = written(scanned, inference_projections(scanned, banked), "packed")
    assert packed.keys() == expected.keys()
    for key, tensor in expected.items():
        np.testing.assert_array_equal(packed[key], tensor, err_msg=key)
    # Factors drawn whole, so each merge moves its kernel (a fresh adapter's B is zero), the base and the
    # factors one collection, as the bank holds them.
    spec = LoRA(rank=2, modules=("q_proj", "v_proj"))
    lone = PretrainedDecoder.from_model(plain, variables).adapt(spec, key=1)
    stacked = PretrainedDecoder.from_model(scanned, variables).adapt(spec, key=1)
    rng = np.random.default_rng(4103)
    tuned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.asarray(rng.normal(size=leaf.shape) / 4, leaf.dtype)
        if str(getattr(path[-1], "key", "")) == "lora_B" else leaf, thaw(lone.variables))
    lone.save(tmp_path / "lone", variables=tuned)
    stacked.save(tmp_path / "adapted", variables=HeldBanks(tuned).place(stacked.model, layout=DEVICE))
    merged, adapted = (load_file(tmp_path / name / "model.safetensors") for name in ("lone", "adapted"))
    assert adapted.keys() == merged.keys()
    for key, tensor in merged.items():
        np.testing.assert_array_equal(adapted[key], tensor, err_msg=key)
    # The adapter's own save reads its factors by their layers' names too, a bank's rows read home as they
    # are written (ledger 4566).
    assert lone.adapter is not None and stacked.adapter is not None
    lone.adapter.save(tuned, tmp_path / "factors")
    stacked.adapter.save(HeldBanks(tuned).place(stacked.model, layout=DEVICE), tmp_path / "banked_factors")
    factors, banked = (load_file(tmp_path / name / "adapter_model.safetensors")
                       for name in ("factors", "banked_factors"))
    assert banked.keys() == factors.keys()
    for key, tensor in factors.items():
        np.testing.assert_array_equal(banked[key], tensor, err_msg=key)


class RecordedBank:
    """A bank that records the shape of each piece of it read to the host (`np.asarray`), its pieces taken
    by indexing as an array's are."""

    def __init__(self, held: np.ndarray, read: list):
        self.held, self.read = held, read
        self.shape, self.dtype = held.shape, held.dtype

    def __getitem__(self, key):
        return RecordedBank(self.held[key], self.read)

    def __array__(self, dtype=None, copy=None):
        self.read.append(self.held.shape)
        return self.held if dtype is None else self.held.astype(dtype)


def test_a_packed_rows_members_each_read_home_their_own_share():
    """A bank's packed row split into its members (`StoredRow.split`, a packed projection's q, k and v) has
    each member sliced where the bank lies and read home alone, its own width of the row, so a row's
    members read it home once between them rather than once each (ledger 4574), each as the row holds it."""
    from dew.nn.backbones.layer_names import StoredRow

    held, read = np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7), []
    members = StoredRow(RecordedBank(held, read), (1,)).split([2, 4, 1])
    for member, (start, stop) in zip(members, ((0, 2), (2, 6), (6, 7)), strict=True):
        np.testing.assert_array_equal(np.asarray(member), held[1][..., start:stop])
    assert read == [(3, 2), (3, 4), (3, 1)]


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
