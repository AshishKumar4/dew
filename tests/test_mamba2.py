"""Mamba-2 against transformers' recorded (tools/mamba2_reference.py, `write_seeded`): its scan
(`mamba2_chunk_scan`, `mamba2_selective_state_update`), one `Mamba2Mixer` layer and a seeded
`Mamba2ForCausalLM` (modeling_mamba2.py) on seeded draws at each site (`ssd_support`'s), in fp32 (bf16 for
the layer too) with their float64 truth. Each of Dew's forms (chunked or recurrent, a cut sequence, chunk
sizes, decoding, padding, packing, forward mode, the model's padded rows and decode) is held to the site's
recording (check 1, `assert_recorded_as_exact`), and a computation that leaves out what a site checks moves
past its rounding (`assert_recorded_moves`). Nothing here imports torch.
"""


import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import decode_through
from reference_error import (
    assert_as_exact,
    assert_moves,
    declared_seeds,
    indexed_recording,
    recording,
    widened,
    with_gradients,
)
from sequence_split_support import SPLIT_ROWS, assert_split_as_whole
from ssd_support import (
    CUTS,
    FIXTURES,
    LAYER_GEOMETRY,
    LAYER_SHAPES,
    PAUSED,
    PREFILL,
    SCAN_STARTS,
    mamba2_layer,
    mamba2_layer_draw,
    mamba2_layer_params,
    mamba2_model,
    mamba2_model_draw,
    mamba2_model_hidden,
    mamba2_model_variables,
    mamba2_site,
    padded_layout,
)

from dew.batch import coordinates
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import AttentionMetadata, MixerCall, ModelInputs
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd, recurrent_ssd
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec

SEEDED = FIXTURES / "ssd_seeded.npz"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


@pytest.fixture(scope="module")
def seeded_reference():
    return recording(SEEDED)


def _recorded(reference, site):
    """The site's declared seeds, and its draw, recorded transformers arm and truth (`indexed_recording`):
    the truth Dew's float64 twin, which the recorder held to transformers' float64 at every seed."""
    seeds = declared_seeds(SEEDED)[site]
    inputs_of, twin = mamba2_site(site)
    return seeds, *indexed_recording(reference, site, seeds, inputs_of=inputs_of,
                                     truth_fn=jax.jit(lambda inputs: twin(site, inputs)))


def assert_recorded_as_exact(reference, site, run=None, label: str | None = None):
    """Dew's `run(inputs)` (the site's own computation by default, Dew's float64 twin's in the inputs' dtype:
    `ssd_support.mamba2_site`) on par with transformers' recorded at the site (check 1)."""
    seeds, draw, recorded, truth = _recorded(reference, site)
    _, twin = mamba2_site(site)
    run = (lambda inputs: twin(site, inputs)) if run is None else run
    assert_as_exact(jax.jit(lambda index, inputs: run(inputs)), recorded, draw,
                    label or f"transformers {site}", seeds=seeds, truth_fn=truth)


def assert_recorded_moves(reference, site, changed, label: str):
    """`changed(inputs)`, the site's computation leaving out what it checks, moved past transformers'
    recorded rounding at the site (`assert_moves`), so the site tells the computation it checks from it."""
    seeds, draw, recorded, truth = _recorded(reference, site)
    assert_moves(recorded, jax.jit(lambda index, inputs: changed(inputs)), draw, label, seeds=seeds,
                 truth_fn=truth)


def test_the_chunked_scan_matches_the_reference(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "scan")


def test_a_carried_state_matches_the_reference(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "carried")


def test_one_recurrent_step_matches_selective_state_update(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "step")


def scan_operands(arrays):
    """A scan draw's operands as the module hands them to the scan: the step through softplus with its
    bias."""
    names = ("x", "dt", "A", "B", "C", "D", "dt_bias")
    x, dt, A, B, C, D, bias = (jnp.asarray(arrays[name]) for name in names)
    return x, jax.nn.softplus(dt + bias), A, B, C, D


def test_the_recurrent_form_matches_the_reference(seeded_reference):
    """The scan token by token from a carried state, against transformers' chunk scan from it ('carried')."""
    def recurrent(arrays):
        output, final = recurrent_ssd(*scan_operands(arrays), jnp.asarray(arrays["h0"]))
        return {"output": output, "final": final}

    assert_recorded_as_exact(seeded_reference, "carried", recurrent, "the recurrent form")


def test_a_sequence_split_in_two_carries_its_state_across_the_cut(seeded_reference):
    """Chunk boundaries that do not line up with the cut: 70 tokens as 30 then 40 at a chunk of 32, the second
    half from the first's final state, against transformers' scan of the whole sequence ('scan')."""
    def cut(arrays):
        x, step, A, B, C, D = scan_operands(arrays)
        head, carried = chunk_ssd(x[:, :30], step[:, :30], A, B[:, :30], C[:, :30], D, None, 32)
        tail, final = chunk_ssd(x[:, 30:], step[:, 30:], A, B[:, 30:], C[:, 30:], D, carried, 32)
        return {"output": jnp.concatenate([head, tail], axis=1), "final": final}

    assert_recorded_as_exact(seeded_reference, "scan", cut, "the cut sequence")


@pytest.mark.parametrize("size", [7, 256])
def test_the_chunk_size_does_not_change_the_scan(seeded_reference, size):
    """The scan at chunks of 7 and of 256 tokens against transformers' scan at its own 32 ('scan')."""
    def chunked(arrays):
        output, final = chunk_ssd(*scan_operands(arrays), None, size)
        return {"output": output, "final": final}

    assert_recorded_as_exact(seeded_reference, "scan", chunked, f"a chunk of {size}")


def fixture_layer_params(reference) -> dict:
    """The ssd.npz layer's weights (`layer.`-prefixed torch names) under Dew's names."""
    return mamba2_layer_params({name.removeprefix("layer."): value for name, value in reference.items()})


def test_the_layer_matches_mamba2_mixer(seeded_reference):
    """The whole mixer on transformers' weights, fresh at each seed: the projection, the conv with its bias,
    the step, the scan with two heads per group, the D skip, the gated norm and the output projection, its
    weights loaded under Dew's names (check 1 against the recorded mixer)."""
    variables = mamba2_layer_params(mamba2_layer_draw(0))
    width = LAYER_SHAPES["hidden"][-1]
    template = nn.unbox(jax.eval_shape(mamba2_layer().init, jax.random.key(0), jnp.zeros((1, 4, width))))
    assert jax.tree.map(jnp.shape, variables) == jax.tree.map(jnp.shape, template)
    assert_recorded_as_exact(seeded_reference, "layer")


ON_A_TPU = pytest.mark.tpu(reason="the layer as a TPU runs it, its scan the Pallas kernel (row 678)")


@pytest.mark.parametrize("site", [pytest.param("layer", marks=ON_A_TPU, id="layer-tpu"),
                                  pytest.param("layer_bf16", marks=ON_A_TPU, id="layer_bf16-tpu")])
def test_the_layer_on_a_tpu_matches_mamba2_mixer(seeded_reference, site):
    """The mixer's forward on a TPU, its scan the route's (`recurrence_chunk.SCAN`: the Pallas kernel,
    `ssd._chunk_scan`), against transformers' recorded mixer, the truth the float64 twin on the host (check
    1): the step a TPU runs, not its lowering's printed calls (row 678)."""
    assert_recorded_as_exact(seeded_reference, site)


@ON_A_TPU
def test_the_layers_gradient_on_a_tpu_is_as_exact_as_its_portable_math():
    """The mixer's output and the gradients of its every weight and its hidden states on a TPU, its scan the
    route's Pallas kernel, against the same mixer with its scan the portable math (`generation.chosen`) on the
    TPU, the truth the float64 mixer on the host (check 1): the backward a TPU runs (row 678)."""
    from dew.nn.kernels import generation

    def draw(seed):
        arrays = mamba2_layer_draw(seed)
        cotangent = np.random.default_rng(np.random.SeedSequence([seed, 3])).standard_normal(
            LAYER_SHAPES["hidden"]).astype(np.float32)
        return mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"]), jnp.asarray(cotangent)

    def run(variables, hidden):
        return mamba2_layer(hidden.dtype).apply(variables, hidden)

    def portable(variables, hidden):
        with generation.chosen(ssd="xla"):
            return run(variables, hidden)

    assert_as_exact(jax.jit(with_gradients(run)), jax.jit(with_gradients(portable)), draw,
                    "the Mamba-2 mixer's gradient on a TPU", seeds=64,
                    truth_fn=jax.jit(with_gradients(portable)))


def test_the_layer_in_bfloat16_matches_the_reference_in_bfloat16(seeded_reference):
    """Compute in bfloat16 with fp32 parameters, against the reference run in bfloat16 end to end; the scan
    itself is fp32 in both (check 1 against the recorded mixer)."""
    assert_recorded_as_exact(seeded_reference, "layer_bf16")


@pytest.mark.check1_shards(seeds=64, per_task=32)
def test_the_layer_decodes_as_it_prefills(seeded_reference):
    """A prefill against the cache and single-token steps after it, against transformers' mixer at each step's
    width ('decoded'): the conv tail and the SSM state both cross the step boundary. Its 64 recorded seeds,
    a decode's prefill and steps compiled at each seed's widths, took 115 s of 143 cold on armada's CI CPU:
    CI draws them in map tasks of 32 (ledger 3821)."""
    module = mamba2_layer()
    variables = mamba2_layer_params(mamba2_layer_draw(0))
    hidden = jnp.asarray(mamba2_layer_draw(0)["hidden"])
    _, cache, _ = decode_through(module, variables, hidden, prefill=PREFILL)
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))

    def decoded(arrays):
        return {"output": decode_through(module, mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"]),
                                         prefill=PREFILL)[0]}

    assert_recorded_as_exact(seeded_reference, "decoded", decoded, "decoded")


def test_the_layer_takes_forward_mode(seeded_reference):
    """`jax.jvp` through the layer, as MeanFlow and consistency training trace a model: its tangent along a
    drawn direction against transformers' (`torch.func.jvp`, 'tangent'). The D skip's custom VJP refused
    forward mode on every target (row 1933), so a hybrid with Mamba-2 layers trained on neither."""
    assert_recorded_as_exact(seeded_reference, "tangent", label="the layer's tangent")


def test_padded_rows_preserve_the_state_and_the_history(seeded_reference):
    """A row paused on padding neither writes its state nor advances its conv history, and its padded slots
    emit zero: each row's stream, the second's with the holes closed, against transformers' mixer on the
    closed rows ('padded')."""
    module = mamba2_layer()
    paused = jnp.asarray(PAUSED)

    def padded(arrays):
        call = MixerCall(AttentionMetadata(coordinates=coordinates(valid=paused)))
        out = module.apply(mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"])[:, :12], call)
        return {"first": out[0], "second": out[1][PAUSED[1]]}

    arrays = mamba2_layer_draw(0)
    call = MixerCall(AttentionMetadata(coordinates=coordinates(valid=paused)))
    out = module.apply(mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"])[:, :12], call)
    assert not jnp.any(out[1][~PAUSED[1]])
    assert_recorded_as_exact(seeded_reference, "padded", padded, "the padded rows")


SPLIT_SEEDS = {"all_valid": 64, "right_padded": 128, "left_padded": 128, "gaps": 128, "packed": 256,
               "empty_shard": 256, "empty_shard_packed": 64}
"""Each row's declared draws, sized by `seeds_for` on armada paired with the unsplit layer: the two arms
share every upstream rounding (the same weights and hidden states), and the gradients of `A_log`, `D`
and the conv's parameters, sums of the whole draw into a few values, are heavy-tailed across draws, so
that the unsplit layer against itself resolves none of them within 512 draws."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SPLIT_ROWS))
def test_a_layer_split_over_the_sequence_runs_padded_and_packed_rows_as_whole_ones(reference, row):
    """Under a sequence axis of 4, the layer runs rows with padding as `held_scan` runs them on one
    device (`assert_split_as_whole`). A pool of processes always carries validity, so this is every
    multi-host run."""
    assert_split_as_whole(mamba2_layer(), fixture_layer_params(reference), row, SPLIT_SEEDS[row],
                          f"mamba2 split over the sequence, {row}", width=LAYER_GEOMETRY["hidden_size"])


@pytest.mark.mesh
@pytest.mark.parametrize("masked", [False, True], ids=["whole rows", "valid rows"])
def test_a_layer_split_over_the_sequence_names_its_scan_for_remat(reference, masked):
    """Under a sequence axis of 4, the layer's scan output carries both names a remat policy keeps it by,
    a mixer's 'context' and a recurrence's 'scan', with or without the rows' validity, as on one device
    (`Mamba2.residuals`): a checkpoint saving either name alone holds more residuals for its backward
    (the pullback's arrays, traced) than one saving nothing."""
    from dew.nn.backbones.decoder_block import RematPolicy

    module = mamba2_layer()
    variables = fixture_layer_params(reference)
    fields = {"valid": jnp.ones((2, 32), bool)} if masked else {}
    call = MixerCall(AttentionMetadata(coordinates=coordinates((2, 32), **fields)))
    drawn = np.random.default_rng(0).standard_normal((2, 32, LAYER_GEOMETRY["hidden_size"]))
    hidden = jnp.asarray(drawn, jnp.float32)
    mesh = MeshSpec(sequence=4).build(jax.devices()[:4])

    def loss(params):
        return jnp.sum(module.apply(params, hidden, call) ** 2)

    def kept(names) -> int:
        """The residual arrays the pullback holds under a policy saving `names`, traced, not run."""
        checkpointed = jax.checkpoint(loss, policy=RematPolicy(save=names).checkpoint_policy())
        with jax.set_mesh(mesh):
            pull = jax.eval_shape(lambda params: jax.vjp(checkpointed, params)[1], variables)
        return len(jax.tree.leaves(pull))

    nothing = kept(())
    assert [name for name in ("context", "scan") if kept((name,)) <= nothing] == []


def test_packed_documents_run_as_if_each_ran_alone(seeded_reference):
    """A row of packed documents computes, for each document, what that document computes alone: neither the
    SSM state nor the conv's taps cross a segment change, against transformers' mixer on each document alone
    ('packed'). The same row without segment ids carries the state and the conv across the cuts, far outside
    the rule."""
    module = mamba2_layer()
    segments = jnp.asarray(np.repeat(np.arange(1, 4), np.diff(CUTS))[None], jnp.int32)
    call = MixerCall(AttentionMetadata(coordinates=coordinates(segments=segments)))

    def packed(arrays):
        return {"output": module.apply(mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"])[:1], call)}

    assert_recorded_as_exact(seeded_reference, "packed", packed, "the packed row")
    assert_recorded_moves(seeded_reference, "packed", lambda arrays: {"output": module.apply(
        mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"])[:1])}, "the row without segment ids")


@pytest.mark.parametrize("form", ["chunked", "recurrent"])
def test_the_chunked_and_recurrent_forms_reset_alike(seeded_reference, form):
    """Document starts drop the state in both forms, including a start on the first token of a chunk and one
    inside it, against transformers' scan on each document alone ('starts'); the scan without the starts far
    outside the rule."""
    starts = jnp.asarray(SCAN_STARTS)

    def reset(arrays):
        operands = (*scan_operands(arrays), jnp.asarray(arrays["h0"]))
        output, final = (recurrent_ssd(*operands, starts=starts) if form == "recurrent"
                         else chunk_ssd(*operands, 32, starts=starts))
        return {"output": output, "final": final}

    assert_recorded_as_exact(seeded_reference, "starts", reset, f"{form} with starts")

    def unreset(arrays):
        output, final = chunk_ssd(*scan_operands(arrays), jnp.asarray(arrays["h0"]), 32)
        return {"output": output, "final": final}

    assert_recorded_moves(seeded_reference, "starts", unreset, "the scan without starts")


def test_padded_packed_rows_reset_at_each_document(seeded_reference):
    """Padding slots and packed documents together: every real token of the padded row computes what its
    document computes alone, the second document starting fresh after the gap, through the model's
    `hidden_and_mtp_inputs` and through `LMObjective.token_scores` with both token fields, against
    transformers' model on each document alone ('model_padded', 'model_scored'). The padded row without its
    segment ids, as the layer computed before it passed segments to its stateful path, sits far outside the
    rule."""
    model = mamba2_model().model
    valid, segments = padded_layout(16)

    def padded(arrays):
        tokens = jnp.asarray(arrays["ids"])[:1, :16]
        hidden = mamba2_model_hidden(model, mamba2_model_variables(arrays), tokens, attention_mask=valid,
                                     segment_ids=segments)
        return {"hidden": hidden[0][valid[0]]}

    assert_recorded_as_exact(seeded_reference, "model_padded", padded, "the padded packed row")

    def unsegmented(arrays):
        tokens = jnp.asarray(arrays["ids"])[:1, :16]
        hidden = mamba2_model_hidden(model, mamba2_model_variables(arrays), tokens, attention_mask=valid)
        return {"hidden": hidden[0][valid[0]]}

    assert_recorded_moves(seeded_reference, "model_padded", unsegmented, "the row without segment ids")

    valid_17, segments_17 = padded_layout(17)
    kept = valid_17[0, :-1]

    def scored(arrays):
        fields = {"attention_mask": jnp.asarray(valid_17), "segment_ids": jnp.asarray(segments_17)}
        scores = LMObjective(model, 16).token_scores(mamba2_model_variables(arrays),
                                                     ModelInputs(jnp.asarray(arrays["ids"])[:1, :17], fields))
        return {"hidden": scores.hidden[0][kept]}

    assert_recorded_as_exact(seeded_reference, "model_scored", scored, "the scored padded row")


def test_the_model_decodes_as_it_scores_in_parallel(seeded_reference):
    """A prefill of four tokens and eight single-token steps, every layer's state riding the flax cache
    collection, against transformers' model on the same twelve tokens at each step's width
    ('model_decoded')."""
    model = mamba2_model().model

    def decoded(arrays):
        params, ids = mamba2_model_variables(arrays), jnp.asarray(arrays["ids"])[:, :12]
        cache = model.apply(params, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
        logits, mutated = model.apply({**params, "cache": cache}, ids[:, :4], decode=True, mutable=["cache"])
        steps = [logits[:, -1]]
        for position in range(4, 12):
            logits, mutated = model.apply({**params, **mutated}, ids[:, position:position + 1], decode=True,
                                          mutable=["cache"])
            steps.append(logits[:, -1])
        return {"logits": jnp.stack(steps, axis=1)}

    params = mamba2_model_variables(mamba2_model_draw(0))
    cache = model.apply(params, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
    assert_recorded_as_exact(seeded_reference, "model_decoded", decoded, "the decoded logits")


def ssd_lm() -> CausalTransformer:
    """Two SSD layers at widths the TPU's Pallas scan takes (`ssd_kernel_runs`): 4 heads of 64
    channels, a state of 64, chunks of 64 over 128 tokens, a residual of 128 apart from every inner
    width, in bf16 as a hybrid's step runs them."""
    config = {"vocab_size": 64, "emb_features": 128, "num_layers": 2, "num_heads": 4, "mlp_features": 0,
              "max_seq_len": 128, "tie_embeddings": False, "qk_norm": False, "dtype": jnp.bfloat16,
              "mixer": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 64, "state_size": 64,
                        "n_groups": 1, "chunk_size": 64}}}
    return CausalTransformer(**config)


def stepped(model):
    """`model`'s training step: the next-token loss's gradient over every parameter and the logits,
    from the parameters and a row of 129 ids, compiled once."""
    def loss(params, ids):
        logits = model.apply({"params": params}, ids[:, :-1])
        wide = jax.nn.log_softmax(logits.astype(jnp.promote_types(logits.dtype, jnp.float32)))
        return -jnp.mean(jnp.take_along_axis(wide, ids[:, 1:, None], axis=-1)), logits
    return jax.jit(jax.grad(loss, has_aux=True))


def rowed(value) -> np.ndarray:
    """`value` as rows over its last axis, float64; a vector one row."""
    value = np.asarray(value, np.float64)
    return value.reshape(-1, value.shape[-1]) if value.ndim > 1 else value[None]


@pytest.mark.skipif(jax.default_backend() != "tpu",
                    reason="needs a tpu, the one backend the ssd kernel runs on")
def test_a_training_step_on_a_tpu_mesh_is_as_exact_with_the_ssd_kernel_as_with_the_xla_scan():
    """A Mamba-2 model's training step compiled on the TPU, its rows split over a mesh of every device
    the process holds (each shard's scan its own Pallas call, `manual_map`): the logits and every
    parameter's gradient against the same step on the XLA scan, its kernels chosen as on a CPU
    (`generation.assumed`), both from the float64 step on the host. Over ORDERS orders of the
    residual stream (tests/residual_orders.py), an exact symmetry, so only the rounding moves: each
    output's RMS within FACTOR of the reference's and its worst row within ROW_FACTOR
    (tests/reference_error.py's over-orders rules); the float64 step under an order first held to
    the identity's within float64 rounding."""
    from reference_error import (
        ORDERS,
        assert_as_exact_over_orders,
        assert_computes_the_oracle,
        assert_rows_as_exact_over_orders,
        chain_roundings,
    )
    from residual_orders import model_axes, orders, permuted, residual_width

    from dew.nn.kernels import generation
    from dew.nn.kernels.ssd import ssd_kernel_runs

    assert ssd_kernel_runs(64, 64, 64, "tpu", dtype=jnp.float32)
    model, devices = ssd_lm(), jax.devices()
    ids = np.asarray(jax.random.randint(jax.random.key(0), (2 * len(devices), 129), 0, 64))
    variables = jax.device_get(nn.unbox(model.init(jax.random.key(1), jnp.asarray(ids[:, :-1]))))
    axes = model_axes(model, jnp.asarray(ids[:, :-1]))
    units = residual_width(variables, axes)

    def ordered(result, order):
        """The step's gradients with the residual's units in `order`, and its logits, which no order
        moves."""
        gradients, logits = result
        return permuted({"params": gradients}, order, axes)["params"], logits

    # The float64 oracle runs on the host: a TPU emulates float64.
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        wide = stepped(model.clone(dtype=jnp.float64))
        exact = jax.device_get(wide(widened(variables["params"]), ids))
        last = orders(units, 2, seed=0)[-1]
        found = jax.device_get(wide(permuted(widened(variables), last, axes)["params"], ids))
        roundings = chain_roundings(jax.make_jaxpr(wide)(widened(variables["params"]), ids))
        for (path, leaf), true in zip(jax.tree_util.tree_leaves_with_path(found),
                                      jax.tree.leaves(ordered(exact, last)), strict=True):
            assert_computes_the_oracle(np.asarray(leaf), np.asarray(true),
                                       f"the residual order {jax.tree_util.keystr(path)}",
                                       roundings=roundings)

    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    placed = jax.device_put(ids, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
    whole = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    kernel, xla = stepped(model), stepped(model)
    distances: dict = {}
    with jax.set_mesh(mesh):
        for order in orders(units, ORDERS, seed=1):
            params = jax.device_put(permuted(variables, order, axes)["params"], whole)
            truth = jax.tree_util.tree_leaves_with_path(ordered(exact, order))
            found = {"dew": jax.device_get(kernel(params, placed))}
            with generation.assumed("cpu"):
                found["reference"] = jax.device_get(xla(params, placed))
            for side, result in found.items():
                for (path, held), leaf in zip(truth, jax.tree.leaves(result), strict=True):
                    name = jax.tree_util.keystr(path)
                    distances.setdefault((name, side), []).append(
                        np.sqrt(np.mean(np.square(rowed(leaf) - rowed(held)), axis=1)))
                    distances[name, "size"] = np.sqrt(np.mean(np.square(rowed(held)), axis=1))
    for name in sorted({name for name, side in distances if side == "dew"}):
        mine, theirs = np.stack(distances[name, "dew"]), np.stack(distances[name, "reference"])
        assert_as_exact_over_orders(np.sqrt(np.mean(np.square(mine), axis=1)),
                                    np.sqrt(np.mean(np.square(theirs), axis=1)), f"the step's {name}")
        assert_rows_as_exact_over_orders(mine, theirs, f"the step's {name} row by row",
                                         sizes=distances[name, "size"])


@pytest.mark.parametrize("scale_wide", [False, True], ids=["weighted-narrow", "weighted-wide"])
def test_a_gated_norm_broadcasts_its_gate_in_either_order(scale_wide):
    """Falcon-H1's gated norm (`MambaRMSNormGated`, modeling_falcon_h1.py:256-282) over two groups computes
    as the RMS norm's arithmetic orders it (`rms_statistics`, `rms_weighted`), where the module computes
    in float64 over fp32 inputs (ledger 3627), bit for bit: norm first, each group normalized in fp32 and
    weighted, the weighted norm rounded to fp32 before a gate broadcast over it multiplies it; gate first,
    over a gate broadcasting the input's leading axis, the norm of the gated values' shape."""
    from dew.nn.mixers.mamba2 import MambaRMSNormGated
    from dew.nn.precision import rms_statistics

    rng = np.random.default_rng(0x3627)
    with jax.enable_x64(new_val=True):
        x = jnp.asarray(rng.normal(size=(2, 4, 8)), jnp.float32)
        gate = jnp.asarray(rng.normal(size=(2, 4, 1)), jnp.float32)
        weight = jnp.asarray(rng.normal(size=(8,)), jnp.float32)
        norm = MambaRMSNormGated(groups=2, gate_after=True, scale_wide=scale_wide, dtype=jnp.float64)
        out = norm.apply({"params": {"weight": weight}}, x, gate)
        grouped = x.reshape(2, 4, 2, 4)
        normalized = (grouped * rms_statistics(grouped, 1e-5)[1]).reshape(x.shape)
        weighted = (normalized * weight if scale_wide
                    else normalized.astype(jnp.float64) * weight.astype(jnp.float64))
        expected = (weighted.astype(jnp.float32) * jax.nn.silu(gate)).astype(jnp.float64)
        assert out.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
        # The gate-first order over a gate that broadcasts the input's leading axis: the norm takes the
        # gated values' shape.
        wide = jnp.asarray(rng.normal(size=(2, 4, 8)), jnp.float32)
        first = MambaRMSNormGated(groups=2, scale_wide=scale_wide, dtype=jnp.float64)
        gated = x[:1] * jax.nn.silu(wide)
        grouped = gated.reshape(2, 4, 2, 4)
        normalized = (grouped * rms_statistics(grouped, 1e-5)[1]).reshape(gated.shape)
        expected = (normalized * weight if scale_wide
                    else normalized.astype(jnp.float64) * weight.astype(jnp.float64)).astype(jnp.float64)
        np.testing.assert_array_equal(np.asarray(first.apply({"params": {"weight": weight}}, x[:1], wide)),
                                      np.asarray(expected))


FUSED_SEEDS = 8
"""The fused decode token's declared draws against the unfused one (check 1): sized by `seeds_for` paired
with the fused step."""


def test_a_decode_token_on_the_fused_step_is_the_layers_own():
    """A Mamba-2 mixer's decode token through the recurrence's fused step (the decode kernel named 'fused',
    interpreted on a CPU: its conv over x, B and C with their bias, the step through softplus, the decay, the
    update and the skip in one launch, `Mamba2._fused_token`; its gated norm after), at a model's widths
    (four heads of 64, a group of 128 states, two heads a program's warpgroup) after a 3-token prefill: its
    output and state on par (check 1) with the layer's own unfused token, from the float64 layer's; its conv
    history bit for bit the unfused token's."""
    from dew.nn.kernels.generation import chosen

    module = Mamba2(emb_features=64, num_heads=4, head_dim=64, state_size=128, n_groups=1, conv_kernel=4,
                    chunk_size=4, norm_eps=1e-5)
    variables = nn.unbox(module.init(jax.random.key(0), jnp.zeros((3, 1, 64))))
    wide = module.clone(dtype=jnp.float64)
    step = MixerCall(decode=True)

    def cached(seed):
        x = jnp.asarray(np.random.default_rng(seed).standard_normal((3, 4, 64)).astype(np.float32))
        allocated = module.apply(variables, x[:, :1], step, mutable=["cache"])[1]
        cache = module.apply({**variables, **allocated}, x[:, :3], step, mutable=["cache"])[1]["cache"]
        return cache, x[:, 3:4]

    def token(layer, cache, x, kernel):
        with chosen(delta_decode=kernel):
            out, written = layer.apply({**variables, "cache": cache}, x, step, mutable=["cache"])
        return out, written["cache"]

    def unfused(layer, cache, x):
        out, written = token(layer, cache, x, "xla")
        return out, written["ssm_state"]

    def fused(cache, x):
        out, written = token(module, cache, x, "fused")
        return out, written["ssm_state"]

    assert_as_exact(fused, functools.partial(unfused, module), cached, "Mamba-2's fused decode token",
                    seeds=FUSED_SEEDS, truth_fn=functools.partial(unfused, wide))
    cache, x = cached(0)
    np.testing.assert_array_equal(np.asarray(token(module, cache, x, "fused")[1]["conv_state"]),
                                  np.asarray(token(module, cache, x, "xla")[1]["conv_state"]))
