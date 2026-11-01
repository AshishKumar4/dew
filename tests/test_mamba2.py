"""Mamba-2 against the transformers 5.16.1 reference.

tools/mamba2_reference.py runs `mamba2_chunk_scan`,
`mamba2_selective_state_update` and one `Mamba2Mixer` layer
(modeling_mamba2.py) on fixed-seed operands and writes what they produced,
in fp32 (bf16 for the layer too) and again in float64; the tests here run
dew's forms on the same operands and the same weights and hold each within
tests/reference_error.py's rule of the reference's own distance from the
float64 run. Where dew's forms are held to each other (chunked against
recurrent, a cut sequence, chunk sizes, decoding, padding, packing), the
one under test is held to the other's distance from dew's own float64 run
of it. Nothing here imports torch.
"""

import functools
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import as_stepped, decode_through
from reference_error import FACTOR, assert_as_exact, declared_seeds, indexed_recording, recording, widened
from sequence_split_support import SPLIT_ROWS, assert_split_as_whole
from ssd_support import (
    LAYER_SHAPES,
    mamba2_draw,
    mamba2_layer,
    mamba2_layer_draw,
    mamba2_layer_params,
    mamba2_layer_run,
    mamba2_run,
)

from dew.accuracy import distance
from dew.batch import coordinates, field_coordinates
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import AttentionMetadata, MixerCall, ModelInputs
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd, recurrent_ssd
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
SEEDED = FIXTURES / "ssd_seeded.npz"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


@pytest.fixture(scope="module")
def seeded_reference():
    return recording(SEEDED)


@pytest.fixture(scope="module")
def geometry():
    return json.loads((FIXTURES / "config.json").read_text())


def outside(wrong, reference, truth, label: str):
    """`wrong` far outside the rule a test holds: more than FACTOR times `reference`'s RMS distance from
    `truth`, so the test tells the computation it checks from the one that leaves its operands out."""
    mine = distance(np.asarray(wrong), np.asarray(truth))
    theirs = distance(np.asarray(reference), np.asarray(truth))
    assert mine > FACTOR * theirs, f"{label}: {mine:.3e} from float64, the rule's {FACTOR} x {theirs:.3e}"


def in_float64(function, *args, **kwargs):
    """`function` run in float64 on its array arguments widened (integers kept), as host arrays."""
    def wide(value):
        return widened(value) if hasattr(value, "dtype") else value

    with jax.enable_x64(new_val=True):
        return jax.tree.map(np.asarray, function(*map(wide, args), **{name: wide(value)
                                                                     for name, value in kwargs.items()}))


def assert_recorded_as_exact(reference, site):
    """The recorded reference and its torch-float64 truth on each fresh draw: the scan's native inputs
    under an orbit, or a mixer's weights and hidden states ('layer', 'layer_bf16')."""
    seeds = declared_seeds(SEEDED)[site]
    layered = site.startswith("layer")
    inputs_of, run_of = (mamba2_layer_draw, mamba2_layer_run) if layered else (mamba2_draw, mamba2_run)
    # The truth is Dew's float64 path, which the recorder held to transformers' float64 at every seed: the
    # mixer's in float64 for both of its sites.
    exact = "layer" if layered else site
    draw, recorded, truth = indexed_recording(reference, site, seeds, inputs_of=inputs_of,
                                              truth_fn=jax.jit(lambda inputs: run_of(exact, inputs)))
    run = jax.jit(lambda index, inputs: run_of(site, inputs))
    assert_as_exact(run, recorded, draw, f"transformers {site}", seeds=seeds, truth_fn=truth)


def test_the_chunked_scan_matches_the_reference(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "scan")


def test_a_carried_state_matches_the_reference(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "carried")


def test_one_recurrent_step_matches_selective_state_update(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "step")


CHUNK_SIZE_SEEDS = 64
"""Sized by seeds_for against the 32-token XLA chunk scan (2026-10-20, armada)."""


def chunk_size_draw(seed):
    """Fresh inputs at the fixture geometry under joint head, P-channel and N-coordinate permutations."""
    arrays = mamba2_draw(seed)
    arrays["dt"] = jax.nn.softplus(jnp.asarray(arrays["dt"]) + jnp.asarray(arrays["dt_bias"]))
    return tuple(arrays[name] for name in ("x", "dt", "A", "B", "C", "D"))


FORMS_SEEDS = 64
"""The declared draws of the recurrent and cut forms against the 32-token XLA chunk scan (check 1), sized by
`seeds_for` on armada: 64 for each, the reference compared with itself."""


def test_the_chunked_and_recurrent_forms_agree():
    """The scan token by token against the chunked scan from a carried state (check 1)."""
    def draw(seed):
        return (*chunk_size_draw(seed), jnp.asarray(mamba2_draw(seed)["h0"]))

    assert_as_exact(jax.jit(recurrent_ssd), jax.jit(lambda *inputs: chunk_ssd(*inputs, 32)), draw,
                    "the recurrent form", seeds=FORMS_SEEDS)


def test_a_sequence_split_in_two_carries_its_state_across_the_cut():
    """Chunk boundaries that do not line up with the cut: 70 tokens as 30 then 40 at a chunk of 32, the
    second half from the first's final state, against the whole sequence's scan (check 1)."""
    def cut(x, step, A, B, C, D):
        head, carried = chunk_ssd(x[:, :30], step[:, :30], A, B[:, :30], C[:, :30], D, None, 32)
        tail, final = chunk_ssd(x[:, 30:], step[:, 30:], A, B[:, 30:], C[:, 30:], D, carried, 32)
        return jnp.concatenate([head, tail], axis=1), final

    assert_as_exact(jax.jit(cut), jax.jit(lambda *inputs: chunk_ssd(*inputs, None, 32)), chunk_size_draw,
                    "the cut sequence", seeds=FORMS_SEEDS)


def test_the_chunk_size_does_not_change_the_scan():
    at_32 = jax.jit(lambda *inputs: chunk_ssd(*inputs, None, 32))
    for size in (7, 256):
        run = jax.jit(lambda *inputs, size=size: chunk_ssd(*inputs, None, size))
        assert_as_exact(run, at_32, chunk_size_draw, f"a chunk of {size}", seeds=CHUNK_SIZE_SEEDS)


MIXER_SEEDS = {"decoded": 64, "tangent": 64, "padded": 64, "packed": 64, "padded packed": 64,
               "scored padded": 64, "model decoded": 64}
"""Each mixer and model comparison's declared draws (check 1), sized by `seeds_for` on armada."""


def layer_draw(seed):
    """A fresh mixer's weights under Dew's names and its hidden states (`ssd_support.mamba2_layer_draw`)."""
    arrays = mamba2_layer_draw(seed)
    return mamba2_layer_params(arrays), jnp.asarray(arrays["hidden"])


def layer(geometry, **overrides) -> Mamba2:
    return Mamba2(emb_features=geometry["hidden_size"], num_heads=geometry["num_heads"],
                  head_dim=geometry["head_dim"], state_size=geometry["state_size"],
                  n_groups=geometry["n_groups"], conv_kernel=geometry["conv_kernel"],
                  chunk_size=geometry["chunk_size"], norm_eps=geometry["layer_norm_epsilon"], **overrides)


def layer_params(reference):
    """The reference layer's state dict under the module's names: kernels
    transposed from torch's [out, in], everything else as stored."""
    weights = {
        name[len("layer.") :]: value
        for name, value in reference.items()
        if name.startswith("layer.")
        and name not in ("layer.hidden", "layer.output", "layer.output_bf16", "layer.output_f64")
    }
    return {
        "params": {
            "in_proj": {"kernel": jnp.asarray(weights["in_proj.weight"].T)},
            "conv1d": {
                "weight": jnp.asarray(weights["conv1d.weight"]),
                "bias": jnp.asarray(weights["conv1d.bias"]),
            },
            "A_log": jnp.asarray(weights["A_log"]),
            "dt_bias": jnp.asarray(weights["dt_bias"]),
            "D": jnp.asarray(weights["D"]),
            "norm": {"weight": jnp.asarray(weights["norm.weight"])},
            "out_proj": {"kernel": jnp.asarray(weights["out_proj.weight"].T)},
        }
    }


def test_the_layer_matches_mamba2_mixer(seeded_reference):
    """The whole mixer on transformers' weights, fresh at each seed: the projection, the conv with its bias,
    the step, the scan with two heads per group, the D skip, the gated norm and the output projection, its
    weights loaded under Dew's names (check 1 against the recorded mixer)."""
    variables = mamba2_layer_params(mamba2_layer_draw(0))
    width = LAYER_SHAPES["hidden"][-1]
    template = jax.eval_shape(mamba2_layer().init, jax.random.key(0), jnp.zeros((1, 4, width)))
    assert jax.tree.map(jnp.shape, variables) == jax.tree.map(jnp.shape, template)
    assert_recorded_as_exact(seeded_reference, "layer")


def test_the_layer_in_bfloat16_matches_the_reference_in_bfloat16(seeded_reference):
    """Compute in bfloat16 with fp32 parameters, against the reference run in bfloat16 end to end; the scan
    itself is fp32 in both (check 1 against the recorded mixer)."""
    assert_recorded_as_exact(seeded_reference, "layer_bf16")


def test_the_layer_decodes_as_it_prefills():
    """A prefill against the cache and single-token steps after it reproduce the parallel forwards, each
    step's at the width it reads (`as_stepped`): the conv tail and the SSM state both cross the step boundary
    (check 1 on fresh mixers)."""
    def decoded(module, variables, hidden):
        return decode_through(module, variables, hidden, prefill=5)[0]

    def parallel(module, variables, hidden):
        return as_stepped(lambda held: module.apply(variables, held), hidden, 5)

    module = mamba2_layer()
    _, cache, _ = decode_through(module, *layer_draw(0), prefill=5)
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
    assert_as_exact(functools.partial(decoded, module), functools.partial(parallel, module), layer_draw,
                    "decoded", seeds=MIXER_SEEDS["decoded"],
                    truth_fn=functools.partial(parallel, mamba2_layer(jnp.float64)))


def test_the_layer_takes_forward_mode():
    """`jax.jvp` through the layer, as MeanFlow and consistency training trace a model: its tangent along a
    seeded direction on par from float64 with the same product its reverse mode gives (the Jacobian
    `jax.jacrev` builds, contracted with the direction), check 1 on fresh mixers over a row of 24 tokens.
    The D skip's custom VJP refused forward mode on every target (row 1933), so a hybrid with Mamba-2 layers
    trained on neither."""
    def draw(seed):
        variables, hidden = layer_draw(seed)
        hidden = hidden[:1, :24]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1933]))
        return variables, hidden, jnp.asarray(rng.standard_normal(hidden.shape, np.float32))

    def tangent(module, variables, x, d):
        return jax.jvp(lambda x: module.apply(variables, x), (x,), (d,))[1]

    def reverse(module, variables, x, d):
        jacobian = jax.jacrev(lambda x: module.apply(variables, x))(x)
        return jnp.tensordot(jacobian, d, axes=d.ndim)

    module = mamba2_layer()
    assert_as_exact(functools.partial(tangent, module), functools.partial(reverse, module), draw,
                    "the layer's tangent", seeds=MIXER_SEEDS["tangent"],
                    truth_fn=functools.partial(tangent, mamba2_layer(jnp.float64)))


PAUSED = np.asarray([[1] * 12, [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]], bool)
"""Two rows of 12 slots, the second paused on padding at five of them."""


def test_padded_rows_preserve_the_state_and_the_history():
    """A row paused on padding neither writes its state nor advances its conv history, and its padded slots
    emit zero: each row's stream, the second's with the holes closed, is what the padded run computes
    (check 1 on fresh mixers)."""
    def padded(module, variables, hidden):
        call = MixerCall(AttentionMetadata(coordinates=coordinates(valid=jnp.asarray(PAUSED))))
        out = module.apply(variables, hidden[:, :12], call)
        return out[0], out[1][PAUSED[1]]

    def closed(module, variables, hidden):
        return (module.apply(variables, hidden[:1, :12])[0],
                module.apply(variables, hidden[1, :12][PAUSED[1]][None])[0])

    module = mamba2_layer()
    variables, hidden = layer_draw(0)
    call = MixerCall(AttentionMetadata(coordinates=coordinates(valid=jnp.asarray(PAUSED))))
    assert not jnp.any(module.apply(variables, hidden[:, :12], call)[1][~PAUSED[1]])
    assert_as_exact(functools.partial(padded, module), functools.partial(closed, module), layer_draw,
                    "the padded rows", seeds=MIXER_SEEDS["padded"],
                    truth_fn=functools.partial(closed, mamba2_layer(jnp.float64)))


SPLIT_SEEDS = {"all_valid": 64, "right_padded": 128, "left_padded": 128, "gaps": 128, "packed": 256,
               "empty_shard": 256, "empty_shard_packed": 64}
"""Each row's declared draws, sized by `seeds_for` on armada paired with the unsplit layer: the two arms
share every upstream rounding (the same weights and hidden states), and the gradients of `A_log`, `D`
and the conv's parameters, sums of the whole draw into a few values, are heavy-tailed across draws, so
that the unsplit layer against itself resolves none of them within 512 draws."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SPLIT_ROWS))
def test_a_layer_split_over_the_sequence_runs_padded_and_packed_rows_as_whole_ones(reference, geometry, row):
    """Under a sequence axis of 4, the layer runs rows with padding as `held_scan` runs them on one
    device (`assert_split_as_whole`). A pool of processes always carries validity, so this is every
    multi-host run."""
    assert_split_as_whole(layer(geometry), layer_params(reference), row, SPLIT_SEEDS[row],
                          f"mamba2 split over the sequence, {row}", width=geometry["hidden_size"])


@pytest.mark.mesh
@pytest.mark.parametrize("masked", [False, True], ids=["whole rows", "valid rows"])
def test_a_layer_split_over_the_sequence_names_its_scan_for_remat(reference, geometry, masked):
    """Under a sequence axis of 4, the layer's scan output carries both names a remat policy keeps it by,
    a mixer's 'context' and a recurrence's 'scan', with or without the rows' validity, as on one device
    (`Mamba2.residuals`): a checkpoint saving either name alone holds more residuals for its backward
    (the pullback's arrays, traced) than one saving nothing."""
    from dew.nn.backbones.decoder_block import RematPolicy

    module = layer(geometry)
    variables = layer_params(reference)
    fields = {"valid": jnp.ones((2, 32), bool)} if masked else {}
    call = MixerCall(AttentionMetadata(coordinates=coordinates((2, 32), **fields)))
    drawn = np.random.default_rng(0).standard_normal((2, 32, geometry["hidden_size"]))
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


CUTS = (0, 20, 22, 70)
"""A row of three packed documents of 20, 2 (shorter than the conv's three-token history) and 48 tokens,
the cuts off the chunk grid."""


def test_packed_documents_run_as_if_each_ran_alone():
    """A row of packed documents computes, for each document, what that document computes alone: neither
    the SSM state nor the conv's taps cross a segment change (check 1 on fresh mixers). The same row
    without segment ids carries the state and the conv across the cuts, far outside the rule."""
    segments = jnp.asarray(np.repeat(np.arange(1, 4), np.diff(CUTS))[None], jnp.int32)

    def packed(module, variables, hidden):
        call = MixerCall(AttentionMetadata(coordinates=coordinates(segments=segments)))
        return module.apply(variables, hidden[:1], call)

    def alone(module, variables, hidden):
        return jnp.concatenate([module.apply(variables, hidden[:1, start:end])
                                for start, end in itertools.pairwise(CUTS)], axis=1)

    module = mamba2_layer()
    assert_as_exact(functools.partial(packed, module), functools.partial(alone, module), layer_draw,
                    "the packed row", seeds=MIXER_SEEDS["packed"],
                    truth_fn=functools.partial(alone, mamba2_layer(jnp.float64)))
    variables, hidden = layer_draw(0)
    with jax.enable_x64(new_val=True):
        truth = np.asarray(alone(mamba2_layer(jnp.float64), widened(variables), widened(hidden)))
    outside(module.apply(variables, hidden[:1]), alone(module, variables, hidden), truth,
            "the row without segment ids")


STARTS = np.zeros((2, 70), bool)
STARTS[0, 32] = STARTS[1, 7] = STARTS[1, 45] = True
"""Document starts on the first token of a chunk and inside one."""


def test_the_chunked_and_recurrent_forms_reset_alike():
    """Document starts drop the state in both forms, including a start on the first token of a chunk and
    one inside it (check 1 against the chunked scan); the scan without the starts far outside the rule."""
    starts = jnp.asarray(STARTS)

    def draw(seed):
        return (*chunk_size_draw(seed), jnp.asarray(mamba2_draw(seed)["h0"]))

    def chunked(*inputs):
        return chunk_ssd(*inputs, 32, starts=starts)

    assert_as_exact(jax.jit(lambda *inputs: recurrent_ssd(*inputs, starts=starts)), jax.jit(chunked), draw,
                    "recurrent with starts", seeds=FORMS_SEEDS)
    inputs = draw(0)
    truth = in_float64(chunked, *inputs)
    outside(chunk_ssd(*inputs, 32)[0], chunked(*inputs)[0], truth[0], "the scan without starts")


def padded_packed_row(length: int):
    """Two documents of six real tokens in `length` slots, padding at slots
    2, 7, 8 and 13: the second document's first token follows the gap at 7
    and 8. Returns the ids, validity and segments of the padded row and of
    the same twelve tokens with the holes closed."""
    holes = [2, 7, 8, 13]
    valid = np.ones((1, length), bool)
    valid[0, holes] = False
    real = np.flatnonzero(valid[0])
    segments = np.zeros((1, length), np.int32)
    segments[0, real[:6]] = 1
    segments[0, real[6:]] = 2
    ids = np.asarray(jax.random.randint(jax.random.key(3), (1, length), 1, 32), np.int32)
    closed = ids[:, valid[0]], segments[:, valid[0]]
    return (jnp.asarray(ids), jnp.asarray(valid), jnp.asarray(segments)), tuple(map(jnp.asarray, closed))


def test_padded_packed_rows_reset_at_each_document():
    """Padding slots and packed documents together: every real token of the padded row computes what it
    computes with the holes closed, the second document starting fresh after the gap, through the model's
    `hidden_and_mtp_inputs` and through `LMObjective.token_scores` with both token fields (check 1 on fresh
    models). The padded row without its segment ids, as the layer computed before it passed segments to its
    stateful path, sits far outside the rule."""
    model = tiny_lm()
    (ids, valid, segments), (closed_ids, closed_segments) = padded_packed_row(16)

    def draw(seed):
        return (model.init(jax.random.key(seed), ids),)

    def hidden(model, params, tokens, **fields):
        # The flat fields laid out once (`field_coordinates`), the layout the trunk reads.
        return model.apply(params, tokens, method=CausalTransformer.hidden_and_mtp_inputs,
                           coordinates=field_coordinates(fields, tokens.shape))[0]

    def padded(model, params):
        return hidden(model, params, ids, attention_mask=valid, segment_ids=segments)[0][valid[0]]

    def closed(model, params):
        return hidden(model, params, closed_ids, segment_ids=closed_segments)[0]

    wide = model.clone(dtype=jnp.float64)
    assert_as_exact(functools.partial(padded, model), functools.partial(closed, model), draw,
                    "the padded packed row", seeds=MIXER_SEEDS["padded packed"],
                    truth_fn=functools.partial(closed, wide))
    (params,) = draw(0)
    with jax.enable_x64(new_val=True):
        truth = np.asarray(closed(wide, widened(params)))
    outside(hidden(model, params, ids, attention_mask=valid)[0][valid[0]], closed(model, params), truth,
            "the row without segment ids")

    (ids_17, valid_17, segments_17), (closed_ids_17, closed_segments_17) = padded_packed_row(17)
    kept = np.asarray(valid_17[0, :-1])
    count = int(kept.sum())

    def scored(model, params):
        fields = {"attention_mask": valid_17, "segment_ids": segments_17}
        return LMObjective(model, 16).token_scores(params, ModelInputs(ids_17, fields)).hidden[0][kept]

    def alone(model, params):
        return LMObjective(model, 12).token_scores(
            params, ModelInputs(closed_ids_17, {"segment_ids": closed_segments_17})).hidden[0][:count]

    assert_as_exact(functools.partial(scored, model), functools.partial(alone, model), draw,
                    "the scored padded row", seeds=MIXER_SEEDS["scored padded"],
                    truth_fn=functools.partial(alone, wide))


def tiny_lm(**overrides) -> CausalTransformer:
    """Two SSD layers with no feed-forward, the reference's block."""
    config = {"vocab_size": 32, "emb_features": 16, "num_layers": 2, "num_heads": 4, "mlp_features": 0,
              "max_seq_len": 16, "tie_embeddings": False, "qk_norm": False,
              "mixer": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 8, "state_size": 5,
                        "n_groups": 2, "chunk_size": 4}}}
    return CausalTransformer(**{**config, **overrides})


def test_the_model_decodes_as_it_scores_in_parallel():
    """A prefill of four tokens and eight single-token steps, every layer's state riding the flax cache
    collection, against the same twelve tokens scored by full passes, each step's at the width it reads
    (`as_stepped`), check 1 on fresh models and tokens."""
    model = tiny_lm()

    def draw(seed):
        ids = jax.random.randint(jax.random.key(seed + 1000), (2, 12), 0, 32)
        return model.init(jax.random.key(seed), ids), ids

    def decoded(model, params, ids):
        cache = model.apply(params, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
        logits, mutated = model.apply({**params, "cache": cache}, ids[:, :4], decode=True, mutable=["cache"])
        steps = [logits[:, -1]]
        for position in range(4, 12):
            logits, mutated = model.apply({**params, **mutated}, ids[:, position:position + 1],
                                          decode=True, mutable=["cache"])
            steps.append(logits[:, -1])
        return jnp.stack(steps, axis=1)

    def full(model, params, ids):
        return as_stepped(lambda held: model.apply(params, held), ids, 4)[:, 3:]

    params, _ = draw(0)
    cache = model.apply(params, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
    assert_as_exact(functools.partial(decoded, model), functools.partial(full, model), draw,
                    "the decoded logits", seeds=MIXER_SEEDS["model decoded"],
                    truth_fn=functools.partial(full, model.clone(dtype=jnp.float64)))


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
    from residual_orders import orders, permuted, residual_width

    from dew.nn.kernels import generation
    from dew.nn.kernels.ssd import ssd_kernel_runs

    assert ssd_kernel_runs(64, 64, 64, "tpu", dtype=jnp.float32)
    model, devices = ssd_lm(), jax.devices()
    ids = np.asarray(jax.random.randint(jax.random.key(0), (2 * len(devices), 129), 0, 64))
    variables = jax.device_get(model.init(jax.random.key(1), jnp.asarray(ids[:, :-1])))
    units = residual_width(variables)

    def ordered(result, order):
        """The step's gradients with the residual's units in `order`, and its logits, which no order
        moves."""
        gradients, logits = result
        return permuted({"params": gradients}, order)["params"], logits

    # The float64 oracle runs on the host: a TPU emulates float64.
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        wide = stepped(model.clone(dtype=jnp.float64))
        exact = jax.device_get(wide(widened(variables["params"]), ids))
        last = orders(units, 2, seed=0)[-1]
        found = jax.device_get(wide(permuted(widened(variables), last)["params"], ids))
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
            params = jax.device_put(permuted(variables, order)["params"], whole)
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
