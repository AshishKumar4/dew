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

import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decode_through
from reference_error import (
    FACTOR,
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    declared_seeds,
    indexed_recording,
    recording,
    widened,
)
from ssd_support import mamba2_draw, mamba2_run

from dew.accuracy import distance
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import AttentionMetadata, MixerCall, ModelInputs, coordinates
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


def as_exact(dew, reference, truth, label: str):
    """Each leaf of `dew` within the rule of `reference`'s distance from `truth`."""
    for index, (mine, theirs, exact) in enumerate(zip(jax.tree.leaves(dew), jax.tree.leaves(reference),
                                                      jax.tree.leaves(truth), strict=True)):
        assert_as_exact_as_the_reference(np.asarray(mine), np.asarray(theirs), np.asarray(exact),
                                         f"{label} {index}")


def operands(reference):
    """The scan's operands as the module hands them to it: the step already
    through softplus with its bias (the reference does that inside)."""
    step = jax.nn.softplus(jnp.asarray(reference["scan.dt"]) + jnp.asarray(reference["scan.dt_bias"]))
    return (
        jnp.asarray(reference["scan.x"]),
        step,
        jnp.asarray(reference["scan.A"]),
        jnp.asarray(reference["scan.B"]),
        jnp.asarray(reference["scan.C"]),
        jnp.asarray(reference["scan.D"]),
    )


def assert_recorded_as_exact(reference, site):
    """The recorded reference and its torch-float64 truth on each fresh native-input/orbit draw."""
    seeds = declared_seeds(SEEDED)[site]
    # The truth is Dew's float64 path, which the recorder held to transformers' float64 at every seed.
    draw, recorded, truth = indexed_recording(reference, site, seeds, inputs_of=mamba2_draw,
                                              truth_fn=jax.jit(lambda inputs: mamba2_run(site, inputs)))
    run = jax.jit(lambda index, inputs: mamba2_run(site, inputs))
    assert_as_exact(run, recorded, draw, f"transformers {site}", seeds=seeds, truth_fn=truth)


def test_the_chunked_scan_matches_the_reference(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "scan")


def test_a_carried_state_matches_the_reference(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "carried")


def test_one_recurrent_step_matches_selective_state_update(seeded_reference):
    assert_recorded_as_exact(seeded_reference, "step")


def test_the_chunked_and_recurrent_forms_agree(reference):
    initial = jnp.asarray(reference["scan.initial"])
    chunked = chunk_ssd(*operands(reference), initial, 32)
    recurrent = recurrent_ssd(*operands(reference), initial)
    as_exact(recurrent, chunked, in_float64(chunk_ssd, *operands(reference), initial, 32), "recurrent")


def test_a_sequence_split_in_two_carries_its_state_across_the_cut(reference):
    """Chunk boundaries that do not line up with the cut: 70 tokens as 30
    then 40 at a chunk of 32, the second half from the first's final state."""
    x, step, A, B, C, D = operands(reference)
    whole = chunk_ssd(x, step, A, B, C, D, None, 32)
    head, carried = chunk_ssd(x[:, :30], step[:, :30], A, B[:, :30], C[:, :30], D, None, 32)
    tail, tail_final = chunk_ssd(x[:, 30:], step[:, 30:], A, B[:, 30:], C[:, 30:], D, carried, 32)
    as_exact((jnp.concatenate([head, tail], axis=1), tail_final), whole,
             in_float64(chunk_ssd, x, step, A, B, C, D, None, 32), "the cut sequence")


CHUNK_SIZE_SEEDS = 64
"""Sized by seeds_for against the 32-token XLA chunk scan (2026-10-20, armada)."""


def chunk_size_draw(seed):
    """Fresh inputs at the fixture geometry under joint head, P-channel and N-coordinate permutations."""
    arrays = mamba2_draw(seed)
    arrays["dt"] = jax.nn.softplus(jnp.asarray(arrays["dt"]) + jnp.asarray(arrays["dt_bias"]))
    return tuple(arrays[name] for name in ("x", "dt", "A", "B", "C", "D"))


def test_the_chunk_size_does_not_change_the_scan():
    at_32 = jax.jit(lambda *inputs: chunk_ssd(*inputs, None, 32))
    for size in (7, 256):
        run = jax.jit(lambda *inputs, size=size: chunk_ssd(*inputs, None, size))
        assert_as_exact(run, at_32, chunk_size_draw, f"a chunk of {size}", seeds=CHUNK_SIZE_SEEDS)


def layer(geometry, **overrides) -> Mamba2:
    return Mamba2(emb_features=geometry["hidden_size"], num_heads=geometry["num_heads"],
                  head_dim=geometry["head_dim"], state_size=geometry["state_size"],
                  n_groups=geometry["n_groups"], conv_kernel=geometry["conv_kernel"],
                  chunk_size=geometry["chunk_size"], norm_eps=geometry["layer_norm_epsilon"], **overrides)


def wide_apply(module, variables, *args, **kwargs):
    """`module` applied in float64 on `variables` and `args` widened, as host arrays."""
    with jax.enable_x64(new_val=True):
        wide = module.clone(dtype=jnp.float64)
        return np.asarray(wide.apply(widened(variables), *widened(args), **kwargs))


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


def test_the_layer_matches_mamba2_mixer(reference, geometry):
    """The whole mixer on the reference's own weights: the projection, the
    conv with its bias, the step, the scan with two heads per group, the
    D skip, the gated norm and the output projection."""
    module = layer(geometry)
    variables = layer_params(reference)
    template = jax.eval_shape(module.init, jax.random.key(0), jnp.zeros((1, 4, geometry["hidden_size"])))
    assert jax.tree.map(jnp.shape, variables) == jax.tree.map(jnp.shape, template)

    out = module.apply(variables, jnp.asarray(reference["layer.hidden"]))

    as_exact(out, reference["layer.output"], reference["layer.output_f64"], "the layer")


def test_the_layer_in_bfloat16_matches_the_reference_in_bfloat16(reference, geometry):
    """Compute in bfloat16 with fp32 parameters, against the reference run
    in bfloat16 end to end; the scan itself is fp32 in both."""
    module = layer(geometry, dtype=jnp.bfloat16)
    out = jnp.asarray(
        module.apply(layer_params(reference), jnp.asarray(reference["layer.hidden"], jnp.bfloat16))
    )
    assert out.dtype == jnp.bfloat16
    as_exact(np.asarray(out, np.float32), reference["layer.output_bf16"], reference["layer.output_f64"],
             "the layer in bfloat16")


def test_the_layer_decodes_as_it_prefills(reference, geometry):
    """A prefill against the cache and single-token steps after it reproduce
    the parallel forward: the conv tail and the SSM state both cross the
    step boundary."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])
    parallel = module.apply(variables, hidden)

    decoded, cache, _ = decode_through(module, variables, hidden, prefill=5)
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
    as_exact(decoded, parallel, wide_apply(module, variables, hidden), "decoded")


def test_padded_rows_preserve_the_state_and_the_history(reference, geometry):
    """A row paused on padding neither writes its state nor advances its
    conv history, and its padded slots emit zero: the other row's stream,
    with the holes closed, is what the padded run computes."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])[:, :12]
    valid = jnp.asarray([[1] * 12, [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]], bool)
    compact = hidden[1][valid[1]][None]

    metadata = AttentionMetadata(coordinates=coordinates(valid=valid))
    padded = jnp.asarray(module.apply(variables, hidden, MixerCall(metadata)))
    plain = jnp.asarray(module.apply(variables, hidden[:1]))
    closed = jnp.asarray(module.apply(variables, compact))

    as_exact(padded[0], plain[0], wide_apply(module, variables, hidden[:1])[0], "the full row")
    as_exact(padded[1][valid[1]], closed[0], wide_apply(module, variables, compact)[0], "the padded row")
    assert not jnp.any(padded[1][~valid[1]])


SPLIT_ROWS = {
    # Every slot real, as a pool of processes fills validity where no producer padded.
    "all_valid": (np.ones(32, bool), None),
    # Right-padded: real tokens first, the tail padding inside the last two shards.
    "right_padded": (np.arange(32) < 21, None),
    # Left-padded: a prompt's padding in front, across the first shard and into the second.
    "left_padded": (np.arange(32) >= 11, None),
    # Gaps inside a document, one at a shard's cut, and a shard holding a single real token, fewer
    # than the conv's three-token history: its next shard reads the ones before it too.
    "gaps": (~np.isin(np.arange(32), [2, 7, 8, 13, 16, 17, 18, 19, 20, 21, 22, 25]), None),
    # Packed documents with gaps between and inside them, one starting on a shard's first real token.
    "packed": (~np.isin(np.arange(32), [5, 6, 14, 15, 30, 31]),
               np.repeat([1, 1, 2, 3], [5, 11, 8, 8])),
    # A whole interior shard of padding with real tokens on both sides: the history and the state
    # cross it unchanged.
    "empty_shard": (~((np.arange(32) >= 8) & (np.arange(32) < 16)), None),
    # The same with one document continuing across the empty shard and another starting after it.
    "empty_shard_packed": (~((np.arange(32) >= 8) & (np.arange(32) < 16)),
                           np.repeat([1, 2], [20, 12])),
}
"""Each row of the sequence split's check: its real tokens and its documents (None for one)."""

SPLIT_BATCH = 8
"""The rows a sequence-split draw holds: every parameter's gradient, the heads' `D` and `A_log` too,
sums over all of them."""

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
    device: a padded slot neither reads nor advances the conv or the state, each real token's conv
    reading the real tokens before it in its document across gaps and shards, and padded slots emit
    zero. Output and input gradient are check 1's against the unsplit layer, from float64, over drawn
    hidden states, and so are the parameters' gradients. A pool of processes always carries
    validity, so this is every multi-host run."""
    module = layer(geometry)
    variables = layer_params(reference)
    valid, segments = SPLIT_ROWS[row]
    fields = {"valid": jnp.asarray(np.broadcast_to(valid, (SPLIT_BATCH, 32)))}
    if segments is not None:
        fields["segments"] = jnp.asarray(np.broadcast_to(segments, (SPLIT_BATCH, 32)), jnp.int32)
    call = MixerCall(AttentionMetadata(coordinates=coordinates(**fields)))
    width = geometry["hidden_size"]

    def draw(seed):
        rng = np.random.default_rng([0x5E9, seed])
        return (rng.standard_normal((SPLIT_BATCH, 32, width)).astype(np.float32),
                rng.standard_normal((SPLIT_BATCH, 32, width)).astype(np.float32))

    def run(model, params, mesh=None):
        def pulled(hidden, cotangent):
            out, pull = jax.vjp(lambda params, hidden: model.apply(params, hidden, call), params, hidden)
            return out, *pull(cotangent)
        if mesh is None:
            return jax.jit(pulled)
        @as_given
        def split(hidden, cotangent):
            with jax.set_mesh(mesh):
                return jax.jit(pulled)(hidden, cotangent)
        return split

    def truth(hidden, cotangent):
        with jax.enable_x64(new_val=True):
            return run(module.clone(dtype=jnp.float64), widened(variables))(*widened((hidden, cotangent)))

    mesh = MeshSpec(sequence=4).build(jax.devices()[:4])
    assert_as_exact(run(module, variables, mesh), run(module, variables), draw,
                    f"mamba2 split over the sequence, {row}", seeds=SPLIT_SEEDS[row], truth_fn=truth)
    split = run(module, variables, mesh)(*draw(0))[0]
    assert not np.any(np.asarray(split)[:, ~valid])


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


def test_packed_documents_run_as_if_each_ran_alone(reference, geometry):
    """A row of packed documents computes, for each document, what that
    document computes alone: neither the SSM state nor the conv's taps
    cross a segment change. Documents of 20, 2 (shorter than the conv's
    three-token history) and 48 tokens, the cuts off the chunk grid. The
    same row without segment ids carries the state and the conv across the
    cuts, far outside the rule."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])[:1]
    cuts = [0, 20, 22, 70]
    segments = jnp.asarray(np.repeat(np.arange(1, 4), np.diff(cuts))[None], jnp.int32)

    packed = module.apply(variables, hidden,
                          MixerCall(AttentionMetadata(coordinates=coordinates(segments=segments))))
    alone = jnp.concatenate([module.apply(variables, hidden[:, start:end])
                             for start, end in itertools.pairwise(cuts)], axis=1)

    truth = np.concatenate([wide_apply(module, variables, hidden[:, start:end])
                            for start, end in itertools.pairwise(cuts)], axis=1)
    as_exact(packed, alone, truth, "the packed row")
    outside(module.apply(variables, hidden), alone, truth, "the row without segment ids")


def test_the_chunked_and_recurrent_forms_reset_alike(reference):
    """Document starts drop the state in both forms, including a start on
    the first token of a chunk and one inside it; the scan without the
    starts far outside the rule."""
    x, step, A, B, C, D = operands(reference)
    starts = jnp.zeros(step.shape[:2], bool).at[0, 32].set(True).at[1, 7].set(True).at[1, 45].set(True)
    initial = jnp.asarray(reference["scan.initial"])
    chunked = chunk_ssd(x, step, A, B, C, D, initial, 32, starts=starts)
    recurrent = recurrent_ssd(x, step, A, B, C, D, initial, starts=starts)
    truth = in_float64(chunk_ssd, x, step, A, B, C, D, initial, 32, starts=starts)
    as_exact(recurrent, chunked, truth, "recurrent with starts")
    outside(chunk_ssd(x, step, A, B, C, D, initial, 32)[0], chunked[0], truth[0], "the scan without starts")


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
    """Padding slots and packed documents together: every real token of the
    padded row computes what it computes with the holes closed, the second
    document starting fresh after the gap, through the model's
    `hidden_and_mtp_inputs` and through `LMObjective.token_scores` with both
    token fields. The padded row without its segment ids, as the layer
    computed before it passed segments to its stateful path, sits far
    outside the rule."""
    model = tiny_lm()
    (ids, valid, segments), (closed_ids, closed_segments) = padded_packed_row(16)
    params = model.init(jax.random.key(0), ids)

    def hidden(tokens, model=model, params=params, **fields):
        return model.apply(params, tokens, method=CausalTransformer.hidden_and_mtp_inputs, **fields)[0]

    closed = hidden(closed_ids, segment_ids=closed_segments)[0]
    padded = hidden(ids, attention_mask=valid, segment_ids=segments)[0]
    with jax.enable_x64(new_val=True):
        truth = np.asarray(hidden(closed_ids, model.clone(dtype=jnp.float64), widened(params),
                                  segment_ids=closed_segments)[0])
    as_exact(padded[valid[0]], closed, truth, "the padded packed row")
    outside(hidden(ids, attention_mask=valid)[0][valid[0]], closed, truth, "the row without segment ids")

    (ids, valid, segments), (closed_ids, closed_segments) = padded_packed_row(17)
    scored = LMObjective(model, 16).token_scores(
        params, ModelInputs(ids, {"attention_mask": valid, "segment_ids": segments}))
    alone = LMObjective(model, 12).token_scores(
        params, ModelInputs(closed_ids, {"segment_ids": closed_segments}))
    kept = valid[0, :-1]
    with jax.enable_x64(new_val=True):
        wide = LMObjective(model.clone(dtype=jnp.float64), 12).token_scores(
            widened(params), ModelInputs(closed_ids, {"segment_ids": closed_segments}))
    count = int(kept.sum())
    as_exact(scored.hidden[0][kept], alone.hidden[0][:count], np.asarray(wide.hidden[0][:count]),
             "the scored padded row")


def tiny_lm(**overrides) -> CausalTransformer:
    """Two SSD layers with no feed-forward, the reference's block."""
    config = {"vocab_size": 32, "emb_features": 16, "num_layers": 2, "num_heads": 4, "mlp_features": 0,
              "max_seq_len": 16, "tie_embeddings": False, "qk_norm": False,
              "mixer": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 8, "state_size": 5,
                        "n_groups": 2, "chunk_size": 4}}}
    return CausalTransformer(**{**config, **overrides})


def test_the_model_decodes_as_it_scores_in_parallel():
    """A prefill of four tokens and eight single-token steps, every layer's
    state riding the flax cache collection, against the same twelve tokens
    scored at once."""
    model = tiny_lm()
    rng = jax.random.key(0)
    ids = jax.random.randint(rng, (2, 12), 0, 32)
    params = model.init(rng, ids)
    full = jnp.asarray(model.apply(params, ids))

    cache = model.apply(params, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
    logits, mutated = model.apply({**params, "cache": cache}, ids[:, :4], decode=True, mutable=["cache"])
    steps = [logits[:, -1]]
    for position in range(4, 12):
        logits, mutated = model.apply({**params, **mutated}, ids[:, position:position + 1],
                                      decode=True, mutable=["cache"])
        steps.append(logits[:, -1])

    with jax.enable_x64(new_val=True):
        truth = np.asarray(model.clone(dtype=jnp.float64).apply(widened(params), ids))
    as_exact(jnp.stack(steps, axis=1), full[:, 3:], truth[:, 3:], "the decoded logits")
