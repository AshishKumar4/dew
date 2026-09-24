"""Mamba-2 against the transformers 5.16.1 reference, at fp32.

tools/mamba2_reference.py runs `mamba2_chunk_scan`,
`mamba2_selective_state_update` and one `Mamba2Mixer` layer
(modeling_mamba2.py) on fixed-seed operands and writes what they produced;
the tests here run dew's forms on the same operands and the same weights
and hold them to the reference. Nothing here imports torch.

Tolerances and the differences actually observed, fp32 on CPU:

- chunked scan, zero state     : output 5.5e-06 on outputs of magnitude up
  to 30, state 7.2e-07, tolerance 1e-5
- chunked scan, carried state  : the same numbers, tolerance 1e-5
- one recurrent step           : output 9.5e-07, state 9.5e-07, tolerance 1e-5
- chunked vs recurrent         : output 3.8e-06, state 1.4e-06, tolerance 1e-5
- the whole layer              : 1.6e-06 on outputs of magnitude up to 3.0
  over 70 tokens (4 heads of 6, 2 groups, state 5), tolerance 1e-5; in
  bfloat16 against the reference's own bfloat16 forward 3.1e-02 (one bf16
  ulp at that magnitude; the reference's bfloat16 sits 8.2e-02 from its
  own fp32, dew's 5.1e-02), tolerance 1e-1
- its decode path              : 8.3e-07 against its parallel forward
- the two-layer model          : 4.8e-07 between a prefill plus single-token
  steps and the parallel logits, of magnitude 3.1
"""

import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import AttentionMetadata, ModelInputs
from dew.nn.mixers.mamba2 import Mamba2, Mamba2Mixer, chunk_ssd, recurrent_ssd, segment_sum
from dew.objectives.lm import LMObjective
from dew.registry import mixers

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
BOUND = 1e-5


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


@pytest.fixture(scope="module")
def geometry():
    return json.loads((FIXTURES / "config.json").read_text())


def largest(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float32) - np.asarray(right, np.float32))))


def operands(reference):
    """The scan's operands as the module hands them to it: the step already
    through softplus with its bias (the reference does that inside)."""
    step = jax.nn.softplus(jnp.asarray(reference["scan.dt"]) + jnp.asarray(reference["scan.dt_bias"]))
    return (jnp.asarray(reference["scan.x"]), step, jnp.asarray(reference["scan.A"]),
            jnp.asarray(reference["scan.B"]), jnp.asarray(reference["scan.C"]), jnp.asarray(reference["scan.D"]))


def test_segment_sum_is_the_masked_cumulative_sum():
    """`out[i, j] = sum_{j < k <= i} x[k]` on and below the diagonal, -inf
    above, the reference's spelling (modeling_mamba2.py:73-90)."""
    x = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    out = segment_sum(x)
    expected = np.full((4, 4), -np.inf, np.float32)
    for i in range(4):
        for j in range(i + 1):
            expected[i, j] = float(np.sum(np.asarray(x)[j + 1:i + 1]))
    assert np.array_equal(np.asarray(out), expected)


def test_the_chunked_scan_matches_the_reference(reference):
    out, final = chunk_ssd(*operands(reference), None, 32)
    assert largest(out, reference["scan.output"]) < BOUND
    assert largest(final, reference["scan.final"]) < BOUND


def test_a_carried_state_matches_the_reference(reference):
    out, final = chunk_ssd(*operands(reference), jnp.asarray(reference["scan.initial"]), 32)
    assert largest(out, reference["scan.output_carried"]) < BOUND
    assert largest(final, reference["scan.final_carried"]) < BOUND


def test_one_recurrent_step_matches_selective_state_update(reference):
    x, step, A, B, C, D = operands(reference)
    out, final = recurrent_ssd(x[:, :1], step[:, :1], A, B[:, :1], C[:, :1], D,
                               jnp.asarray(reference["scan.initial"]))
    assert largest(out[:, 0], reference["step.output"]) < BOUND
    assert largest(final, reference["step.state"]) < BOUND


def test_the_chunked_and_recurrent_forms_agree(reference):
    initial = jnp.asarray(reference["scan.initial"])
    chunked, chunked_final = chunk_ssd(*operands(reference), initial, 32)
    recurrent, recurrent_final = recurrent_ssd(*operands(reference), initial)
    assert largest(chunked, recurrent) < BOUND
    assert largest(chunked_final, recurrent_final) < BOUND


def test_a_sequence_split_in_two_carries_its_state_across_the_cut(reference):
    """Chunk boundaries that do not line up with the cut: 70 tokens as 30
    then 40 at a chunk of 32, the second half from the first's final state."""
    x, step, A, B, C, D = operands(reference)
    whole, final = chunk_ssd(x, step, A, B, C, D, None, 32)
    head, carried = chunk_ssd(x[:, :30], step[:, :30], A, B[:, :30], C[:, :30], D, None, 32)
    tail, tail_final = chunk_ssd(x[:, 30:], step[:, 30:], A, B[:, 30:], C[:, 30:], D, carried, 32)
    assert largest(jnp.concatenate([head, tail], axis=1), whole) < BOUND
    assert largest(tail_final, final) < BOUND


def test_the_chunk_size_does_not_change_the_scan(reference):
    x, step, A, B, C, D = operands(reference)
    at_32, final_32 = chunk_ssd(x, step, A, B, C, D, None, 32)
    at_7, final_7 = chunk_ssd(x, step, A, B, C, D, None, 7)
    at_256, final_256 = chunk_ssd(x, step, A, B, C, D, None, 256)
    assert largest(at_7, at_32) < BOUND and largest(final_7, final_32) < BOUND
    assert largest(at_256, at_32) < BOUND and largest(final_256, final_32) < BOUND


def layer(geometry, **overrides) -> Mamba2:
    return Mamba2(emb_features=geometry["hidden_size"], num_heads=geometry["num_heads"],
                  head_dim=geometry["head_dim"], state_size=geometry["state_size"],
                  n_groups=geometry["n_groups"], conv_kernel=geometry["conv_kernel"],
                  chunk_size=geometry["chunk_size"], norm_eps=geometry["layer_norm_epsilon"], **overrides)


def layer_params(reference):
    """The reference layer's state dict under the module's names: kernels
    transposed from torch's [out, in], everything else as stored."""
    weights = {name[len("layer."):]: value for name, value in reference.items()
               if name.startswith("layer.") and name not in ("layer.hidden", "layer.output", "layer.output_bf16")}
    return {"params": {
        "in_proj": {"kernel": jnp.asarray(weights["in_proj.weight"].T)},
        "conv1d": {"weight": jnp.asarray(weights["conv1d.weight"]), "bias": jnp.asarray(weights["conv1d.bias"])},
        "A_log": jnp.asarray(weights["A_log"]), "dt_bias": jnp.asarray(weights["dt_bias"]),
        "D": jnp.asarray(weights["D"]),
        "norm": {"weight": jnp.asarray(weights["norm.weight"])},
        "out_proj": {"kernel": jnp.asarray(weights["out_proj.weight"].T)}}}


def test_the_layer_matches_mamba2_mixer(reference, geometry):
    """The whole mixer on the reference's own weights: the projection, the
    conv with its bias, the step, the scan with two heads per group, the
    D skip, the gated norm and the output projection. Largest observed
    difference 1.6e-06 over 70 tokens on outputs of magnitude up to 3.0."""
    module = layer(geometry)
    variables = layer_params(reference)
    template = jax.eval_shape(module.init, jax.random.key(0), jnp.zeros((1, 4, geometry["hidden_size"])))
    assert jax.tree.map(jnp.shape, variables) == jax.tree.map(jnp.shape, template)

    out = module.apply(variables, jnp.asarray(reference["layer.hidden"]))

    assert largest(out, reference["layer.output"]) < BOUND


def test_the_layer_in_bfloat16_matches_the_reference_in_bfloat16(reference, geometry):
    """Compute in bfloat16 with fp32 parameters, against the reference run
    in bfloat16 end to end; the scan itself is fp32 in both. Largest
    observed difference 3.1e-02 on outputs of magnitude up to 3.0, one
    bf16 ulp at that magnitude; the two round at different points, and
    each sits further from the fp32 output than they do from each other
    (8.2e-02 for the reference, 5.1e-02 for dew)."""
    module = layer(geometry, dtype=jnp.bfloat16)
    out = jnp.asarray(module.apply(layer_params(reference), jnp.asarray(reference["layer.hidden"], jnp.bfloat16)))
    assert out.dtype == jnp.bfloat16
    assert largest(out, reference["layer.output_bf16"]) < 1e-1


def test_the_layer_decodes_as_it_prefills(reference, geometry):
    """A prefill against the cache and single-token steps after it reproduce
    the parallel forward: the conv tail and the SSM state both cross the
    step boundary. Largest observed difference 8.3e-07 against the parallel
    forward on the same 70 tokens."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])
    parallel = module.apply(variables, hidden)

    cache = module.apply(variables, hidden[:, :1], decode=True, mutable=["cache"])[1]["cache"]
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
    prefill = 5
    out, mutated = module.apply({**variables, "cache": cache}, hidden[:, :prefill],
                                decode=True, mutable=["cache"])
    steps = [out]
    for position in range(prefill, hidden.shape[1]):
        out, mutated = module.apply({**variables, **mutated}, hidden[:, position:position + 1],
                                    decode=True, mutable=["cache"])
        steps.append(out)

    assert largest(jnp.concatenate(steps, axis=1), parallel) < BOUND


def test_the_decode_state_is_a_fixed_size(reference, geometry):
    """The mixer's analogue of the KV cache does not grow with the tokens:
    the conv tail is the last K-1 columns and the SSM state one
    `[H, P, N]` block per row, whatever was decoded."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])
    cache = module.apply(variables, hidden[:, :1], decode=True, mutable=["cache"])[1]["cache"]
    _, mutated = module.apply({**variables, "cache": cache}, hidden, decode=True, mutable=["cache"])
    conv_dim = geometry["num_heads"] * geometry["head_dim"] + 2 * geometry["n_groups"] * geometry["state_size"]
    assert jax.tree.map(jnp.shape, mutated["cache"]) == {
        "conv_state": (2, conv_dim, geometry["conv_kernel"] - 1),
        "ssm_state": (2, geometry["num_heads"], geometry["head_dim"], geometry["state_size"])}


def test_padded_rows_preserve_the_state_and_the_history(reference, geometry):
    """A row paused on padding neither writes its state nor advances its
    conv history, and its padded slots emit zero: the other row's stream,
    with the holes closed, is what the padded run computes."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])[:, :12]
    valid = jnp.asarray([[1] * 12, [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]], bool)
    compact = hidden[1][valid[1]][None]

    padded = jnp.asarray(module.apply(variables, hidden, attention_metadata=AttentionMetadata(valid=valid)))
    plain = jnp.asarray(module.apply(variables, hidden[:1]))
    closed = jnp.asarray(module.apply(variables, compact))

    assert largest(padded[0], plain[0]) < BOUND
    assert largest(padded[1][valid[1]], closed[0]) < BOUND
    assert not jnp.any(padded[1][~valid[1]])


def test_packed_documents_run_as_if_each_ran_alone(reference, geometry):
    """A row of packed documents computes, for each document, what that
    document computes alone: neither the SSM state nor the conv's taps
    cross a segment change. Documents of 20, 2 (shorter than the conv's
    three-token history) and 48 tokens, the cuts off the chunk grid.
    Largest observed difference 8.3e-07 on outputs of magnitude up to 2.9;
    the same row without segment ids, which carries the state and the conv
    across the cuts, is 2.9 away."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])[:1]
    cuts = [0, 20, 22, 70]
    segments = jnp.asarray(np.repeat(np.arange(1, 4), np.diff(cuts))[None], jnp.int32)

    packed = module.apply(variables, hidden, segment_ids=segments)
    alone = jnp.concatenate([module.apply(variables, hidden[:, start:end])
                             for start, end in itertools.pairwise(cuts)], axis=1)

    assert largest(packed, alone) < BOUND
    assert largest(module.apply(variables, hidden), alone) > 1e-1


def test_the_chunked_and_recurrent_forms_reset_alike(reference):
    """Document starts drop the state in both forms, including a start on
    the first token of a chunk and one inside it. Largest observed
    difference 1.9e-06 on outputs up to 30 and 2.4e-07 on the state; the
    scan without the starts is 3.4 away."""
    x, step, A, B, C, D = operands(reference)
    starts = jnp.zeros(step.shape[:2], bool).at[0, 32].set(True).at[1, 7].set(True).at[1, 45].set(True)
    initial = jnp.asarray(reference["scan.initial"])
    chunked, chunked_final = chunk_ssd(x, step, A, B, C, D, initial, 32, starts=starts)
    recurrent, recurrent_final = recurrent_ssd(x, step, A, B, C, D, initial, starts=starts)
    assert largest(chunked, recurrent) < BOUND
    assert largest(chunked_final, recurrent_final) < BOUND
    assert largest(chunked, chunk_ssd(x, step, A, B, C, D, initial, 32)[0]) > 1e-1


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
    token fields. Observed 0.0 apart in fp32. The padded row without its
    segment ids, and the layer before it passed segments to its stateful
    path, sit 2.7 away on states of magnitude 2.7."""
    model = tiny_lm()
    (ids, valid, segments), (closed_ids, closed_segments) = padded_packed_row(16)
    params = model.init(jax.random.key(0), ids)

    def hidden(tokens, **fields):
        return model.apply(params, tokens, method=CausalTransformer.hidden_and_mtp_inputs, **fields)[0]

    closed = hidden(closed_ids, segment_ids=closed_segments)[0]
    padded = hidden(ids, attention_mask=valid, segment_ids=segments)[0]
    assert largest(padded[valid[0]], closed) < BOUND
    carried = hidden(ids, attention_mask=valid)[0]
    assert largest(carried[valid[0]], closed) > 1e-1

    (ids, valid, segments), (closed_ids, closed_segments) = padded_packed_row(17)
    scored = LMObjective(model, 16).token_scores(
        params, ModelInputs(ids, {"attention_mask": valid, "segment_ids": segments}))
    alone = LMObjective(model, 12).token_scores(
        params, ModelInputs(closed_ids, {"segment_ids": closed_segments}))
    kept = valid[0, :-1]
    assert largest(scored.hidden[0][kept], alone.hidden[0][:int(kept.sum())]) < BOUND


def test_the_kind_builds_from_the_configs_fields():
    record = {"kind": "mamba2", "num_heads": 4, "head_dim": 6, "state_size": 5, "n_groups": 2,
              "conv_kernel": 4, "chunk_size": 8}
    mixer = mixers.from_record(record)
    assert isinstance(mixer, Mamba2Mixer)
    assert mixer.n_groups == 2 and mixer.chunk_size == 8
    with pytest.raises(ValueError):
        mixers.from_record({**record, "linear_num_heads": 4})


def tiny_lm(**overrides) -> CausalTransformer:
    """Two SSD layers with no feed-forward, the reference's block."""
    config = {"vocab_size": 32, "emb_features": 16, "num_layers": 2, "num_heads": 4, "mlp_features": 0,
              "max_seq_len": 16, "tie_embeddings": False, "qk_norm": False,
              "mixer": {"kind": "mamba2", "num_heads": 4, "head_dim": 8, "state_size": 5,
                        "n_groups": 2, "chunk_size": 4}}
    return CausalTransformer(**{**config, **overrides})


def test_a_layer_without_a_feed_forward_holds_the_mixer_alone():
    model = tiny_lm()
    variables = jax.eval_shape(model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    layer_0 = variables["params"]["layers_0"]
    assert set(layer_0) == {"input_layernorm", "self_attn"}
    assert set(layer_0["self_attn"]) == {"A_log", "D", "conv1d", "dt_bias", "in_proj", "norm", "out_proj"}
    with pytest.raises(ValueError, match="0 for a layer without"):
        jax.eval_shape(tiny_lm(mlp_features=-1).init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))


def test_the_model_decodes_as_it_scores_in_parallel():
    """A prefill of four tokens and eight single-token steps, every layer's
    state riding the flax cache collection, against the same twelve tokens
    scored at once. Largest observed logit difference 4.8e-07 on logits of
    magnitude 3.1."""
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

    assert largest(jnp.stack(steps, axis=1), full[:, 3:]) < BOUND
