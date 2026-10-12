"""Analytical counts on real rows, composed by the objective, not the HLO.

Closed-form arithmetic catches padded lengths, total-expert instead of
active-expert counts, omitted loop passes, accidental remat duplication,
packed documents treated as one attention square, and a vocabulary head
incorrectly charged to a decision model.
"""

import json
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.extend import core as jaxpr_core
from recording import RecordingTracker

from dew.data.text import ByteTokenizer
from dew.decision.head import DecisionHead, JointSchemaHead
from dew.decision.layout import DecisionInputs, Specials, StateFirstLayout
from dew.decision.objective import DecisionObjective
from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import GatedMLP, RematPolicy
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.jepa import FactorizedTokenStack, JepaEncoder, JepaPredictor, TokenStack
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.flops import (
    MissingFlopsCount,
    attention_flops,
    counted_flops,
    delta_flops,
    dense_training_flops,
    projection_flops,
)
from dew.nn.gemma4_moe import Gemma4Experts
from dew.nn.gpt_oss import GptOssMLP
from dew.nn.inputs import ModelInputs
from dew.nn.kda import KimiDeltaAttention
from dew.nn.linear import GatedDeltaNet, recurrent_delta_rule, recurrent_gated_delta_rule
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mamba import Mamba, selective_scan
from dew.nn.mixers.mamba2 import Mamba2, recurrent_ssd
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.moe import SparseMLP
from dew.nn.protocols import Counted, CountedLogits
from dew.nn.ssm import BidirectionalS5Layer, S5Layer, _scanned_directions
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.rl.preference import DPOObjective
from dew.telemetry.instrumentation import StepFlops
from dew.training import MeshSpec
from dew.training.distributed import DevicePrefetchIterator
from dew.training.interval import Interval
from dew.training.model_flops import FORWARD_FLOPS, TOKENS, TRAINING_FLOPS, prepare_counts, step_counts

LENGTHS = np.asarray([2, 5], np.int64)


def decoder(**kwargs):
    return CausalTransformer(vocab_size=64, emb_features=8, num_heads=2, num_kv_heads=1,
                             head_dim=4, num_layers=3, mlp_features=16, **kwargs)


def test_dense_and_attention_equations_use_each_real_length():
    assert projection_flops(LENGTHS, 8, 16) == 2 * 8 * 16 * 7
    assert attention_flops(LENGTHS, 2, 4, 4, causal=True) == 2 * 2 * 4 * (4 + 25)
    assert attention_flops(LENGTHS, 2, 4, 4, causal=False) == 4 * 2 * 4 * (4 + 25)
    assert projection_flops(np.array([2**32], np.int64), 8192, 16384) == 2.0 * 8192 * 16384 * 2**32
    assert attention_flops(np.array([2**32], np.int64), 2, 4, 4, causal=True) == 16.0 * 2**64


def test_windowed_and_chunked_attention_do_not_count_unread_pairs():
    lengths = np.array([8], np.int64)
    assert attention_flops(lengths, 2, 4, 4, causal=True, window=4) == 32 * (8 * 4 - 4 * 4 / 2)
    assert attention_flops(lengths, 2, 4, 4, causal=True, chunk=4) == 16 * (4 * 4 + 4 * 4)


@pytest.mark.parametrize("lengths", [np.array([-1]), np.array([1.5]), np.array([[2, 5]])])
def test_invalid_lengths_are_not_silently_cast(lengths):
    with pytest.raises(ValueError, match="nonnegative integer"):
        projection_flops(lengths, 8, 16)


def test_causal_mixer_counts_queries_shared_keys_and_output_gate():
    mixer = CausalSelfAttention(emb_features=8, num_heads=2, num_kv_heads=1, head_dim=4, max_seq_len=16)
    attention = 16 * (4 + 25)
    assert mixer.model_flops(LENGTHS) == 2 * 8 * (8 + 4 + 4 + 8) * 7 + attention
    assert mixer.clone(kv_shared=True).model_flops(LENGTHS) == 2 * 8 * (8 + 8) * 7 + attention
    assert mixer.clone(output_gate=True).model_flops(LENGTHS) == mixer.model_flops(LENGTHS) + 2 * 8 * 8 * 7


@pytest.mark.parametrize("width,heads", [(1024, 16), (4096, 32)])
def test_qwen_delta_shapes_count_rule_projections_and_conv(width, heads):
    mixer = GatedDeltaNet(width, heads, heads, 128, 128, chunk_size=64)
    tokens = 7
    state_and_chunk = 6 * heads * 128 * 128 * tokens
    projected = 2 * width * (5 * heads * 128 + 2 * heads) * tokens
    convolution = 2 * (3 * heads * 128) * 4 * tokens
    assert delta_flops(LENGTHS, heads, 128, 128) == state_and_chunk
    assert mixer.model_flops(LENGTHS) == projected + convolution + state_and_chunk
    assert mixer.clone(chunk_size=128).model_flops(LENGTHS) == mixer.model_flops(LENGTHS)


@pytest.mark.parametrize("module", [Mamba(8, 16), Mamba2(8, 2, 8), KimiDeltaAttention(8, 2, 4)])
def test_other_recurrences_declare_additive_real_token_work(module):
    assert isinstance(module, Counted)
    one = module.model_flops(np.asarray([1], np.int64))
    assert one > 0
    assert module.model_flops(LENGTHS) == 7 * one


def test_s5_real_maps_complex_recurrence_and_bidirectional_output():
    one = S5Layer(8, 4)
    assert one.model_flops(LENGTHS) == 8 * 8 * 4 * 7
    assert BidirectionalS5Layer(8, 4).model_flops(LENGTHS) == 2 * one.model_flops(LENGTHS) + 4 * 8 * 8 * 7


def test_dit_counts_routed_tokens_modulation_and_output_but_not_remat():
    model = SimpleDiT(emb_features=8, num_heads=2, num_layers=2, patch_size=2)
    block = 24 * 8 * 8 * 7 + 4 * 8 * 29 + 12 * 8 * 8 * 2
    output = 2 * 8 * 4 * 3 * 7
    inputs = 2 * 12 * 8 * 7 + 4 * 4 * 8 * 8 * 2 + 2 * 32 * 32 * 2
    assert model.image_flops(LENGTHS, 3) == 2 * block + output + inputs
    assert model.clone(remat='full').image_flops(LENGTHS, 3) == model.image_flops(LENGTHS, 3)
    routed = model.clone(routes=((0.5, 0, 0),))
    shortened = 24 * 8 * 8 * 4 + 4 * 8 * (1 + 9) + 12 * 8 * 8 * 2
    assert routed.image_flops(LENGTHS, 3) == shortened + block + output + inputs
    hybrid = HybridSSMAttentionDiT(emb_features=8, num_heads=2, num_layers=2, patch_size=2,
                                   ssm_state_dim=4, ssm_attention_ratio="all-ssm")
    ssm = BidirectionalS5Layer(8, 4).model_flops(LENGTHS)
    assert hybrid.image_flops(LENGTHS, 3) == 2 * (ssm + 16 * 8 * 8 * 7 + 12 * 8 * 8 * 2) + output + inputs


def test_jepa_supplies_context_target_and_factorized_geometry_explicitly():
    stack = TokenStack(8, 2, num_heads=2)
    encoder = JepaEncoder(emb_features=8, num_layers=2, num_heads=2)
    assert encoder.image_flops(LENGTHS, 3) == stack.model_flops(LENGTHS) + 2 * 16**2 * 3 * 8 * 7
    factorized = FactorizedTokenStack(8, 2, num_heads=2)
    with pytest.raises(MissingFlopsCount, match="spatiotemporal_flops"):
        factorized.model_flops(np.array([10]))
    assert factorized.spatiotemporal_flops(np.array([2]), np.array([5])) == (
        stack.model_flops(np.array([5, 5])) + stack.model_flops(np.array([2] * 5)))
    predictor = JepaPredictor(emb_features=16, predictor_features=8, num_layers=2, num_heads=2)
    assert predictor.multisequence_flops(np.array([3]), np.array([2])) == (
        stack.model_flops(np.array([5])) + 2 * 16 * 8 * 5)


def test_diffusion_supplies_patch_channels_and_refuses_missing_auxiliary_work():
    model = SimpleDiT(emb_features=8, num_layers=2, num_heads=2, patch_size=2)
    objective = DiffusionObjective(model, presets.Flow(), inputs=InputSpec(sample=Field("images", (4, 8, 3))),
                                   autoencoder=None, ema_decay=None)
    batch = {"images": np.zeros((2, 4, 8, 3), np.float32)}
    np.testing.assert_array_equal(objective.real_lengths(batch), [8, 8])
    assert objective.model_flops(None, batch) == model.image_flops(np.array([8, 8]), 3)


def test_dense_mlp_gate_and_active_experts_are_counted():
    assert GatedMLP(16, 8).model_flops(LENGTHS) == 6 * 8 * 16 * 7
    assert GatedMLP(16, 8, activation="gelu").model_flops(LENGTHS) == 4 * 8 * 16 * 7
    routed = SparseMLP(num_experts=8, top_k=2, hidden_features=16, out_features=8)
    assert routed.model_flops(LENGTHS) == (2 * 8 * 8 + 2 * 6 * 8 * 16 + 2 * 2 * 8) * 7
    more = routed.clone(num_experts=16)
    assert more.model_flops(LENGTHS) - routed.model_flops(LENGTHS) == 2 * 8 * 8 * 7
    with pytest.raises(MissingFlopsCount, match="admitted routing"):
        routed.clone(capacity_factor=0.5).model_flops(LENGTHS)


def test_backbone_composes_layers_loop_passes_and_a_separate_lm_head():
    model = decoder()
    assert isinstance(model, Counted) and isinstance(model, CountedLogits)
    layer = (2 * 8 * 24 + 6 * 8 * 16) * 7 + 16 * (4 + 25)
    assert model.model_flops(LENGTHS) == 3 * layer
    assert model.logits_flops(LENGTHS) == 2 * 8 * 64 * 7
    assert model.clone(loop=Loop(3, layers=(1, 3))).model_flops(LENGTHS) == 7 * layer
    assert model.clone(scan_layers=True, remat=RematPolicy()).model_flops(LENGTHS) == 3 * layer
    assert model.clone(tie_embeddings=False).logits_flops(LENGTHS) == model.logits_flops(LENGTHS)


def test_hybrid_backbone_reuses_its_mixer_factory_geometry():
    mixer = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=4, linear_value_head_dim=4)
    model = decoder(mixer=mixer)
    layer = GatedDeltaNet(8, 2, 2, 4, 4).model_flops(LENGTHS) + GatedMLP(16, 8).model_flops(LENGTHS)
    assert model.model_flops(LENGTHS) == 3 * layer


def packed_batch():
    tokens = np.array([[1, 2, 3, 4, 0, 0, 0], [5, 6, 7, 0, 0, 0, 0]], np.int32)
    documents = np.array([[1, 1, 2, 2, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]], np.int32)
    return {"text": ModelInputs(tokens, token_fields={"segment_ids": documents,
                                                    "attention_mask": tokens != 0})}


def test_lm_packing_counts_independent_attention_squares_and_real_predictions():
    model = decoder()
    objective = LMObjective(model, seq_len=6, pad_id=0)
    batch = packed_batch()
    np.testing.assert_array_equal(objective.real_lengths(batch), [2, 2, 3])
    expected = model.model_flops(np.array([2, 2, 3])) + model.logits_flops(np.array([2, 2]))
    assert objective.model_flops(model, batch) == expected
    assert objective.model_flops(None, batch) == expected


def test_preference_geometry_is_not_mistaken_for_a_single_lm_sequence():
    objective = DPOObjective(decoder(), seq_len=6)
    batch = {"input_ids": np.ones((2, 2, 7), np.int32)}
    assert prepare_counts(objective, batch, count_model=False) is batch
    with pytest.raises(MissingFlopsCount, match=r"DPOObjective.*auxiliary"):
        objective.model_flops(None, batch)


def test_decision_head_charges_options_and_evidence_not_a_vocabulary():
    questions, options = np.array([1, 2]), np.array([2, 6])
    laya = DecisionHead(8, layers=1)
    expected = 24 * 8 * 8 * 7 + 4 * 8 * 29 + (2 * 8 * 8 + 2 * 8) * 8
    assert laya.decision_flops(LENGTHS, questions, options) == expected
    clef = JointSchemaHead(8, width=16, routing_layers=1, layers=0, feedforward=32)
    projected = 2 * 8 * 16 * (7 + 2 * 3 + 2 * 8 + 2)
    evidence = 4 * 16 * 16 * (7 + 8) + 4 * 16 * (2 * 2 + 5 * 6) + 4 * 16 * 32 * 8
    scorer = (8 * 16 * 16 + 2 * 16 + 6 * 16 + 2 * 8) * 8
    pooling = 2 * 8 * (2 * (1 + 2 * 2) + 5 * (2 + 2 * 6))
    assert clef.decision_flops(LENGTHS, questions, options) == projected + evidence + scorer + pooling
    with pytest.raises(MissingFlopsCount, match=r"JointSchemaHead.*decision_flops"):
        clef.model_flops(LENGTHS)


def test_missing_family_is_named_instead_of_falling_back_to_hlo():
    with pytest.raises(MissingFlopsCount, match=r"Dense.*train/mfu"):
        counted_flops(nn.Dense(8), LENGTHS)


def test_host_row_counts_add_to_objectives_total_without_padding():
    objective = LMObjective(decoder(), seq_len=6, pad_id=0)
    batch = packed_batch()
    counted = prepare_counts(objective, batch, count_model=True)
    np.testing.assert_array_equal(counted[TOKENS], [4, 3])
    assert counted[FORWARD_FLOPS].sum() == objective.model_flops(None, batch)
    assert counted["text"] is batch["text"]
    with pytest.raises(ValueError, match="reserved"):
        prepare_counts(objective, counted, count_model=True)


def test_decision_objective_supplies_the_layout_geometry_to_the_head():
    model = decoder()
    head = DecisionHead(8, layers=1)
    objective = DecisionObjective(model, head=head, tokenizer=ByteTokenizer(),
                                  specials=Specials(begin=None, separator=10, marker=0,
                                                    marker_text="\x00", pad=255),
                                  layout=StateFirstLayout(max_len=16))
    batch = {"valid": np.arange(8)[None, :] < LENGTHS[:, None],
             "questions": np.array([[True, False], [True, True]]),
             "options": np.array([[[True, True, False], [False, False, False]],
                                  [[True, True, True], [True, True, True]]])}
    np.testing.assert_array_equal(objective.real_lengths(batch), LENGTHS)
    wanted = model.model_flops(LENGTHS) + head.decision_flops(LENGTHS, np.array([1, 2]), np.array([2, 6]))
    assert objective.model_flops(None, batch) == wanted
    assert objective.model_flops(objective.model, batch) == wanted


@pytest.mark.mesh(devices=2)
def test_prefetch_counts_on_worker_then_step_reduces_global_rows():
    objective = LMObjective(decoder(), seq_len=6, pad_id=0)
    batch = packed_batch()
    threads = []

    def prepare(batch):
        threads.append(threading.current_thread().name)
        return prepare_counts(objective, batch, count_model=True)

    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    with DevicePrefetchIterator(iter([batch]), mesh, prepare=prepare) as source:
        placed = next(source)
        measured = jax.jit(step_counts)(placed)
        assert float(measured[TOKENS]) == 7
        assert float(measured[FORWARD_FLOPS]) == objective.model_flops(None, batch)
    assert threads == ["dew-prefetch"]


def test_interval_accumulates_work_with_its_loss_and_resets_only_at_logs():
    interval = Interval(last_log_time=0, last_saved=None)
    loss = jnp.asarray(1.0)
    for forward, tokens in ((100, 4), (200, 3)):
        interval.count({"rows": np.ones((2, 1))}, StepFlops(10, 20), loss,
                        jnp.asarray(a=True), None, {FORWARD_FLOPS: jnp.asarray(forward),
                                                   TRAINING_FLOPS: jnp.asarray(3 * forward),
                                                   TOKENS: jnp.asarray(tokens)})
    np.testing.assert_array_equal(np.asarray([interval.book.training, interval.book.tokens]), [900, 7])
    interval.saved(2)
    np.testing.assert_array_equal(np.asarray([interval.book.training, interval.book.tokens]), [900, 7])
    interval.logged(3.0)
    np.testing.assert_array_equal(np.asarray([interval.book.training, interval.book.tokens]), [0, 0])
    assert not interval.model_counted and not interval.tokens_counted


def forward_products(function, *operands, state_mac: bool = False):
    """Independent shape count of forward dots/convolutions, not an HLO estimate.

    Elementwise ops and normalization are excluded, so no tolerance can
    hide a missing Dense. Noncausal, unpadded fixtures make attention's
    logical product equal its executed product; chunked recurrence model
    FLOPs instead follow their documented algorithm-independent formula.
    """
    return float(jaxpr_products(jax.make_jaxpr(function)(*operands), state_mac=state_mac))


def jaxpr_products(graph, *, state_mac=False):
    """Count a scan body once per iteration, and one executed cond branch."""
    if isinstance(graph, dict):
        return sum(jaxpr_products(part, state_mac=state_mac) for part in graph.values())
    if isinstance(graph, tuple | list):
        return sum(jaxpr_products(part, state_mac=state_mac) for part in graph)
    if not hasattr(graph, "eqns"):
        if hasattr(graph, "jaxpr"):
            return jaxpr_products(graph.jaxpr, state_mac=state_mac)
        return 0
    total, producers = 0, {}
    consumers = {}
    for equation in graph.eqns:
        for variable in equation.invars:
            if not isinstance(variable, jaxpr_core.Literal):
                consumers.setdefault(variable, []).append(equation)
    for equation in graph.eqns:
        primitive = equation.primitive.name
        if primitive == "scan":
            total += equation.params["length"] * jaxpr_products(equation.params["jaxpr"], state_mac=state_mac)
        elif primitive == "cond":
            total += max(jaxpr_products(branch, state_mac=state_mac)
                         for branch in equation.params["branches"])
        elif primitive == "dot_general":
            left = equation.invars[0].aval.shape
            contracted = equation.params["dimension_numbers"][0][0]
            terms = np.prod([left[axis] for axis in contracted], dtype=np.int64)
            factor = 2
            if np.issubdtype(equation.outvars[0].aval.dtype, np.complexfloating):
                factor = 8
                for operand in equation.invars:
                    if np.issubdtype(operand.aval.dtype, np.floating):
                        factor = 4
                    source = producer(producers, operand)
                    if source is not None and source.primitive.name == "convert_element_type" and (
                            np.issubdtype(source.invars[0].aval.dtype, np.floating)):
                        factor = 4  # Real input against complex weights.
                if all(user.primitive.name == "real" for user in consumers.get(equation.outvars[0], [])):
                    factor = 4  # Only the real component of a complex read is returned.
            total += factor * np.prod(equation.outvars[0].aval.shape, dtype=np.int64) * terms
        elif primitive in ("ragged_dot", "ragged_dot_general"):
            terms = equation.invars[0].aval.shape[-1]
            total += 2 * np.prod(equation.outvars[0].aval.shape, dtype=np.int64) * terms
        elif primitive == "conv_general_dilated":
            kernel = equation.invars[1].aval.shape
            dims = equation.params["dimension_numbers"].rhs_spec
            total += (2 * np.prod(equation.outvars[0].aval.shape, dtype=np.int64)
                      * np.prod([kernel[axis] for axis in dims[1:]], dtype=np.int64))
        elif state_mac and primitive == "reduce_sum":
            product = producer(producers, equation.invars[0])
            if product is not None and product.primitive.name == "mul":
                total += 2 * np.prod(product.outvars[0].aval.shape, dtype=np.int64)
        elif primitive == "mul" and convolution_tap(equation, producers):
            total += 2 * np.prod(equation.outvars[0].aval.shape, dtype=np.int64)
        elif state_mac and primitive == "mul":
            left, right = (var.aval.shape for var in equation.invars)
            if (len(left) >= 2 and len(right) >= 2 and left[-1] == 1 and right[-2] == 1
                    and left[-2] > 1 and right[-1] > 1):
                total += 2 * np.prod(equation.outvars[0].aval.shape, dtype=np.int64)
        else:
            total += jaxpr_products(equation.params, state_mac=state_mac)
        for variable in equation.outvars:
            producers[variable] = equation
    return total


def producer(producers, variable):
    """Scalar constants have no producer and Literal is intentionally unhashable."""
    return None if isinstance(variable, jaxpr_core.Literal) else producers.get(variable)


def convolution_tap(equation, producers):
    """The explicit depthwise conv's [D,K] column times a window of the
    activations shifted along time, [B,T,D] (the time-major layout) or [B,D,T].

    Dew avoids conv_general_dilated for this op; the two slices still expose
    its tap geometry independently of the model's declared kernel size.
    """
    if len(equation.outvars[0].aval.shape) != 3:
        return False
    tapped = shifted = False
    for variable in equation.invars:
        if isinstance(variable, jaxpr_core.Literal):
            continue
        source = producer(producers, variable)
        while source is not None and source.primitive.name in (
                "broadcast_in_dim", "convert_element_type", "squeeze", "reshape"):
            source = producer(producers, source.invars[0])
        if source is not None and source.primitive.name == "slice":
            original, selected = source.invars[0].aval.shape, source.outvars[0].aval.shape
            if len(original) == 2 and original[1] > 1 and selected == (original[0], 1):
                tapped = True
            narrowed = [axis for axis in range(len(original)) if original[axis] != selected[axis]]
            if len(original) == 3 and len(narrowed) == 1 and narrowed[0] in (1, 2):
                shifted = True
    return tapped and shifted


@pytest.mark.parametrize("family", ["decoder", "loop", "mlp", "attention", "dit", "jepa"])
def test_analytical_count_equals_every_forward_dot_and_convolution(family):
    lengths = np.array([4, 4], np.int64)
    if family in ("decoder", "loop"):
        model = decoder(causal=False, attention_impl="reference",
                        loop=Loop(2) if family == "loop" else None)
        operands = (jnp.ones((2, 4), jnp.int32),)
        count = model.model_flops(lengths) + model.logits_flops(lengths)
    elif family == "mlp":
        model = GatedMLP(16, 8)
        operands = (jnp.ones((2, 4, 8)),)
        count = model.model_flops(lengths)
    elif family == "attention":
        model = CausalSelfAttention(8, 2, 1, 4, 16, causal=False, attention_impl="reference")
        operands = (jnp.ones((2, 4, 8)),)
        count = model.model_flops(lengths)
    elif family == "dit":
        model = SimpleDiT(emb_features=8, num_heads=2, num_layers=2, patch_size=2,
                          attention_impl="reference")
        operands = (jnp.ones((2, 4, 4, 3)), jnp.ones((2,)))
        count = model.image_flops(lengths, 3)
    else:
        model = JepaEncoder(emb_features=8, num_heads=2, num_layers=2, patch_size=2,
                            attention_impl="reference")
        operands = (jnp.ones((2, 4, 4, 3)),)
        count = model.image_flops(lengths, 3)
    variables = model.init(jax.random.key(201), *operands)
    assert count == forward_products(model.apply, variables, *operands)


@pytest.mark.parametrize("kind", ["gdn", "kda", "ssd", "mamba"])
def test_recurrent_state_macs_are_algorithm_independent_and_scan_aware(kind):
    """Three delta MACs, two SSD/Mamba MACs, executed once per real token."""
    batch, length, heads, key, value = 2, 7, 2, 3, 5
    q = jnp.ones((batch, length, heads, key))
    v = jnp.ones((batch, length, heads, value))
    beta = jnp.ones((batch, length, heads))
    if kind in ("gdn", "kda"):
        g = jnp.zeros((batch, length, heads, key) if kind == "kda" else (batch, length, heads))
        rule = recurrent_delta_rule if kind == "kda" else recurrent_gated_delta_rule
        counted = forward_products(rule, q, q, v, g, beta, state_mac=True)
        assert counted == 6 * batch * length * heads * key * value
    elif kind == "ssd":
        b = jnp.ones((batch, length, heads, key))
        counted = forward_products(recurrent_ssd, v, beta, jnp.ones((heads,)), b, b,
                                    jnp.ones((heads,)), state_mac=True)
        assert counted == 4 * batch * length * heads * key * value
    else:
        channels = 5
        x = jnp.ones((batch, length, channels))
        b = jnp.ones((batch, length, key))
        def rule(x, dt, a, b, c, d):
            return selective_scan(x, dt, a, b, c, d, chunk_size=1)
        counted = forward_products(rule, x, x, jnp.ones((channels, key)), b, b,
                                    jnp.ones((channels,)), state_mac=True)
        assert counted == 4 * batch * length * channels * key


@pytest.mark.parametrize("model,padding_work", [
    (GatedDeltaNet(8, 2, 2, 4, 3), 0), (KimiDeltaAttention(8, 2, 4), 0),
    (Mamba(8, 16, state_size=3, chunk_size=1), 0),
    (Mamba2(8, 2, 4, state_size=3, chunk_size=1), 2 * 2 * 14 * 4 * 3),
    (MultiHeadLatentAttention(8, 2, 16, 4, 3, 2, 2, 3, causal=False, attention_impl="reference"), 0),
])
def test_remaining_recurrent_and_latent_modules_match_token_paths(model, padding_work):
    """One-token recurrence has no chunk-prep products.

    Mamba-2 computes throwaway convolution outputs over K-1 padded history
    positions, before slicing. That exact 2*B*conv_features*K*(K-1) jaxpr
    overhead is not MFU; model plus it must equal the full traced product
    count, with no tolerance to hide a missing learned projection.
    """
    x = jnp.ones((2, 1, 8))
    variables = model.init(jax.random.key(202), x)
    assert model.model_flops(np.array([1, 1])) + padding_work == forward_products(
        model.apply, variables, x, state_mac=True)


@pytest.mark.parametrize("kind", ["sparse", "gpt_oss", "gemma4"])
def test_active_expert_products_match_the_executed_forward(kind):
    lengths = np.array([4, 4])
    x = jnp.ones((2, 4, 8))
    if kind == "sparse":
        model = SparseMLP(4, 2, 16, 8, implementation="xla")
    elif kind == "gpt_oss":
        model = GptOssMLP(8, 16, 4, 2, implementation="xla")
    else:
        model = Gemma4Experts(4, 2, 16, 8, implementation="xla")
    operands = (x, x) if kind == "gemma4" else (x,)
    variables = model.init(jax.random.key(203), *operands)
    # GPT-OSS spells the final weighted expert contraction as mul+reduce.
    scalar_mix = 2 * 2 * 8 * 8 if kind == "gpt_oss" else 0
    assert model.model_flops(lengths) == forward_products(model.apply, variables, *operands) + scalar_mix


def test_microbatches_two_sum_real_tokens_before_interval_throughput():
    """An accumulation window counts its two different real batches, not padded rows."""
    import optax

    from dew.training import Trainer

    objective = LMObjective(decoder(), seq_len=6, pad_id=0)
    interval = Interval(last_log_time=0, last_saved=None)
    for inputs in (np.array([[1, 2, 3, 0, 0, 0, 0]], np.int32),
                   np.array([[1, 2, 3, 4, 5, 0, 0]], np.int32)):
        counted = prepare_counts(objective, {"text": inputs}, count_model=True)
        aux = step_counts(jax.tree.map(jnp.asarray, counted))
        interval.count(counted, StepFlops(10, 20), jnp.asarray(1.0), jnp.asarray(a=True), None, aux)
    trainer = Trainer(objective, optax.sgd(0.01), key=0, accumulation=2)
    totals = np.asarray([interval.book.training, interval.book.tokens])
    assert totals[1] == 8
    metrics = trainer._throughput(2, 2, 2, interval.flops,
                                  training_flops=float(totals[0]), tokens=float(totals[1]))
    assert metrics["train/tokens"] * 2 == 8
    assert metrics["train/tokens_per_s"] == 4


def test_accumulation_two_logs_the_real_tokens_of_both_microbatches():
    import optax

    from dew.training import Trainer

    class Batches:
        batch, records, val, steps_per_epoch = 8, 16, None, 2

        def train(self, partition):
            for length in (3, 5):
                row = np.arange(7) < length
                yield {"text": np.broadcast_to(row.astype(np.int32), (self.batch, 7))}

    tracker = RecordingTracker()
    objective = LMObjective(decoder().clone(num_layers=1), seq_len=6, pad_id=0)
    trainer = Trainer(objective, optax.sgd(0.01), key=0, accumulation=2, tracker=tracker)
    trainer.fit(Batches(), steps=2, log_every=2)
    ticks = [scalars for _, scalars in tracker.scalars if "train/tokens" in scalars]
    assert len(ticks) == 1
    assert ticks[0]["train/tokens"] * 2 == 8 * (3 + 5)
    assert ticks[0]["train/tokens_per_s"] > 0


def test_s5_write_read_matches_the_native_scanned_path():
    """CPU/GPU chunked S5 executes extra pole-power GEMMs, which belong to HFU.
    The native TPU scan implements the recurrent function directly and its
    real-input/complex-state maps exactly match the model write/read count."""
    module = S5Layer(8, 3)
    x = jnp.ones((2, 7, 8))
    variables = module.init(jax.random.key(204), x)
    operators = module.bind(variables).operators()
    def forward(x, operators):
        return _scanned_directions(x, (operators,), (False,), x.dtype)
    assert module.model_flops(np.array([7, 7])) == forward_products(forward, x, operators)


@pytest.mark.parametrize("kind", ["laya", "clef"])
def test_decision_heads_match_every_forward_product_including_pooling(kind):
    states = jnp.ones((1, 4, 8))
    inputs = DecisionInputs(tokens=jnp.ones((1, 4), jnp.int32), valid=jnp.ones((1, 4), bool),
        kinds=jnp.zeros((1, 1), jnp.int32), questions=jnp.ones((1, 1), bool),
        spans=jnp.array([[[0, 2]]]), option_spans=jnp.array([[[[2, 3], [3, 4]]]]),
        options=jnp.ones((1, 1, 2), bool))
    model = (DecisionHead(8, layers=1, attention_impl="reference") if kind == "laya" else
             JointSchemaHead(8, width=8, heads=2, layers=1, routing_layers=1, feedforward=16,
                              attention_impl="reference"))
    table = jnp.ones((16, 8)) if kind == "clef" else None
    variables = model.init(jax.random.key(205), states, inputs, table)
    count = model.decision_flops(np.array([4]), np.array([1]), np.array([2]))
    # Clef writes its field/option cosine as multiply+reduce, not dot_general.
    scalar_cosine = 2 * 8 * 2 if kind == "clef" else 0
    assert count == forward_products(model.apply, variables, states, inputs, table) + scalar_cosine


def test_chunked_ssm_dit_forward_equals_model_maps_plus_known_pole_products():
    """S5's native chunk path has pole-prep and in-chunk products, outside model FLOPs.
    Their independent shape count admits no tolerance for a missing learned map."""
    model = HybridSSMAttentionDiT(emb_features=8, num_heads=2, num_layers=1, patch_size=2,
                                   ssm_state_dim=3, ssm_attention_ratio="all-ssm")
    image, time = jnp.ones((2, 4, 4, 3)), jnp.ones((2,))
    variables = model.init(jax.random.key(206), image, time)
    count = model.image_flops(np.array([4, 4]), 3)
    # Two directions: 2 pole-picking products C*C*C*N and one 2C-wide
    # real-state product B*2C*2C*N, each MAC two real FLOPs.
    extra = 2 * (4 * 4**3 * 3 + 8 * 2 * 4**2 * 3)
    assert count + extra == forward_products(model.apply, variables, image, time)


@pytest.mark.parametrize("inputs,outputs,rank", [(8, 16, 2), (4096, 14336, 16)])
def test_peft_training_work_does_not_charge_a_frozen_weight_gradient(inputs, outputs, rank):
    lengths = np.array([3, 5], np.int64)
    expected = (4 * inputs * outputs + 6 * rank * (inputs + outputs)) * 8
    assert dense_training_flops(lengths, inputs, outputs, trainable=False, adapter_rank=rank) == expected
    assert dense_training_flops(lengths, inputs, outputs) == 3 * projection_flops(lengths, inputs, outputs)


def test_training_work_assembly_is_distinct_from_forward_model_count():
    objective = LMObjective(decoder(), seq_len=6, pad_id=0)
    batch = packed_batch()
    assert objective.training_flops(None, batch) == 3 * objective.model_flops(None, batch)


def test_qwen35_reference_uses_the_counted_real_token_backbone_convention():
    """The reference workload has no vocabulary projection, only its backbone.

    Translate its committed released config without allocating any weights;
    8x3072 real rows and the same 6HDkDv recurrent convention must give the
    number both device runners publish, rather than an independent literal.
    """
    from tools.reference_runs.qwen35 import FLOPS_PER_STEP

    path = Path(__file__).parent / "fixtures" / "hf" / "qwen35-0.8b" / "config.json"
    config = json.loads(path.read_text())
    model = translate_config(config["text_config"]).value
    counted = int(3 * counted_flops(model, np.full((8,), 3072, np.int64)))
    assert counted == FLOPS_PER_STEP, (counted, FLOPS_PER_STEP, counted - FLOPS_PER_STEP)


def _kernels(variables) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Every Dense kernel's shape a module's init builds, by its path."""
    return {path[:-1]: value.shape for path, value in flatten_dict(variables["params"]).items()
            if path[-1] == "kernel"}


@pytest.mark.parametrize("fused", [False, True])
def test_delta_net_builds_exactly_the_projections_it_counts(fused):
    model = GatedDeltaNet(32, 2, 4, 8, 8, fused_in_proj=fused)
    variables = jax.eval_shape(model.init, jax.random.key(82), jnp.ones((1, 1, 32)))
    assert _kernels(variables) == {(name,): shape for name, shape in model.projections().items()}


@pytest.mark.parametrize("activation,packed", [("swiglu", False), ("swiglu", True), ("gelu", False)])
def test_mlp_builds_its_projections_and_packs_its_group_from_them(activation, packed):
    model = GatedMLP(48, 32, activation=activation, packed=packed)
    variables = jax.eval_shape(model.init, jax.random.key(83), jnp.ones((1, 1, 32)))
    declared = {(name,): shape for name, shape in model.projections().items()}
    groups = model.apply(variables, method="projection_groups")
    if activation == "gelu":
        assert groups == ()
    else:
        (group,) = groups
        assert group.widths == tuple(model.projections()[name][1] for name in group.members)
        if packed:
            del declared[("gate_proj",)], declared[("up_proj",)]
            declared[(group.packed,)] = (32, sum(group.widths))
    assert _kernels(variables) == declared


@pytest.mark.parametrize("packed,k_eq_v,kv_shared,output_gate,value_dim", [
    (False, False, False, False, 8), (True, False, False, True, 8), (False, False, False, False, 4),
    (False, True, False, False, 8), (False, False, True, False, 8)])
def test_attention_builds_only_the_projections_it_declares(packed, k_eq_v, kv_shared, output_gate, value_dim):
    model = CausalSelfAttention(32, 4, 2, 8, 8, packed=packed, k_eq_v=k_eq_v, kv_shared=kv_shared,
                                kv_store_key="provider", output_gate=output_gate, value_head_dim=value_dim,
                                attention_impl="reference")
    keys, values = jnp.ones((1, 1, 2, 8)), jnp.ones((1, 1, 2, value_dim))
    store = {"provider": (keys, values, jnp.zeros((1, 1), jnp.int32))}
    variables = jax.eval_shape(lambda: model.init(jax.random.key(84), jnp.ones((1, 1, 32)), kv_store=store))
    declared = {(name,): shape for name, shape in model.projections().items()}
    groups = model.apply(variables, method="projection_groups")
    if kv_shared or k_eq_v:
        assert groups == ()
    else:
        (group,) = groups
        assert group.widths == (64 if output_gate else 32, 16, 2 * value_dim)
        if packed:
            for name in group.members:
                del declared[(name,)]
            declared[(group.packed,)] = (32, sum(group.widths))
    assert _kernels(variables) == declared


def test_packed_attention_refuses_a_layer_without_its_own_values():
    model = CausalSelfAttention(32, 4, 2, 8, 8, packed=True, k_eq_v=True, attention_impl="reference")
    with pytest.raises(ValueError, match="packed qkv_proj"):
        jax.eval_shape(model.init, jax.random.key(85), jnp.ones((1, 1, 32)))


@pytest.mark.parametrize("size,layers,mlp,linear,attention", [
    ("0.8B", 24,
     {"gate_proj": (1024, 3584), "up_proj": (1024, 3584), "down_proj": (3584, 1024)},
     {"in_proj_qkv": (1024, 6144), "in_proj_z": (1024, 2048), "in_proj_b": (1024, 16),
      "in_proj_a": (1024, 16), "out_proj": (2048, 1024)},
     {"q_proj": (1024, 4096), "k_proj": (1024, 512), "v_proj": (1024, 512), "o_proj": (2048, 1024)}),
    ("9B", 32,
     {"gate_proj": (4096, 12288), "up_proj": (4096, 12288), "down_proj": (12288, 4096)},
     {"in_proj_qkv": (4096, 8192), "in_proj_z": (4096, 4096), "in_proj_b": (4096, 32),
      "in_proj_a": (4096, 32), "out_proj": (4096, 4096)},
     {"q_proj": (4096, 8192), "k_proj": (4096, 1024), "v_proj": (4096, 1024), "o_proj": (4096, 4096)}),
], ids=["qwen35-0.8b", "qwen35-9b"])
def test_qwen35_layers_build_the_projections_the_frozen_base_was_measured_on(size, layers, mlp, linear,
                                                                              attention):
    """The FP8 frozen base's A/B at 316cce053 covered these layer projections
    (`dense_matrices` there, its packed alternatives aside); built from the
    one declaration they are unchanged, so that measurement still applies."""
    path = Path(__file__).parent / "fixtures" / "hf" / "qwen35-0.8b" / "config.json"
    config = json.loads(path.read_text())["text_config"]
    if size == "9B":
        config.update(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32, num_attention_heads=16,
                      num_key_value_heads=4, linear_num_value_heads=32, tie_word_embeddings=False,
                      layer_types=["linear_attention"] * 3 + ["full_attention"] * 1)
        config["layer_types"] *= 8
    model = translate_config(config).value
    variables = jax.eval_shape(model.init, jax.random.key(86), jnp.ones((1, 8), jnp.int32))
    built = {path: shape for path, shape in _kernels(variables).items() if path[0].startswith("layers_")}
    expected = {(f"layers_{index}", owner, name): shape for index in range(layers)
                for owner, sites in (("mlp", mlp), ("self_attn", attention if index % 4 == 3 else linear))
                for name, shape in sites.items()}
    assert built == expected
