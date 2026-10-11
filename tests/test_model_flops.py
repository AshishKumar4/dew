"""Analytical counts on real rows, composed by the objective, not the HLO.

Closed-form arithmetic catches padded lengths, total-expert instead of
active-expert counts, omitted loop passes, accidental remat duplication,
packed documents treated as one attention square, and a vocabulary head
incorrectly charged to a decision model.
"""

import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.data.text import ByteTokenizer
from dew.decision.head import DecisionHead, JointSchemaHead
from dew.decision.layout import Specials, StateFirstLayout
from dew.decision.objective import DecisionObjective
from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import GatedMLP, RematPolicy
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.jepa import FactorizedTokenStack, JepaEncoder, JepaPredictor, TokenStack
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.flops import MissingFlopsCount, attention_flops, counted_flops, delta_flops, projection_flops
from dew.nn.inputs import ModelInputs
from dew.nn.kda import KimiDeltaAttention
from dew.nn.linear import GatedDeltaNet
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mamba import Mamba
from dew.nn.mixers.mamba2 import Mamba2
from dew.nn.moe import SparseMLP
from dew.nn.protocols import Counted, CountedLogits
from dew.nn.ssm import BidirectionalS5Layer, S5Layer
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.rl.preference import DPOObjective
from dew.telemetry.instrumentation import StepFlops
from dew.training import MeshSpec
from dew.training.distributed import DevicePrefetchIterator
from dew.training.model_flops import FORWARD_FLOPS, TOKENS, prepare_counts, step_counts
from dew.training.trainer import _Interval

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
    state_and_chunk = (4 * heads * 128 * 128 + 2 * heads * 64 * (128 + 128)) * tokens
    projected = 2 * width * (5 * heads * 128 + 2 * heads) * tokens
    convolution = 2 * (3 * heads * 128) * 4 * tokens
    assert delta_flops(LENGTHS, heads, 128, 128, 64) == state_and_chunk
    assert mixer.model_flops(LENGTHS) == projected + convolution + state_and_chunk


@pytest.mark.parametrize("module", [Mamba(8, 16), Mamba2(8, 2, 8), KimiDeltaAttention(8, 2, 4)])
def test_other_recurrences_declare_additive_real_token_work(module):
    assert isinstance(module, Counted)
    one = module.model_flops(np.asarray([1], np.int64))
    assert one > 0
    assert module.model_flops(LENGTHS) == 7 * one


def test_s5_real_maps_complex_recurrence_and_bidirectional_output():
    one = S5Layer(8, 4)
    assert one.model_flops(LENGTHS) == (8 * 8 * 4 + 8 * 4 + 2 * 8) * 7
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
    assert routed.model_flops(LENGTHS) == (2 * 8 * 8 + 2 * 6 * 8 * 16) * 7
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
    assert clef.decision_flops(LENGTHS, questions, options) == projected + evidence + scorer
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
    interval = _Interval(last_log_time=0, last_saved=None)
    loss = jnp.asarray(1.0)
    for forward, tokens in ((100, 4), (200, 3)):
        interval.count({"rows": np.ones((2, 1))}, StepFlops(10, 20), loss,
                        jnp.asarray(a=True), {FORWARD_FLOPS: jnp.asarray(forward),
                                             TOKENS: jnp.asarray(tokens)})
    np.testing.assert_array_equal(np.asarray(interval.book[3:]), [300, 7])
    interval.saved(2)
    np.testing.assert_array_equal(np.asarray(interval.book[3:]), [300, 7])
    interval.logged(3.0)
    np.testing.assert_array_equal(np.asarray(interval.book[3:]), [0, 0])
    assert not interval.model_counted and not interval.tokens_counted
