"""Analytical counts on real rows, composed by the objective, not the HLO.

Closed-form arithmetic catches padded lengths, total-expert instead of
active-expert counts, omitted loop passes, accidental remat duplication,
packed documents treated as one attention square, and a vocabulary head
incorrectly charged to a decision model.
"""

import dataclasses
import json
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.extend import core as jaxpr_core
from recording import RecordingTracker

from dew.data.chat import ROLES_KEY, Role
from dew.data.dataset import Loader
from dew.data.text import ByteTokenizer
from dew.decision.head import DecisionHead, JointSchemaHead
from dew.decision.layout import DecisionInputs, Specials, StateFirstLayout
from dew.decision.objective import DecisionObjective
from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.interop.hf_decoders import translate_config
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import RematPolicy
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
from dew.nn.kernels.generation import assumed, gpu_target
from dew.nn.kernels.recurrence_chunk import recurrent_delta_rule
from dew.nn.linear import GatedDeltaNet, recurrent_gated_delta_rule
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mamba import Mamba, selective_scan
from dew.nn.mixers.mamba2 import Mamba2, recurrent_ssd
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.mlp import GatedMLP
from dew.nn.moe import SparseMLP
from dew.nn.protocols import Counted, CountedDecisions, CountedLogits, CountedTraining
from dew.nn.ssm import BidirectionalS5Layer, S5Layer, _scanned_directions
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.lm.objective import batch_text
from dew.telemetry.instrumentation import StepFlops
from dew.training import MeshSpec
from dew.training.distributed import DevicePrefetchIterator
from dew.training.interval import Interval
from dew.training.model_flops import TOKENS, TRAINING_FLOPS, _TokenCount, prepare_counts, step_counts

LENGTHS = np.asarray([2, 5], np.int64)


def decoder(**kwargs):
    return CausalTransformer(vocab_size=64, emb_features=8, num_heads=2, num_kv_heads=1,
                             head_dim=4, num_layers=3, mlp_features=16, **kwargs)


def test_integer_token_counts_carry_across_words_on_a_partitioned_batch():
    from jax.sharding import NamedSharding, PartitionSpec

    rows = _TokenCount.words([2**32 - 1, 2, 2**40, 0])
    mesh = MeshSpec().build(jax.devices()[:2])
    placed = jax.device_put(rows, NamedSharding(mesh, PartitionSpec("data")))
    total = jax.jit(_TokenCount.sum)(placed)
    assert _TokenCount.host(total) == 2**40 + 2**32 + 1
    assert total.dtype == jnp.uint32


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


def test_causal_mixer_counts_queries_shared_keys_and_output_gate():
    mixer = CausalSelfAttention(MixerContext(emb_features=8, num_heads=2, num_kv_heads=1, head_dim=4,
                                             max_seq_len=16))
    attention = 16 * (4 + 25)
    assert mixer.model_flops(LENGTHS) == 2 * 8 * (8 + 4 + 4 + 8) * 7 + attention
    shared = mixer.clone(ctx=dataclasses.replace(mixer.ctx, kv_shared=True))
    assert shared.model_flops(LENGTHS) == 2 * 8 * (8 + 8) * 7 + attention
    gated = mixer.clone(ctx=dataclasses.replace(mixer.ctx, output_gate=True))
    assert gated.model_flops(LENGTHS) == mixer.model_flops(LENGTHS) + 2 * 8 * 8 * 7


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
    with pytest.raises(MissingFlopsCount):
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
    with pytest.raises(MissingFlopsCount):
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


def test_decision_head_charges_options_and_evidence_not_a_vocabulary():
    questions, options = np.array([1, 2]), np.array([2, 6])
    laya = DecisionHead(8, layers=1)
    expected = 24 * 8 * 8 * 7 + 4 * 8 * 29 + (2 * 8 * 8 + 2 * 8) * 8
    assert laya.decision_flops(LENGTHS, questions, options) == expected
    joint = JointSchemaHead(8, width=16, routing_layers=1, layers=0, feedforward=32)
    projected = 2 * 8 * 16 * (7 + 2 * 3 + 2 * 8 + 2)
    evidence = 4 * 16 * 16 * (7 + 8) + 4 * 16 * (2 * 2 + 5 * 6) + 4 * 16 * 32 * 8
    scorer = (8 * 16 * 16 + 2 * 16 + 6 * 16 + 2 * 8) * 8
    pooling = 2 * 8 * (2 * (1 + 2 * 2) + 5 * (2 + 2 * 6))
    assert joint.decision_flops(LENGTHS, questions, options) == projected + evidence + scorer + pooling
    assert isinstance(joint, CountedDecisions) and not isinstance(joint, Counted)
    with pytest.raises(MissingFlopsCount):
        counted_flops(joint, LENGTHS)


def test_host_row_counts_add_to_objectives_total_without_padding():
    """Each row commits the targets its loss scores: two documents of two tokens score one each, and
    one of three scores two, while the FLOP count keeps their inputs."""
    objective = LMObjective(decoder(), seq_len=6, pad_id=0)
    batch = packed_batch()
    counted = prepare_counts(objective, batch, count_model=True)
    np.testing.assert_array_equal(counted[TOKENS], _TokenCount.words([2, 2]))
    assert counted[TRAINING_FLOPS].sum() == objective.training_flops(None, batch)
    assert counted["text"] is batch["text"]
    with pytest.raises(ValueError):
        prepare_counts(objective, counted, count_model=True)


def packed_windows(questions: int = 2) -> dict[str, np.ndarray]:
    """Two packed windows of eight tokens, each holding one row of `LENGTHS`
    tokens whose questions all start at its first token."""
    return {"segments": (np.arange(8)[None, :] < LENGTHS[:, None]).astype(np.int32),
            "spans": np.zeros((2, questions, 2), np.int32)}


def test_decision_objective_supplies_the_layout_geometry_to_the_head():
    model = decoder()
    head = DecisionHead(8, layers=1)
    objective = DecisionObjective(model, head=head, tokenizer=ByteTokenizer(),
                                  specials=Specials(begin=None, separator=10, marker=0,
                                                    marker_text="\x00", pad=255),
                                  layout=StateFirstLayout(max_len=16))
    batch = {**packed_windows(), "questions": np.array([[True, False], [True, True]]),
             "options": np.array([[[True, True, False], [False, False, False]],
                                  [[True, True, True], [True, True, True]]])}
    np.testing.assert_array_equal(objective.real_lengths(batch), LENGTHS)
    wanted = model.model_flops(LENGTHS) + head.decision_flops(LENGTHS, np.array([1, 2]), np.array([2, 6]))
    assert objective.model_flops(None, batch) == wanted
    assert objective.model_flops(objective.model, batch) == wanted


def test_adapted_decision_backbone_counts_training_work_without_claiming_an_lm_head():
    class Trunk(nn.Module):
        @nn.compact
        def hidden_states(self, tokens, *, train=False, **fields):
            return nn.Dense(8, use_bias=False)(nn.Embed(16, 8)(tokens))

        def __call__(self, tokens, *, train=False):
            return self.hidden_states(tokens, train=train)

        @nn.nowrap
        def model_flops(self, lengths):
            return projection_flops(lengths, 8, 8)

        @nn.nowrap
        def parameter_flops(self, lengths):
            return self.model_flops(lengths)

        @nn.nowrap
        def matrix_usage(self, path, lengths, *, logits=False):
            return np.zeros_like(lengths) if logits else lengths

    model = Trunk()
    variables = model.init(jax.random.key(0), jnp.ones((2, 8), jnp.int32))
    adapter = LoRA(1, ("Dense_0",)).apply(model, variables, key=1)
    assert isinstance(adapter.model, CountedTraining)
    assert not isinstance(adapter.model, CountedLogits)
    head = DecisionHead(8, layers=0)
    objective = DecisionObjective(adapter, head=head, tokenizer=ByteTokenizer(),
                                  specials=Specials(begin=None, separator=10, marker=0,
                                                    marker_text="\x00", pad=255),
                                  layout=StateFirstLayout(max_len=8))
    batch = {**packed_windows(questions=1), "questions": np.ones((2, 1), bool),
             "options": np.ones((2, 1, 2), bool)}
    frozen = 4 * 8 * 8 * np.sum(LENGTHS)
    factors = 6 * (8 + 8) * np.sum(LENGTHS)
    head_work = head.decision_flops(LENGTHS, np.ones(2, np.int64), np.full(2, 2, np.int64))
    assert objective.training_flops(None, batch) == frozen + factors + 3 * head_work


@pytest.mark.parametrize("mtp_weight", [None, 1.0])
def test_lm_counts_only_the_prediction_depths_the_loss_trains(mtp_weight):
    objective = LMObjective(decoder(num_nextn_predict_layers=1), seq_len=6, mtp_weight=mtp_weight)
    batch = packed_batch()
    trunk = (objective.model.model_flops(objective.real_lengths(batch))
             + objective.model.logits_flops(objective._prediction_lengths(batch)))
    depths = objective.model.prediction_flops(objective.real_lengths(batch),
        prediction_lengths=objective._prediction_rows(objective.model, batch)) if mtp_weight else 0
    assert objective.training_flops(None, batch) == 3 * (trunk + depths)


@pytest.mark.parametrize("packed", [False, True])
def test_prediction_depth_work_excludes_pad_holes_and_unscored_document_boundaries(packed):
    model = decoder(num_nextn_predict_layers=2)
    objective = LMObjective(model, seq_len=6, pad_id=0, mtp_weight=1.0)
    ids = np.array([[1, 2, 0, 3, 4, 5, 0]], np.int32)
    fields = {"attention_mask": ids != 0}
    if packed:
        fields["segment_ids"] = np.array([[1, 1, 0, 2, 2, 2, 0]], np.int32)
    batch = {"text": ModelInputs(ids, token_fields=fields)}
    rows = objective._prediction_rows(model, batch)
    np.testing.assert_array_equal(rows[0][0], [1, 2] if packed else [3])
    np.testing.assert_array_equal(rows[0][1], [1])
    np.testing.assert_array_equal(rows[1][0], [1])
    np.testing.assert_array_equal(rows[1][1], [] if packed else [0])
    lengths = objective.real_lengths(batch)
    desired = (model.model_flops(lengths) + model.logits_flops(np.array([3]))
               + model.prediction_flops(lengths, prediction_lengths=rows))
    assert objective.model_flops(None, batch) == desired
    assert desired < (model.model_flops(lengths) + model.logits_flops(np.array([3]))
                      + model.prediction_flops(lengths))


def test_an_assistant_only_run_counts_the_assistant_targets_its_loss_scores_at_every_depth():
    """With `loss_role`, the targets the FLOP count reads at each depth are the ones the loss weighs
    on the device: an assistant target, its document's, past no pad hole. The others are no work the
    loss trains."""
    model = decoder(num_nextn_predict_layers=2)
    objective = LMObjective(model, seq_len=6, pad_id=0, mtp_weight=1.0, loss_role=Role.ASSISTANT)
    ids = np.array([[1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 7]], np.int32)
    roles = np.zeros(ids.shape, np.int8)
    roles[:, 3:6] = Role.ASSISTANT
    batch = {"text": ModelInputs(ids, token_fields={"attention_mask": ids != 0}), ROLES_KEY: roles}
    prepared = batch_text(batch)
    for depth in range(3):
        weights = (objective._row_weights(prepared, jnp.asarray(roles), jnp.float32) if depth == 0 else
                   objective._depth_weights(prepared, jnp.asarray(roles), jnp.float32, depth))
        np.testing.assert_array_equal(objective._prediction_lengths(batch, depth),
                                      np.sum(np.asarray(weights), axis=-1))
    rows = objective._prediction_rows(model, batch)
    assert [int(np.sum(scored)) for _, scored in rows] == [
        int(np.sum(objective._depth_weights(prepared, jnp.asarray(roles), jnp.float32, depth)))
        for depth in (1, 2)]


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
        assert _TokenCount.host(measured[TOKENS]) == 4
        assert float(measured[TRAINING_FLOPS]) == objective.training_flops(None, batch)
    assert threads == ["dew-prefetch"]


def test_interval_accumulates_work_with_its_loss_and_resets_only_at_logs():
    """A log takes the interval's work and tokens and starts the next
    interval's at zero, and the run's lifetime tokens keep both logs'."""
    interval = Interval(last_log_time=0, last_saved=None, tokens_before_log=5)
    loss = jnp.asarray(1.0)

    def counted(steps):
        for forward, tokens in steps:
            interval.count({"rows": np.ones((2, 1))}, StepFlops(10, 20), loss,
                           jnp.asarray(a=True), None, {TRAINING_FLOPS: jnp.asarray(3 * forward),
                                                      TOKENS: jnp.asarray(_TokenCount.words(tokens))})

    counted(((100, 4), (200, 3)))
    assert float(interval.book.training) == 900 and _TokenCount.host(interval.book.tokens) == 7
    interval.saved(2)
    assert float(interval.book.training) == 900 and _TokenCount.host(interval.book.tokens) == 7
    taken = interval.take(3, loss, {}, jnp.asarray(a=True), (None, None), {}, {})
    assert float(taken.reading.training) == 900 and _TokenCount.host(taken.reading.tokens) == 7
    assert float(interval.book.training) == 0 and _TokenCount.host(interval.book.tokens) == 0
    assert not interval.model_counted and not interval.tokens_counted
    assert interval.real_tokens == 12
    counted(((50, 2),))
    assert interval.real_tokens == 14


def forward_products(function, *operands, state_mac: bool = False):
    """Independent shape count of forward dots/convolutions, not an HLO estimate.

    Elementwise ops and normalization are excluded, so no tolerance can
    hide a missing Dense. Noncausal, unpadded fixtures make attention's
    logical product equal its executed product; chunked recurrence model
    FLOPs instead follow their documented algorithm-independent formula.
    The forward is traced on its reference path, as on a CPU, on every
    device: a GPU's kernels are custom calls, whose products no jaxpr shows.
    """
    with assumed("cpu"):
        traced = jax.make_jaxpr(function)(*operands)
    return float(jaxpr_products(traced, state_mac=state_mac))


def executed(cond) -> list:
    """The branches of `cond` a run may take: of a platform choice (`jax.lax.platform_dependent`), the
    one an accelerator runs."""
    platforms = cond.params.get("branches_platforms")
    if platforms is None:
        return list(cond.params["branches"])
    return [cond.params["branches"][next(index for index, names in enumerate(platforms)
                                         if names is None or "cuda" in names)]]


def jaxpr_products(graph, *, state_mac=False):
    """Count a scan body once per iteration, and one executed cond branch: of a platform choice
    (`jax.lax.platform_dependent`), the branch an accelerator runs, the model's own products, where a
    CPU's sums them in float64 and spreads a bias by a product (`dew.nn.precision`)."""
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
            total += max(jaxpr_products(branch, state_mac=state_mac) for branch in executed(equation))
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
        model = CausalSelfAttention(MixerContext(8, 2, 1, 4, 16, causal=False, attention_impl="reference"))
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
    """Three delta MACs and the one-pass update's readout (`readout_products`), two SSD/Mamba MACs,
    executed once per real token."""
    batch, length, heads, key, value = 2, 7, 2, 3, 5
    q = jnp.ones((batch, length, heads, key))
    v = jnp.ones((batch, length, heads, value))
    beta = jnp.ones((batch, length, heads))
    if kind in ("gdn", "kda"):
        g = jnp.zeros((batch, length, heads, key) if kind == "kda" else (batch, length, heads))
        rule = recurrent_delta_rule if kind == "kda" else recurrent_gated_delta_rule
        counted = forward_products(rule, q, q, v, g, beta, state_mac=True)
        assert counted == 6 * batch * length * heads * key * value + 2 * batch * length * heads * key
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


@pytest.mark.parametrize("model", [
    GatedDeltaNet(8, 2, 2, 4, 3), KimiDeltaAttention(8, 2, 4), Mamba(8, 16, state_size=3, chunk_size=1),
    Mamba2(8, 2, 4, state_size=3, chunk_size=1),
    MultiHeadLatentAttention(8, 2, 16, 4, 3, 2, 2, 3, causal=False, attention_impl="reference"),
])
def test_remaining_recurrent_and_latent_modules_match_token_paths(model):
    """One-token recurrence has no chunk-prep products, and the convs
    compute no outputs over their zero history: the model's count is the
    full traced product count, with no tolerance to hide a missing learned
    projection."""
    x = jnp.ones((2, 1, 8))
    variables = model.init(jax.random.key(202), x)
    assert (model.model_flops(np.array([1, 1])) + readout_products(model, 2)
            == forward_products(model.apply, variables, x, state_mac=True))


def readout_products(model, tokens: int) -> int:
    """The `q . k` product a delta rule's token takes beside its analytic rule's three state products
    (`delta_flops`): the token scan is the decode step's one-pass update (`recurrence.update`), whose
    output reads the old state and adds the token's own write by `q . k`, two a key dimension and a value
    head."""
    if isinstance(model, GatedDeltaNet):
        return 2 * tokens * model.num_v_heads * model.head_k_dim
    if isinstance(model, KimiDeltaAttention):
        return 2 * tokens * model.num_heads * model.head_dim
    return 0


@pytest.mark.parametrize("kind", ["sparse", "latent", "gpt_oss", "gemma4"])
def test_active_expert_products_match_the_executed_forward(kind):
    """A latent MoE combines its experts' outputs at the latent width, before
    projecting them up to the residual's."""
    lengths = np.array([4, 4])
    x = jnp.ones((2, 4, 8))
    if kind == "sparse":
        model = SparseMLP(4, 2, 16, 8, implementation="xla")
    elif kind == "latent":
        model = SparseMLP(4, 2, 16, 8, implementation="xla", latent_features=4)
    elif kind == "gpt_oss":
        model = GptOssMLP(8, 16, 4, 2, implementation="xla")
    else:
        model = Gemma4Experts(4, 2, 16, 8, implementation="xla")
    operands = (x, x) if kind == "gemma4" else (x,)
    variables = model.init(jax.random.key(203), *operands)
    # GPT-OSS spells the final weighted expert contraction as mul+reduce.
    scalar_mix = 2 * 2 * 8 * 8 if kind == "gpt_oss" else 0
    assert model.model_flops(lengths) == forward_products(model.apply, variables, *operands) + scalar_mix


def test_accumulation_two_logs_the_scored_targets_of_both_microbatches():
    """Rows of 3 and 5 real tokens score 2 and 4 targets each: an update of both microbatches
    commits their targets, not their inputs."""
    import optax

    from dew.training import Trainer

    def batches(partition):
        for length in (3, 5):
            row = np.arange(7) < length
            yield {"text": np.broadcast_to(row.astype(np.int32), (8, 7))}

    tracker = RecordingTracker()
    objective = LMObjective(decoder().clone(num_layers=1), seq_len=6, pad_id=0)
    trainer = Trainer(objective, optax.sgd(0.01), key=0, accumulation=2, tracker=tracker)
    trainer.fit(Loader(batches, None, batch=8, records=16), steps=2, log_every=2)
    ticks = [scalars for _, scalars in tracker.scalars if "train/tokens" in scalars]
    assert len(ticks) == 1
    assert ticks[0]["train/tokens"] * 2 == 8 * (2 + 4)
    assert ticks[0]["train/tokens_per_s"] > 0


def test_a_token_budget_runs_until_the_targets_its_loss_scored_reach_it():
    """Eight padded rows of 3 real tokens hold 24 inputs and score 16 targets a step, so a budget of
    40 tokens ends at the step whose targets pass it, the third, not the second, where the inputs
    would pass it."""
    import optax

    from dew.training import Trainer

    def batches(partition):
        while True:
            yield {"text": np.broadcast_to((np.arange(7) < 3).astype(np.int32), (8, 7))}

    objective = LMObjective(decoder().clone(num_layers=1), seq_len=6, pad_id=0)
    state = Trainer(objective, optax.sgd(0.01), key=0).fit(Loader(batches, None, batch=8, records=None),
                                                           tokens=40, log_every=1)
    assert int(state.step) == 3


def test_a_run_on_a_device_without_a_known_peak_still_reports_its_training_flops():
    """train/flops, the step's analytical training work over the mesh, needs no device peak: a run
    logs every step's count, the objective's own for its batch, on any device, and train/mfu only
    where the device's peak is known, so on the CPU lane, whose devices have none, the count alone.
    The counts are sums of float32 per-row shares, exact below 2**21 (a share is the count over
    eight rows, three more bits), so they compare exactly."""
    import optax

    from dew.training import Trainer

    batch = {"text": np.broadcast_to((np.arange(7) < 5).astype(np.int32), (8, 7))}
    tracker = RecordingTracker()
    objective = LMObjective(decoder().clone(num_layers=1), seq_len=6, pad_id=0)
    work = objective.training_flops(None, batch)
    assert 0 < work < 2**21
    trainer = Trainer(objective, optax.sgd(0.01), key=0, tracker=tracker)
    trainer.fit(Loader(lambda partition: iter([batch, batch]), None, batch=8, records=16), steps=2,
                log_every=1)
    logged = [scalars for _, scalars in tracker.scalars if "train/loss" in scalars]
    assert [scalars["train/flops"] for scalars in logged] == [work, work]
    peak = gpu_target(jax.devices()[0].device_kind).bf16_peak
    assert all(("train/mfu" in scalars) == (peak is not None) for scalars in logged)


def test_s5_write_read_matches_the_native_scanned_path():
    """CPU/GPU chunked S5 executes extra pole-power GEMMs, which belong to HFU.
    The native TPU scan implements the recurrent function directly and its
    real-input/complex-state maps exactly match the model write/read count."""
    module = S5Layer(8, 3)
    x = jnp.ones((2, 7, 8))
    variables = module.init(jax.random.key(204), x)
    operators = module.bind(variables).operators(jnp.float32)
    def forward(x, operators):
        return _scanned_directions(x, (operators,), (False,), x.dtype)
    assert module.model_flops(np.array([7, 7])) == forward_products(forward, x, operators)


@pytest.mark.parametrize("kind", ["laya", "joint"])
def test_decision_heads_match_every_forward_product_including_pooling(kind):
    states = jnp.ones((1, 4, 8))
    inputs = DecisionInputs(tokens=jnp.ones((1, 4), jnp.int32), segments=jnp.ones((1, 4), jnp.int32),
        kinds=jnp.zeros((1, 1), jnp.int32), questions=jnp.ones((1, 1), bool),
        spans=jnp.array([[[0, 2]]]), option_spans=jnp.array([[[[2, 3], [3, 4]]]]),
        options=jnp.ones((1, 1, 2), bool))
    model = (DecisionHead(8, layers=1, attention_impl="reference") if kind == "laya" else
             JointSchemaHead(8, width=8, heads=2, layers=1, routing_layers=1, feedforward=16,
                              attention_impl="reference"))
    table = jnp.ones((16, 8)) if kind == "joint" else None
    variables = model.init(jax.random.key(205), states, inputs, table)
    count = model.decision_flops(np.array([4]), np.array([1]), np.array([2]))
    # The joint head writes its field/option cosine as multiply+reduce, not dot_general.
    scalar_cosine = 2 * 8 * 2 if kind == "joint" else 0
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


def test_qwen35_reference_uses_the_counted_real_token_backbone_convention(monkeypatch):
    """The reference workload has no vocabulary projection, only its backbone.

    Translate its committed released config without allocating any weights;
    8x3072 real rows and the same 6HDkDv recurrent convention must give the
    work record both device runners publish.
    """
    from tool_support import REPO_ROOT, load

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    writer = load("reference_runs/work")
    runner = load("reference_runs/dew_qwen35_step")

    path = Path(__file__).parent / "fixtures" / "hf" / "qwen35-0.8b" / "config.json"
    config = json.loads(path.read_text())
    model = translate_config(config["text_config"]).value
    batch = {"tokens": np.zeros((8, 3072), np.int32)}
    objective = runner.BackboneLoss(model, None, np.ones(model.emb_features, np.float32), 3072)
    record = writer.measure(objective, batch)
    assert record.training_flops_per_step == 3 * counted_flops(model, np.full((8,), 3072, np.int64))
    assert record.training_flops_per_step == objective.training_flops(None, batch)
    assert record.tokens_per_step == 8 * 3072
