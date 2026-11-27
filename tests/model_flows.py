"""The GPU tier's model flows (row 5461, the owner's ruling of 2026-11-11): each model family that takes a
distinct default route on a GPU, built at widths its native kernels take, its training step's loss and
gradients and its decode loop's logits on par with the reference path's (check 1), both against float64
truths its run computes on the GPU's FP64 units (bible §8), every program from its prepare.

A flow's prepare (`DEW_TIER_PREPARE`, on the node's CPU, no GPU drawn) compiles its arms' programs for the
GPU the node's serialized topology names (`dew.training.ahead.compile_programs`) and its float64 truths'
and terms' programs beside them (`truths_ahead.compiled_truths`), and holds its programs' routes to the
ones the flow is for (`Flow.native`, read as the GPU's target would choose them, `generation.routes`). Its
run loads those programs and judges them: it traces, lowers and compiles nothing of its own and computes
every truth from the programs its prepare compiled, but where a qualification run sizes its sites
(`check1_wire.QUALIFY`), each computing its sizing draws' truths (`reference_error._qualified`). Under
the lane's deterministic ops check 1 also holds each arm to its own bits on the same draw
(`reference_error._of_its_inputs`)."""

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import pickle
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path

import lane_environment
import jax
import jax.numpy as jnp
import kernel_matrix
import numpy as np
from arm_arithmetic import drawn_on_the_host
from generation_support import TOPOLOGY, node_target
from gpu_flow import Cell, run_cells
from model_support import float64_twin
from reference_error import as_given, assert_as_exact
from tier_wire import TRUTHS, preparing, unit_needs
from truths_ahead import admitted_needs, compiled_memory, compiled_truths

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.categorical import token_log_probs
from dew.nn.kernels import generation

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


PARAMS = "params.pickle"
"""The file of a flow's store holding the parameters its prepare drew (`Flow.params`)."""

PORTABLE = "portable.json"
"""A flow's store's record that no kernel off the reference path claims a choice it is for on the node's
GPU (`claimed`), at the stamp and GPU of the prepare that found it: its prepare stores nothing else, and its
run at that stamp on that GPU skips it."""

@functools.cache
def generic_kernels() -> dict[str, frozenset[str]]:
    """Each choice's kernels of the reference path: its ops' generic ones (`selection.Kernel.generic`), read
    from Dew's ops as they define them (`kernel_matrix.ops`)."""
    return {str(choice): frozenset(kernel.name for op in ops for kernel in op.kernels if kernel.generic)
            for choice, ops in kernel_matrix.ops().items()}


@functools.cache
def reference_kernels() -> dict[str, str]:
    """The reference arm's choices (`generation.chosen`): each choice's generic kernel that requires nothing,
    the path that takes every call its op's others turn down."""
    found = {}
    for choice, ops in kernel_matrix.ops().items():
        names = {kernel.name for op in ops for kernel in op.kernels if kernel.generic and not kernel.supports}
        if len(names) != 1:
            raise ValueError(f"{choice}'s ops hold {sorted(names) or 'no'} generic kernel requiring nothing: "
                             "the reference arm runs one")
        found[str(choice)] = names.pop()
    return found


@dataclasses.dataclass(frozen=True)
class Flow:
    """One model family's flow: the released config of `fixture` with `widths` set (its text config's for a
    wrapper), computing in `dtype`; `native` each program's (`KINDS`: its training step, its decode loop)
    choices and the kernel each must take on a GPU a native kernel claims it on (a name, `NATIVE` or `EVERY`,
    `held_routes`); under the lane's
    deterministic ops and matmul precision where `lane`, as a user's process runs otherwise, without them
    and at the default precision (`precision`); `rows` sequences of `length` tokens, decoded from `prefill`
    of them a token a step."""

    name: str
    fixture: str
    widths: Mapping[str, object]
    dtype: str
    native: Mapping[str, Mapping[str, str]]
    lane: bool = True
    rows: int = 2
    length: int = 256
    prefill: int = 248
    packed: bool = False
    """Whether its routed experts are packed NVFP4 banks (`dew.nn.kernels.formats.NvFp4`, row 8492), frozen
    beside its parameters as a packed load holds them (`packed_variables`): its training step trains its
    parameters alone, and its truths read each bank's exact matrix (`exact_banks`)."""
    chosen: Mapping[str, Mapping[str, str]] = dataclasses.field(default_factory=dict)
    """Each program's kernels its Dew arm names (`generation.chosen`), by its kind: a kernel a user names
    where its selector takes another at the flow's shapes."""
    seeds: Mapping[str, int] = dataclasses.field(default_factory=lambda: {"train": 8, "decode": 8})
    """Each kind's declared count (`KINDS`), as a qualification run sizes it on the GPUs
    (`reference_error._qualified`, row 6198): the larger of an A100's and a GH200's (DC-SIZE2-A100 and
    -SM90, 2026-11-14)."""
    @property
    def precision(self) -> str | None:
        """The default matmul precision its programs trace at (`jax_default_matmul_precision`): the lane's,
        whose fp32 products are whole (tests/lane_environment.py), or a user's, JAX's own default (None),
        under which a port takes an fp32 call's products in TF32 (`recurrence_chunk._port_precision`)."""
        return "highest" if self.lane else None

    @cached_property
    def config(self) -> dict:
        """The family's released config at the flow's widths."""
        config = json.loads((FIXTURES / self.fixture / "config.json").read_text())
        if "text_config" in config:
            config["text_config"] = {**config["text_config"], **self.widths}
        else:
            config.update(self.widths)
        return config

    @cached_property
    def model(self) -> CausalTransformer:
        """The flow's model, every kernel its selectors' choice."""
        from dew.interop.hf_decoders import translate_config
        from dew.registry import models

        fields = {key: value for key, value in translate_config(self.config).items()
                  if key != "dtype"}
        model = models.build("causal_transformer", **fields, dtype=self.dtype)
        assert isinstance(model, CausalTransformer)
        return model

    @cached_property
    def params(self):
        """The flow's parameters, float32, drawn once on the host's CPU (a GPU's normal transform differs from
        the CPU's in its last bits, so a draw on the GPU would be other parameters than its prepare's) where
        the flow prepares, or where no tier's store holds them; a run reads its prepare's (`PARAMS`), and
        initializes no model. The initializer runs compiled, one program: run eagerly, it compiled each of
        its operations apart, 1,416 programs and 70 s over the flows' prepare on armada (`drawn`)."""
        return drawn(self)

    def initialized(self) -> tuple[jax.Array, ...]:
        """What the model's `init` reads beside its key: token ids of a draw's shape, zeros."""
        return (jnp.zeros((self.rows, self.length), jnp.int32),)

    def taken(self, variables):
        """What a flow's programs take of its model's variables: its parameters, and where `packed` its
        experts' banks beside them (`packed_variables`)."""
        return packed_variables(variables) if self.packed else variables["params"]

    def variables(self, taken) -> dict:
        """The model's variables of what its programs take (`taken`)."""
        return dict(taken) if self.packed else {"params": taken}

    def trained(self, taken):
        """What its training step differentiates of what its programs take: its parameters."""
        return taken["params"] if self.packed else taken

    def with_trained(self, taken, params) -> dict:
        """The model's variables of what its programs take, `params` its parameters."""
        return {**taken, "params": params} if self.packed else {"params": params}

    @property
    def vocab(self) -> int:
        config = self.config.get("text_config", self.config)
        return int(config["vocab_size"])

    def draw(self, seed: int):
        """A draw's inputs: the flow's parameters and `rows` sequences of token ids, NumPy's from the seed."""
        ids = np.random.default_rng([5461, seed]).integers(0, self.vocab, (self.rows, self.length), np.int32)
        return self.params, ids

    def train(self, model) -> Callable:
        """`model`'s training step: each token's next-token loss, `[rows, length - 1]`, and every parameter's
        gradient of their mean: the losses its loss sums, each an element of a row, where their mean alone is
        one value a draw, which no count of draws check 1 takes resolves (a GH200's flows were Inconclusive on
        it at 40 draws, every gradient on par)."""
        def step(taken, ids):
            def loss(params):
                logits = model.apply(self.with_trained(taken, params), ids)
                losses = -token_log_probs(logits[:, :-1], ids[:, 1:])
                return jnp.mean(losses), losses
            (_, losses), gradients = jax.value_and_grad(loss, has_aux=True)(self.trained(taken))
            return losses, gradients
        return step

    def decode(self, model) -> Callable:
        """`model`'s decode loop: its first `prefill` tokens at once, then one token a step, each step's last
        logits, `[rows, 1 + length - 1 - prefill, vocab]`."""
        def decoded(taken, ids):
            variables = self.variables(taken)
            _, made = model.apply(variables, self.rows, method=CausalTransformer.init_cache,
                                  mutable=["cache"])
            logits, mutated = model.apply({**variables, "cache": made["cache"]}, ids[:, :self.prefill],
                                          decode=True, mutable=["cache"])
            steps = [logits[:, -1:]]
            for position in range(self.prefill, self.length - 1):
                logits, mutated = model.apply({**variables, "cache": mutated["cache"]},
                                              ids[:, position:position + 1], decode=True, mutable=["cache"])
                steps.append(logits[:, -1:])
            return jnp.concatenate(steps, axis=1)
        return decoded

    def decode_truth(self, model) -> Callable:
        """The decode loop's float64 truth: one forward over the sequence, read at the steps' positions."""
        def truth(taken, ids):
            logits = model.apply(self.variables(taken), ids[:, :self.length - 1])
            return logits[:, self.prefill - 1:]
        return truth

    kinds = ("train", "decode")
    """The flow's programs: its training step's loss and gradients, and its decode loop's logits."""

    @property
    def inputs(self) -> list[jax.ShapeDtypeStruct]:
        """The shapes of a draw's inputs beside the parameters (`draw`): its token ids."""
        return [jax.ShapeDtypeStruct((self.rows, self.length), np.int32)]

    def program(self, kind: str, model) -> Callable:
        """`model`'s program `kind` (`kinds`)."""
        return self.train(model) if kind == "train" else self.decode(model)

    def truth(self, kind: str, model) -> Callable:
        """The float64 truth of the program `kind` over `model`, its float64 twin: where `packed`, over each
        bank's exact matrix (`exact_banks`), frozen as the bank is."""
        program = self.train(model) if kind == "train" else self.decode_truth(model)
        return (lambda taken, ids: program(exact_banks(taken), ids)) if self.packed else program


@dataclasses.dataclass(frozen=True)
class ImageFlow:
    """A denoiser's flow (`dew.nn.backbones.unet.Unet`, the diffusion run's default model) of two levels
    of `features` channels, the deeper attending, over `channels`-channel latents, computing in `dtype`: its
    training step and its forward over `rows` latents of `size` square pixels and their timesteps, as `Flow`
    holds a decoder's, `native` each program's choices a native kernel takes."""

    name: str
    dtype: str
    native: Mapping[str, Mapping[str, str]]
    lane: bool = True
    features: tuple[int, ...] = (64, 128)
    channels: int = 4
    rows: int = 2
    size: int = 16
    chosen: Mapping[str, Mapping[str, str]] = dataclasses.field(default_factory=dict)
    """Each program's kernels its Dew arm names (`Flow.chosen`)."""
    seeds: Mapping[str, int] = dataclasses.field(default_factory=lambda: {"train": 8, "forward": 8})
    """Each kind's declared count (`kinds`)."""

    kinds = ("train", "forward")
    """The flow's programs: its training step's per-element losses and gradients, and its forward's output."""

    @property
    def precision(self) -> str | None:
        """As `Flow.precision`."""
        return "highest" if self.lane else None

    @cached_property
    def model(self):
        """The flow's UNet, its attention its selectors' choice, the deeper stage
        attending."""
        from dew.nn.attention_blocks import Stage
        from dew.nn.backbones.unet import Unet

        return Unet(output_channels=self.channels, emb_features=2 * self.features[-1],
                    feature_depths=self.features, attention_configs=(None, Stage(heads=4)), num_res_blocks=1,
                    norm_groups=8, dtype=self.dtype)

    @property
    def inputs(self) -> list[jax.ShapeDtypeStruct]:
        """The shapes of a draw's inputs beside the parameters (`draw`): its latents, their timesteps and
        their targets."""
        latents = (self.rows, self.size, self.size, self.channels)
        return [jax.ShapeDtypeStruct(latents, np.float32), jax.ShapeDtypeStruct((self.rows,), np.float32),
                jax.ShapeDtypeStruct(latents, np.float32)]

    @cached_property
    def params(self):
        """The flow's variables, float32, drawn once on the host's CPU, or its prepare's (`Flow.params`): its
        parameters and its time embedding's frequencies (the `constants` collection), which its programs
        read and differentiate none of (`drawn`)."""
        return drawn(self)

    @staticmethod
    def taken(variables):
        """What a flow's programs take of its model's variables (`params`): all of them."""
        return variables

    def initialized(self) -> tuple[jax.Array, ...]:
        """What the model's `init` reads beside its key: latents and timesteps of a draw's shapes, zeros."""
        latents, times, _ = self.inputs
        return jnp.zeros(latents.shape, latents.dtype), jnp.zeros(times.shape, times.dtype)

    def draw(self, seed: int):
        """A draw's inputs: the flow's parameters, latents and targets of a unit normal's and timesteps in
        [0, 1), NumPy's from the seed."""
        rng = np.random.default_rng([5461, seed])
        latents, times, targets = self.inputs
        return (self.params, rng.standard_normal(latents.shape).astype(np.float32),
                rng.random(times.shape).astype(np.float32),
                rng.standard_normal(targets.shape).astype(np.float32))

    def train(self, model) -> Callable:
        """`model`'s training step: each element's squared error against its target, `[rows, size, size,
        channels]`, and every parameter's gradient of their mean (`Flow.train`'s per-element losses)."""
        def step(variables, latents, times, targets):
            def loss(params):
                denoised = model.apply({**variables, "params": params}, latents, times)
                losses = jnp.square(denoised - targets)
                return jnp.mean(losses), losses
            (_, losses), gradients = jax.value_and_grad(loss, has_aux=True)(variables["params"])
            return losses, gradients
        return step

    def forward(self, model) -> Callable:
        """`model`'s forward: its output over the latents at their timesteps."""
        def forward(variables, latents, times, targets):
            del targets
            return model.apply(variables, latents, times)
        return forward

    def program(self, kind: str, model) -> Callable:
        """`model`'s program `kind` (`kinds`)."""
        return self.train(model) if kind == "train" else self.forward(model)

    def truth(self, kind: str, model) -> Callable:
        """The float64 truth of the program `kind` over `model`, its float64 twin: the program itself."""
        return self.program(kind, model)


type ModelFlow = Flow | ImageFlow
"""A flow of any family (`FLOWS`): a decoder's, or a denoiser's."""


EXPERT_BANKS = ("gate_up_proj", "down_proj")
"""GPT-OSS's routed experts' matrices, each `[experts, in, out]`, which a packed load holds as NVFP4 banks
(`dew.interop.quantized_loads.with_banks`)."""


def packed_variables(variables) -> dict:
    """`variables`' parameters with each routed expert matrix (`EXPERT_BANKS`) packed into an NVFP4 bank
    (`NvFp4.quantized`) in `QUANTIZATION` at its path, taken out of the parameters, as a packed load holds
    it (`quantized_loads.with_banks`), which the expert layers read (`dew.nn.dense.packed_matrix`)."""
    from flax import linen as nn
    from flax.traverse_util import flatten_dict, unflatten_dict

    from dew.nn.kernels.formats import NvFp4
    from dew.quantization import QUANTIZATION

    # Its parameters' values, each partitioning's box taken off, as a load holds them.
    params = flatten_dict(dict(nn.unbox(variables["params"])))
    banks = {path: NvFp4.quantized(jnp.asarray(params.pop(path))) for path in list(params)
             if path[-1] in EXPERT_BANKS}
    return {"params": unflatten_dict(params), QUANTIZATION: unflatten_dict(banks)}


def exact_banks(taken) -> dict:
    """What a packed flow's programs take (`packed_variables`) with each NVFP4 bank its exact matrix, in its
    tensor scale's dtype (float64 where check 1 widens it), computed in that dtype from the bank's bits: each
    code's E2M1 value read from its stored word by integer arithmetic and casts alone (`NvFp4.codes`), times
    its block's scale and its expert's, which float64 holds exactly (2 + 3 + 24 significant bits); still
    frozen where the bank lay, so a truth differentiates the parameters alone."""
    from dew.nn.kernels.formats import NvFp4
    from dew.quantization import QUANTIZATION

    def exact(bank: NvFp4) -> jax.Array:
        dtype = jnp.result_type(bank.scale)
        codes = bank.codes(dtype)
        blocks = jnp.repeat(jnp.swapaxes(jnp.asarray(bank.scales), -1, -2).astype(dtype), 16, axis=-2)
        return codes * blocks * jnp.asarray(bank.scale, dtype)[..., None, None]

    return {**taken, QUANTIZATION: jax.tree.map(exact, taken[QUANTIZATION],
                                                is_leaf=lambda leaf: isinstance(leaf, NvFp4))}


def drawn(flow: ModelFlow):
    """What the flow's programs take of its model's variables (`ModelFlow.taken`), float32, drawn once on the
    host's CPU by its compiled initializer where the flow prepares, or where no tier's store holds them; a
    run's its prepare's (`PARAMS`)."""
    if TRUTHS in os.environ and not preparing():
        return pickle.loads((store_of(flow) / PARAMS).read_bytes())
    return drawn_on_the_host(lambda seed: flow.taken(jax.jit(flow.model.init)(jax.random.key(seed),
                                                                              *flow.initialized())),
                             5461, held=True)


KINDS = Flow.kinds
"""A decoder flow's programs (`Flow.kinds`)."""


NATIVE = "native"
"""A program's route taken by a kernel off the reference path (`generic_kernels`), among the kernels its
choice's selections report: a recurrence's reports its forward's and its training's kernels alike, whichever
runs, so a route set cannot tell one pass from another (check 1 judges the passes that run)."""

EVERY = "every"
"""A program's route whose every selection is off the reference path: a choice each pass selects as it runs
(the template's mainloop, its forward's and its backward's alike), so a pass on the plain-JAX path is seen."""

TEMPLATE = {"attention": NATIVE, "attention_mainloop": EVERY}
"""An attention program's routes on the template: its attention off the reference path, and every pass of
the template on a GPU's own mainloop, not its plain-JAX one (`templates.attention_route.MAINLOOP`), which a
template call would otherwise take unseen."""

DENSE = {"hidden_size": 512, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 128,
         "intermediate_size": 1024, "num_hidden_layers": 2, "vocab_size": 512,
         "max_position_embeddings": 1024, "attention_bias": False}
"""A decoder's widths at which the GPU's attention kernels take its calls: heads of 128, as many key heads as
query heads, so its decode folds onto the decode kernel (`attention_kernels.decodes`)."""

GEMMA2 = {**DENSE, "query_pre_attn_scalar": 128, "sliding_window": 64}
"""Gemma 2's alternating window of 64 keys and its softcaps, at `DENSE`'s widths."""

GPT_OSS = {**DENSE, "sliding_window": 64, "num_local_experts": 16, "num_experts_per_tok": 16,
           "intermediate_size": 128}
"""GPT-OSS's sinks and alternating window at `DENSE`'s widths, every token routed to every one of its experts:
no routing decision two arms could take apart. Sixteen experts 128 wide hold the parameters two 1024 wide
would, and give the router's bias sixteen values a draw where two left check 1 Inconclusive on it at 40 draws
on an A100 (pooled ratio 1.25 in [0.74, 2.01])."""

GPT_OSS_PACKED = {**GPT_OSS, "intermediate_size": 256}
"""`GPT_OSS` with its experts 256 wide, so each bank's depth is whole 256s and its width whole 64s, which the
packed GEMM's NVFP4 mainloop takes (`dew.nn.kernels.gemm.NVFP4`'s `packed`)."""

MLA = {**DENSE, "head_dim": 64, "kv_lora_rank": 128, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
       "qk_head_dim": 128, "v_head_dim": 64, "first_k_dense_replace": 2}
"""DeepSeek-V2's latent attention, its value heads narrower than its query heads, every layer's MLP dense (its
`head_dim` the rope slice's, as DeepSeek points it)."""

KDA = {"hidden_size": 512, "num_hidden_layers": 2, "head_dim": 128, "num_attention_heads": 4,
       "num_key_value_heads": 4, "intermediate_size": 1024, "vocab_size": 512, "first_k_dense_replace": 2,
       "num_experts": 2, "num_experts_per_token": 2,
       "linear_attn_config": {"full_attn_layers": [2], "head_dim": 128, "kda_layers": [1], "num_heads": 4,
                              "short_conv_kernel_size": 4}}
"""Kimi Linear's delta attention, its keys 128 wide (the template's and the port's), every MLP dense."""

GDN = {"hidden_size": 512, "num_hidden_layers": 2, "head_dim": 128, "num_attention_heads": 4,
       "num_key_value_heads": 4, "intermediate_size": 1024, "vocab_size": 512,
       "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
       "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 2,
       "linear_num_value_heads": 4, "mlp_only_layers": [0, 1], "max_position_embeddings": 1024}
"""Qwen3-Next's gated delta net and short conv, its keys and values 128 wide, every MLP dense."""

SSD = {"hidden_size": 256, "num_hidden_layers": 2, "num_heads": 8, "head_dim": 64, "state_size": 64,
       "chunk_size": 64, "n_groups": 1, "time_step_rank": 16, "vocab_size": 512}
"""Mamba-2's state-space duality in chunks of 64, the SSD template's and the mamba_ssm port's."""

SELECTIVE = {"hidden_size": 64, "intermediate_size": 128, "state_size": 16, "time_step_rank": 8,
             "num_hidden_layers": 2, "vocab_size": 512}
"""Mamba's selective scan at the selective template's least widths (tests/test_selective_template.py)."""

LIGHTNING = {"hidden_size": 512, "num_hidden_layers": 2, "head_dim": 128, "num_attention_heads": 4,
             "num_key_value_heads": 4, "intermediate_size": 512, "vocab_size": 512, "block_size": 64,
             "attn_type_list": [0, 1], "num_local_experts": 2, "num_experts_per_tok": 2,
             "max_position_embeddings": 1024}
"""MiniMax-Text-01's lightning attention, then its softmax layer, every token routed to both its experts."""

FLOWS = (
    Flow("dense bf16, the lane's", "llama-tiny", DENSE, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}),
    # A user's attention a native kernel's, the cell's own on each GPU: FlashAttention's, or on an A100
    # 'flash_inference' (its backward the template's, FlashAttention-2's failing check 1 there, row 13288).
    Flow("dense bf16, a user's", "llama-tiny", DENSE, "bfloat16",
         {"train": {"attention": NATIVE}, "decode": {"attention": NATIVE, "decode": NATIVE}}, lane=False),
    Flow("dense fp32", "llama-tiny", DENSE, "float32", {"train": {}, "decode": {}}),
    Flow("window and softcap, the lane's", "gemma2-tiny", GEMMA2, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}),
    Flow("window and softcap, a user's", "gemma2-tiny", GEMMA2, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}, lane=False),
    Flow("sinks, the lane's", "gpt-oss-tiny", GPT_OSS, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}, seeds={"train": 128, "decode": 8}),
    Flow("sinks, a user's", "gpt-oss-tiny", GPT_OSS, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}, lane=False,
         seeds={"train": 128, "decode": 8}),
    # Its training named onto the packed GEMM, which its selector takes at a decode step's rows alone.
    Flow("experts packed NVFP4", "gpt-oss-tiny", GPT_OSS_PACKED, "bfloat16",
         {"train": {"nvfp4": EVERY}, "decode": {"nvfp4": NATIVE}}, packed=True,
         chosen={"train": {"nvfp4": "packed"}}),
    Flow("latent attention, the lane's", "deepseek-v2-tiny", MLA, "bfloat16",
         {"train": TEMPLATE, "decode": {}}),
    Flow("latent attention, a user's", "deepseek-v2-tiny", MLA, "bfloat16",
         {"train": {"attention": NATIVE}, "decode": {}}, lane=False),
    Flow("delta attention (KDA)", "kimi-linear-tiny", KDA, "bfloat16",
         {"train": {**TEMPLATE, "gated_delta_rule": NATIVE, "short_conv": NATIVE},
          "decode": {"gated_delta_rule": NATIVE, "short_conv": NATIVE, "delta_decode": "step"}},
         seeds={"train": 64, "decode": 8}),
    Flow("gated delta net and its short conv", "qwen3-next-tiny", GDN, "bfloat16",
         {"train": {**TEMPLATE, "gated_delta_rule": NATIVE, "short_conv": NATIVE},
          "decode": {**TEMPLATE, "decode": NATIVE, "gated_delta_rule": NATIVE, "short_conv": NATIVE,
                     "delta_decode": "step"}}, seeds={"train": 64, "decode": 8}),
    Flow("state-space duality (SSD), bf16", "mamba2-tiny", SSD, "bfloat16",
         {kind: {"ssd": NATIVE, "short_conv": NATIVE} for kind in KINDS}, seeds={"train": 16, "decode": 8}),
    Flow("state-space duality (SSD), fp32, a user's", "mamba2-tiny", SSD, "float32",
         {kind: {"ssd": NATIVE, "short_conv": NATIVE} for kind in KINDS}, lane=False,
         seeds={"train": 32, "decode": 8}),
    Flow("selective scan", "mamba-tiny", SELECTIVE, "bfloat16",
         {kind: {"selective_scan": NATIVE, "short_conv": NATIVE} for kind in KINDS}),
    Flow("lightning attention", "minimax-tiny", LIGHTNING, "bfloat16",
         {"train": {**TEMPLATE, "gated_delta_rule": NATIVE},
          "decode": {**TEMPLATE, "decode": NATIVE, "gated_delta_rule": NATIVE}}),
    ImageFlow("UNet denoiser bf16", "bfloat16", {"train": {}, "forward": {}}),
    ImageFlow("UNet denoiser fp32", "float32", {"train": {}, "forward": {}}),
)
"""Every model flow, by the route it is for (docs/design/testing.md: a kernel's native run is a flow's
cell): each family that takes a distinct default route on a GPU, its bf16 attention flows twice, under the
lane's deterministic ops (the template on a GPU's own mainloop) and as a user's process runs it (a
library's fused kernel where one takes the call, FlashAttention among them, which refuses the lane's), the
dense fp32 flow on XLA's declared fp32 route, and the SSD port in fp32 too, as a user's process runs it (its
products in TF32 at the default precision, which the lane's whole fp32 products refuse).
Lightning attention's decay without the delta rule takes the recurrence template on a Hopper or an Ampere GPU
(an A100's once its output launch's phases were stated apart, row 6233), and its training on a Hopper GPU
(the fast backward's intermediates in fp32's three parts, `mainloop_backward.fast_backward_form`), an A100's
training the portable math's at its 128-wide heads, whose fast backward's launches pass an A100's registers
(`template.backward_holds`): fla's port hands it to the portable math (`recurrence_chunk.PORT_SUPPORTS`).
A latent attention flow's decode loop attends in XLA (the latent decode's own form), and a KDA flow's decode
step's softmax layer too: no native attention route of theirs there. A UNet denoiser's flows (`ImageFlow`, the
diffusion run's default model) hold its convolutions', norms' and attention's default routes, bf16 and fp32
(row 7941): no native kernel claims a choice of theirs, so they run on every GPU."""


def flow_named(name: str) -> ModelFlow:
    return next(flow for flow in FLOWS if flow.name == name)


CPU_BACKEND_FLAGS = ("--xla_cpu_scheduler_type",)
"""The lane's flags of the CPU backend's own (tests/lane_environment.py) that a GPU flow's programs read
nothing of and a prepare's CPU process sets apart from its run's GPU one; a CPU program's schedule they do
decide, so `dew.training.ahead.computation_flags` keeps them, and drops the host's device count itself
(`ahead.HOST_FLAGS`)."""


def flow_flags(flow: ModelFlow) -> str:
    """The XLA flags a flow compiles its programs under in its prepare and loads them under in its run, the
    same in both (`BuildTarget.held` reads them, in their order): the process's that decide what a program
    computes (`dew.training.ahead.computation_flags`: no phase's own, as a cache directory a process sets
    after its backend opened, nor the host's device count), but the CPU backend's own (`CPU_BACKEND_FLAGS`),
    then its variant's on a cuda lane, the lane's repeatable flags where `lane`, else a user's
    (`lane_environment.REPEATABLE_GPU_FLAGS`, `USERS_GPU_FLAGS`)."""
    from dew.training.ahead import computation_flags

    variant = lane_environment.REPEATABLE_GPU_FLAGS if flow.lane else lane_environment.USERS_GPU_FLAGS
    named = (*CPU_BACKEND_FLAGS, *(flag.split("=")[0] for flag in variant))
    flags = lane_environment.without_flags(" ".join(computation_flags()), *named)
    return " ".join([flags, *variant]).strip()


def users_process() -> bool:
    """Whether this process computes as a user's does (`lane_environment.USERS_ARITHMETIC`): the process a
    flow not of the lane's runs in, prepared and run (`Flow.lane`)."""
    return os.environ.get(lane_environment.USERS_ARITHMETIC) == "1"


class WrongProcess(AssertionError):
    """A flow in a process not opened under its variant's flags or matmul precision (`held_process`)."""


def held_process(flow: ModelFlow) -> None:
    """Raise unless this process is of the flow's variant (row 5461): XLA parses its flags once a process and
    JAX reads its precision as it starts, so a flow holds only where its process's XLA runtime runs under the
    flow's flags (`flow_flags`) and JAX multiplies at the flow's precision (`Flow.precision`). A prepare's
    process, which Dew's import precedes jaxlib's (`PREPARED_APART`), shows its flags hold for every runtime
    XLA loads (`dew._devices.flags_hold`); a run's, which tests/conftest.py configures before it imports JAX
    and the tier starts in its variant's environment, that XLA_FLAGS carried them while its backend was
    closed (`flags_set`), what a process that loaded jaxlib before Dew can show."""
    from dew._devices import flags_hold, flags_set

    flags = flow_flags(flow)
    held = flags_hold(flags) if preparing() else flags_set(flags)
    if not held or jax.config.jax_default_matmul_precision != flow.precision:
        raise WrongProcess(f"{flow.name}: its process opened its backend under XLA_FLAGS "
                           f"{os.environ.get('XLA_FLAGS', '')!r} at matmul precision "
                           f"{jax.config.jax_default_matmul_precision!r}, not {flags!r} at "
                           f"{flow.precision!r}")


def store_of(flow: ModelFlow) -> Path:
    """The flow's directory in the tier's store (`TRUTHS`), refused where the tier names none."""
    if TRUTHS not in os.environ:
        raise RuntimeError(f"a model flow reads its truths and programs from {TRUTHS}, which its part's "
                           "prepare writes: none is set")
    return Path(os.environ[TRUTHS]) / hashlib.sha256(flow.name.encode()).hexdigest()[:24]


def stamp() -> str:
    """The commit (`map_judge.commit`) and JAX the flows' truths' programs are compiled and read at: a program
    another compiled is never run (`truths_ahead.compiled_truths`)."""
    from map_judge import commit

    return f"{commit()} jax {jax.__version__} jaxlib {importlib.metadata.version('jaxlib')}"


def _flat(flow: ModelFlow, program: Callable):
    """`program(params, *inputs)` as a function of the parameters' leaves and the draw's inputs
    (`ModelFlow.inputs`), and their shapes: a program compiled ahead takes and returns arrays
    (`compile_programs`)."""
    leaves, tree = jax.tree.flatten(flow.params)
    count = len(leaves)

    def flat(*arguments):
        return program(jax.tree.unflatten(tree, arguments[:count]), *arguments[count:])
    shapes = [jax.ShapeDtypeStruct(np.shape(leaf), np.asarray(leaf).dtype) for leaf in leaves]
    return flat, [*shapes, *flow.inputs]


def programs_of(flow: ModelFlow) -> dict[str, tuple[Callable, list]]:
    """The flow's programs by name, each of the parameters' leaves and the draw's inputs: each of its kinds
    (`ModelFlow.kinds`), Dew's (`dew`) and the reference path's (`reference`)."""
    model = flow.model

    def named(kind: str, program: Callable) -> Callable:
        # Dew's arm traced under the kernels the flow names for it (`chosen`), its reference's as given.
        def dew(*arguments):
            with generation.chosen(**flow.chosen.get(kind, {})):
                return program(*arguments)
        return dew

    return {f"{kind} {arm}": _flat(flow, named(kind, flow.program(kind, model)) if arm == "dew"
                                   else flow.program(kind, model))
            for kind in flow.kinds for arm in ("dew", "reference")}


class Unreached(AssertionError):
    """A flow whose Dew arm did not take a route it is for on a GPU a native kernel claims it on
    (`held_routes`)."""


def claimed(flow: ModelFlow, found: Mapping[str, list]) -> dict[str, set[str]]:
    """Each program's choices the flow is for that a kernel off the reference path claims on the GPU its
    routes (`found`, each program's by its kind) were traced for: a route such a kernel took, or one whose
    `preferred` (what it runs where nothing refuses the call, `generation.Route`) is such a kernel turned
    down for a limit a row tracks lifting, so a kernel refused by a fit or a launch it does not yet make
    reddens its flow while its row is open. A refusal no implementation lifts (row `INHERENT`), or the GPU
    measuring the reference path the faster (`measured`), releases the claim. Read from the routes, never
    a list kept beside the kernels: a kernel that comes to claim a GPU is the flow's to take there."""
    generic = generic_kernels()

    def claims(route) -> bool:
        choice = str(route.choice)
        if route.kernel not in generic[choice]:
            return True  # a kernel off the reference path took the call
        refusal = route.refusal
        return route.preferred not in generic[choice] and (
            refusal is None or (refusal.row != generation.INHERENT and refusal.kind != "measured"))

    return {kind: {str(route.choice) for route in found[kind]
                   if str(route.choice) in choices and claims(route)}
            for kind, choices in flow.native.items()}


def held_routes(flow: ModelFlow, found: Mapping[str, list], target: str) -> None:
    """Raise unless each of the flow's programs, traced for the GPU generation `target` (`found`, each
    program's routes by its kind), took every route it is for that a native kernel claims there
    (`claimed`): a kernel off the reference path among its choice's routes (`NATIVE`), every one of them so
    (`EVERY`), or the kernel it names. A choice no native kernel claims on `target` runs the portable math,
    which check 1 holds as it holds any route."""
    generic = generic_kernels()
    native = claimed(flow, found)
    for kind, choices in flow.native.items():
        taken = {}
        for route in found[kind]:
            taken.setdefault(str(route.choice), set()).add(route.kernel)
        for choice, kernel in choices.items():
            if choice not in native[kind]:
                continue
            kernels = taken.get(choice, set())
            held = (kernels - generic[choice] if kernel == NATIVE else
                    kernels if kernel == EVERY and not kernels & generic[choice] else kernels & {kernel})
            if not held or (kernel not in (NATIVE, EVERY) and kernels - {kernel}):
                refused = sorted({f"{route.preferred}: {route.refusal.code}" for route in found[kind]
                                  if str(route.choice) == choice and route.refusal is not None})
                raise Unreached(f"{flow.name}: its {kind}'s {choice} took {sorted(kernels) or 'nothing'} on "
                                f"{target}, which a native kernel claims, not {kernel}; refused: {refused}")


def held_reference(flow: ModelFlow, found: list) -> None:
    """Raise unless the flow's reference arm ran the reference path alone (`generic_kernels`)."""
    generic = generic_kernels()
    if native := sorted({(str(route.choice), route.kernel) for route in found
                         if route.kernel not in generic.get(str(route.choice), ())}):
        raise AssertionError(f"{flow.name}: its reference arm ran {native}, not its reference path")


def _compiled(flow: ModelFlow, named: Mapping[str, tuple[Callable, list]], arm: str, target):
    """The arm's programs of `named` (`programs_of`) compiled for `target` (`compile_programs`), one a call so
    each program's routes are its own, each program's routes by its kind (`ModelFlow.kinds`), and the names
    of those its compile refuses to prepare, each with the refusal's reason (`report.CompiledOnDevice`: a
    convolution, whose algorithm only its GPU picks), which its run compiles on its GPU (`loaded`): their
    routes traced all the same. None of the programs where every one is refused."""
    from dew.nn.kernels.templates.report import CompiledOnDevice
    from dew.training.ahead import AheadPrograms, compile_programs

    compiled, found, on_device = {}, {}, {}
    for kind in flow.kinds:
        name = f"{kind} {arm}"
        with generation.routes() as found[kind]:
            # Over the node's own topology, tuned as its run's compile would be (row 8422).
            try:
                compiled[kind] = compile_programs({name: named[name]}, target.generation,
                                                  target.multiprocessors, topology=os.environ[TOPOLOGY])
            except CompiledOnDevice as refused:
                on_device.update(dict.fromkeys(refused.programs or (name,), str(refused)))
    if not compiled:
        return None, found, on_device
    return AheadPrograms(next(iter(compiled.values())).target, {
        name: program for each in compiled.values() for name, program in each.programs.items()},
        tuned=all(each.tuned for each in compiled.values()),
        at_run=tuple(name for each in compiled.values() for name in each.at_run)), found, on_device


ON_DEVICE = "on_device.json"
"""A flow's file in its store naming the programs its prepare could not prepare (`report.CompiledOnDevice`),
each arm's, by the refusal's reason, which its run compiles on its GPU and names (`loaded`)."""


def prepare(flow: ModelFlow) -> bool:
    """The flow's prepare: its programs compiled for the node's GPU into its store, Dew's arm's routes held
    to the ones it is for (`held_routes`), the reference arm's to its reference path; then every draw's truth
    (`judge`). Whether it prepared: none where no kernel off the reference path claims a choice the flow is
    for on the node's GPU (`claimed`), which its store records (`PORTABLE`) for its run to skip."""
    held_process(flow)
    target = node_target()
    directory = store_of(flow)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PORTABLE).unlink(missing_ok=True)
    named = programs_of(flow)
    dew, found, on_device = _compiled(flow, named, "dew", target)
    if any(flow.native.values()) and not any(claimed(flow, found).values()):
        # No kernel off the reference path claims a choice the flow is for on this GPU: its run's to skip.
        (directory / PORTABLE).write_text(json.dumps({"stamp": stamp(), "target": target.generation}))
        return False
    (directory / PARAMS).write_bytes(pickle.dumps(flow.params))
    held_routes(flow, found, target.generation)
    with generation.chosen(**reference_kernels()):
        reference, found, refused = _compiled(flow, named, "reference", target)
    held_reference(flow, [route for kind in flow.kinds for route in found[kind]])
    for arm, programs in (("dew", dew), ("reference", reference)):
        (directory / f"{arm}.programs").unlink(missing_ok=True)
        if programs is not None:
            programs.save(directory / f"{arm}.programs")
    (directory / ON_DEVICE).write_text(json.dumps({"programs": {**on_device, **refused}, "stamp": stamp()}))
    for kind in flow.kinds:
        judge(flow, {}, kind, target)
    programs = need(flow, dew, reference, directory / "truths")
    (directory / NEED).write_text(json.dumps(programs))
    unit_needs(programs["programs"])
    return True


NEED = "need.json"
"""A flow's file in its store stating its run's programs' device memory in the allocator's terms (`need`),
by which the GPU tier admits a part's worker to a slot (`dew.training.memory.admitted` of each)."""


def need(flow: ModelFlow, dew, reference, truths: Path) -> dict[str, list[dict[str, int | str]]]:
    """A flow's run's programs as the allocator admits them (`truths_ahead.admitted_needs`): its own programs'
    (Dew's and its reference's, each `Executable.memory`, labelled `<flow>: <program>`, as `train dew`), the
    state its run holds while its truths compute, and its truths' (`truths_ahead.compiled_memory`, each
    labelled by its site), so a unit's needs say which program asks for what (row 13580)."""
    # A program its run compiles on its GPU (`ON_DEVICE`) states none ahead, as one whose compile states
    # none (`Executable.memory`).
    arms = [{**executable.memory._asdict(), "label": f"{flow.name}: {name}"}
            for programs in (dew, reference) if programs is not None
            for name, executable in programs.programs.items() if executable.memory is not None]
    return {"programs": admitted_needs(arms, compiled_memory(truths))}


PREPARED_APART = """import sys
import dew  # before jaxlib loads, so XLA_FLAGS' flags are proven held (`dew._devices.flags_hold`)
import model_flows
model_flows.prepared_here(sys.argv[1])
"""
"""A flow's prepare process's program (`prepared_apart`)."""


def prepared_apart(flow: ModelFlow) -> dict | None:
    """The flow's prepare (`prepare`) in a process of its own (`PREPARED_APART`), started under its XLA
    flags (`flow_flags`) and this process's arithmetic, its variant's (`users_process`): XLA parses its flags
    once a process, as it starts, and the programs it compiles state them (`BuildTarget`). What it took where
    it prepared (`measured`), None where the node's GPU does not run the flow."""
    from process_support import fresh

    return fresh(PREPARED_APART, flow.name, flags=flow_flags(flow))


def prepared_here(name: str) -> None:
    """The flow `name` prepared in this process, what it took printed (`measured`), null where the node's GPU
    does not run it: `PREPARED_APART`'s program."""
    import time

    began = time.perf_counter()
    flow = flow_named(name)
    print(json.dumps(measured(flow, began) if prepare(flow) else None))


def loaded(flow: ModelFlow) -> dict[str, Callable]:
    """The flow's programs its prepare compiled, loaded onto this GPU, each a function of the parameters and
    the draw's inputs whose results are fetched to the host, as check 1 holds an arm's (`reference_error`):
    no arm's outputs stay on the GPU while the next program runs, which the flow's needs count none of
    (`need`)."""
    from truths_ahead import AT_RUN

    from dew.training.ahead import AheadPrograms

    directory = store_of(flow)
    on_device = json.loads((directory / ON_DEVICE).read_text())["programs"]
    held: dict[str, Callable] = {
        name: program for arm in ("dew", "reference") if (directory / f"{arm}.programs").exists()
        for name, program in AheadPrograms.load(directory / f"{arm}.programs").loaded().items()}
    # Each program its prepare could not prepare compiled on this GPU as it first runs, under this process's
    # flags, traced as its prepare traced it (the reference arm under its reference kernels), named in the
    # run's summary as its prepare declared (`truths_ahead.AT_RUN`), in the flow's truths' store, as an at-run
    # site's truths are (the store a flow's run opens names its records, tests/conftest.py's).
    named = programs_of(flow) if on_device else {}
    with compiled_truths(directory / "truths", prepare=False, stamp=stamp()) as truths:
        for name, why in on_device.items():
            kernels = reference_kernels() if name.endswith(" reference") else {}
            held[name] = jax.jit(functools.partial(_traced_under, named[name][0], kernels))
            truths.unprepared(f"{flow.name} {name}: {why}", AT_RUN)
    return {name: as_given(lambda params, *inputs, program=program: jax.device_get(
                program(*jax.tree.leaves(params), *inputs))) for name, program in held.items()}


def _traced_under(program: Callable, kernels: Mapping[str, str], *arguments):
    """`program(*arguments)` traced with `kernels` named (`generation.chosen`)."""
    with generation.chosen(**kernels):
        return program(*arguments)


def judge(flow: ModelFlow, programs: Mapping[str, Callable], kind: str, target=None) -> None:
    """Check 1 of the flow's program `kind` (`ModelFlow.kinds`) against its float64 truth, computed on this
    GPU by the programs the tier's prepare compiled for it (`truths_ahead.compiled_truths`); compiled there
    where this process prepares (`programs` none), for `target` (the node's GPU, `node_target`; this
    process's devices where none)."""
    truth = flow.truth(kind, float64_twin(flow.model))
    with compiled_truths(store_of(flow) / "truths", prepare=preparing(), stamp=stamp(), target=target):
        assert_as_exact(programs.get(f"{kind} dew"), programs.get(f"{kind} reference"), flow.draw,
                        f"{flow.name}: {kind}", seeds=flow.seeds[kind], truth_fn=truth)


def run(flow: ModelFlow) -> int:
    """The flow on this GPU from its prepared programs and truths: check 1 of each program its run judges
    (`ModelFlow.kinds`), each a cell (`gpu_flow.run_cells`, which reports the cells that ran); its device's
    peak bytes after them, which the test reports."""
    programs = loaded(flow)
    run_cells([Cell(f"{flow.name}: {kind}", functools.partial(judge, flow, programs, kind))
               for kind in flow.kinds])
    return int((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0))


def measured(flow: ModelFlow, began: float) -> dict:
    """What a prepare since `began` took (`prepared_apart`): its seconds, its process's peak resident bytes
    and the bytes it wrote to the flow's store, which size a tier part's prepare on its node."""
    import resource
    import time

    written = sum(path.stat().st_size for path in store_of(flow).rglob("*") if path.is_file())
    return {"seconds": time.perf_counter() - began, "written_bytes": written,
            "peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def portable_here(flow: ModelFlow) -> bool:
    """Whether the flow's prepare found nothing it is for claimed on this GPU (`PORTABLE`), at this stamp:
    a record another stamp or GPU wrote is refused, never read as this run's."""
    record = store_of(flow) / PORTABLE
    if not record.is_file():
        return False
    found = json.loads(record.read_text())
    here = {"stamp": stamp(), "target": generation.device_generation()}
    if found != here:
        raise ValueError(f"{flow.name}: its store's portable record is {found}'s, not this run's {here}")
    return True
