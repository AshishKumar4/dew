"""The GPU tier's model flows (row 5461, the owner's ruling of 2026-11-11): each model family that takes a
distinct default route on a GPU, built at widths its native kernels take, its training step's loss and
gradients and its decode loop's logits on par with the reference path's (check 1), both against float64
truths its prepare computed, from programs its prepare compiled for the node's GPU.

A flow's prepare (`DEW_TIER_PREPARE`, on the node's CPU, no GPU drawn) computes every draw's float64 truth
into the tier's store (`reference_error.stored_truths`), compiles its programs for the GPU the node's
serialized topology names (`dew.training.ahead.compile_programs`) beside them, and holds its programs'
routes to the ones the flow is for (`Flow.native`, read as the GPU's target would choose them,
`generation.routes`). Its run loads those programs and judges them: it traces, lowers and compiles
nothing of its own and computes no truth. Under the lane's deterministic ops check 1 also holds each arm to
its own bits on the same draw (`reference_error._of_its_inputs`)."""

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
from gpu_flow import Cell, run_cells
from model_support import float64_twin
from reference_error import as_given, assert_as_exact, stored_truths

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.categorical import token_log_probs
from dew.nn.kernels import generation

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

PREPARE = "DEW_TIER_PREPARE"
"""Set to 1, a flow prepares rather than runs (`preparing`): on the node's CPU, before its GPU run."""

TRUTHS = "DEW_TIER_TRUTHS"
"""The directory a part's prepare writes its flows' truths and programs to and its run reads them from."""

PARAMS = "params.pickle"
"""The file of a flow's store holding the parameters its prepare drew (`Flow.params`)."""

TOPOLOGY = "DEW_NODE_TOPOLOGY"
"""The node's GPUs' serialized topology (dew-ops' session hands a prepare its node's): the GPU a flow's
programs are compiled for."""

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


def preparing() -> bool:
    """Whether this process prepares the tier's flows (`PREPARE`)."""
    return os.environ.get(PREPARE) == "1"


@dataclasses.dataclass(frozen=True)
class Flow:
    """One model family's flow: the released config of `fixture` with `widths` set (its text config's for a
    wrapper), computing in `dtype`; `native` each program's (`KINDS`: its training step, its decode loop)
    choices on a GPU and the kernel each must take there (a name, `NATIVE` or `EVERY`); under the lane's
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
    seeds: int = 8
    families: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    """Each choice the flow is for that takes a native kernel on some GPUs alone: the instruction families
    (`generation.family`) of those GPUs, elsewhere the portable math; a choice it names none for takes one on
    every GPU."""

    @property
    def precision(self) -> str | None:
        """The default matmul precision its programs trace at (`jax_default_matmul_precision`): the lane's,
        whose fp32 products are whole (tests/lane_environment.py), or a user's, JAX's own default (None),
        under which a port takes an fp32 call's products in TF32 (`recurrence_chunk._port_precision`)."""
        return "highest" if self.lane else None

    def native_on(self, target: str) -> dict[str, dict[str, str]]:
        """Each program's choices that take a native kernel on the GPU generation `target` (`families`), each
        with the kernel it must take there."""
        family = generation.family(target)
        return {kind: {choice: kernel for choice, kernel in choices.items()
                       if family in self.families.get(choice, (family,))}
                for kind, choices in self.native.items()}

    def runs_on(self, target: str) -> bool:
        """Whether the flow runs on the GPU generation `target`: where a choice it is for is native there, or
        where it is for none (a precision's flow on its declared routes)."""
        return not any(self.native.values()) or any(self.native_on(target).values())

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
        """The flow's model, every kernel its selectors' choice (`attention_impl` auto)."""
        from dew.interop.hf_decoders import translate_config
        from dew.registry import models

        fields = {key: value for key, value in translate_config(self.config).items()
                  if key not in ("dtype", "attention_impl")}
        model = models.build("causal_transformer", **fields, dtype=self.dtype, attention_impl="auto")
        assert isinstance(model, CausalTransformer)
        return model

    @cached_property
    def params(self):
        """The flow's parameters, float32, drawn once on the host's CPU (a GPU's normal transform differs from
        the CPU's in its last bits, so a draw on the GPU would be other parameters than its prepare's) where
        the flow prepares, or where no tier's store holds them; a run reads its prepare's (`PARAMS`), and
        initializes no model. The initializer runs compiled, one program: run eagerly, it compiled each of
        its operations apart, 1,416 programs and 70 s over the flows' prepare on armada."""
        if TRUTHS in os.environ and not preparing():
            return pickle.loads((store_of(self) / PARAMS).read_bytes())
        with jax.default_device(jax.devices("cpu")[0]):
            tokens = jnp.zeros((self.rows, self.length), jnp.int32)
            return jax.device_get(jax.jit(self.model.init)(jax.random.key(5461), tokens)["params"])

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
        def step(params, ids):
            def loss(params):
                logits = model.apply({"params": params}, ids)
                losses = -token_log_probs(logits[:, :-1], ids[:, 1:])
                return jnp.mean(losses), losses
            (_, losses), gradients = jax.value_and_grad(loss, has_aux=True)(params)
            return losses, gradients
        return step

    def decode(self, model) -> Callable:
        """`model`'s decode loop: its first `prefill` tokens at once, then one token a step, each step's last
        logits, `[rows, 1 + length - 1 - prefill, vocab]`."""
        def decoded(params, ids):
            variables = {"params": params}
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
        def truth(params, ids):
            logits = model.apply({"params": params}, ids[:, :self.length - 1])
            return logits[:, self.prefill - 1:]
        return truth


KINDS = ("train", "decode")
"""A flow's programs: its training step's loss and gradients, and its decode loop's logits."""


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
    Flow("dense bf16, a user's", "llama-tiny", DENSE, "bfloat16",
         {"train": {"attention": "flash"}, "decode": {"attention": "flash", "decode": NATIVE}}, lane=False),
    Flow("dense fp32", "llama-tiny", DENSE, "float32", {"train": {}, "decode": {}}),
    Flow("window and softcap, the lane's", "gemma2-tiny", GEMMA2, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}),
    Flow("window and softcap, a user's", "gemma2-tiny", GEMMA2, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}, lane=False),
    Flow("sinks, the lane's", "gpt-oss-tiny", GPT_OSS, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}),
    Flow("sinks, a user's", "gpt-oss-tiny", GPT_OSS, "bfloat16",
         {"train": TEMPLATE, "decode": {**TEMPLATE, "decode": NATIVE}}, lane=False),
    Flow("latent attention, the lane's", "deepseek-v2-tiny", MLA, "bfloat16",
         {"train": TEMPLATE, "decode": {}}),
    Flow("latent attention, a user's", "deepseek-v2-tiny", MLA, "bfloat16",
         {"train": {"attention": "flash"}, "decode": {}}, lane=False),
    Flow("delta attention (KDA)", "kimi-linear-tiny", KDA, "bfloat16",
         {"train": {**TEMPLATE, "gated_delta_rule": NATIVE, "short_conv": NATIVE},
          "decode": {"gated_delta_rule": NATIVE, "short_conv": NATIVE, "delta_decode": "step"}}),
    Flow("gated delta net and its short conv", "qwen3-next-tiny", GDN, "bfloat16",
         {"train": {**TEMPLATE, "gated_delta_rule": NATIVE, "short_conv": NATIVE},
          "decode": {**TEMPLATE, "decode": NATIVE, "gated_delta_rule": NATIVE, "short_conv": NATIVE,
                     "delta_decode": "step"}}),
    Flow("state-space duality (SSD), bf16", "mamba2-tiny", SSD, "bfloat16",
         {kind: {"ssd": NATIVE, "short_conv": NATIVE} for kind in KINDS}),
    Flow("state-space duality (SSD), fp32, a user's", "mamba2-tiny", SSD, "float32",
         {kind: {"ssd": NATIVE, "short_conv": NATIVE} for kind in KINDS}, lane=False),
    Flow("selective scan", "mamba-tiny", SELECTIVE, "bfloat16",
         {kind: {"selective_scan": NATIVE, "short_conv": NATIVE} for kind in KINDS}),
    Flow("lightning attention", "minimax-tiny", LIGHTNING, "bfloat16",
         {"train": {**TEMPLATE, "gated_delta_rule": NATIVE},
          "decode": {**TEMPLATE, "decode": NATIVE, "gated_delta_rule": NATIVE}},
         families={"gated_delta_rule": ("wgmma",)}),
)
"""Every model flow, by the route it is for (docs/design/testing.md: a kernel's native run is a flow's
cell): each family that takes a distinct default route on a GPU, its bf16 attention flows twice, under the
lane's deterministic ops (the template on a GPU's own mainloop) and as a user's process runs it (a
library's fused kernel where one takes the call, FlashAttention among them, which refuses the lane's), the
dense fp32 flow on XLA's declared fp32 route, and the SSD port in fp32 too, as a user's process runs it (its
products in TF32 at the default precision, which the lane's whole fp32 products refuse).
Lightning attention's decay without the delta rule takes the recurrence template on a Hopper GPU alone: fla's
port hands it to the portable math (`recurrence_chunk.PORT_SUPPORTS`). A latent attention flow's decode loop
attends in XLA (the latent decode's own form), and a KDA flow's decode step's softmax layer too: no native
attention route of theirs there."""


def flow_named(name: str) -> Flow:
    return next(flow for flow in FLOWS if flow.name == name)


HOST_FLAGS = ("--xla_force_host_platform_device_count", "--xla_cpu_scheduler_type")
"""The lane's flags of the host's own backend (tests/lane_environment.py), which compute nothing of a GPU's
program and part a prepare's CPU process from its run's GPU one."""


def flow_flags(flow: Flow) -> str:
    """The XLA flags a flow compiles its programs under in its prepare and loads them under in its run, the
    same in both (`BuildTarget.held` reads them, in their order): the process's that decide what a program
    computes (`dew.training.ahead.computation_flags`: no phase's own, as a cache directory a process sets
    after its backend opened), but the host backend's own (`HOST_FLAGS`), then its variant's on a cuda lane,
    the lane's repeatable flags where `lane`, else a user's (`lane_environment.REPEATABLE_GPU_FLAGS`,
    `USERS_GPU_FLAGS`)."""
    from dew.training.ahead import computation_flags

    variant = lane_environment.REPEATABLE_GPU_FLAGS if flow.lane else lane_environment.USERS_GPU_FLAGS
    named = (*HOST_FLAGS, *(flag.split("=")[0] for flag in variant))
    flags = lane_environment.without_flags(" ".join(computation_flags()), *named)
    return " ".join([flags, *variant]).strip()


def users_process() -> bool:
    """Whether this process computes as a user's does (`lane_environment.USERS_ARITHMETIC`): the process a
    flow not of the lane's runs in, prepared and run (`Flow.lane`)."""
    return os.environ.get(lane_environment.USERS_ARITHMETIC) == "1"


class WrongProcess(AssertionError):
    """A flow in a process not opened under its variant's flags or matmul precision (`held_process`)."""


def held_process(flow: Flow) -> None:
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


def store_of(flow: Flow) -> Path:
    """The flow's directory in the tier's store (`TRUTHS`), refused where the tier names none."""
    if TRUTHS not in os.environ:
        raise RuntimeError(f"a model flow reads its truths and programs from {TRUTHS}, which its part's "
                           "prepare writes: none is set")
    return Path(os.environ[TRUTHS]) / hashlib.sha256(flow.name.encode()).hexdigest()[:24]


def stamp() -> str:
    """The commit (`map_judge.commit`) and JAX the flows' truths are prepared and read at: a truth another
    computed is never read (`reference_error.stored_truths`)."""
    from map_judge import commit

    return f"{commit()} jax {jax.__version__} jaxlib {importlib.metadata.version('jaxlib')}"


def _flat(flow: Flow, program: Callable):
    """`program(params, ids)` as a function of the parameters' leaves and the ids, and their shapes: a program
    compiled ahead takes and returns arrays (`compile_programs`)."""
    leaves, tree = jax.tree.flatten(flow.params)

    def flat(*arguments):
        return program(jax.tree.unflatten(tree, arguments[:-1]), arguments[-1])
    shapes = [jax.ShapeDtypeStruct(np.shape(leaf), np.asarray(leaf).dtype) for leaf in leaves]
    return flat, [*shapes, jax.ShapeDtypeStruct((flow.rows, flow.length), np.int32)]


def _target():
    """The node's GPU (`TOPOLOGY`): its generation, its kind's (`generation.gpu_target`), and the facts its
    serialized device states of itself, its multiprocessors and its block's shared memory among them
    (`generation.runtime_facts`), read with no GPU visible, as a prepare runs, so nothing of the CUDA
    driver's: what its run's load holds the programs to (`BuildTarget.held`)."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus

    # The CUDA plugin, which a serialized topology deserializes through, initialized with no GPU and no
    # client (`dew.training.ahead.host_gpus`).
    compile_only_gpus("sm80", 1)
    device = topologies.TopologyDescription.deserialize(Path(os.environ[TOPOLOGY]).read_bytes()).devices[0]
    return generation.runtime_facts(generation.gpu_target(device.device_kind), device,
                                    read=lambda ordinal, attribute: None)


def programs_of(flow: Flow) -> dict[str, tuple[Callable, list]]:
    """The flow's programs by name, each of the parameters' leaves and the ids: its training step and decode
    loop, Dew's (`dew`) and the reference path's (`reference`)."""
    model = flow.model
    return {f"{kind} {arm}": _flat(flow, getattr(flow, kind)(model))
            for kind in ("train", "decode") for arm in ("dew", "reference")}


class Unreached(AssertionError):
    """A flow whose Dew arm did not take a route it is for, or took a native kernel where it holds the route
    portable (`held_routes`)."""


def held_routes(flow: Flow, found: Mapping[str, list], target: str) -> None:
    """Raise unless each of the flow's programs, traced for the GPU generation `target` (`found`, each
    program's routes by its kind), took every route it is for there (`Flow.native_on`): a kernel off the
    reference path among its choice's routes (`NATIVE`), every one of them so (`EVERY`), or the kernel it
    names; and the portable math alone on a choice native on other GPUs only (`Flow.families`), so a kernel
    that comes to take it is a flow's to cover."""
    generic = generic_kernels()
    native = flow.native_on(target)
    for kind, choices in flow.native.items():
        taken = {}
        for route in found[kind]:
            taken.setdefault(str(route.choice), set()).add(route.kernel)
        for choice, kernel in choices.items():
            kernels = taken.get(choice, set())
            if choice not in native[kind]:
                if kernels - generic[choice]:
                    raise Unreached(f"{flow.name}: its {kind}'s {choice} took {sorted(kernels)} on {target}, "
                                    "where the flow holds it portable (`Flow.families`): name "
                                    f"{generation.family(target)} there")
                continue
            held = (kernels - generic[choice] if kernel == NATIVE else
                    kernels if kernel == EVERY and not kernels & generic[choice] else kernels & {kernel})
            if not held or (kernel not in (NATIVE, EVERY) and kernels - {kernel}):
                raise Unreached(f"{flow.name}: its {kind}'s {choice} took {sorted(kernels) or 'nothing'} on "
                                f"{target}, not {kernel}: the flow does not reach the route it is for")


def held_reference(flow: Flow, found: list) -> None:
    """Raise unless the flow's reference arm ran the reference path alone (`generic_kernels`)."""
    generic = generic_kernels()
    if native := sorted({(str(route.choice), route.kernel) for route in found
                         if route.kernel not in generic.get(str(route.choice), ())}):
        raise AssertionError(f"{flow.name}: its reference arm ran {native}, not its reference path")


def _compiled(named: Mapping[str, tuple[Callable, list]], arm: str, target):
    """The arm's programs of `named` (`programs_of`) compiled for `target` (`compile_programs`), one a call so
    each program's routes are its own, and each program's routes by its kind."""
    from dew.training.ahead import AheadPrograms, compile_programs

    compiled, found = {}, {}
    for kind in KINDS:
        name = f"{kind} {arm}"
        with generation.routes() as found[kind]:
            compiled[kind] = compile_programs({name: named[name]}, target.generation, target.multiprocessors)
    return AheadPrograms(compiled[KINDS[0]].target, {
        name: program for each in compiled.values() for name, program in each.programs.items()}), found


def prepare(flow: Flow) -> bool:
    """The flow's prepare: its programs compiled for the node's GPU into its store, Dew's arm's routes held
    to the ones it is for, the reference arm's to its reference path; then every draw's truth (`judge`).
    Whether it prepared: none where the node's GPU does not run the flow (`Flow.runs_on`)."""
    held_process(flow)
    target = _target()
    if not flow.runs_on(target.generation):
        return False
    directory = store_of(flow)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PARAMS).write_bytes(pickle.dumps(flow.params))
    named = programs_of(flow)
    dew, found = _compiled(named, "dew", target)
    dew.save(directory / "dew.programs")
    held_routes(flow, found, target.generation)
    with generation.chosen(**reference_kernels()):
        reference, found = _compiled(named, "reference", target)
    reference.save(directory / "reference.programs")
    held_reference(flow, [route for kind in KINDS for route in found[kind]])
    for kind in KINDS:
        judge(flow, {}, kind)
    return True


PREPARED_APART = """import sys
import dew  # before jaxlib loads, so XLA_FLAGS' flags are proven held (`dew._devices.flags_hold`)
import model_flows
model_flows.prepared_here(sys.argv[1])
"""
"""A flow's prepare process's program (`prepared_apart`)."""


def prepared_apart(flow: Flow) -> dict | None:
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


def loaded(flow: Flow) -> dict[str, Callable]:
    """The flow's programs its prepare compiled, loaded onto this GPU, each a function of the parameters and
    the ids."""
    from dew.training.ahead import AheadPrograms

    directory = store_of(flow)
    held = {name: program for arm in ("dew", "reference")
            for name, program in AheadPrograms.load(directory / f"{arm}.programs").loaded().items()}
    return {name: as_given(lambda params, ids, program=program: program(*jax.tree.leaves(params), ids))
            for name, program in held.items()}


def judge(flow: Flow, programs: Mapping[str, Callable], kind: str) -> None:
    """Check 1 of the flow's program `kind` (`KINDS`) against its float64 truth from the tier's store
    (prepared there where this process prepares, `programs` none)."""
    truth_model = float64_twin(flow.model)
    truth = flow.train(truth_model) if kind == "train" else flow.decode_truth(truth_model)
    with stored_truths(store_of(flow) / "truths", prepare=preparing(), stamp=stamp()):
        assert_as_exact(programs.get(f"{kind} dew"), programs.get(f"{kind} reference"), flow.draw,
                        f"{flow.name}: {kind}", seeds=flow.seeds, truth_fn=truth)


def run(flow: Flow) -> int:
    """The flow on this GPU from its prepared programs and truths: check 1 of each program, each a cell
    (`gpu_flow.run_cells`, which reports the cells that ran); its device's peak bytes after them, which the
    test reports."""
    programs = loaded(flow)
    run_cells([Cell(f"{flow.name}: {kind}", functools.partial(judge, flow, programs, kind))
               for kind in KINDS])
    return int((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0))


def measured(flow: Flow, began: float) -> dict:
    """What a prepare since `began` took (`prepared_apart`): its seconds, its process's peak resident bytes
    and the bytes it wrote to the flow's store, which size a tier part's prepare on its node."""
    import resource
    import time

    written = sum(path.stat().st_size for path in store_of(flow).rglob("*") if path.is_file())
    return {"seconds": time.perf_counter() - began, "written_bytes": written,
            "peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
