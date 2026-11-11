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
import subprocess
import sys
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path

import lane_environment
import jax
import jax.numpy as jnp
import kernel_matrix
import numpy as np
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
    wrapper), computing in `dtype`; `native` each choice its training step or decode runs on a GPU and the
    kernel it must take there (a name, or `NATIVE`: any kernel off the reference path); under the lane's
    deterministic ops and matmul precision where `lane`, as a user's process runs otherwise, without them
    and at the default precision (`precision`); `rows` sequences of `length` tokens, decoded from `prefill`
    of them a token a step."""

    name: str
    fixture: str
    widths: Mapping[str, object]
    dtype: str
    native: Mapping[str, str]
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

    def native_on(self, target: str) -> dict[str, str]:
        """The choices the flow is for that take a native kernel on the GPU generation `target` (`families`),
        each with the kernel it must take there."""
        family = generation.family(target)
        return {choice: kernel for choice, kernel in self.native.items()
                if family in self.families.get(choice, (family,))}

    def runs_on(self, target: str) -> bool:
        """Whether the flow runs on the GPU generation `target`: where a choice it is for is native there, or
        where it is for none (a precision's flow on its declared routes)."""
        return not self.native or bool(self.native_on(target))

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
        """The flow's parameters, float32, drawn once on the host's CPU: a GPU's normal transform differs from
        the CPU's in its last bits, so a draw on the GPU would be other parameters than its prepare's."""
        with jax.default_device(jax.devices("cpu")[0]):
            tokens = jnp.zeros((self.rows, self.length), jnp.int32)
            return jax.device_get(self.model.init(jax.random.key(5461), tokens)["params"])

    @property
    def vocab(self) -> int:
        config = self.config.get("text_config", self.config)
        return int(config["vocab_size"])

    def draw(self, seed: int):
        """A draw's inputs: the flow's parameters and `rows` sequences of token ids, NumPy's from the seed."""
        ids = np.random.default_rng([5461, seed]).integers(0, self.vocab, (self.rows, self.length), np.int32)
        return self.params, ids

    def train(self, model) -> Callable:
        """`model`'s training step: its next-token loss and every parameter's gradient."""
        def step(params, ids):
            def loss(params):
                logits = model.apply({"params": params}, ids)
                return -jnp.mean(token_log_probs(logits[:, :-1], ids[:, 1:]))
            return jax.value_and_grad(loss)(params)
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


NATIVE = "native"
"""A flow's route taken by any kernel off the reference path (`generic_kernels`)."""

DENSE = {"hidden_size": 512, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 128,
         "intermediate_size": 1024, "num_hidden_layers": 2, "vocab_size": 512,
         "max_position_embeddings": 1024, "attention_bias": False}
"""A decoder's widths at which the GPU's attention kernels take its calls: heads of 128, as many key heads as
query heads, so its decode folds onto the decode kernel (`attention_kernels.decodes`)."""

GEMMA2 = {**DENSE, "query_pre_attn_scalar": 128, "sliding_window": 64}
"""Gemma 2's alternating window of 64 keys and its softcaps, at `DENSE`'s widths."""

GPT_OSS = {**DENSE, "sliding_window": 64, "num_local_experts": 2, "num_experts_per_tok": 2}
"""GPT-OSS's sinks and alternating window at `DENSE`'s widths, every token routed to both its experts: no
routing decision two arms could take apart."""

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
    Flow("dense bf16, the lane's", "llama-tiny", DENSE, "bfloat16", {"attention": NATIVE, "decode": NATIVE}),
    Flow("dense bf16, a user's", "llama-tiny", DENSE, "bfloat16", {"attention": "flash", "decode": NATIVE},
         lane=False),
    Flow("dense fp32", "llama-tiny", DENSE, "float32", {}),
    Flow("window and softcap, the lane's", "gemma2-tiny", GEMMA2, "bfloat16", {"attention": NATIVE}),
    Flow("window and softcap, a user's", "gemma2-tiny", GEMMA2, "bfloat16", {"attention": NATIVE},
         lane=False),
    Flow("sinks, the lane's", "gpt-oss-tiny", GPT_OSS, "bfloat16", {"attention": NATIVE}),
    Flow("sinks, a user's", "gpt-oss-tiny", GPT_OSS, "bfloat16", {"attention": NATIVE}, lane=False),
    Flow("latent attention, the lane's", "deepseek-v2-tiny", MLA, "bfloat16", {"attention": NATIVE}),
    Flow("latent attention, a user's", "deepseek-v2-tiny", MLA, "bfloat16", {"attention": NATIVE},
         lane=False),
    Flow("delta attention (KDA)", "kimi-linear-tiny", KDA, "bfloat16", {"gated_delta_rule": NATIVE}),
    Flow("gated delta net and its short conv", "qwen3-next-tiny", GDN, "bfloat16",
         {"gated_delta_rule": NATIVE, "short_conv": NATIVE, "delta_decode": "step"}),
    Flow("state-space duality (SSD), bf16", "mamba2-tiny", SSD, "bfloat16", {"ssd": NATIVE}),
    Flow("state-space duality (SSD), fp32, a user's", "mamba2-tiny", SSD, "float32", {"ssd": NATIVE},
         lane=False),
    Flow("selective scan", "mamba-tiny", SELECTIVE, "bfloat16", {"selective_scan": NATIVE}),
    Flow("lightning attention", "minimax-tiny", LIGHTNING, "bfloat16", {"gated_delta_rule": NATIVE},
         families={"gated_delta_rule": ("wgmma",)}),
)
"""Every model flow, by the route it is for (docs/design/testing.md: a kernel's native run is a flow's
cell): each family that takes a distinct default route on a GPU, its bf16 attention flows twice, under the
lane's deterministic ops (the template) and as a user's process runs it (FlashAttention, which refuses
them), the dense fp32 flow on XLA's declared fp32 route, and the SSD port in fp32 too, as a user's process
runs it (its products in TF32 at the default precision, which the lane's whole fp32 products refuse).
Lightning attention's decay without the delta rule takes the recurrence template on a Hopper GPU alone: fla's
port hands it to the portable math (`recurrence_chunk.PORT_SUPPORTS`)."""


def flow_named(name: str) -> Flow:
    return next(flow for flow in FLOWS if flow.name == name)


HOST_FLAGS = ("--xla_force_host_platform_device_count", "--xla_cpu_scheduler_type")
"""The lane's flags of the host's own backend (tests/lane_environment.py), which compute nothing of a GPU's
program and part a prepare's CPU process from its run's GPU one."""


def flow_flags(flow: Flow) -> str:
    """The XLA flags a flow compiles its programs under in its prepare and loads them under in its run, the
    same in both (`BuildTarget.held` reads them, in their order): the process's, but the host backend's own
    (`HOST_FLAGS`), then its variant's on a cuda lane, the lane's repeatable flags where `lane`, else a
    user's (`lane_environment.REPEATABLE_GPU_FLAGS`, `USERS_GPU_FLAGS`)."""
    variant = lane_environment.REPEATABLE_GPU_FLAGS if flow.lane else lane_environment.USERS_GPU_FLAGS
    named = (*HOST_FLAGS, *(flag.split("=")[0] for flag in variant))
    flags = lane_environment.without_flags(os.environ.get("XLA_FLAGS", ""), *named)
    return " ".join([flags, *variant]).strip()


def users_process() -> bool:
    """Whether this process computes as a user's does (`lane_environment.USERS_ARITHMETIC`): the process a
    flow not of the lane's runs in, prepared and run (`Flow.lane`)."""
    return os.environ.get(lane_environment.USERS_ARITHMETIC) == "1"


class WrongProcess(AssertionError):
    """A flow in a process not opened under its variant's flags or matmul precision (`held_process`)."""


def held_process(flow: Flow) -> None:
    """Raise unless this process is of the flow's variant (row 5461): XLA parses its flags once a process and
    JAX reads its precision as it starts, so a flow holds only where its process opened its backend under
    the flow's flags, as XLA_FLAGS carried them while it was closed (`dew._devices.flags_set`), and multiplies
    at the flow's precision (`Flow.precision`)."""
    from dew._devices import flags_set

    flags = flow_flags(flow)
    if not flags_set(flags) or jax.config.jax_default_matmul_precision != flow.precision:
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
    """The commit and JAX the flows' truths are prepared and read at: a truth another computed is never
    read (`reference_error.stored_truths`)."""
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return f"{commit} jax {jax.__version__} jaxlib {importlib.metadata.version('jaxlib')}"


def _flat(flow: Flow, program: Callable):
    """`program(params, ids)` as a function of the parameters' leaves and the ids, and their shapes: a program
    compiled ahead takes and returns arrays (`compile_programs`)."""
    leaves, tree = jax.tree.flatten(flow.params)

    def flat(*arguments):
        return program(jax.tree.unflatten(tree, arguments[:-1]), arguments[-1])
    shapes = [jax.ShapeDtypeStruct(np.shape(leaf), np.asarray(leaf).dtype) for leaf in leaves]
    return flat, [*shapes, jax.ShapeDtypeStruct((flow.rows, flow.length), np.int32)]


def _target():
    """The node's GPU (`TOPOLOGY`): its generation and multiprocessors, as its serialized topology names
    them."""
    from jax.experimental import topologies

    from dew.training.ahead import compile_only_gpus

    # The CUDA plugin, which a serialized topology deserializes through, initialized with no GPU and no
    # client (`dew.training.ahead.host_gpus`).
    compile_only_gpus("sm80", 1)
    device = topologies.TopologyDescription.deserialize(Path(os.environ[TOPOLOGY]).read_bytes()).devices[0]
    return generation.device_target(device)


def programs_of(flow: Flow) -> dict[str, tuple[Callable, list]]:
    """The flow's programs by name, each of the parameters' leaves and the ids: its training step and decode
    loop, Dew's (`dew`) and the reference path's (`reference`)."""
    model = flow.model
    return {f"{kind} {arm}": _flat(flow, getattr(flow, kind)(model))
            for kind in ("train", "decode") for arm in ("dew", "reference")}


class Unreached(AssertionError):
    """A flow whose Dew arm did not take a route it is for, or took a native kernel where it holds the route
    portable (`held_routes`)."""


def held_routes(flow: Flow, found: list, target: str) -> None:
    """Raise unless the flow's Dew arm, traced for the GPU generation `target`, took every route it is for
    there (`Flow.native_on`): a kernel off the reference path, or the one it names, among the kernels its
    choice ran (a decode's prefill over its cache's keys may run another); and the portable math alone on a
    choice native on other GPUs only (`Flow.families`), so a kernel that comes to take it is a flow's to
    cover."""
    taken = {}
    for route in found:
        taken.setdefault(str(route.choice), set()).add(route.kernel)
    native = flow.native_on(target)
    for choice, kernel in flow.native.items():
        kernels = taken.get(choice, set())
        if choice not in native:
            if kernels - generic_kernels()[choice]:
                raise Unreached(f"{flow.name}: its {choice} took {sorted(kernels)} on {target}, where the "
                                "flow holds it portable (`Flow.families`): name "
                                f"{generation.family(target)} there")
            continue
        held = kernels - generic_kernels()[choice] if kernel == NATIVE else kernels & {kernel}
        if not held:
            raise Unreached(f"{flow.name}: its {choice} took {sorted(kernels) or 'nothing'} on {target}, not "
                            f"{kernel}: the flow does not reach the route it is for")


def held_reference(flow: Flow, found: list) -> None:
    """Raise unless the flow's reference arm ran the reference path alone (`generic_kernels`)."""
    generic = generic_kernels()
    if native := sorted({(str(route.choice), route.kernel) for route in found
                         if route.kernel not in generic.get(str(route.choice), ())}):
        raise AssertionError(f"{flow.name}: its reference arm ran {native}, not its reference path")


def prepare(flow: Flow) -> bool:
    """The flow's prepare: its programs compiled for the node's GPU into its store, Dew's arm's routes held
    to the ones it is for, the reference arm's to its reference path; then every draw's truth (`judge`).
    Whether it prepared: none where the node's GPU does not run the flow (`Flow.runs_on`)."""
    from dew.training.ahead import compile_programs

    held_process(flow)
    target = _target()
    if not flow.runs_on(target.generation):
        return False
    directory = store_of(flow)
    directory.mkdir(parents=True, exist_ok=True)
    named = programs_of(flow)
    with generation.routes() as found:
        compile_programs({name: program for name, program in named.items() if name.endswith("dew")},
                         target.generation, target.multiprocessors).save(directory / "dew.programs")
    held_routes(flow, found, target.generation)
    with generation.routes() as found, generation.chosen(**reference_kernels()):
        compile_programs({name: program for name, program in named.items() if name.endswith("reference")},
                         target.generation, target.multiprocessors).save(directory / "reference.programs")
    held_reference(flow, found)
    judge(flow, {})
    return True


def prepared_apart(flow: Flow) -> dict | None:
    """The flow's prepare (`prepare`) in a process of its own, started under its XLA flags (`flow_flags`) and
    this process's arithmetic, its variant's (`users_process`): XLA parses its flags once a process, as it
    starts, and the programs it compiles state them (`BuildTarget`). What it took where it prepared
    (`measured`), None where the node's GPU does not run the flow."""
    from process_support import fresh

    return fresh(Path(__file__), flow.name, flags=flow_flags(flow))


def loaded(flow: Flow) -> dict[str, Callable]:
    """The flow's programs its prepare compiled, loaded onto this GPU, each a function of the parameters and
    the ids."""
    from dew.training.ahead import AheadPrograms

    directory = store_of(flow)
    held = {name: program for arm in ("dew", "reference")
            for name, program in AheadPrograms.load(directory / f"{arm}.programs").loaded().items()}
    return {name: as_given(lambda params, ids, program=program: program(*jax.tree.leaves(params), ids))
            for name, program in held.items()}


def judge(flow: Flow, programs: Mapping[str, Callable]) -> None:
    """Check 1 of the flow's training step and decode loop, each against its float64 truth from the tier's
    store (prepared there where this process prepares, `programs` none)."""
    truth_model = float64_twin(flow.model)
    with stored_truths(store_of(flow) / "truths", prepare=preparing(), stamp=stamp()):
        for kind, truth in (("train", flow.train(truth_model)), ("decode", flow.decode_truth(truth_model))):
            assert_as_exact(programs.get(f"{kind} dew"), programs.get(f"{kind} reference"), flow.draw,
                            f"{flow.name}: {kind}", seeds=flow.seeds, truth_fn=truth)


def run(flow: Flow) -> int:
    """The flow on this GPU from its prepared programs and truths: check 1 of each program; its device's
    peak bytes after them, which the test reports."""
    judge(flow, loaded(flow))
    return int((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0))


def measured(flow: Flow, began: float) -> dict:
    """What a prepare since `began` took (`prepared_apart`): its seconds, its process's peak resident bytes
    and the bytes it wrote to the flow's store, which size a tier part's prepare on its node."""
    import resource
    import time

    written = sum(path.stat().st_size for path in store_of(flow).rglob("*") if path.is_file())
    return {"seconds": time.perf_counter() - began, "written_bytes": written,
            "peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


if __name__ == "__main__":
    import time

    began = time.perf_counter()
    flow = flow_named(sys.argv[1])
    print(json.dumps(measured(flow, began) if prepare(flow) else None))
