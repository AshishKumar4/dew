"""The attention selectors' data (`dew.nn.attention.resolve_implementation`): what a call is to them
(`AttentionCase`), each attention kernel's requirements (its `SUPPORTS`), the target's preference,
and the two selectors, attention's (`ATTENTION`) and a decode step's (`DECODE`), which
`dew.nn.kernels.selection.select` evaluates while a program traces."""

import dataclasses
import importlib
import importlib.util

import jax.numpy as jnp
import numpy as np

from dew.devices import deterministic_ops_requested

from .kernels import decode_attention, generation
from .kernels.generation import INHERENT, Refusal, bf16_dot_runs, capability, device_generation, gpu_placed
from .kernels.selection import Kernel, Op, Requirement
from .kernels.templates import attention_mosaic_program, attention_route
from .kernels.templates.attention_triton import DTYPES as TEMPLATE_DTYPES

CUDNN_DTYPES = (jnp.bfloat16, jnp.float16)


def cudnn_max_head_dim() -> int:
    """The widest heads cuDNN's fused bf16/fp16 kernel takes on this GPU: 256 on Hopper and later, 128
    before (jax's `check_is_flash_attention`, jax/_src/cudnn/fused_attention_stablehlo.py, which
    raises past it)."""
    named = capability(device_generation())
    return 256 if named is not None and named >= (9, 0) else 128


FLASH_MAX_HEAD_DIM = 256
# Its backward takes heads up to 192 wide on sm8x parts other than sm80: "FlashAttention backward for
# head dim > 192 requires A100/A800 or H100/H800", measured on an RTX 4080.
FLASH_MAX_BACKWARD_HEAD_DIM = 192
# Packed rows run as varlen sequences, each row's documents and its padding after them, so a row's
# share of the grid is its declared document cap whatever it holds. At Qwen3.5-0.8B's attention (8 x 3072, 8
# query heads over 2 of 256, causal) a cap of 64 ran forward and backward in 2.7 to 4.2 ms on an H100
# and 6.7 to 13.7 on an A100 against the document mask's 12.3 to 12.7 and 22.7 to 24.7; the row
# length's cap took 80 and 160 (docs/performance.md).
FLASH_PACKED_SEGMENTS = 64
# tokamax's Triton attention computes heads 8 wide or narrower wrong: bf16 and fp16 heads of 1 to 8,
# causal or not over 4 to 200 tokens, lay 0.37 to 5.8 of the output's size from float64 on an A100, where
# xla's lay 2e-3, and its own heads of 9 to 128 within twice xla's (ledger 1542).
TRITON_NARROWEST = 9


def flash_architectures() -> tuple[str, ...] | None:
    """The GPU generations the installed dew_flash_attn holds code for, as the wheel reports them
    (its build's CUDA architectures), or None where it is not installed."""
    if importlib.util.find_spec('dew_flash_attn') is None:
        return None
    return importlib.import_module('dew_flash_attn').ARCHITECTURES


def flash_built_for(device: str, architectures: tuple[str, ...]) -> bool:
    """Whether a wheel built for `architectures` holds code the generation `device` runs: a CUDA
    binary for X.Y runs on X.Z from Z = Y up, so its own, or sm80's on any 8.x part."""
    own = capability(device)
    built = [capability(architecture) for architecture in architectures]
    return own is not None and any(named is not None and named[0] == own[0] and named[1] <= own[1]
                                   for named in built)


# Splash's tile size. At 1024 the forward kernel holds 4 MiB of fp32 logits
# plus its operands. On a TPU v6e, Qwen3-0.6B's attention (8 x 1024 tokens,
# 16 query heads over 8, 128 wide, causal) took 1.28 ms forward and backward
# at 1024 with the fused backward, against 1.81 at 512 with separate dq and
# dkv kernels and 1.42 at 512 fused, with the same errors against fp32
# (docs/performance.md). `splash.block_sizes` narrows this to a divisor of
# each sequence, as the mask blocking needs.
SPLASH_BLOCK = 1024
# The kernel tiles the key axis by lanes: the compute block must be a whole
# number of them (`{bkv_compute=} must be a multiple of {NUM_LANES=}`,
# splash_attention_kernel.py:970-971) and the mask blocking must divide both
# sequences (splash_attention_mask_info.py:565-571), so a length that is not
# a multiple of this has no legal block size and never reaches splash.
SPLASH_LANES = 128
# The two input dtypes a TPU matmul takes; fp16 has no MXU path. The kernel
# accumulates in fp32 whatever comes in: the logits carry
# preferred_element_type=float32, and the running max, sum and output stay
# fp32 to the last block, which is the reference softmax.
SPLASH_DTYPES = (jnp.bfloat16, jnp.float32)
# The shortest sequence 'auto' hands splash. tools/qualify_splash.py on one
# v6e (jax 0.11.1, bf16, 16 query heads over 8 key heads of width 128, 16384
# causal tokens a step), a training step: splash 3.24 ms against XLA's 2.62
# at 256 keys, 2.57 against 4.64 at 512, 28.9 against 144.2 at 16384; width
# 64 crosses at the same length. Softcap, sinks and packed ids cost splash
# at most 16% more from 512 to 16384.
SPLASH_MIN_LENGTH = 512


@dataclasses.dataclass(frozen=True)
class AttentionCase:
    """What attention's selector reads of a call (`select_attention`): its shapes `[B, T, N, D]`
    over `[B, S, K, D]` keys, dtypes and validity, which comes as coordinates, never a mask a cache
    built: causality and a window, packed documents (a call's own or another sequence's, `aligned`
    to their last keys under causality or a window, `CrossDocuments`), key lengths and key starts
    (a windowed decode step's first key, `key_value_seq_starts`), and `spans_only`, what only the
    template's spans read (`template.spans`): `blocks`, a causal call's block or image groups, whose queries
    also see their block's later keys, or its keys' validity or the places its window counts, which the
    generic paths take as their mask; `mask` is an arbitrary mask only the generic path takes.
    `arithmetic` is a call asking what only the
    reference path computes (`reference_only`), `narrows` one jax.nn's xla kernel would compute
    below its query's precision or its process's CPU arithmetic (`_xla_kernel_narrows`,
    `_xla_kernel_widens_less`), `folds`
    one of few query positions `folded_attention` reads on a GPU, and `splash_mask` one whose mask
    splash's descriptor describes, formed only where splash may run, under a name that may choose it
    on a TPU (`TPU_BACKEND`), and False elsewhere."""

    queries: int
    keys: int
    heads: int
    kv_heads: int
    head_dim: int
    dtype: np.dtype
    key_dtype: np.dtype
    causal: bool
    window: int | None
    documents: str | None
    aligned: bool
    max_segments: int | None
    lengths: bool
    starts: bool
    blocks: bool
    spans_only: bool
    mask: bool
    bias: bool
    sinks: bool
    softcap: float | None
    arithmetic: bool
    narrows: bool
    folds: bool
    dropout: bool
    splash_mask: bool
    shards: int

    @property
    def cell(self) -> tuple[str | None, str]:
        """The call's cell among attention's declared ones (`DECLARED`): the target's instruction family and
        attention's one kind of call, its forward and its training alike, as the kernel is chosen before
        the call is differentiated (`attention_route.attend`'s `custom_vjp`)."""
        return generation.family(device_generation()), 'attention'


_Need = Requirement[AttentionCase]
"""A requirement of a `AttentionCase`, the selectors' case here."""


def _flash_widest() -> int:
    """The widest heads FlashAttention's backward takes on the target: 192 on sm8x parts other than
    sm80 (FlashAttention's own `is_sm8x`), else `FLASH_MAX_HEAD_DIM`."""
    named = capability(device_generation())
    return (FLASH_MAX_BACKWARD_HEAD_DIM if named is not None and named[0] == 8 and named[1] > 0
            else FLASH_MAX_HEAD_DIM)


def _on_gpu(case: AttentionCase) -> bool:
    return generation.backend() == 'gpu'


def _flash_installed(case: AttentionCase) -> bool:
    # A step traced for an assumed GPU on a host without the wheel takes the wheel its target
    # installs: what the host lacks shows as what it ran (`_hosted`), not as the target's refusal.
    return flash_architectures() is not None or generation.ASSUMED.get() is not None


def _flash_built(case: AttentionCase) -> bool:
    # A step traced for an assumed GPU takes the wheel its target installs, as `_flash_installed` holds.
    architectures = flash_architectures()
    return (architectures is None or generation.ASSUMED.get() is not None
            or flash_built_for(device_generation(), architectures))


_ARITHMETIC = _Need('arithmetic', lambda case: not case.arithmetic,
                          "the call asks for arithmetic only the reference path computes "
                          "(`reference_only`)", row=INHERENT)
_ON_GPU = _Need('backend', _on_gpu, "it runs on a GPU", row=INHERENT)

FLASH_SUPPORTS = (
    _ON_GPU,
    _Need('document_bound', lambda case: case.documents is None or case.max_segments is not None,
                "the call packs documents with no bound on a row's (`Coordinates.max_segments`), which its "
                "packed sequences need", row=INHERENT),
    _Need('packed_documents', lambda case: case.documents is None or case.max_segments is None
                or case.max_segments <= FLASH_PACKED_SEGMENTS,
                lambda case: f"the call packs up to {case.max_segments} documents a row, past the "
                             f"{FLASH_PACKED_SEGMENTS} its empty sequences cost less than the document mask",
                row='measured: flash packed-document cap (docs/performance.md)', kind='measured'),
    _Need('dtype', lambda case: case.dtype in CUDNN_DTYPES, lambda case: f"the query is {case.dtype}",
                row=INHERENT),
    _Need('head_dim', lambda case: case.head_dim % 8 == 0 and case.head_dim <= _flash_widest(),
                lambda case: f"the heads are {case.head_dim} wide, past the {_flash_widest()} its "
                             "backward takes here", row=INHERENT),
    _Need('square', lambda case: not (case.causal or case.window is not None)
                or case.queries == case.keys or case.documents == 'cross',
                "a causal or windowed call has another number of keys than of queries, which FlashAttention "
                "aligns to the last key", row=INHERENT),
    _Need('mask', lambda case: not case.mask, "the call has an arbitrary mask", row=INHERENT),
    _Need('spans_only', lambda case: not case.spans_only,
                "the call has block groups, key validity or counted places, which neither its causal flag "
                "nor its packed sequences express", row=INHERENT),
    _Need('key_lengths', lambda case: not case.lengths, "the call ends its rows' keys by lengths",
                row='roadmap: flash attention over key lengths'),
    _Need('key_starts', lambda case: not case.starts, "the call starts its rows' keys by starts",
                row='roadmap: flash attention over key lengths'),
    _Need('bias', lambda case: not case.bias, "the call has a bias (ALiBi's slopes among them)",
                row='roadmap: flash attention ALiBi'),
    _Need('softcap', lambda case: case.softcap is None, "the call has a logit softcap",
                row='roadmap: flash attention softcap'),
    _Need('deterministic', lambda case: not deterministic_ops_requested(),
                "--xla_gpu_deterministic_ops is set, which its backward's atomic sum of the query gradient "
                "does not keep", row=INHERENT),
    _Need('dropout', lambda case: not case.dropout,
          "the call drops attention weights, which FlashAttention's path here does not take",
          row='roadmap: attention dropout in FlashAttention'),
    _ARITHMETIC,
    _Need('not_installed', _flash_installed, "dew_flash_attn is not installed", row=INHERENT),
    _Need('binary_arch', _flash_built,
                lambda case: f"the installed dew_flash_attn holds no code for {device_generation()}",
                row='roadmap: flash wheel architectures'),
)
"""FlashAttention's requirements (`flash_attention`): a GPU, bf16 or fp16 heads a multiple of 8 up to the
width its backward takes, a causal or windowed call square or over another sequence's documents (which it
aligns as FlashAttention aligns them), no arbitrary mask, block groups, key lengths, bias or softcap, no
deterministic ops, and a wheel holding code for the target."""

CUDNN_SUPPORTS = (
    _ON_GPU,
    _Need('capability', lambda case: bf16_dot_runs(), "cuDNN takes bf16 and fp16 from sm80",
                row=INHERENT),
    _Need('dtype', lambda case: case.dtype in CUDNN_DTYPES, lambda case: f"the query is {case.dtype}",
                row=INHERENT),
    _Need('head_dim', lambda case: case.head_dim % 8 == 0 and case.head_dim <= cudnn_max_head_dim(),
                lambda case: f"the heads are {case.head_dim} wide, past the {cudnn_max_head_dim()} it "
                             "tiles here", row=INHERENT),
    _Need('softcap', lambda case: case.softcap is None, "the call has a logit softcap",
                row=INHERENT),
    _Need('sinks', lambda case: not case.sinks, "the call has attention sinks", row=INHERENT),
    _Need('bidirectional_window', lambda case: case.window is None or case.causal or case.aligned,
                "cuDNN keeps a window behind the query only", row=INHERENT),
    _Need('documents', lambda case: case.documents is None or case.aligned,
                "cuDNN takes no segment ids, a call's packed documents only as their mask, a dense bias "
                "that costs more than the xla kernel's mask (`_attention_kernel` builds it where cuDNN is "
                "named)", row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('mask', lambda case: not case.mask,
                "cuDNN takes an arbitrary mask as a dense bias, which costs more than the xla kernel's "
                "mask but in a forward-only cache call (`kernel_for_materialized_mask`)",
                row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('spans_only', lambda case: not case.spans_only,
                "cuDNN takes block groups, key validity or counted places only as the mask the call's spans "
                "mean, a dense bias, as it takes an arbitrary mask",
                row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('key_starts', lambda case: not case.starts,
                "cuDNN takes key starts only as their mask, a dense bias, where the decode kernel reads "
                "each row's span of the cache alone (`decode_attention.attend`)",
                row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('deterministic', lambda case: not deterministic_ops_requested(),
                "--xla_gpu_deterministic_ops is set, under which XLA's cuDNN attention backward crashes "
                "(openxla/xla#46500)", row=INHERENT),
    _Need('dropout', lambda case: not case.dropout,
          "the call drops attention weights, which cuDNN's fused path here does not take",
          row='roadmap: attention dropout in cuDNN'),
    _ARITHMETIC,
)
"""cuDNN's fused attention's requirements (`cudnn_attention`): a GPU of sm80 or later (cuDNN refuses
bf16 and fp16 below it: "SDPA FP16/BF16 requires SM80"), one of its two dtypes, heads it tiles, no
softcap or sinks, a window only behind the query, no documents but another sequence's aligned ones
(which it takes as their mask, `_attention_kernel`), no block groups or key starts, and no deterministic
ops. Every
fused kernel runs at the query's head width, values included (`widen_value_heads`)."""

TRITON_SUPPORTS = (
    _ON_GPU,
    _Need('capability', lambda case: bf16_dot_runs(), "tokamax's Triton kernel needs sm80",
                row=INHERENT),
    _Need('dtype', lambda case: case.dtype in CUDNN_DTYPES,
                lambda case: f"the query is {case.dtype}, where it was measured on bf16 and fp16 alone",
                row='measured: tokamax Triton attention (docs/performance.md)', kind='measured'),
    _Need('fp16', lambda case: case.dtype != np.dtype(np.float16),
                "the query is float16, whose dQ tokamax's backward narrows before the scale the route "
                "applies outside it (`triton_attention`), so a finite gradient's store can overflow; the "
                "Mosaic template takes the call once it replaces this route",
                row='ledger 1676'),
    _Need('narrow', lambda case: case.head_dim <= 64,
                lambda case: f"the heads are {case.head_dim} wide, where it measured no faster than "
                             "cuDNN",
                row='measured: tokamax Triton attention (docs/performance.md)', kind='measured'),
    _Need('head_dim', lambda case: case.head_dim >= TRITON_NARROWEST,
                lambda case: f"the heads are {case.head_dim} wide, narrower than the {TRITON_NARROWEST} it "
                             "computes right",
                row='ledger 1542'),
    *(_Need(code, test, detail, row=INHERENT) for code, test, detail in (
        ('window', lambda case: case.window is None, "the call has a sliding window"),
        ('mask', lambda case: not case.mask, "the call has an arbitrary mask"),
        ('spans_only', lambda case: not case.spans_only,
         "the call has block groups, key validity or counted places"),
        ('bias', lambda case: not case.bias, "the call has a bias"),
        ('sinks', lambda case: not case.sinks, "the call has attention sinks"),
        ('softcap', lambda case: case.softcap is None, "the call has a logit softcap"),
        ('documents', lambda case: case.documents is None, "the call packs documents"),
        ('key_lengths', lambda case: not case.lengths, "the call ends its keys by lengths"),
        ('key_starts', lambda case: not case.starts, "the call starts its keys by starts"),
    )),
    _Need('dropout', lambda case: not case.dropout, "tokamax's Triton kernel takes no probability dropout",
          row=INHERENT),
    _ARITHMETIC,
)
"""tokamax's Triton flash kernel's requirements (`triton_attention`): a GPU of sm80 or later, bf16 (its
fp16 dQ store a declared gap, ledger 1676), heads at least `TRITON_NARROWEST` wide and at most 64 wide
where it was measured faster (an RTX 4080's SimpleDiT-B attention 5.62 to 4.33 ms, its causal 3-layer
decoder's 1.82 to 1.24; Qwen3-0.6B's 128-wide 9.13 to 9.24), and no window, mask, block groups, bias,
sinks, softcap, documents or key lengths or starts."""

TPU_BACKEND = _Need('backend', lambda case: generation.backend() == 'tpu', "it runs on a TPU", row=INHERENT,
                    kind='measured')
"""Splash's first requirement, a TPU, which the case builder also reads before it builds splash's mask
descriptor (`AttentionCase.splash_mask`), the one fact that costs host-side work."""

TPU_SUPPORTS = (
    # Elsewhere it runs interpreted, where 'tpu' is named.
    TPU_BACKEND,
    _Need('bias', lambda case: not case.bias, "splash takes no bias", row=INHERENT, kind='measured'),
    _Need('spans_only', lambda case: not case.spans_only,
          "splash's descriptor describes no mask of the call's own data, which block groups, key validity "
          "and counted places are; a named tpu call takes them as their mask on its pallas flash kernel "
          "(`_attention_kernel`)", row='measured: the tpu path takes a call mask on its pallas flash kernel',
          kind='measured'),
    _Need('key_lengths', lambda case: not (case.lengths or case.starts),
                "splash takes key lengths and starts only as a mask it cannot describe", row=INHERENT,
                kind='measured'),
    # Named, the tpu path takes the documents' mask on its pallas flash kernel (`_attention_kernel`).
    _Need('aligned', lambda case: not case.aligned,
                "splash aligns causality to the first key, not a cache call's last",
                row='roadmap: splash over cache documents', kind='measured'),
    *(_Need(code, test, detail, row=INHERENT, kind='measured') for code, test, detail in (
        ('dtype', lambda case: case.dtype in SPLASH_DTYPES, "splash takes bf16 and fp32"),
        ('shards', lambda case: case.shards == 1,
         "an automatic sequence axis gathers the keys, and a pallas call would gather the queries back"),
        ('tiles', lambda case: (case.queries % SPLASH_LANES == 0 and case.keys % SPLASH_LANES == 0
                                and min(case.queries, case.keys) >= SPLASH_MIN_LENGTH
                                and case.splash_mask),
         "splash's blocking does not tile the lengths, or its descriptor cannot describe the mask"),
    )),
    _Need('dropout', lambda case: not case.dropout, "splash takes no probability dropout", row=INHERENT),
    _ARITHMETIC,
)
"""Splash's requirements (`tpu_attention`): a TPU, no bias, block groups or key lengths, one of its two
dtypes, lengths its blocking tiles and at least `SPLASH_MIN_LENGTH`, a mask its descriptor describes, no
documents aligned to their last keys (splash's causality aligns to the first), and no automatic
sequence axis. A call the measured ones turn down still runs when 'tpu' is named, on the pallas flash
kernel with the call's mask, interpreted off a TPU."""

XLA_SUPPORTS = (
    _Need('sinks', lambda case: not case.sinks, "jax.nn's attention has no sinks", row=INHERENT,
                kind='handoff'),
    _Need('softcap', lambda case: case.softcap is None, "jax.nn's attention has no softcap",
                row=INHERENT, kind='handoff'),
    _Need('narrows', lambda case: not case.narrows,
                "jax.nn's xla kernel would compute the call below its query's precision or its process's "
                "CPU arithmetic",
                row=INHERENT, kind='handoff'),
    _Need('arithmetic', lambda case: not case.arithmetic,
                "the call asks for arithmetic only the reference path computes (`reference_only`)",
                row=INHERENT, kind='handoff'),
    _Need('dropout', lambda case: not case.dropout, "jax.nn's attention takes no probability dropout",
          row=INHERENT, kind='handoff'),
)
"""jax.nn's xla attention's requirements: the generic kernel, which hands what it cannot honour to
the reference path."""


TEMPLATE_SUPPORTS = (
    _Need('dtype', lambda case: case.dtype in TEMPLATE_DTYPES and case.key_dtype == case.dtype,
          lambda case: f"the query is {case.dtype} and the keys are {case.key_dtype}, where the template "
                       "takes one dtype of fp32, bf16 and fp16 on its operands", row=INHERENT),
    _Need('mask', lambda case: not case.mask,
          "the call has an arbitrary mask, which the template reads as spans and never as a mask",
          row=INHERENT),
    _Need('bias', lambda case: not case.bias,
          "the call has a bias, a dense term beside the spans that carry the template's validity",
          row=INHERENT),
    _Need('shards', lambda case: case.shards == 1,
          "an automatic sequence axis gathers the keys, and a pallas call would gather the queries back",
          row=INHERENT),
    _Need('dropout', lambda case: not case.dropout or attention_route.drops(),
          attention_route.DROPS_NOT, row='1521'),
    _ARITHMETIC,
)
"""The attention template's requirements (`attention_route.attend`, the 'template' kernel): one dtype of fp32,
bf16 and fp16 on the queries and keys alike, no arbitrary mask and no bias, no sharded sequence axis, and the
call's default arithmetic. It takes every other call form: causality, windows, documents, key lengths and
starts, block groups, sinks, a softcap, and probability dropout off a GPU (`attention_route.drops`). The
call runs it by default on a family whose Mosaic template has a routed mainloop, where its declared old
route does not take the call (`_attention_preferred`); elsewhere where a call names it
(`implementation='template'`, or `generation.chosen(attention='template')`)."""


DECLARED: dict[tuple[str | None, str], tuple[str, str]] = {
    ('wgmma', 'attention'): (
        'flash', "FlashAttention-3 measured faster than the Mosaic template on an H100 (llama-d128 bf16, "
                 "2 x 4096, 32 query heads over 8 key heads, CUPTI on an exclusive node): forward 0.588 "
                 "against 0.761 ms causal and 1.094 against 1.296 full, training forward and backward 2.53 "
                 "against 11.45 causal and 4.49 against 16.48 full (receipts DERIVED-benchx-h100, "
                 "BWDSTAGES-h100)"),
}
"""Attention's declared cells (bible §4, owner 10-25; `Op.declared`), by (instruction family, kind of call)
(`AttentionCase.cell`): where an old route measured faster than the family's Mosaic template, the old route
and its receipt. Attention has one kind of call: its kernel is chosen before the call is differentiated
(`attention_route.attend`'s `custom_vjp`), so its forward and its training run one kernel, and the receipts
cover both. Every other call of a family whose Mosaic template has a routed mainloop
(`attention_mosaic_program.bundle`) takes the template; a family with none (mma.sync's today) takes the
op's order by legality. Nothing is ranked at runtime: the list shrinks as the template wins."""


def _attention_preferred(case: AttentionCase) -> str:
    """The kernel `case` runs where its cell declares no old route that takes it (`DECLARED`): splash on a
    TPU and xla on a CPU; on a GPU xla for a query no fused kernel takes the dtype of (the Mosaic template
    takes 16-bit operands alone, so no family routes it another); the template on a family whose Mosaic
    template has a routed mainloop, which hands a call it does not take to the next kernel by legality; and
    elsewhere the op's order by legality, FlashAttention first (measured faster than cuDNN and tokamax's
    Triton kernel on an A100 and an A10), a call with block groups, key validity or counted places, which no
    old kernel's spans express, taking the generic path over the mask its spans mean (ledger 1537)."""
    backend = generation.backend()
    if backend != 'gpu':
        return 'tpu' if backend == 'tpu' else 'xla'
    if case.dtype not in CUDNN_DTYPES:
        # No fused GPU kernel takes another dtype, so no measurement prefers one.
        return 'xla'
    if attention_mosaic_program.bundle(device_generation(), case.dtype) is not None:
        return 'template'
    return ATTENTION.kernels[0].name


def _hosted(kernel: str) -> str:
    """What this host runs for the target's `kernel`: cuDNN's, FlashAttention's or tokamax's Triton
    kernel where this process places the call on its own GPU, else jax.nn's xla attention, as a step
    traced for an assumed GPU on a CPU lowers (`gpu_placed`); Dew's own Pallas kernels lower for the
    target on any host."""
    return 'xla' if kernel in ('cudnn', 'flash', 'triton') and not gpu_placed() else kernel


ATTENTION = Op[AttentionCase]('attention', (
    Kernel[AttentionCase]('flash', FLASH_SUPPORTS),
    Kernel[AttentionCase]('triton', TRITON_SUPPORTS),
    Kernel[AttentionCase]('cudnn', CUDNN_SUPPORTS),
    Kernel[AttentionCase]('tpu', TPU_SUPPORTS),
    Kernel[AttentionCase]('xla', XLA_SUPPORTS, generic=True),
    Kernel[AttentionCase]('reference', (), generic=True),
    Kernel[AttentionCase]('template', TEMPLATE_SUPPORTS),
), _attention_preferred, _hosted, DECLARED)
"""Attention's selector: a declared cell's old route where it takes the call (`DECLARED`), else the call's
kernel (`_attention_preferred`: the template on a family with a routed Mosaic mainloop), then flash, triton,
cuDNN, splash, xla and the reference path, the first that takes the call. The template comes last, after
the reference path, which takes every call, so it runs where the preference names it (a templated family's
calls) or a call names it."""



def _decode_preferred(case: AttentionCase) -> str:
    """The kernel a decode call runs: on a GPU the decode kernel, the template's decode mainloop (measured
    1.57 times as fast as cuDNN in geometric mean on an A100 over Qwen3-0.6B's 28 decode shapes; no cell
    declares cuDNN, `generation.TARGETS`' note), which hands a call it does not take to the next kernel by
    legality; xla elsewhere."""
    return 'decode' if generation.backend() == 'gpu' else 'xla'


DECODE = Op[AttentionCase]('decode', (
    Kernel[AttentionCase]('decode', decode_attention.SUPPORTS),
    Kernel[AttentionCase]('cudnn', CUDNN_SUPPORTS),
    Kernel[AttentionCase]('xla', XLA_SUPPORTS, generic=True),
    Kernel[AttentionCase]('reference', (), generic=True),
), _decode_preferred, _hosted)
"""The selector of a call of few query positions over key lengths (`decodes`): the decode kernel
(`decode_attention.attend`), cuDNN and the folded xla attention (`folded_attention`)."""


def decodes(case: AttentionCase) -> bool:
    """Whether a call is the decode selector's (`DECODE`): a few query positions over key lengths, with
    no mask, key validity, bias, causality, window, documents or dropout. Plain, they are the keys' own
    heads, which the folded attention reads (`decode_geometry.folds`) and differentiates; with sinks or a
    softcap, which the decode kernel applies itself, any query heads that group over the key heads on a GPU,
    which the reference path differentiates."""
    if case.dropout or not (case.lengths and case.queries <= decode_attention.QUERIES
            and not (case.mask or case.spans_only or case.bias or case.causal) and case.window is None
            and case.documents is None):
        return False
    if case.sinks or case.softcap is not None:
        return generation.backend() == 'gpu' and case.heads % case.kv_heads == 0
    return case.folds and case.heads == case.kv_heads


def kernel_refusals(op: Op[AttentionCase], kernel: str, case: AttentionCase) -> list[Refusal]:
    """Every requirement of `op`'s `kernel` that `case` fails, as refusals, in order."""
    return [requirement.refusal(case) for requirement in op.kernel(kernel).failed(case)]
