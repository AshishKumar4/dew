"""The attention selectors' data (`dew.nn.attention.resolve_implementation`): what a call is to them
(`AttentionCase`), each attention kernel's requirements (its `SUPPORTS`), the target's preference,
and the two selectors, attention's (`ATTENTION`) and a decode step's (`DECODE`), which
`dew.nn.kernels.selection.select` evaluates while a program traces."""

import dataclasses
import functools
import importlib
import importlib.util
import math

import jax.numpy as jnp
import numpy as np

from dew._devices import deterministic_ops_requested

from .kernels import decode_attention, generation
from .kernels.generation import INHERENT, Choice, bf16_dot_runs, capability, device_generation, gpu_placed
from .kernels.selection import Kernel, Op, Requirement
from .kernels.templates import attention_quantized, attention_route
from .kernels.templates.attention_triton import DTYPES as TEMPLATE_DTYPES
from .precision import wide_cpu_arithmetic

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


def flash_lowered() -> bool:
    """Whether FlashAttention's call lowers here for the call's target: the installed wheel holds code for
    it, on this process's own GPU (`gpu_placed`) or, within `assumed`, for the assumed GPU's compile-only
    devices, attached or not: the wheel's FFI targets lower and compile for them on any host, the target
    stated to it (`flash_target`: FlashAttention-3 or -2, and the forward's split of its keys, by the
    target's generation and multiprocessors, not the host's)."""
    architectures = flash_architectures()
    if architectures is None:
        return False
    found = generation.assumption()
    return gpu_placed() or (found is not None and generation.backend() == 'gpu'
                            and flash_built_for(found.generation, architectures))


def flash_target() -> tuple[str, int] | None:
    """The GPU a FlashAttention call lowers for, as the wheel takes it (`dew_flash_attn.targeting`): an
    assumed one's generation and multiprocessors, None for this process's own."""
    if generation.assumption() is None:
        return None
    target = generation.gpu_target()
    return None if target.multiprocessors is None else (target.generation, target.multiprocessors)


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
    below its query's precision (`_xla_kernel_narrows`), `folds`
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
    bias_hook: bool
    """Whether the call's bias is one the template reads as a score hook in fp32 (`attention.
    template_attention`): 4-D, one term a key `[B or 1, H or 1, 1, K]` (ALiBi's) or dense `[B or 1, H or 1,
    Q, K]`, its rows and heads the call's or one for them all (a key head's `[B, Hk, Q, K]` broadcasts to
    none of them, row 2578), of at most fp32's width."""
    dense_bias: bool
    """Whether the call's bias varies by query (`[B or 1, H or 1, Q, K]`), which the template reads a tile at
    a time (`template.Biased`)."""
    sinks: bool
    softcap: float | None
    arithmetic: bool
    narrows: bool
    folds: bool
    dropout: bool
    splash_mask: bool
    shards: int
    form: str = "fast"
    """The form the template computes in where it runs the call (`attention_route.FORMS`), as the
    run's numerics name it (`generation.numerics().forms`): the old routes compute their own arithmetic
    and take only 'fast', so an accurate call runs the template, or the reference path where the
    template does not take it."""
    rows: int = 1
    """The call's rows, every leading axis of the query's `[B, T, N, D]` (a batch, its chunks)."""
    drop_placed: bool = False
    """Whether its dropout draws at a sequence slice's global queries or keys (`template.DropPlaces`), which
    the Mosaic dropping walk does not read (`attention_route.drops`)."""

    @property
    def cell(self) -> tuple[str | None, str]:
        """The call's cell among its selector's declared ones: attention's (`DECLARED`) the target's
        instruction family and the kind of call, a 16-bit call's or an fp32 call's ('float32 attention'), each
        its forward and its training alike, as the kernel is chosen before the call is differentiated
        (`attention_route.attend`'s `custom_vjp`); a decode step's (`DECODE_DECLARED`, `decodes`) the
        device's generation and 'decode' alike, as its kernels' speeds part by generation within a family.
        An fp32 call longer than its receipts timed (`FP32_MEASURED`) is a cell of its own, which none
        declares: a receipt is evidence for the shapes it measured alone, and past them the call takes the
        template (row 5283)."""
        kind = 'attention' if self.dtype in CUDNN_DTYPES else f"{jnp.dtype(self.dtype).name} attention"
        if decodes(self):
            return device_generation(), kind.replace('attention', 'decode')
        if self.dtype not in CUDNN_DTYPES and max(self.queries, self.keys) > FP32_MEASURED:
            kind += f" past {FP32_MEASURED} tokens"
        return generation.family(device_generation()), kind


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
    # installs: what the host lacks refuses the program it is traced into (`_unlowered`), not the call.
    return flash_architectures() is not None or generation.assumption() is not None


def _flash_built(case: AttentionCase) -> bool:
    # A step traced for an assumed GPU takes the wheel its target installs, as `_flash_installed` holds.
    architectures = flash_architectures()
    return (architectures is None or generation.assumption() is not None
            or flash_built_for(device_generation(), architectures))


_ARITHMETIC = _Need('arithmetic', lambda case: not case.arithmetic,
                          "the call asks for arithmetic only the reference path computes "
                          "(`reference_only`)", row=INHERENT)
_ON_GPU = _Need('backend', _on_gpu, "it runs on a GPU", row=INHERENT)
_FAST = _Need('form', lambda case: case.form == "fast", "no accurate form", row=INHERENT)
_FAST_HANDOFF = _Need('form', lambda case: case.form == "fast", "no accurate form", row=INHERENT,
                      kind='handoff')

FLASH_SUPPORTS = (
    _FAST,
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
    # Its backward's D is rowsum(dO·O) of the rounded output, so a row whose sink holds most of its
    # softmax, dP - D the sink's share of dP, takes O's rounding at 2^-8 over that share (512 times the
    # reference's error at packed rows with a window, gate-g02-judged-h100, row 2410); the template's D
    # cancels.
    _Need('sinks', lambda case: not case.sinks,
          "the call has attention sinks, whose rows FlashAttention's backward misreads where a sink holds "
          "most of the softmax", row=INHERENT),
    _ARITHMETIC,
    _Need('not_installed', _flash_installed, "dew_flash_attn is not installed", row=INHERENT),
    _Need('binary_arch', _flash_built,
                lambda case: f"the installed dew_flash_attn holds no code for {device_generation()}",
                row='roadmap: flash wheel architectures'),
)
"""FlashAttention's requirements (`flash_attention`): a GPU, bf16 or fp16 heads a multiple of 8 up to the
width its backward takes, a causal or windowed call square or over another sequence's documents (which it
aligns as FlashAttention aligns them), no arbitrary mask, block groups, key lengths, bias, softcap or sinks,
no deterministic ops, and a wheel holding code for the target."""

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
                "mask; a window's band and a forward-only cache call over several queries, which it is "
                "measured faster at, reach it as that bias (`attention.mask_kernel`'s `as_bias`)",
                row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('spans_only', lambda case: not case.spans_only,
                "cuDNN takes block groups, key validity or counted places only as the mask the call's spans "
                "mean, a dense bias, as it takes an arbitrary mask",
                row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('key_starts', lambda case: not case.starts,
                "cuDNN takes key starts only as their mask, a dense bias, where the decode kernel reads "
                "each row's span of the cache alone (`decode_attention.attend`)",
                row='measured: cuDNN dense-mask bias (docs/performance.md)', kind='measured'),
    _Need('dropout', lambda case: not case.dropout,
          "the call drops attention weights, which cuDNN's fused path here does not take",
          row='roadmap: attention dropout in cuDNN'),
    _ARITHMETIC,
)
"""cuDNN's fused attention's requirements (`cudnn_attention`): a GPU of sm80 or later (cuDNN
refuses bf16 and fp16 below it: "SDPA FP16/BF16 requires SM80"), one of its two dtypes, heads it tiles, no
softcap or sinks, a window only behind the query, no documents but another sequence's aligned ones (which
it takes as their mask, `_attention_kernel`), and no block groups or key starts. They are its backward's
too, under deterministic ops as without: XLA runs cuDNN's deterministic algorithm there. Every fused kernel
runs at the query's head width, values included (`widen_value_heads`)."""

TRITON_SUPPORTS = (
    _FAST,
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
    _FAST,
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

_FITS = _Need('fits', lambda case: _dense_fits(case),
              lambda case: f"jax.nn's xla attention holds the call's scores whole, {_dense_peak(case)} bytes "
                           f"at once in a training step with its operands, past the "
                           f"{generation.device_memory()} a device holds", row='5283', kind='measured')
"""A differentiated call on jax.nn's xla attention ('xla', and 'inference''s training) fits a device's
memory (`_dense_fits`): one that does not takes the template, its temporaries growing with the tokens, not
their square (row 5283). A call naming the kernel runs it, as a step compiled ahead for a larger device
than this host's does."""

XLA_SUPPORTS = (
    _Need('sinks', lambda case: not case.sinks, "jax.nn's attention has no sinks", row=INHERENT,
                kind='handoff'),
    _Need('softcap', lambda case: case.softcap is None, "jax.nn's attention has no softcap",
                row=INHERENT, kind='handoff'),
    _Need('narrows', lambda case: not case.narrows,
                "jax.nn's xla kernel would compute the call below its query's precision",
                row=INHERENT, kind='handoff'),
    _Need('arithmetic', lambda case: not case.arithmetic,
                "the call asks for arithmetic only the reference path computes (`reference_only`)",
                row=INHERENT, kind='handoff'),
    _Need('dropout', lambda case: not case.dropout, "jax.nn's attention takes no probability dropout",
          row=INHERENT, kind='handoff'),
    _Need('dtypes', lambda case: case.key_dtype == case.dtype,
          lambda case: f"jax.nn's attention takes a query of its keys' dtype, not {case.dtype} over "
                       f"{case.key_dtype} (`dot_product_attention`'s `_check_shape_and_dtype`)",
          row=INHERENT, kind='handoff'),
    _Need('wide_cpu', lambda case: not (wide_cpu_arithmetic() and generation.backend() == 'cpu'),
          "a CPU under Dew's wide arithmetic (`WIDE_CPU_ARITHMETIC`) sums fp32 products in float64 "
          "(`precision.accumulated`), where jax.nn's attention sums them in fp32: its reference path keeps "
          "them, so a run under it records the products it computed (a process without the setting runs "
          "jax.nn's kernel on a CPU as on any platform)",
          row=INHERENT, kind='handoff'),
    _Need('float64', lambda case: case.dtype != np.float64,
          "jax.nn's attention takes its softmax in float32 whatever its operands' dtype "
          "(`_dot_product_attention_core`'s `astype(np.float32)`), which a float64 call does not keep",
          row=INHERENT, kind='handoff'),
    _Need('cpu_float16', lambda case: not (case.dtype == np.float16 and generation.backend() == 'cpu'),
          "XLA's CPU compiler has no float16 dot into float32 (F16_F16_F32), the algorithm jax.nn's "
          "attention names for a float16 query (`_dot_product_attention_core`), past its trace-time fallback",
          row=INHERENT, kind='handoff'),
    _FITS,
)
"""jax.nn's xla attention's requirements: the generic kernel, which hands what it cannot honour to
the reference path, and refuses a call whose scores do not fit a device's memory (`_FITS`)."""


def _dense_fits(case: AttentionCase) -> bool:
    """Whether jax.nn's xla attention fits a device's memory for `case`: its training step's peak with its
    operands, as Dew's footprint model derives it (`_dense_peak`), within the bytes a device holds
    (`generation.device_memory`);
    where the device states none (an assumed TPU), it fits. A training step's, as a call's kernel is chosen
    before it is differentiated (`attention_route.attend`'s `custom_vjp`); the call's global shapes, a
    device's share where no sharding splits them (a split call fits the more)."""
    memory = generation.device_memory()
    return memory is None or _dense_peak(case) <= memory


def _dense_peak(case: AttentionCase) -> int:
    """The most bytes jax.nn's xla attention holds at once in a training step of `case`: its temporaries'
    peak beside its operands (`footprint.gradient_peak`, on the platform the call is chosen for; 12 to 13 a
    score in fp32, its logits, weights and their cotangents), and the operands, the query, keys and values."""
    return _dense_peak_of(case.rows, case.queries, case.heads, case.keys, case.kv_heads, case.head_dim,
                          jnp.dtype(case.dtype).name, case.causal, generation.backend())


@functools.cache
def _dense_peak_of(rows: int, queries: int, heads: int, keys: int, kv_heads: int, width: int, dtype: str,
                   causal: bool, platform: str) -> int:
    """`_dense_peak` of a call of these shapes, derived once a shape: the program traced over shapes alone,
    a matrix product's workspace (MiBs) none beside its scores."""
    import jax

    from dew.training import footprint

    query = jax.ShapeDtypeStruct((rows, queries, heads, width), dtype)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), dtype)

    def loss(params):
        with jax.softmax_custom_jvp(new_val=True):
            return jax.nn.dot_product_attention(*params, is_causal=causal, implementation='xla'), ()

    operands = (query, key, key)
    peak = footprint.gradient_peak(loss, operands, (), platform=platform, workspace=footprint.Workspace(0, 0))
    return peak.bytes + sum(math.prod(operand.shape) * jnp.dtype(dtype).itemsize for operand in operands)


TEMPLATE_SUPPORTS = (
    _Need('dtype', lambda case: case.dtype in TEMPLATE_DTYPES and case.key_dtype == case.dtype,
          lambda case: f"the query is {case.dtype} and the keys are {case.key_dtype}, where the template "
                       "takes one dtype of fp32, bf16 and fp16 on its operands", row=INHERENT),
    _Need('mask', lambda case: not case.mask,
          "the call has an arbitrary mask, which the template reads as spans and never as a mask",
          row=INHERENT),
    _Need('bias', lambda case: not case.bias or case.bias_hook,
          "the call's bias is not 4-D one term a key or a query's every key, its rows and heads the call's "
          "or one for them all, or is wider than fp32, which the template's score hook reads it in",
          row=INHERENT),
    _Need('shards', lambda case: case.shards == 1,
          "an automatic sequence axis gathers the keys, and a pallas call would gather the queries back",
          row=INHERENT),
    _Need('dropout',
          lambda case: not case.dropout or attention_route.drops(case.dtype, placed=case.drop_placed),
          attention_route.DROPS_NOT, row='1521'),
    _ARITHMETIC,
)
"""The attention template's requirements (`attention_route.attend`, the 'template' kernel): one dtype of fp32,
bf16 and fp16 on the queries and keys alike, no arbitrary mask and no bias, no sharded sequence axis, and the
call's default arithmetic. It takes every other call form: causality, windows, documents, key lengths and
starts, block groups, sinks, a softcap, and probability dropout off a GPU or over 16-bit operands on one
(`attention_route.drops`). A call
on a GPU or a TPU runs it by default, where its cell's declared old route does not take the call
(`_attention_preferred`, `DECLARED`), and anywhere a call names it (`implementation='template'`, or
`generation.chosen(attention='template')`)."""


FP32_MEASURED = 1024
"""The longest call DECLARED's fp32 receipts timed, its queries and its keys alike: a training step's
attention at Qwen3-8B's widths, 1 x 1024 (CELLS-h100, CELLS-a100); a longer fp32 call is a cell none declares
(`AttentionCase.cell`)."""


FLASH = Kernel[AttentionCase]('flash', FLASH_SUPPORTS)
XLA = Kernel[AttentionCase]('xla', (_FAST_HANDOFF, *XLA_SUPPORTS), generic=True)
INFERENCE = Kernel[AttentionCase]('inference', (_FAST, *TEMPLATE_SUPPORTS, _Need(
    'spans_only', lambda case: not case.spans_only,
    "the call's visibility is spans alone, which a differentiated call's xla attention would build "
    "densely: the template takes it", row=INHERENT), _FITS))
"""The template for a call no derivative is taken of and xla's attention for one differentiated."""

DECLARED: dict[tuple[str | None, str], tuple[Kernel[AttentionCase], str]] = {
    ('wgmma', 'attention'): (
        FLASH, "FlashAttention-3 measured faster than the Mosaic template on an H100 (llama-d128 bf16, "
                 "2 x 4096, 32 query heads over 8 key heads, CUPTI on an exclusive node): forward 0.588 "
                 "against 0.761 ms causal and 1.094 against 1.296 full, training forward and backward 2.53 "
                 "against 11.45 causal and 4.49 against 16.48 full (receipts DERIVED-benchx-h100, "
                 "BWDSTAGES-h100); a training step's attention at Qwen3-8B's widths, 2 x 4096 bf16, 2.44 "
                 "against 11.38 ms causal and 4.26 against 15.61 full (CELLS-h100)"),
    ('wgmma', 'float32 attention'): (
        INFERENCE, "jax.nn's xla attention measured faster than the template's fp32 route on an H100 for a "
                     "training step's attention at Qwen3-8B's widths, 1 x 1024 fp32, 2.08 against 26.36 ms "
                     "causal and 2.19 against 46.21 full (CELLS-h100); the template's fp32 forward as bf16 "
                     "parts faster than xla's for a call no derivative is taken of where `template_infers` "
                     "(M-ET-16, W18-gh200-1)"),
    ('mma.sync', 'attention'): (
        FLASH, "FlashAttention-2 measured faster than the template on an A100: a training step's attention "
                 "at Qwen3-8B's widths, 2 x 4096 bf16, 5.74 against 14.96 ms causal and 9.87 against 25.74 "
                 "full (CELLS-a100, CUPTI, 11 interleaved rounds)"),
    ('mma.sync', 'float32 attention'): (
        XLA, "jax.nn's xla attention measured faster than the template's fp32 route on an A100 at "
               "Qwen3-8B's widths (32/8 heads x 128, CUPTI medians, one warm process, W76-a100, M-ET-28): a "
               "training step's attention 5.94 against 226.4 ms at 1 x 1024 causal and 6.00 against 373.0 "
               "full, 102.3 against 3002.7 at 1 x 4096 causal and 101.6 against 5832.6 full; its forward "
               "3.13 against 8.26 ms at 1 x 1024 causal and 48.9 against 110.3 at 1 x 4096"),
}
"""Attention's declared cells (bible §4, owner 10-25; `Op.declared`), by (instruction family, kind of call)
(`AttentionCase.cell`): where an old route measured faster than the template, the old route and its receipt.
A call's kind is its dtype's, 16-bit or fp32: its kernel is chosen before the call is differentiated
(`attention_route.attend`'s `custom_vjp`), so its forward and its training run one kernel, and the receipts
cover both. Every other call takes the template (`_attention_preferred`). Nothing is ranked at runtime: the
list shrinks as the template wins."""


def template_infers(dtype, width: int, causal: bool) -> bool:
    """Whether the template's forward measured faster than xla's attention for a call of `dtype` no
    derivative is taken of, `width` wide, where its kernel is 'inference'. An fp32 call's (its operands as
    bf16 parts, the 'float32 attention' cell's): at width 64 everywhere (0.22 to 0.55 of xla's time, 8192
    causal 3.54 against 16.47 ms) and causal at 128 (0.57 to 0.80), at an RMS error from float64 30 to 60
    times xla's lower; not at 128 full (1.19 to 1.37 of xla's, one consumer of 64 columns, its QK products'
    parts run twice), nor at widths unmeasured (M-ET-16, W18-gh200-1, GH200, CUPTI median of 7, 2048 and
    8192 tokens). A 16-bit call's, which takes 'inference' for its dense bias alone (`_attention_preferred`):
    everywhere."""
    if jnp.dtype(dtype) != jnp.float32:
        return True
    return width == 64 or (width == 128 and causal)


def _attention_preferred(case: AttentionCase) -> str:
    """The kernel `case` runs where its cell declares no old route that takes it (`DECLARED`): the template
    on every GPU, which hands a call it does not take to the next kernel by legality; splash on a TPU, until
    a receipt (CELLS-v5e) measures the template there (theme A10); xla on a CPU, which has no instruction
    family. A bias that varies by query (`AttentionCase.dense_bias`) trains on the template where it is
    causal on warpgroup MMA alone, and elsewhere infers on it and trains on xla's attention ('inference')
    where that fits a device (`_FITS`), else on the template too:
    the template's forward took 0.32 to 0.72 of xla's time on an H100 and 0.47 to 1.04 on an A100 in 22 to
    73% less memory, a table every row shares or one a row, causal and full; its training step 0.62 to 0.96
    causal on an H100, but 1.07 to 1.48 full there and 1.12 to 2.23 on an A100 (M-DD-8, W55-gh200 and
    W57-a100: bf16, 32/8 heads x 128, 2048 and 8192 tokens, B 1 and 4). An accurate call trains on the
    template whatever its bias, as 'inference' takes the fast form alone (xla's attention computes no
    other). A call only spans express (`AttentionCase.spans_only`: block groups, key validity, runs or counted
    places) runs the template on a GPU and a TPU, which alone reads them as spans: every fused kernel there
    refuses it and the reference path builds its dense mask, `[B, 1, S, S]` (rows 1495, 1953); on a CPU it
    runs as any call there does, jax.nn's xla attention over the dense mask they mean (`visibility_mask`)
    where its scores fit, else the template (a forked decision row's call took the template's plain-JAX
    mainloop, and a run recorded fast its accurate route, once its forks' runs reached the call). A call
    whose products a quantized model's rule owns (`attention_quantized.owner`; an adapter's or an input
    quantizer's interception owns its projections alone, never the attention's products) and quantizes runs
    the template's quantized attention where it takes the rule and the call (`attention_quantized.takes`),
    else the reference path, which forms each product whole before Qwix quantizes it; a rule quantizing no
    activation leaves them whole (`attention_quantized.whole`)."""
    backend = generation.backend()
    if (rule := attention_quantized.owner()) is not None and not attention_quantized.whole(rule):
        # A quantized model's rule owns the call's products, quantizing the weights by each row's largest:
        # the template's anchored walk rounds them at that maximum where it takes the rule and the call
        # (`attention_quantized`, row 4858: no sinks, no draw), else the reference path forms them.
        if attention_quantized.takes(rule) and not (case.sinks or case.dropout):
            return 'template'
        return 'reference'
    if case.spans_only and backend in ('gpu', 'tpu'):
        return 'template'
    if (backend == 'gpu' and case.dense_bias and case.form == "fast"
            and not (case.causal and generation.gpu_target().architecture.wgmma)):
        # Its training xla's attention, where that fits (`_FITS`).
        return 'inference' if _dense_fits(case) else 'template'
    if backend not in ('gpu', 'tpu'):
        # jax.nn's xla attention, the faster on a CPU (a training step's time 0.3 to 0.7 of the template's
        # plain-JAX mainloop's at 4 x 256 x 4 x 32, 1 x 512 x 4 x 64, 2 x 1024 x 8 x 64 and 1 x 4096 x 8 x 64,
        # 1.1 at 1 x 2048 x 3 x 32, et-cpuattn; row 4826 makes the mainloop competitive), where its scores
        # fit (`_dense_fits`); else the template, whose temporaries grow with the tokens, not their square:
        # the reference path after xla in the order holds them whole too.
        return 'xla' if _dense_fits(case) else 'template'
    return {'gpu': 'template', 'tpu': 'tpu'}.get(backend, 'xla')


def _unlowered(kernel: str) -> str | None:
    """Why this host cannot lower the target's `kernel` for a GPU a step is traced for on it, None where it
    lowers it: FlashAttention where its call lowers for no GPU here (`flash_lowered`: its wheel holds no code
    for the target); cuDNN's where this host loads no cuDNN, without which its call traces nothing ("cuDNN
    not found", a CPU environment's host), or where the program is compiled for any GPU of the generation
    (`generation.Assumption.portable`), as a cuDNN plan compiled with no device holds to its GPU's model
    alone: one compiled for a canonical sm90 failed on a GH200 ("CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
    cudnnFinalize Descriptor Failed", FS-CANONICAL-CUDNN-gh200), one compiled for that GH200's own topology
    ran there exactly (FS-DEVICELESS-C3B-gh200). The call is refused, not lowered as another kernel: a step
    compiled ahead with XLA's attention in its place was not the GPU's program (an H100's ran XLA's attention,
    its fp32 scores 4.3 GB the run's step never holds, row 3792; every CPU's compile ahead did, row 4616).
    Dew's own Pallas kernels and tokamax's Triton kernel (`generation.assumed`'s cross-compile) lower for
    the target on any host; on its own GPU every kernel does."""
    if gpu_placed():
        return None
    if kernel == 'flash' and not flash_lowered():
        return "FlashAttention's wheel here holds no code for the target"
    found = generation.assumption()
    if kernel != 'cudnn' or found is None:
        return None
    if found.portable:
        return (f"a cuDNN plan compiled with no device holds to its GPU's model alone, and this program is "
                f"compiled for any {found.generation}: compile it for the run host's own GPUs "
                "(`dew.training.ahead.host_gpus`)")
    return None if _cudnn_loads() else "this host loads no cuDNN library"


def _cudnn_loads() -> bool:
    """Whether this process loads cuDNN: the CUDA plugin found its library (read once a GPU's devices,
    compile-only or attached, are made: before, the plugin has loaded none)."""
    from jax._src.lib import cuda_versions

    try:
        return cuda_versions is not None and cuda_versions.cudnn_get_version() > 0
    except RuntimeError:
        return False


ATTENTION = Op[AttentionCase](Choice.ATTENTION, (
    FLASH,
    Kernel[AttentionCase]('triton', TRITON_SUPPORTS),
    Kernel[AttentionCase]('cudnn', (_FAST, *CUDNN_SUPPORTS)),
    Kernel[AttentionCase]('tpu', TPU_SUPPORTS),
    XLA,
    Kernel[AttentionCase]('reference', (), generic=True),
    Kernel[AttentionCase]('template', TEMPLATE_SUPPORTS),
    INFERENCE,
), _attention_preferred, _unlowered, DECLARED)
"""Attention's selector: a declared cell's old route where it takes the call (`DECLARED`), else the call's
kernel (`_attention_preferred`: the template on a GPU or a TPU, xla on a CPU), then flash, triton, cuDNN,
splash, xla and the reference path, the first that takes the call. The old routes take only the fast
form (`AttentionCase.form`): an accurate call runs the template, or the reference path on a CPU; a
decode call keeps its own kernels whatever the form (`DECODE`). The template comes last, after the
reference path, which takes every call, so it runs where the preference names it or a call names it; and
'inference' after it, the template for a call no derivative is taken of and xla's attention for one
differentiated (`dew.nn.attention.forward_differentiable`'s `primal_alone`), where a cell declares it or
a dense bias's preference names it."""



def _decode_preferred(case: AttentionCase) -> str:
    """The kernel a decode call runs: on a GPU the hand-written decode kernel (`decode_attention`; the
    template's decode does not yet match it, rows 4042 to 4045), which hands a call it does not take to the
    next kernel by legality, a declared cell's old route first (`DECODE_DECLARED`); xla elsewhere."""
    return 'decode' if generation.backend() == 'gpu' else 'xla'


_FP32_DECODE = ("jax.nn's xla attention measured faster than the decode kernel and the attention template's "
                "decode at every fp32 decode shape timed, its products IEEE fp32 as theirs (`highest`; wall "
                "a step over 50 steps in one loop, interleaved rounds, 1 to 128 rows over 4096 and 32768 "
                "keys, 32 or 16 query heads over 8, 128 or 64 wide): {}")
DECODE_XLA = Kernel[AttentionCase]('xla', XLA_SUPPORTS, generic=True)
"""A decode step's xla attention (the folded attention, `folded_attention`)."""

DECODE_DECLARED: dict[tuple[str | None, str], tuple[Kernel[AttentionCase], str]] = {
    ('sm80', 'float32 decode'): (DECODE_XLA, _FP32_DECODE.format(
        "on an A100 2.63 to 3.64 times the decode kernel's speed, 51.2 against 140.5 us at one row over "
        "4096 keys (ET-DFP32B-a100), and 25 to 161 times the template's, 52.5 against 5215.5 us there "
        "(ET-4312-a100)")),
    ('sm90', 'float32 decode'): (DECODE_XLA, _FP32_DECODE.format(
        "on an H100 2.75 to 3.60 times the decode kernel's speed, 23.8 against 72.4 us at one row over "
        "4096 keys (ET-DFP32B-h100), and 31 to 250 times the template's, 23.6 against 3669.2 us there "
        "(ET-4312-h100)")),
}
"""The decode selector's declared cells (`Op.declared`), by (device generation, kind of call), each an old
route measured faster than the decode kernel at every shape a receipt timed on the generation, and that
receipt. cuDNN declares none: faster at most 16-bit rows, it measured slower at some on each generation timed
(an A100's 8 rows over 32768 keys, 756 against 734 us, ET-DCELL-a100; an H100 PCIe's one row over 4096, 40.5
to 65 against 35.6, ET-DFP32-h100-pcie)."""


DECODE = Op[AttentionCase](Choice.DECODE, (
    Kernel[AttentionCase]('decode', decode_attention.SUPPORTS),
    Kernel[AttentionCase]('cudnn', (*CUDNN_SUPPORTS, decode_attention.PRODUCTS)),
    DECODE_XLA,
    Kernel[AttentionCase]('reference', (), generic=True),
    Kernel[AttentionCase]('template', (*TEMPLATE_SUPPORTS, decode_attention.PRODUCTS)),
), _decode_preferred, _unlowered, DECODE_DECLARED)
"""The selector of a call of few query positions over key lengths (`decodes`): the decode kernel
(`decode_attention.attend`), cuDNN and the folded xla attention (`folded_attention`), a declared cell's
kernel first (`DECODE_DECLARED`), each fused kernel only where it computes the run's declared products
(`decode_attention.PRODUCTS`); the attention template's decode (`attention_decode.attend`, through
`attention_route.attend`) where a call or a run's context names it (`generation.chosen(decode='template')`),
until it measures faster than the decode kernel (rows 4042 to 4045, 4075)."""


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
