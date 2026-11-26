"""The kernel each of Dew's ops runs.

Each op Dew dispatches (attention, decode, the dense and grouped products, the norms and gated products,
the recurrences' scans) picks its kernel for the call it gets: the kernels that take the call, in the op's
order, and the kernel its cell on this device declares measured fastest. Two surfaces override that pick,
one mechanism under both:

- `use(Kernels(attention="xla"))` or `use(attention="xla")`, a block: each call traced within it runs the
  kernel the block names for its op, whatever the op's pick or the call's own (nested blocks keep the
  outer block's other choices).
- `implementation=` on a layer or a call (`dew.nn.norms.RMSNorm(implementation="pallas")`,
  `dew.nn.attention.scaled_dot_product_attention(..., implementation="cudnn")`): that call's own kernel,
  which a block's choice for its op overrides.

A kernel named for a call it does not take raises `KernelRefused`, a `ValueError` naming the first
requirement the call fails (and every one it fails, `KernelRefused.refusals`), rather than running another
kernel in silence. `choices` are the ops a block
names; `dew.gpu_facts()` tells what the device issues.
"""

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Literal

from dew.nn.kernels.generation import CHOICES as choices, chosen
from dew.nn.kernels.selection import KernelRefused

__all__ = ["KernelRefused", "Kernels", "choices", "use"]


@dataclasses.dataclass(frozen=True)
class Kernels:
    """The kernel each op runs, one field an op (`choices`), its kernels the field's values: None leaves the
    op's pick for each call. A run's (`--diagnostics.kernels.attention xla`, a choice of its kernels) and a
    block's alike (`use(Kernels(attention="xla"))`), where a mapping of names held any string a parse never
    checked (api03)."""

    attention: Literal['cudnn', 'flash', 'inference', 'reference', 'template', 'tpu', 'triton',
                       'xla'] | None = None
    attention_mainloop: Literal['mosaic', 'portable', 'triton'] | None = None
    blockwise_fp8: Literal['mosaic', 'xla'] | None = None
    decode: Literal['cudnn', 'decode', 'reference', 'template', 'xla'] | None = None
    delta_decode: Literal['fused', 'step', 'xla'] | None = None
    delta_gates: Literal['apart', 'joined'] | None = None
    diagonal_scan: Literal['mosaic_gpu', 'xla'] | None = None
    gate_up: Literal['apart', 'joined'] | None = None
    gated_delta_rule: Literal['fla', 'mosaic_gpu', 'xla'] | None = None
    gated_mlp: Literal['apart', 'fused', 'wide'] | None = None
    gemm_epilogue: Literal['fused', 'xla'] | None = None
    gemm_prologue: Literal['fused', 'xla'] | None = None
    grouped_matmul: Literal['apart', 'cutlass', 'mosaic', 'triton', 'xla'] | None = None
    layer_scan: Literal['plain', 'scanned'] | None = None
    linear_ce: Literal['mosaic', 'xla'] | None = None
    low_rank: Literal['fused', 'xla'] | None = None
    # Keyword-only: a positional call of the fields before it keeps naming the ones it named.
    merge: Literal['mosaic', 'triton'] | None = dataclasses.field(default=None, kw_only=True)
    mxfp8: Literal['mosaic', 'xla'] | None = None
    narrow_copies: Literal['off', 'on'] | None = None
    nvfp4: Literal['packed', 'xla'] | None = None
    nvfp4_block: Literal['mosaic', 'xla'] | None = None
    paged_decode: Literal['gathered', 'template', 'tpu', 'triton'] | None = None
    qkv: Literal['apart', 'joined'] | None = None
    rms_norm: Literal['pallas', 'xla'] | None = None
    selective_scan: Literal['mosaic_gpu', 'xla'] | None = None
    short_conv: Literal['mosaic_gpu', 'xla'] | None = None
    sparse_attention: Literal['template', 'xla'] | None = None
    ssd: Literal['mosaic_gpu', 'pallas', 'tpu', 'xla'] | None = None
    swiglu: Literal['pallas', 'xla'] | None = None
    syrk: Literal['mosaic', 'xla'] | None = None
    wkv4: Literal['mosaic_gpu', 'xla'] | None = None

    def named(self) -> dict[str, str]:
        """The ops it names and the kernel each runs, those it leaves to their picks omitted."""
        return {field.name: kernel for field in dataclasses.fields(self)
                if (kernel := vars(self)[field.name]) is not None}


@contextlib.contextmanager
def use(kernels: Kernels | None = None, /, **names: str) -> Iterator[None]:
    """Within the block, each op `kernels` names, and each op `names` names over it, runs that kernel
    (`dew.nn.kernels.generation.chosen`, every call traced within)."""
    with chosen(**{**(kernels.named() if kernels is not None else {}), **names}):
        yield
