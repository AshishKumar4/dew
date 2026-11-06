"""The kernel each of Dew's ops runs.

Each op Dew dispatches (attention, decode, the dense and grouped products, the norms and gated products,
the recurrences' scans) picks its kernel for the call it gets: the kernels that take the call, in the op's
order, and the kernel its cell on this device declares measured fastest. Two surfaces override that pick,
one mechanism under both:

- `use(attention="xla")`, a block: each call traced within it runs the kernel the block names for its op,
  whatever the op's pick or the call's own (nested blocks keep the outer block's other choices).
- `implementation=` on a layer or a call (`dew.nn.norms.RMSNorm(implementation="pallas")`,
  `dew.nn.attention.scaled_dot_product_attention(..., implementation="cudnn")`): that call's own kernel,
  which a block's choice for its op overrides.

A kernel named for a call it does not take raises `KernelRefused`, a `ValueError` listing every
requirement the call fails, rather than running another kernel in silence. `choices` are the ops a block
names; `dew.gpu_facts()` tells what the device issues.
"""

from dew.nn.kernels.generation import CHOICES as choices, chosen as use
from dew.nn.kernels.selection import KernelRefused

__all__ = ["KernelRefused", "choices", "use"]
