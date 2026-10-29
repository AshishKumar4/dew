"""The attention template's Blackwell mainloop (`dew.nn.kernels.templates.attention_blackwell`), first rung:
the full and causal forward of 16-bit operands, interpreted on the CPU by Mosaic GPU's own interpreter,
which models tensor memory and tcgen05's products, held to check 1 against the template's reference from
float64 truth; and compiled for a B200 on this CPU within its shared and tensor memory, no register
spilled. The interpreter is coverage, not proof of the compiled kernel's synchronization; a B200's or a
B300's run is."""

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_blackwell

pytestmark = pytest.mark.kernel

SEEDS = 8
"""Each site's declared draws, check 1's fewest (`reference_error.seeds_for`)."""
CASES = {"full": False, "causal": True}
"""The call forms this rung takes, by whether they are causal."""
SHAPES = {64: (1, 200, 4, 2), 128: (1, 136, 2, 1)}
"""Each head width's rows, tokens (not whole tiles, so the call pads, past one key tile), and query and key
heads (grouped)."""


def _draw(shape, width: int, dtype):
    rows, tokens, heads, kv_heads = shape

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.normal(size=(rows, tokens, count, width)), dtype)
                     for count in (heads, kv_heads, kv_heads))
    return draw


def _attended(causal: bool, tokens: int, rows: int, *, reference: bool = False, **options):
    """The forward of a call of `rows` rows of `tokens` tokens, causal or full, on this mainloop (`options`
    its own), or the template's reference where `reference` (check 1 widens its inputs for truth)."""
    spans = template.spans(tokens, tokens, rows, causal=causal)

    def run(query, key, value):
        if reference:
            return template.reference(query, key, value, spans)
        return attention_blackwell.forward(query, key, value, spans, **options)
    return run


INTERPRETED_CASES = ("full",)
"""The call forms the interpreter runs: a causal call's per-block walk fails Mosaic GPU's interpreter at the
pinned JAX (an XLA RET_CHECK compiling a role's thread, a tuple of 8 leaves given 9 shardings), which its
compile for a B200 does not (row 2422)."""


@INTERPRETS
@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("width", list(SHAPES))
@pytest.mark.parametrize("case", INTERPRETED_CASES)
def test_the_blackwell_forward_holds_check_1_interpreted(case, width, natural):
    """The forward of a `case` call of heads `width` wide, interpreted: as exact as the template's reference
    from float64 truth, its padded tokens (masked past the keys) and its grouped heads included."""
    shape = SHAPES[width]
    rows, tokens = shape[:2]
    ours = _attended(CASES[case], tokens, rows, natural=natural, interpret=True)
    reference = _attended(CASES[case], tokens, rows, reference=True)
    assert_as_exact(ours, reference, _draw(shape, width, jnp.bfloat16), f"blackwell {case} {width}",
                    seeds=SEEDS)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", list(SHAPES))
def test_the_blackwell_forward_compiles_for_a_b200_within_its_resources(width):
    """The causal forward of heads `width` wide at the default tiles, compiled by Mosaic for a compile-only
    B200 on this CPU through ptxas (tests/kernel_compile.py): within the multiprocessor's shared memory
    and its 512 columns of tensor memory, no register spilled."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    spans = template.spans(1024, 1024, 2, causal=True)

    def call():
        def forward(query, key, value):
            return attention_blackwell.forward(query, key, value, spans)

        return forward, (jax.ShapeDtypeStruct((2, 1024, 8, width), jnp.bfloat16),
                         jax.ShapeDtypeStruct((2, 1024, 2, width), jnp.bfloat16),
                         jax.ShapeDtypeStruct((2, 1024, 2, width), jnp.bfloat16))

    with generation.assumed("sm100"):
        found = compile_mosaic_kernels([support.KernelCase(f"blackwell attention {width}", call)], "sm100")
    assert len(found) == 1 and found[0].get("kernels"), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
