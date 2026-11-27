"""The generated bodies' check-1 matrix (tests/kernel_matrix.py): each site the algebra's generated, checked
bodies take (`dew.algebra.generated`) run with the choice `generated` named "generated", at its component's
declared cases, on par with the same site named "written", from its declaration's float64 truth.

The sites are the templates' own functions, each its component's declared function: the selective scan's decay
and readout (`recurrence_algebra.selective_decay`, `selective_read`) and the derived split-KV merge
(`attention_algebra.merge`). Every cell asserts the generated body ran (`generated.taken`): a site that fell
back to its written math would hold check 1 trivially. The cases are each component's declared ones
(`dew.algebra.component_cases`), keyed by the generation they were generated for, the one this family names.
"""

import functools

import jax.numpy as jnp
import pytest
from generation_support import pinned
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew.algebra import component_cases, generate, generated
from dew.nn.kernels import generation

CHOICES = frozenset({"generated"})
"""The `generation.CHOICES` entries the cells run: each cell names `generated` both ways."""
KERNELS = {}
"""The kernels the cells run under test, by name: none, as `generated` is no `selection.Op`'s but the sites'
own choice of their body, which has no kernels to list."""


def supported(case: dict[str, str]) -> bool:
    """Whether the site serves `case` (tests/kernel_matrix.py): its component is declared for the case's
    generation."""
    return case["generation"] in generate.DECLARATIONS[f"{component_cases.__name__}.{case['site']}"].cases.targets


CASES = (
    ('sm90', 'selective_decay'),
    ('sm90', 'selective_read'),
    ('sm90', 'split_kv_merge'),
)
"""The cells' cases, each its axis values in `AXES`' order (`kernel_matrix.covering("generated")`'s pairs)."""


@cell("generated", covers=lambda case: dict(zip(AXES["generated"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pinned(case[0])) for case in CASES], ids="-".join)
def a_generated_body_is_on_par(case: tuple[str, ...]) -> None:
    """`case`'s site at each of its component's declared cases with its generated body named, on par with
    the site named written (check 1) from the declaration's float64 truth, its generated body counted as
    run at every case."""
    named = dict(zip(AXES["generated"], case, strict=True))
    qualified = f"{component_cases.__name__}.{named['site']}"
    form = generate.DECLARATIONS[qualified]
    assert isinstance(form, generate.ComponentForm)
    site = form.function  # Each declared function is its site's own call.

    def under(kernel: str):
        def run(*values):
            with generation.chosen(generated=kernel):
                return site(*map(jnp.asarray, values))
        return run

    for operands in form.cases.operands:
        with generated.taken() as counts:
            assert_as_exact(under("generated"), under("written"),
                            functools.partial(form.cases.draw, operands=operands),
                            f"{named['site']} generated ({named['generation']})", seeds=form.cases.seeds,
                            truth_fn=form.truth)
        assert counts and set(counts) == {(qualified, generated.GENERATED)}, counts
