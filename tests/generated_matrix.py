"""The generated bodies' check-1 matrix (tests/kernel_matrix.py): each site the algebra's generated, checked
bodies take (`dew.algebra.generated`) run with the choice `generated` named "generated", at its component's
declared cases, on par with the same site named "written", from its declaration's float64 truth.

The sites are the templates' own functions, each its component's declared function: the selective scan's
decay, readout, backward step lift and adjoint, plain or at its lifted step
(`recurrence_algebra.selective_decay`, `selective_read`, `selective_dx_scale`, `selective_adjoint`), the
recurrence skip's pullback (`skipped_back`), the gated delta rule's value pullback and key memory decay
(`gated_values_back`, `key_memory_decay`) and the derived split-KV merge (`attention_algebra.merge`); the
token stage's declared function composes the written decay and update, so its site is called apart
(`selective.token_stage`, `SITES`). Every cell asserts the generated body ran (`generated.taken`): a site that
fell back to its written math would hold check 1 trivially. The cases are each component's declared ones
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
from dew.nn.kernels.templates.recurrence import selective

CHOICES = frozenset({"generated"})
"""The `generation.CHOICES` entries the cells run: each cell names `generated` both ways."""
KERNELS = {}
"""The kernels the cells run under test, by name: none, as `generated` is no `selection.Op`'s but the sites'
own choice of their body, which has no kernels to list."""


def _token(*operands):
    """The selective scan's token stage (`selective.token_stage`) over its declared flat operands
    (`component_cases.selective_token`'s ABI), B's scalars one token's chunk row."""
    count, kept = divmod(len(operands) - 2, 3)
    states, rates, b = (list(operands[part * count:(part + 1) * count]) for part in range(3))
    step, value, *rest = operands[3 * count:]
    return tuple(selective.token_stage(states, rates, jnp.stack(b)[None], 0, step, value,
                                       rest[0] if kept else None))


SITES = {"selective_token": _token}
"""A site whose component's declared function is not its own call, by its component's name."""


def supported(case: dict[str, str]) -> bool:
    """Whether the site serves `case` (tests/kernel_matrix.py): its component is declared for the case's
    generation."""
    declared = generate.DECLARATIONS[f"{component_cases.__name__}.{case['site']}"]
    return case["generation"] in declared.cases.targets


CASES = (
    ('sm90', 'selective_decay'),
    ('sm90', 'selective_read'),
    ('sm90', 'split_kv_merge'),
    ('sm90', 'selective_token'),
    ('sm90', 'selective_dx_scale'),
    ('sm90', 'skipped_back'),
    ('sm90', 'key_memory_decay'),
    ('sm90', 'gdn_value_pullback'),
    ('sm90', 'selective_adjoint'),
    ('sm90', 'selective_lifted_adjoint'),
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
    site = SITES.get(named["site"], form.function)

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
