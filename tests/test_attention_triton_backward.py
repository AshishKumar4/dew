"""The Pallas-Triton mainloop's backward (`attention_triton.backward`) and the plain-JAX one's over the
template core's call forms and runs, on par with `attention.reference` from float64 in both forms, here
interpreted and on a GPU: tests/test_attention_triton.py's helpers, a module of its own so a GPU tier's
prepare of a file a unit fits its deadline (row 12671)."""
import jax.numpy as jnp
import pytest
from test_attention_template import CASES, RUN_CASES
from test_attention_triton import BACKWARD_FORMS, PINNED, TEMPLATE_FORMS, check_backward

from dew.nn.kernels.templates import attention_triton

pytestmark = pytest.mark.usefixtures("interpreted_as_an_a100")


@pytest.mark.parametrize("options", TEMPLATE_FORMS.values(), ids=TEMPLATE_FORMS)
@pytest.mark.parametrize("config", PINNED[:3], ids=str)
@pytest.mark.parametrize("form", BACKWARD_FORMS)
def test_the_backward_is_on_par_over_every_call_form(form, config, options):
    """Each call form's keys, fp32, no hooks, at pinned dQ and dK/dV configs, in both forms."""
    check_backward(CASES[form], (), jnp.float32, config, **options)


@pytest.mark.parametrize("options", TEMPLATE_FORMS.values(), ids=TEMPLATE_FORMS)
@pytest.mark.parametrize("config", PINNED[:2], ids=str)
@pytest.mark.parametrize("form", RUN_CASES)
def test_the_plain_backward_is_on_par_over_runs(form, config, options):
    """Each query's runs on the plain-JAX mainloop: its query walk takes its ranges as the forward does, and
    its key walk reads the query blocks some range of which holds its keys, in both forms."""
    check_backward(RUN_CASES[form], (), jnp.float32, config, mainloop="portable", **options)


@pytest.mark.parametrize("form", [form for form in CASES if form not in BACKWARD_FORMS])
def test_the_backward_is_on_par_over_the_core_s_other_call_forms(form):
    """The rest of the template core's call forms, the dK/dV launch at warps and stages of its own."""
    check_backward(CASES[form], (), jnp.float32, PINNED[0], attention_triton.Tiles(32, 32, 8, 3))
