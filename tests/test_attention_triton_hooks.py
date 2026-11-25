"""The Pallas-Triton mainloop's backward (`attention_triton.backward`) under every score hook, its
parameters' and initial state's gradients on par with `attention.reference` from float64 in both forms,
here interpreted and on a GPU: tests/test_attention_triton.py's helpers, a module of its own so a GPU
tier's prepare of a file a unit fits its deadline (row 12671: this sweep alone took 2,954 s of its file's
6,746)."""
import pytest
from test_attention_template import CASES
from test_attention_triton import DTYPE_IDS, PINNED, SCORE_HOOKS, TEMPLATE_FORMS, check_backward

from dew.nn.kernels.templates import attention_triton

pytestmark = pytest.mark.usefixtures("interpreted_as_an_a100")


@pytest.mark.parametrize("options", TEMPLATE_FORMS.values(), ids=TEMPLATE_FORMS)
@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("hooks", SCORE_HOOKS.values(), ids=SCORE_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook(hooks, dtype, options):
    """Each score hook over causal packed documents under a window: the gradients of its parameters and
    of the initial state too, in both forms."""
    check_backward(CASES["causal-documents-window"], hooks, dtype, PINNED[0], **options)
