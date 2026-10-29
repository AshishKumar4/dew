"""The recurrences on the GPU (G14, the owner's rule: no unit tests on GPUs): each native recurrence route's
cases the CPU does not interpret, run in one process (`gpu_flow`), each cell its own file's property:

- The Mosaic mainloop's forward and gradient on par from float64 with their references at the cases the
  CPU leaves to a GPU (test_recurrence_mosaic's `native`, test_recurrence_mosaic_gradient), every hook
  it takes token by token, and a two-role walk's training on an H100.
- The template finite and on par where chunked forms break, every form, regime and product
  (test_recurrence_template_stability).
- The portable backward as a GPU compiles it, and the SSD port's forward and gradient, finite and on par
  under every regime (test_recurrence_stability).
- The short conv's kernels as exact as XLA's conv at the training shapes (test_short_conv).
"""

import functools

import jax.numpy as jnp
import pytest
import test_recurrence_mosaic as forward
import test_recurrence_mosaic_gradient as gradient
import test_recurrence_stability as stability
import test_recurrence_template_stability as template_stability
import test_short_conv as short_conv
from gpu_flow import Cell, run_cells

from dew.nn.kernels.templates.recurrence import mainloop

pytestmark = pytest.mark.gpu(reason="each recurrence route's native kernel: the Mosaic mainloop, the ports")


def on_its_family(check):
    """`check` where this GPU is of a family the mainloop's bundles claim (an A100's, an H100's)."""
    def run():
        if mainloop.family_of() is None:
            pytest.skip("the mainloop's bundles run compiled on a GPU of the family each claims")
        check()
    return run


def test_the_recurrences_run_on_this_gpu():
    """Every recurrence cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together; every direct mainloop call keeps its own bound."""
    templates_finite = template_stability.the_template_stays_finite_and_on_par_where_chunked_forms_break
    run_cells([
        *(Cell(f"the Mosaic forward is as exact as its reference: {name}",
               functools.partial(forward.test_the_mosaic_forward_is_as_exact_as_its_reference, *values))
          for name, values in forward.native(forward.FORWARD_GRID, forward.INTERPRETED_FORWARD)),
        *(Cell(f"every hook the Mosaic mainloop takes is as exact as the recurrence: {hooks} {packed}",
               on_its_family(functools.partial(
                   forward.every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token,
                   hooks, packed)))
          for hooks in forward.TAKEN for packed in (False, True)),
        *(Cell(f"the Mosaic gradient is as exact as its reference: {name}",
               functools.partial(gradient.test_the_mosaic_gradient_is_as_exact_as_its_reference, *values))
          for name, values in forward.native(gradient.GRADIENT_GRID, gradient.INTERPRETED_GRADIENT)),
        *(Cell(f"a walk of two roles is as exact as its reference: {name}",
               functools.partial(gradient.test_a_walk_of_two_roles_is_as_exact_as_its_reference, *values),
               "sm90")
          for name, values in forward.native(gradient.WALK_GRID, gradient.INTERPRETED_WALKS)),
        *(Cell(f"the template stays finite and on par where chunked forms break: {name} {regime} {product}",
               functools.partial(templates_finite, name, regime, product))
          for name, regime, product in template_stability.CASES),
        *(Cell(f"the portable backward as a GPU compiles it is as exact as the recurrence's: {name} {regime}",
               functools.partial(stability.test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences,
                                 name, regime))
          for name, regime in stability.ACCELERATED),
        *(Cell(f"the SSD port stays finite and exact where chunked forms break: {regime}",
               functools.partial(stability.the_ssd_port_stays_finite_and_exact_where_chunked_forms_break,
                                 regime))
          for regime in stability.REGIMES),
        *(Cell(f"the SSD port's gradient is as exact as the recurrence's: {regime}",
               functools.partial(stability.the_ssd_ports_gradient_is_as_exact_as_the_recurrences, regime))
          for regime in stability.REGIMES),
        *(Cell(f"the short conv's kernels are as exact as XLA's at the training shapes: {model} "
               f"{dtype.__name__}{' packed' if packed else ''}",
               functools.partial(short_conv.the_kernels_are_as_exact_as_xla_at_the_training_shapes,
                                 model, dtype, packed))
          for model in sorted(short_conv.CONV_SHAPES) for dtype in (jnp.bfloat16, jnp.float32)
          for packed in (False, True)),
    ])
