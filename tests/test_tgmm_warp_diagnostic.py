"""Temporary source instrumentation for the gathered-gradient lowering failure."""

import traceback

import jax
import test_gemm_gather


def test_gathered_gradient_load_source():
    jax.config.update("jax_traceback_filtering", "off")
    try:
        test_gemm_gather.test_each_rings_gathered_kernel_gradient_lowers_for_cuda("sm100", (True, True))
    except ValueError as error:
        evidence = [str(error)]
        current = error.__traceback__
        while current is not None:
            frame = current.tb_frame
            if frame.f_code.co_name == "_get_lowering_rule_wg":
                context = frame.f_locals["ctx"]
                evidence.append(f"FAILING_GET_OUTPUT {context.avals_out}")
                evidence.append(f"FAILING_GET_SEMANTICS {context.module_ctx.primitive_semantics}")
            if frame.f_code.co_name == "lower_jaxpr_to_mosaic_gpu":
                equation = frame.f_locals.get("eqn")
                if equation is not None and equation.primitive.name == "get":
                    evidence.append(f"FAILING_GET_EQUATION {equation}")
                    source = equation.source_info.traceback.as_python_traceback()
                    for location in traceback.extract_tb(source):
                        if "/dew/nn/" in location.filename and location.name in (
                                "staged_at", "_staged_halves", "zeroed", "tile_of"):
                            evidence.append(f"FAILING_GET_SOURCE {location}")
            current = current.tb_next
        raise AssertionError("\n".join(evidence)) from None
