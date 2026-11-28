"""Probe (not for landing; run natively on a GH200): row 14172's witnesses at the fix."""


def test_probe_14172_cells():
    import test_gemm_nvfp4

    test_gemm_nvfp4.a_packed_nvfp4_prefills_finish_is_its_products_finished_apart()
    test_gemm_nvfp4.an_owed_products_native_fold_rounds_once_below_the_normals()
    test_gemm_nvfp4.a_packed_nvfp4_call_placed_on_its_hosts_cpu_keeps_a_subnormal_scale()
