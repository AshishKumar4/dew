"""dew.telemetry.trace: a device's traced window, its busy time the union of its kernels."""

from dew.telemetry.trace import window_split


def test_a_traced_window_splits_into_compute_exposed_collectives_and_idle():
    """One device's kernels, in microseconds: an all-reduce half hidden
    behind compute, a gap a batch's host-to-device copy ends, and a gap
    compute ends. benchmark_step and the reference runs both report these
    figures, and the window is their sum."""
    events = [("loop_add_fusion", 0, 10), ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 5, 20),
              ("MemcpyH2D", 25, 26), ("gemm_fusion_dot", 26, 30), ("loop_multiply_fusion", 40, 50)]
    split = window_split([(name, start * 1000, end * 1000) for name, start, end in events])

    assert {key: value / 1000 for key, value in split.items()
            if key in ("window", "busy", "compute", "communication", "exposed_communication",
                       "idle_input", "idle_host", "AllReduce", "AllGather")} == {
        "window": 50, "busy": 35, "compute": 25, "communication": 15,
        "exposed_communication": 10, "idle_input": 5, "idle_host": 10, "AllReduce": 15,
        "AllGather": 0}
    assert split["window"] == (split["compute"] + split["exposed_communication"]
                               + split["idle_input"] + split["idle_host"])


def test_a_nested_last_kernel_does_not_shorten_a_traced_window():
    split = window_split([("nested", 30.5, 40.25), ("long_kernel", 10.25, 100.75)])
    assert (split["busy"], split["compute"], split["window"]) == (90.5, 90.5, 90.5)
    assert split["idle_input"] == split["idle_host"] == 0
