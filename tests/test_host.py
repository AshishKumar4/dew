"""The host's CPU class, the one key the shared XLA bundles and the CI's records take a host by
(tools/armada/host.py's `cpu_class`), and the execution limits a run that read one keys (its `limit`)."""

from tool_support import load

host = load("host", "tools/armada")
cpu_class = host.cpu_class

ZEN4 = "vendor_id\t: AuthenticAMD\ncpu family\t: 25\nmodel\t\t: 17\nmodel name\t: AMD EPYC 9654\n" * 2
NEOVERSE = ("processor\t: 0\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\nCPU architecture: 8\n"
            "CPU part\t: 0xd4f\n")


def test_a_host_s_class_is_its_microarchitecture_byte_for_byte_as_the_records_have_keyed_it():
    """x86's vendor, family and model, as every record of the CI's test cache was keyed before the class
    moved here, so none goes cold; an Arm host's implementer, architecture and part, which the records
    named "unknown-unknown-unknown", one class for every Arm host; and the CI's own reading of the host
    is this one."""
    assert cpu_class(ZEN4) == "AuthenticAMD-25-17"
    assert cpu_class(NEOVERSE) == "0x41-8-0xd4f"
    assert load("ci", "tools/armada").run_keys.cpu_class() == cpu_class()


def test_a_v1_memory_controllers_limit_is_the_memory_limit_a_run_keys(tmp_path, monkeypatch):
    """A run that read its memory limit keys the limit binding it (`limit`), cgroup v1's memory controller's
    (`memory.limit_in_bytes`, at that controller's own mount) as v2's: Dew's one walk of them
    (`_cgroups.memory_limits`), where CI read v2's alone and keyed a v1 container's limit 'max' (row
    13035)."""
    reader = host.cgroups()
    membership = tmp_path / "cgroup"
    membership.write_text("4:cpu,cpuacct:/job\n5:memory:/job\n")
    (tmp_path / "memory/job").mkdir(parents=True)
    (tmp_path / "memory/job/memory.limit_in_bytes").write_text("1073741824\n")
    real_limits = reader.memory_limits
    monkeypatch.setattr(reader, "memory_limits", lambda root: real_limits(root, membership))
    monkeypatch.setenv("DEW_CGROUP_ROOT", str(tmp_path))
    assert host.limit("limit:memory") == "1073741824"
