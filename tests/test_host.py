"""The host's CPU class, the one key the shared XLA bundles and the CI's records take a host by
(`dew._host.cpu_class`)."""

from tool_support import load

from dew._host import cpu_class

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
    assert load("ci", "tools/armada").cpu_class() == cpu_class()
