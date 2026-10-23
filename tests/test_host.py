"""The host's CPU class, the one key Dew's compilation cache, the shared XLA bundles and the CI's records
take a host by (`dew._host.cpu_class`)."""

import jax
from tool_support import load

import dew.cache
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


def test_dew_keeps_each_cpu_class_s_programs_apart_unless_a_directory_is_named(monkeypatch, tmp_path):
    """XLA:CPU compiles for the host that compiles and JAX's key leaves the host out, so a home directory
    two classes share would hand one a program that does not link there: Dew's default directory is the
    class's own. A directory JAX is configured with is used as it is."""
    configured = jax.config.jax_compilation_cache_dir
    try:
        jax.config.update("jax_compilation_cache_dir", None)
        homes = {}
        for named in ("AuthenticAMD-25-17", "AuthenticAMD-25-1"):
            monkeypatch.setattr(dew.cache, "cpu_class", lambda named=named: named)
            homes[named] = dew.cache.default_compilation_cache_dir()
        jax.config.update("jax_compilation_cache_dir", str(tmp_path / "named"))
        kept = dew.cache.default_compilation_cache_dir()
    finally:
        jax.config.update("jax_compilation_cache_dir", configured)
    assert len(set(homes.values())) == 2
    assert all(home.endswith(f"/{named}") for named, home in homes.items())
    assert kept == str(tmp_path / "named")
