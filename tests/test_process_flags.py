"""A run's XLA flags take effect as JAX opens its backend (`dew.devices.flags_hold`): one holds where
XLA would read its value, and a process prepared with flags after its backend opened is refused
(`dew.training.prepare_process`), not appended to no effect (row 1901)."""

import jax
import pytest

from dew.devices import flags_hold
from dew.training import prepare_process


def test_a_flag_holds_as_xla_reads_it_its_last_occurrence_winning(monkeypatch):
    """A run's XLA flag holds where XLA would read its value: a later occurrence of the flag overrides
    an earlier one, and a bare flag reads as true."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true --xla_dump_hlo_as_text "
                                    "--xla_gpu_deterministic_ops=false")
    assert not flags_hold("--xla_gpu_deterministic_ops=true")
    assert flags_hold("--xla_gpu_deterministic_ops=false --xla_dump_hlo_as_text")


def test_flags_set_after_the_backend_opened_are_refused_where_the_process_is_prepared(monkeypatch):
    """XLA reads its flags as JAX opens its backend: once it has, preparing the process with flags
    that do not already hold is refused, not appended to no effect; flags that hold are accepted."""
    jax.devices()
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_hlo_as_text")
    with pytest.raises(ValueError):
        prepare_process(xla_flags="--xla_gpu_deterministic_ops=true")
    assert flags_hold("--xla_dump_hlo_as_text") and not flags_hold("--xla_gpu_deterministic_ops=true")
