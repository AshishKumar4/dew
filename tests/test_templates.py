"""dew.nn.kernels.templates.base: a template's configs are the ones a block of the device holds, its
heuristic its cost model's fastest of them, and a call's implementation the one estimated fastest."""

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from dew.nn.kernels import generation
from dew.nn.kernels.templates.base import Template, Unfitted, choose


@dataclasses.dataclass(frozen=True, order=True)
class Block:
    rows: int


@dataclasses.dataclass(frozen=True)
class Doubling(Template):
    """A launch over `x [rows, width]` whose program stages `rows` of fp32 rows and costs `seconds`."""

    config_cls: ClassVar[type[Block]] = Block
    ROWS: ClassVar[tuple[str, ...]] = ("x",)
    name: ClassVar[str] = "doubling"
    seconds: dict = dataclasses.field(default_factory=dict, hash=False, compare=False)

    def _fwd(self, x, *, config, return_residuals=False):
        return x * 2, None

    def space(self, ba):
        return [Block(rows) for rows in (8, 16, 32, 64)]

    def staged_bytes(self, ba, config):
        return config.rows * ba.arguments["x"].shape[1] * 4

    def cost(self, ba, config, target):
        return self.seconds.get(config.rows, 1.0)


def bound(op, width=512):
    return op.bind(jax.ShapeDtypeStruct((4096, width), jnp.float32))


def test_a_templates_configs_are_the_ones_a_block_of_the_device_holds():
    """At 512 fp32 columns 64 rows stage 128 KiB: past an A10's 99 KiB block, within an A100's 163."""
    op = Doubling()
    with generation.assumed("sm86"):
        assert op._get_autotuning_configs(bound(op)) == {Block(8), Block(16), Block(32)}
    with generation.assumed("sm80"):
        assert op._get_autotuning_configs(bound(op)) == {Block(8), Block(16), Block(32), Block(64)}


def test_a_templates_heuristic_is_its_fastest_fitting_config_the_first_of_equals():
    """The cost model's fastest config that fits, where several tie the first of the space's order, so
    the pick never rides a set's iteration."""
    fewer = Doubling(seconds={8: 4.0, 16: 3.0, 32: 2.0, 64: 1.0})
    with generation.assumed("sm86"):
        assert fewer._get_heuristics_config(bound(fewer)) == Block(32)
        assert Doubling()._get_heuristics_config(bound(Doubling())) == Block(8)
        assert Doubling(seconds={16: 0.5, 32: 0.5})._get_heuristics_config(bound(Doubling())) == Block(16)
    with generation.assumed("sm80"):
        assert fewer._get_heuristics_config(bound(fewer)) == Block(64)


def test_a_target_that_reports_no_block_takes_the_least_any_gpu_holds():
    """A GPU generation the table holds no facts for, and that is not this device, budgets 99 KiB; a
    CPU interprets the kernels as an A100, so budgets an A100's block."""
    assert generation.shared_budget(generation.GpuTarget("sm100", "NVIDIA B200")) == generation.SHARED_BUDGET
    assert generation.shared_budget(generation.TARGETS["sm90"]) == generation.TARGETS["sm90"].shared_limit
    sm80 = generation.TARGETS["sm80"].shared_limit
    assert generation.shared_budget(generation.GpuTarget("cpu", "cpu")) == sm80


def test_a_call_no_config_holds_is_refused_naming_its_bytes_and_the_budget():
    op = Doubling()
    with generation.assumed("sm86"), pytest.raises(Unfitted) as refused:
        op._get_heuristics_config(bound(op, width=8192))
    assert (refused.value.least, refused.value.budget, refused.value.generation) == (
        262144, 99 * 1024, "sm86")


@dataclasses.dataclass(frozen=True)
class Slow(Doubling):
    name: ClassVar[str] = "slow"

    def cost(self, ba, config, target):
        return 10.0


@dataclasses.dataclass(frozen=True)
class Hopper(Doubling):
    name: ClassVar[str] = "hopper"

    def eligible(self, ba, target):
        return target.wgmma and ba.arguments["x"].dtype == jnp.bfloat16

    def cost(self, ba, config, target):
        return 0.1


def test_a_call_runs_the_implementation_estimated_fastest_on_the_device_or_the_one_chosen():
    """Among the implementations that run the call on the device, the one whose fastest fitting
    config is estimated fewest seconds, reported as the call's route; `generation.chosen` names
    another, and one that does not run the call there is refused. Hopper's runs bf16 on sm90 alone."""
    x = jax.ShapeDtypeStruct((4096, 512), jnp.float32)
    implementations = [Slow(), Doubling(), Hopper()]
    with generation.assumed("sm80"), generation.routes() as found:
        assert type(choose("decode", implementations, x)) is Doubling
        with generation.chosen(decode="slow"):
            assert type(choose("decode", implementations, x)) is Slow
        with generation.chosen(decode="hopper"), pytest.raises(ValueError):
            choose("decode", implementations, x)
    assert [(route.kernel, route.preferred) for route in found] == [
        ("doubling", "doubling"), ("slow", "doubling")]
    with generation.assumed("sm90"):
        assert type(choose("decode", implementations, x)) is Doubling
        assert type(choose("decode", implementations, x.update(dtype=jnp.bfloat16))) is Hopper
