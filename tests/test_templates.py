"""dew.nn.kernels.templates.base: a template's configs are the ones a block of the device holds, its
heuristic its cost model's fastest of them (a fixed space's, as the Triton launches keep), and a call's
implementation, as a selector over them takes it (`dew.nn.kernels.selection`), the first in its order
that runs the call."""

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from dew.nn.kernels import generation
from dew.nn.kernels.selection import Kernel, KernelRefused, Op, Requirement, select
from dew.nn.kernels.templates.base import Call, Template, Unfitted, call


@dataclasses.dataclass(frozen=True, order=True)
class Block:
    rows: int


@dataclasses.dataclass(frozen=True)
class Doubling(Template):
    """A launch over `x [rows, width]` whose program stages `rows` of fp32 rows and costs `seconds`."""

    config_cls: ClassVar[type[Block]] = Block
    name: ClassVar[str] = "doubling"
    seconds: dict = dataclasses.field(default_factory=dict, hash=False, compare=False)

    def _fwd(self, x, *, config, return_residuals=False):
        return x * 2, None

    def space(self, ba):
        return [Block(rows) for rows in (8, 16, 32, 64)]

    def staged_bytes(self, ba, config, target):
        return config.rows * ba.arguments["x"].shape[1] * 4

    def cost(self, ba, config, target):
        return self.seconds.get(config.rows, 1.0)


def bound(op, width=512):
    return op.bind(jax.ShapeDtypeStruct((4096, width), jnp.float32))


def test_a_templates_configs_are_the_ones_a_block_of_the_device_holds():
    """At 512 fp32 columns 64 rows stage 128 KiB: past an A10's 99 KiB block, within an A100's 163."""
    op = Doubling()
    with generation.assumed("sm86"):
        assert op.fitting(bound(op), generation.gpu_target()) == (Block(8), Block(16), Block(32))
    with generation.assumed("sm80"):
        assert op.fitting(bound(op), generation.gpu_target()) == (Block(8), Block(16), Block(32), Block(64))


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


@dataclasses.dataclass(frozen=True)
class Unfitting(Doubling):
    """An implementation whose every program stages more than any block holds."""

    name: ClassVar[str] = "unfitting"

    def staged_bytes(self, ba, config, target):
        return 1 << 30


@dataclasses.dataclass(frozen=True)
class Unpriced(Doubling):
    """An implementation with no cost facts on its target, as XLA on a TPU."""

    name: ClassVar[str] = "unpriced"

    def cost(self, ba, config, target):
        raise ValueError("no cost facts")


def choose(implementations, x) -> Template:
    """The implementation of `implementations` a selector over them runs for a call on `x`: each its
    kernel where it runs the call (`Template.eligible`), in their order."""
    op = Op("decode", tuple(Kernel(
        implementation.name,
        (Requirement[Call]("eligible", lambda case, name=implementation.name: case.eligible(name),
                           "it does not run the call here", row=generation.INHERENT),))
        for implementation in implementations))
    bound = call(implementations, x)
    return bound.implementation(select(op, bound).kernel)


def test_a_call_runs_the_first_implementation_that_runs_it_on_the_device_or_the_one_chosen():
    """Among the implementations that run the call on the device, the first in their order, reported
    as the call's route, whatever their estimates; `generation.chosen` names another, and one that does
    not run the call there is refused, naming why. Hopper's runs bf16 on sm90 alone."""
    x = jax.ShapeDtypeStruct((4096, 512), jnp.float32)
    implementations = [Hopper(), Doubling(), Slow()]
    with generation.assumed("sm80"), generation.routes() as found:
        assert type(choose(implementations, x)) is Doubling
        with generation.chosen(decode="slow"):
            assert type(choose(implementations, x)) is Slow
        with generation.chosen(decode="hopper"), pytest.raises(KernelRefused) as refused:
            choose(implementations, x)
    assert [refusal.code for refusal in refused.value.refusals] == ["eligible"]
    assert [(route.kernel, route.preferred) for route in found] == [
        ("doubling", "doubling"), ("slow", "slow")]
    with generation.assumed("sm80"):
        # The selector estimates none: one whose cost needs facts the target lacks takes its call.
        assert type(choose([Unpriced(), Doubling()], x)) is Unpriced
    with generation.assumed("sm90"):
        assert type(choose(implementations, x)) is Doubling
        assert type(choose(implementations, x.update(dtype=jnp.bfloat16))) is Hopper


def _selected(kernels: dict, generic: tuple = (), named: str | None = None) -> str:
    """The kernel a selector over `kernels` (name to whether it takes the call), the `generic` ones
    (taking every call) after them, runs, `named` as `generation.chosen` names it; a kernel `None`
    names takes no call but by name, as CUTLASS's grouped GEMM (`grouped_matmul.NAMED`)."""
    def requirement(takes):
        if takes is None:
            return Requirement[None]("named", lambda case: False, "", row=generation.INHERENT,
                                     kind="measured")
        return Requirement[None]("takes", lambda case: takes, "", row=generation.INHERENT)

    ops = [Kernel[None](name, (requirement(takes),)) for name, takes in kernels.items()]
    ops += [Kernel[None](name, (), generic=True) for name in generic]
    with generation.assumed("sm90"), generation.chosen(decode=named):
        return select(Op("decode", tuple(ops)), None).kernel


def test_a_selector_runs_the_first_kernel_that_takes_the_call_and_a_named_one_only_by_name():
    """The selector runs the first of its kernels that takes the call, in its fixed order, a generic one
    only where no other does; a kernel kept off every call runs only where named."""
    assert _selected({"a": True, "b": True}) == "a"
    assert _selected({"a": False, "b": True}, ("xla",)) == "b"
    assert _selected({"a": False, "b": False}, ("xla",)) == "xla"
    assert _selected({"named": None, "b": True}, ("xla",)) == "b"
    assert _selected({"named": None}, ("xla",)) == "xla"
    assert _selected({"named": None, "b": True}, ("xla",), named="named") == "named"
