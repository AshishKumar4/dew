"""dew.nn.kernels.templates.base: a template's configs are the ones a block of the device holds, its
heuristic its cost model's fastest of them, and a call's implementation, as a selector over them ranks
it (`dew.nn.kernels.selection`), the one estimated fastest."""

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
    ROWS: ClassVar[tuple[str, ...]] = ("x",)
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
    kernel where it runs the call (`Template.eligible`), costed by its estimate (`Call.seconds`)."""
    op = Op("decode", tuple(Kernel(
        implementation.name,
        (Requirement[Call]("eligible", lambda case, name=implementation.name: case.eligible(name),
                           "it does not run the call here", row=generation.INHERENT),),
        cost=lambda case, name=implementation.name: case.seconds(name))
        for implementation in implementations))
    bound = call(implementations, x)
    return bound.implementation(select(op, bound).kernel)


def test_a_call_runs_the_implementation_estimated_fastest_on_the_device_or_the_one_chosen():
    """Among the implementations that run the call on the device, the one whose fastest fitting
    config is estimated fewest seconds, reported as the call's route; `generation.chosen` names
    another, and one that does not run the call there is refused, naming why. Hopper's runs bf16 on sm90
    alone."""
    x = jax.ShapeDtypeStruct((4096, 512), jnp.float32)
    implementations = [Slow(), Doubling(), Hopper()]
    with generation.assumed("sm80"), generation.routes() as found:
        assert type(choose(implementations, x)) is Doubling
        with generation.chosen(decode="slow"):
            assert type(choose(implementations, x)) is Slow
        with generation.chosen(decode="hopper"), pytest.raises(KernelRefused) as refused:
            choose(implementations, x)
    assert [refusal.code for refusal in refused.value.refusals] == ["eligible"]
    assert [(route.kernel, route.preferred) for route in found] == [
        ("doubling", "doubling"), ("slow", "slow")]
    with generation.assumed("sm80"), generation.chosen(decode="doubling"):
        # A chosen implementation estimates no other: one whose cost needs facts the target lacks, or
        # whose programs fit no block of it, leaves the chosen one its call.
        assert type(choose([Unpriced(), Doubling()], x)) is Doubling
        assert type(choose([Unfitting(), Doubling()], x)) is Doubling
    with generation.assumed("sm90"):
        assert type(choose(implementations, x)) is Doubling
        assert type(choose([Unpriced()], x)) is Unpriced
        assert type(choose(implementations, x.update(dtype=jnp.bfloat16))) is Hopper


def _ranked(costs: dict, generic: dict | None = None, named: str | None = None) -> str:
    """The kernel a selector over kernels of `costs` (each taking the call; None states no cost) runs,
    the `generic` ones (name to cost) after them, `named` as `generation.chosen` names it."""
    takes = (Requirement[None]("takes", lambda case: True, "", row=generation.INHERENT),)

    def priced(cost):
        return None if cost is None else (lambda case: cost)

    kernels = [Kernel[None](name, takes, cost=priced(cost)) for name, cost in costs.items()]
    kernels += [Kernel[None](name, takes, generic=True, cost=priced(cost))
                for name, cost in (generic or {}).items()]
    with generation.assumed("sm90"), generation.chosen(decode=named):
        return select(Op("decode", tuple(kernels)), None).kernel


def test_a_selector_ranks_every_costed_kernel_and_runs_an_unqualified_one_only_by_name():
    """Where each kernel that takes the call states a cost, the selector runs the cheapest, a generic
    kernel stating one among them (XLA's program beating a launch); a kernel costing no finite seconds,
    which no measurement qualified, runs only where named, the call handed to the generic kernel where no
    other qualifies; an op whose kernels do not all state costs keeps its declared order whole."""
    assert _ranked({"a": 2.0, "b": 1.0}) == "b"
    assert _ranked({"a": 2.0, "b": 1.0}, {"xla": 0.5}) == "xla"
    assert _ranked({"a": 2.0, "b": 2.0}) == "a"
    assert _ranked({"a": float("inf"), "b": 3.0}, {"xla": None}) == "b"
    assert _ranked({"a": float("inf"), "b": float("inf")}, {"xla": None}) == "xla"
    assert _ranked({"a": float("inf"), "b": 3.0}, named="a") == "a"
    assert _ranked({"a": None, "b": 1.0}, {"xla": 0.5}) == "a"
