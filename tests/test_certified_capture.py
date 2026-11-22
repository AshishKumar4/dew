"""A launch certified aside is certified from its own call (`report.capture`): its program's trace traces it
once, into the jaxpr certification replays and compiles beside it, and the program it lowers is the one it
lowers with no certification pending (row 8720)."""

from process_support import fresh, keeping_picks

CAPTURED_STEP = """
import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from dew.nn.kernels import tuning
from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.templates import report

traced = []


@jax.custom_vjp
def scaled(x, w):
    return x * w


scaled.defvjp(lambda x, w: (x * w, (x, w)), lambda kept, grad: (grad * kept[1], grad * kept[0]))


@dataclasses.dataclass(frozen=True)
class Scaled(tuning.KernelOp):
    HOOKS = ("hook",)
    name = "scaled"
    config_cls = int

    def _fwd(self, x, w, hook, *, config, return_residuals=False):
        traced.append(config)
        return hook(scaled(x, w)), None

    def _get_heuristics_config(self, ba):
        return 1

    def _get_autotuning_configs(self, ba):
        return {1}

    def identified(self, name, value):
        return name


placed, mode = sys.argv[1], sys.argv[2]


def loss(x, w, t):
    # Its hook closes over an operand of the step's own trace (`closed`), which no launch traced alone holds;
    # else over a constant.
    held = t.sum() if mode == "closed" else np.float32(2.0)
    return Scaled()(x, w, lambda y: y * held).sum()


if placed == "sharded":
    mesh = Mesh(np.array(jax.devices()[:4]), ("d",))
    gradient = jax.value_and_grad(lambda x, w, t: jax.lax.psum(loss(x, w, t), "d"), argnums=(0, 1))
    sharded = PartitionSpec("d")
    step = jax.jit(jax.shard_map(gradient, mesh=mesh, in_specs=(sharded, sharded, PartitionSpec()),
                                 out_specs=(PartitionSpec(), (sharded, sharded))))
else:
    mesh = None
    step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
shapes = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in ((8, 16), (8, 16), (2,))]
with report.certifying(compile_only_gpus("sm90")[0]):
    if mesh is not None:
        jax.set_mesh(mesh)
    if mode == "certified":
        text = step.trace(*shapes).lower().as_text()
    else:
        # Certified aside as a compile's lowering is (`OuterProgram._lowered_aside`), settled at its end.
        with report.certifying_aside() as aside:
            text = step.trace(*shapes).lower().as_text()
        report.settled_aside(aside)
        assert not report.refused_launches()
    found = [[each.op, each.stored] for each in report.certifications()]
print(json.dumps({"text": text, "traced": len(traced), "certified": found,
                  "uncertified": [str(why) for why in report.uncertified().values()]}))
"""
"""A launch of a custom derivative in a step's gradient traced and lowered with each launch certified on a
compile-only H100: aside, in a block a compile's lowering certifies in (`report.certifying_aside`;
`captured`, and `closed`, its hook closing over an operand of the step's trace, not a constant), or in turn
(`certified`); whole, or a shard's inside a
shard_map of four devices (`sharded`). The program's text, how many times the launch was traced, and what
was certified."""


def stepped(tmp_path, placed: str, mode: str) -> dict:
    """`CAPTURED_STEP` in a process of its own, its cache and the verdicts kept beside it its own: no
    process reads another's verdict."""
    return fresh(CAPTURED_STEP, placed, mode, platforms="cpu", cache_dir=tmp_path / f"{placed}-{mode}",
                 flags=f"{keeping_picks()} --xla_force_host_platform_device_count=4")


def test_a_launch_certified_aside_is_traced_once_and_its_program_is_the_one_it_lowers_unpending(tmp_path):
    """A step's launch certified aside (`report.capture`) is traced once, its program's trace's call
    replayed for certification, where certifying it in turn traces it again alone (`_lowered_launch`); its
    program, its gradient taken through the launch's custom derivative, lowers byte for byte as with no
    certification pending, whole and as a shard inside a shard_map; and it is certified either way. A hook
    closing over an operand of the step's own trace, which no launch traced alone could read, is certified
    aside, its operand replayed as an argument."""
    for placed in ("whole", "sharded"):
        captured, certified, closed = (stepped(tmp_path, placed, mode)
                                       for mode in ("captured", "certified", "closed"))
        assert captured["text"] == certified["text"], placed
        assert (captured["traced"], certified["traced"], closed["traced"]) == (1, 2, 1), placed
        for found in (captured, certified, closed):
            assert found["certified"] == [["Scaled", False]] and not found["uncertified"], (placed, found)


TEMPLATE_GRADIENT = """
import hashlib
import json
import sys

import jax
import jax.numpy as jnp
from kernel_compile import lowered_for

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.templates import report

# A shape whose derived configs all hold on an H100: no refusal steps a launch in turn that aside takes as
# holding until its program's compile settles it, which a lowering alone never does.
shapes = [jax.ShapeDtypeStruct(shape, jnp.bfloat16)
          for shape in ((1, 512, 4, 64), (1, 512, 2, 64), (1, 512, 2, 64))]


def loss(query, key, value):
    attended = scaled_dot_product_attention(query, key, value, causal=True, implementation="template")
    return attended.astype(jnp.float32).sum()


step = jax.value_and_grad(loss, argnums=(0, 1, 2))
with report.certifying(compile_only_gpus("sm90")[0]):
    if sys.argv[1] == "captured":
        with report.certifying_aside() as aside:
            text = lowered_for(step, shapes, "sm90").as_text()
        report.settled_aside(aside)
        assert not report.refused_launches()
    else:
        text = lowered_for(step, shapes, "sm90").as_text()
    found = sorted([each.op, each.pulled, sorted((name, held[0]) for name, held in each.kernels.items())]
                   for each in report.certifications())
print(json.dumps({"text": hashlib.sha256(text.encode()).hexdigest(), "certified": found,
                  "uncertified": [str(why) for why in report.uncertified().values()]}))
"""
"""An H100's attention template's gradient (its launches' custom derivatives) at a shape whose configs all
hold, each launch certified on a compile-only H100 aside (`captured`) or in turn: its program's text's digest
and each certification's op and kernels' modules."""


def test_a_templates_gradient_certified_aside_lowers_as_certified_in_turn_and_certifies_its_kernels(tmp_path):
    """An attention template's forward and backward launches certified aside from the step's own calls
    (`report.capture`) lower the step byte for byte as certifying each in turn does, and certify the same
    kernels, by the same modules: the replay is the launch its program lowers. At a shape no config is
    refused at: a refused one is stepped from in turn at the trace, aside at its program's compile."""
    captured, certified = (fresh(TEMPLATE_GRADIENT, mode, platforms="cpu", cache_dir=tmp_path / mode,
                                 flags=keeping_picks()) for mode in ("captured", "certified"))
    assert captured["text"] == certified["text"]
    assert captured["certified"] and captured["certified"] == certified["certified"], (captured, certified)
    assert not captured["uncertified"], captured["uncertified"]


PROGRAM_LOWERING = """
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.generation import assumed, compile_only_gpus, gpu_target
from dew.nn.kernels.options import owned_program
from dew.nn.kernels.templates import report
from dew.training.runtime import ready_process

events = {"hit": 0, "miss": 0}


def heard(event, *args, **kwargs):
    kind = event.rsplit("/", 1)[-1]
    if "/pallas/mosaic_gpu/lowering_cache/" in event and kind in events:
        events[kind] += 1


jax.monitoring.register_event_listener(heard)
# Readied as a run is, its Mosaic GPU lowerings kept beside its compilation cache (`_cache.keep_lowerings`).
ready_process()
mesh = Mesh(np.array(compile_only_gpus("sm90")), ("devices",))
shapes = [jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=NamedSharding(mesh, PartitionSpec()))
          for shape in ((1, 512, 4, 64), (1, 512, 2, 64), (1, 512, 2, 64))]


def loss(query, key, value):
    attended = scaled_dot_product_attention(query, key, value, causal=True, implementation="template")
    return attended.astype(jnp.float32).sum()


step = owned_program(jax.value_and_grad(loss, argnums=(0, 1, 2)))
# On the step's mesh, as a run's trace is (`jax.set_mesh`), which its replays are lowered on.
with (report.certifying(compile_only_gpus("sm90")[0]), assumed("sm90"), jax.set_mesh(mesh),
      mgpu_core.artificial_shared_memory_limit(gpu_target("sm90").shared_limit)):
    # Traced as its compile traces it, each launch certified aside and settled before it lowers: the
    # replays lowered.
    with report.certifying_aside() as aside:
        traced = step.trace(*shapes)
    report.settled_aside(aside)
    replayed = dict(events)
    traced.lower()
print(json.dumps({"replayed": replayed, "lowered": {kind: events[kind] - replayed[kind] for kind in events}}))
"""
"""An H100's attention template's gradient through Dew's compile point on a compile-only H100, its launches
certified aside from its trace's calls (`report.capture`): the Mosaic GPU lowering cache's misses and hits
over its trace's replays and over its program's lowering after them."""


def test_a_programs_kernels_certified_aside_are_lowered_once(tmp_path):
    """A program's kernels certified aside are lowered once: each replay's lowering (`replay.replayed`) is
    the one its program's lowering reads back from the Mosaic GPU lowering cache, under the same JAX
    settings, the replay's thread taking the trace's own (`_cache.carried`), its default device among them
    (row 11803: a replay set its device as JAX's default, which the program's lowering on the trace's thread
    left unset, so each of an every-template decoder's 44 kernels was lowered again, a third of its
    program's lowering)."""
    found = fresh(PROGRAM_LOWERING, platforms="cpu", cache_dir=tmp_path / "cache", flags=keeping_picks())
    assert found["replayed"]["miss"], found
    assert found["lowered"]["miss"] == 0 and found["lowered"]["hit"], found
