"""A GPU compile's buffers read from its buffer assignment (`dew.training.live_buffers`): XLA:GPU's heap
simulation, which XLA:CPU keeps none of, its scanned stacks, and a compile-only twin's probe of the device's
own compile."""
import jax
import jax.experimental.topologies as topologies
import jax.numpy as jnp
import pytest
from jax import ad_checkpoint
from jax._src.lib import xla_client
from jax.sharding import PartitionSpec
from live_buffers_support import counted, temporaries

from dew.training.live_buffers import live_intervals, peak_live

pytestmark = pytest.mark.gpu(reason="XLA:GPU's buffer assignment: its heap simulation and early exit")


def _step(x, w):
    return jnp.tanh(x @ w) @ w


def test_a_compiles_intervals_hold_its_peak_buffers_at_each_allocations_peak():
    """At the first position a trace's charged intervals hold their most bytes, those covering it are the
    buffers `peak_live` names of it, as XLA's own count of its peak reads the same simulation."""
    def step(x, w):
        for _ in range(3):
            x = jnp.tanh(x @ w)
        return x.sum(), x

    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    compiled = jax.jit(jax.grad(step, has_aux=True)).lower(tile, tile).compile()
    intervals, live = live_intervals(compiled), peak_live(compiled)
    assert intervals and all(each.complete for each in intervals)
    peaked = []
    for trace in {each.trace for each in intervals}:
        within = [each for each in intervals if each.trace == trace]
        at = max(range(max(each.end for each in within)), key=lambda at, within=within: counted(within, at))
        peaked += [each for each in within if each.charged and each.start <= at < each.end]
    assert temporaries(peaked) == temporaries(live)


def test_a_scans_stacked_residual_is_named_by_what_its_body_writes():
    """A scan saves each step's named residual into a stacked output, a loop state element whose buffer is
    the zeros a broadcast makes (row 13304): that buffer, live at the gradient's peak, is named by the
    residual the body writes into it."""
    def step(xs, w):
        def body(carry, x):
            hidden = ad_checkpoint.checkpoint_name(jnp.tanh(x @ w), "hidden")
            return carry + jnp.sum(jnp.sin(hidden)), None

        saved = jax.checkpoint(body, policy=jax.checkpoint_policies.save_only_these_names("hidden"))
        return jax.lax.scan(saved, 0.0, xs)[0]

    rows = jax.ShapeDtypeStruct((16, 512, 256), jnp.float32)
    weight = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    live = peak_live(jax.jit(jax.grad(step, argnums=1)).lower(rows, weight).compile())
    assert live is not None
    stacked = [buffer for buffer in live if buffer.shape == (16, 512, 256)]
    assert ("hidden",) in {buffer.names for buffer in stacked}, stacked


def test_a_probe_on_the_devices_twin_states_the_devices_own_compile():
    """A compile for a compile-only twin of the attached device, its target config the device's own, so jax
    compiles it through the device's client, whose executor autotunes as the device's own compile does,
    stopping after buffer assignment: it states the memory and the buffers live at its peak as the device's
    own whole compile of the program does (the probe `dew.training.memory`'s admission reads)."""
    devices = jax.devices()[:1]
    topology = topologies.get_topology_desc(
        platform="cuda", target_config=xla_client.get_topology_for_devices(devices).target_config,
        topology="1x1x1")
    twin = jax.sharding.NamedSharding(topologies.make_mesh(topology, (1,), ("x",)), PartitionSpec())
    def tiles(sharding) -> list[jax.ShapeDtypeStruct]:
        return [jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=sharding)] * 2

    probe = jax.jit(_step).lower(*tiles(twin)).compile(
        {"xla_gpu_experimental_early_exit": "EARLY_EXIT_POINT_AFTER_BUFFER_ASSIGNMENT"})
    attached = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, ("x",)), PartitionSpec())
    whole = jax.jit(_step).lower(*tiles(attached)).compile()
    assert probe.memory_analysis().temp_size_in_bytes == whole.memory_analysis().temp_size_in_bytes
    assert peak_live(whole) is not None
    assert peak_live(probe) == peak_live(whole)
