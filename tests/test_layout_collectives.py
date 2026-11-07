"""A layout's CPU collective can run after its judge's concurrent draws."""

import subprocess
import sys


def test_a_layouts_collective_has_its_gang_after_the_draws_ahead():
    from process_support import worker_env

    environment = worker_env(8)
    environment["PJRT_NPROC"] = "8"
    done = subprocess.run([sys.executable, __file__, "worker"], env=environment,
                          capture_output=True, text=True, timeout=45, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def _worker(hold: float = 8):
    import threading

    import jax
    import jax.numpy as jnp
    import numpy as np
    from tool_support import load

    jax.config.update("jax_enable_x64", True)
    tool = load("layout_parity")
    _, case = tool.model_case("jepa", "float32", {}, {})
    armed, released = threading.Event(), threading.Event()
    entered = (threading.Event(), threading.Event())

    def held(index, _value):
        if armed.is_set():
            entered[int(index)].set()
            released.wait()

    @jax.jit
    def pending(matrix):
        return jax.lax.fori_loop(0, 100, lambda _, value: value @ matrix, matrix)

    @jax.jit
    def draw(index, matrix):
        jax.debug.callback(held, index, matrix[0, 0])
        return matrix + 1

    def exchange(value):
        return jax.lax.ppermute(value, "devices", [(i, (i + 1) % 8) for i in range(8)])

    collective = jax.pmap(exchange, axis_name="devices")
    matrices = [jax.device_put(np.eye(512, dtype=dtype)) for dtype in (np.float32, np.float64)]
    indices = [jax.device_put(np.asarray(i, np.int32)) for i in range(2)]
    values = jax.device_put(np.arange(8, dtype=np.int32))
    for index, matrix in zip(indices, matrices, strict=True):
        pending(matrix).block_until_ready()
        draw(index, matrix).block_until_ready()
    collective(values).block_until_ready()

    judge = tool.Judge(case, tool.References(), "a collective after the draws ahead")
    judge.AHEAD = 1

    def reference(seed):
        matrix = pending(matrices[seed])
        draw(indices[seed], matrix).block_until_ready()
        return tool.Drawn(0.0, {}, {})

    judge._drawn_reference = reference
    judge._drawn_anchor = lambda seed, witnessed: tool.Drawn(0.0, {}, {})

    def release_draws():
        assert all(event.wait(15) for event in entered)
        # Keep both programs in flight while the caller tries to finish ahead(). The eight-device
        # collective needs the whole client pool, so its caller must reach it after these finish.
        threading.Event().wait(hold)
        released.set()

    releaser = threading.Thread(target=release_draws, daemon=True)
    releaser.start()
    armed.set()
    judge.ahead(0)
    assert all(event.wait(15) for event in entered)
    assert released.is_set()
    actual = np.asarray(collective(values))
    np.testing.assert_array_equal(actual, np.asarray([7, 0, 1, 2, 3, 4, 5, 6], np.int32))
    released.set()
    releaser.join()
    judge._pool.shutdown()


if __name__ == "__main__":
    assert sys.argv[1:] == ["worker"]
    _worker()
