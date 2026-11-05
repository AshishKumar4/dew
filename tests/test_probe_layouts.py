import jax
import numpy as np
import pytest
from test_inference import make_lm_run

import dew


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("spec", [{"sequence": 2}, {"fsdp": 2, "stage": 2}, {"fsdp": 4}])
def test_probe(tmp_path, spec):
    from dew.inference import Server
    from dew.sampling import Sampling
    from dew.training import Layout, MeshSpec

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run, ema_decay=None, fields={"num_layers": 2})
    layout = Layout(min_shard=1, tolerance=1.0)
    sampling = Sampling(temperature=1.0, eos_token_ids=255)
    devices = jax.devices()[:4]
    alone = dew.pipeline(str(run), mesh=MeshSpec().build(devices[:1]), layout=layout)
    placed = dew.pipeline(str(run), mesh=MeshSpec(**spec).build(devices), layout=layout)
    a = alone("the ", 4, key=3, sampling=sampling).host()
    b = placed("the ", 4, key=3, sampling=sampling).host()
    server = Server.from_task(placed, slots=2, capacity=16)
    t = server.submit("the ", 4, key=3)
    server.run()
    c = t.result().host()
    raise AssertionError(repr((a.tokens, b.tokens, c.tokens, a.raw_log_probs, b.raw_log_probs, c.raw_log_probs)))
