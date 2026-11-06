import dataclasses, sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np
from test_linear_attention import layer, layer_params, FIXTURES
import json
from dew.batch import Entered, coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
reference = dict(np.load(FIXTURES / "gated_delta_net.npz"))
geometry = json.loads((FIXTURES / "config.json").read_text())
module = layer(geometry); variables = layer_params(reference)
width = geometry["hidden_size"]
for lengths, enters in (([5], [0]), ([5, 4], [0, 0]), ([5, 3], [0, 1]), ([5, 3, 2], [0, 1, 1]), ([5, 3, 2, 4, 3], [0, 1, 1, 0, 4])):
    documents = np.concatenate([np.full(l, n, np.int32) for n, l in enumerate(lengths, 1)] + [np.zeros(2, np.int32)])
    tokens = len(documents); valid = documents > 0
    entered = Entered(jnp.asarray(documents[None]), jnp.asarray([enters], jnp.int32))
    c = coordinates(valid=jnp.asarray(valid[None]), segments=jnp.asarray(documents[None]))
    call = MixerCall(AttentionMetadata(coordinates=dataclasses.replace(c, entered=entered)))
    hidden = jnp.asarray(np.random.default_rng(0).normal(size=(1, tokens, width)), jnp.float32)
    got = np.asarray(module.apply(variables, hidden, call))
    begins = np.cumsum([0, *lengths])
    errs = []
    for n, (l, r) in enumerate(zip(lengths, enters)):
        own = hidden[:, begins[n]:begins[n]+l]
        if r == 0:
            want = np.asarray(module.apply(variables, own))
        else:
            sh = hidden[:, begins[r-1]:begins[r-1]+lengths[r-1]]
            want = np.asarray(module.apply(variables, jnp.concatenate([sh, own], 1)))[:, sh.shape[1]:]
        errs.append(float(np.max(np.abs(got[:, begins[n]:begins[n]+l] - want))))
    print(lengths, enters, ["%.2e" % e for e in errs])
