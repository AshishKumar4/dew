import itertools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Aux, Dataset, Field, InputSpec, Objective, Trainer


class Regression(Objective):
    model = nn.Dense(1)
    inputs = InputSpec(Field("x", (1,)))

    def init(self, key, variables=None):
        return self.model.init(key, jnp.ones((1, 1)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        loss = jnp.mean((prediction - batch["y"]) ** 2)
        return loss, Aux(metrics={"mse": loss})


x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
batch = {"x": x, "y": 2 * x + 1}
data = Dataset(train=lambda partition: itertools.repeat(batch),
               val=None, records=32, batch=32)
objective = Regression()
trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
print(objective.model.apply(state.params, jnp.array([[0.0], [1.0]])))
