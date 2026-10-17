# Live kernels

The preview Worker calls production's Durable Objects (the pool, the snapshot registry, the preparer and the model hosts) by `script_name`, and the two scripts deploy apart, so an RPC of a shared object is only ever extended: its reply keeps its shape, and a parameter it gains is optional (`test/contract.test.mjs` runs each build against the other's objects).
