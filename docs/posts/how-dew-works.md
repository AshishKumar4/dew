# How Dew works, from a model to kernels on the GPU

A new page, [How Dew works](../design/architecture.md), follows one training run through Dew, from the Flax model and data you write down to the kernels that run on the GPU, and checks each step against the code.

It covers the parts Dew built itself. The algebra engine rewrites a traced program's math into a cheaper equal form at compile time, using an e-graph and a price for each form. The template engine generates each GPU kernel from the device's own facts, with its math, its main loop and its architecture-specific steps kept apart. The page also covers how a layer's call finds its kernel, how a kernel becomes PTX and then a cubin inside XLA's executable, and how the accuracy gate holds every kernel to a float64 truth.

Where the code and the design specs disagree, the page follows the code and says so, and a table at the end gives the state of each part: on by default, built but off, in progress or planned.
