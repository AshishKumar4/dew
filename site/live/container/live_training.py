"""Bounds a live training cell to a couple of minutes on a shared 4-vCPU host.

A training context runs Dew itself, not the stand-ins of model_client.
`install` has IPython run each cell with its calls' callees passed through
`capped`, whatever name it calls them by: a preset of `dew.data.presets`, called or given
to a dataset's `pipe`, and a dataset's own `batch(n)` make batches of at most
BATCH rows, and a Dew `fit` runs at most STEPS steps, logging at least four
times. Each cap prints what it
changed before the run's own output, so the cell's first lines say how this
run differs from the code on the page.
"""

import ast
import inspect

BATCH, STEPS = 8, 20


def batch(asked):
    """The batch a cell asked for, lowered to BATCH; a token budget's batch is not counted in rows."""
    if (asked or 0) > BATCH:
        print(f"Live run: batch {asked} -> {BATCH}, so it fits a shared 4-vCPU host.")
        return BATCH
    return asked


def fitted(steps, log_every):
    """A `fit` call's steps and log_every: more than STEPS steps lowered to STEPS, logging at least four
    times."""
    changed, cut = [], steps > STEPS
    if cut:
        changed.append(f"steps {steps} -> {STEPS}")
        steps = STEPS
    if log_every > max(1, steps // 4):
        changed.append(f"log_every {log_every} -> {max(1, steps // 4)}")
        log_every = max(1, steps // 4)
    if changed:
        why = "so it finishes in about two minutes" if cut else "to log the run as it goes"
        print(f"Live run: {', '.join(changed)}, {why}.")
    return {"steps": steps, "log_every": log_every}


def capped(call):
    """What a cell's call of `call` reaches: `call` itself, so the call runs in the cell's own frame (a
    `locals()` or an `exec` reads the cell's, however it is named), unless `call` is a Dew function this
    caps, then one that caps its arguments first: a preset's batch (`batch`), called or through a
    dataset's `pipe`, a dataset's `batch(n)` rows, a `fit`'s steps and log_every, the latter its own
    default where the cell names none (`fitted`). Whatever name the cell calls it by, and however it
    passes them."""
    module = getattr(call, "__module__", None) or ""
    name = getattr(call, "__name__", "")
    capping_one = module == "dew.data.presets" or name in ("pipe", "batch", "fit")
    if module.split(".")[0] != "dew" or not capping_one:
        return call

    def capping(*args, **options):
        if (module == "dew.data.presets" or name == "pipe") and "batch" in options:
            options["batch"] = batch(options["batch"])
        elif module == "dew.data.pipeline" and name == "batch":
            if args:
                args = (batch(args[0]), *args[1:])
            elif "size" in options:
                options["size"] = batch(options["size"])
        elif name == "fit" and "steps" in options:
            default = inspect.signature(call).parameters["log_every"].default
            options.update(fitted(options["steps"], options.get("log_every", default)))
        return call(*args, **options)

    return capping


class Capped(ast.NodeTransformer):
    """A cell with each of its calls' callees passed through `capped` first, as IPython runs every cell of
    a training context (`install`): which function a name holds is known only as the cell runs."""

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        # This module, as a cell reaches it without binding a name of its own.
        module = ast.Call(ast.Name("__import__", ast.Load()), [ast.Constant("live_training")], [])
        callee = ast.Call(ast.Attribute(module, "capped", ast.Load()), [node.func], [])
        return ast.Call(callee, node.args, node.keywords)


def install():
    """Cap every later cell of this kernel: IPython transforms a cell before it runs any of it, so the
    cell that installs this is not one of them."""
    from IPython import get_ipython

    get_ipython().ast_transformers.append(Capped())
